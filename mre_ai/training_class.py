#!/usr/bin/env python

import time
import copy
import logging
from pathlib import Path
import warnings
import argparse
from collections import defaultdict
import pickle as pkl
import numpy as np
from itertools import chain
import xarray as xr
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data.sampler import RandomSampler
from torchsummary import summary
from tensorboardX import SummaryWriter
from mre_ai.wave_pred import WaveDataset, HyperNetworkDataset
from mre_ai.pytorch_arch_deeplab_2d import DeepLab2D
from mre_ai.pytorch_arch_siren import CNNSirenHypernet
from mre_ai.phantom import PhantomDataset
import holoviews as hv
from holoviews import opts
from holoviews.operation.datashader import datashade, shade, dynspread, rasterize
from pyviz_med.visual import PyPatient


class ModelTrainer:
    '''Main class for training an ML model via pytorch.  This class requires a kwarg
    dictionary to set all the needed parameters.'''

    def __init__(self, cfg_dict):

        self.default_cfg()
        self.process_kwargs(cfg_dict)
        self.set_random_seeds()
        self.load_and_split_datasets()
        self.init_dataloaders()
        self.load_model_and_device()

    def train(self):
        '''Run the main training loop.'''

        if self.cfg['dry_run']:
            self.dry_run_output()

        else:
            self.init_optimizer()
            self.init_loss()
            self.init_output()
            self.train_model()
            self.eval_model()
            self.write_outputs()

    def default_cfg(self):
        '''Set the default values for the expected kwargs.'''

        self.cfg = {'input_data': None, 'output_path': None, 'run_version': None, 'verbose_level':
                    logging.ERROR, 'train_list': None, 'val_list': None, 'test_list': None,
                    'test_group_label': None, 'train_trans': True, 'train_aug': True,
                    'train_sample': 'shuffle', 'val_trans': True, 'val_aug': False, 'val_sample':
                    'shuffle', 'test_trans': True, 'test_aug': False, 'seed': 100, 'worker_init_fn':
                    'default', 'batch_size': 50, 'lr': 1e-2, 'step_size': 20, 'gamma': 0.1,
                    'num_epochs': 40, 'dry_run': False, 'loss': 'l2', 'model_arch': 'DeepLab2D',
                    'n_layers': 3, 'in_channels': 1, 'out_channels_final': 1, 'channel_growth':
                    False, 'transfer_layer': False, 'transform': False, 'train_color_aug': False,
                    'val_color_aug': False, 'test_color_aug': False, 'dbc_weight': 1e7,
                    'latent_weight': 1e-1, 'hypo_weight': 1e2, 'eval_label': 'pred', 'DL_Dataset':
                    'PhantomDataset', 'bounds': (0, 5), 'pretrain': None}

    def process_kwargs(self, kwargs):
        '''Replace the default cfg values with ones from the kwargs.'''

        for key in kwargs:
            if key not in self.cfg.keys():
                raise KeyError(f'{key} is not a valid cfg key')
            val = self.str2bool(kwargs[key])
            self.cfg[key] = val

        critical_keys = ['input_data', 'output_path', 'run_version']
        for key in critical_keys:
            if self.cfg[key] is None:
                raise KeyError(f'"{key}" must be specified in kwargs.')

    def str2bool(self, val):
        '''string to bool helper function.'''
        if type(val) is not str:
            return val
        elif val.lower() in ("yes", "true", "t"):
            return True
        elif val.lower() in ("no", "false", "f"):
            return False
        else:
            return val

    def set_random_seeds(self):
        torch.manual_seed(self.cfg['seed'])
        np.random.seed(self.cfg['seed'])

    def load_and_split_datasets(self):
        '''Load up the required datasets and split them into train/val/test.'''
        self.load_main_ds()

        if all(i is None for i in [self.cfg['train_list'], self.cfg['val_list'],
                                   self.cfg['test_list']]):
            self.train_list = [str(a).zfill(6) for a in range(0, 12000)]
            self.val_list = [str(a).zfill(6) for a in range(12000, 16000)]
            self.test_list = [str(a).zfill(6) for a in range(16000, 20000)]

        else:
            self.train_list = self.cfg['train_list']
            self.val_list = self.cfg['val_list']
            self.test_list = self.cfg['test_list']

        if self.cfg['test_group_label'] is None:
            self.cfg['test_group_label'] = 'default_group'

        # This is going to get cumbersome quickly, need a better way to hotswap Datasets
        if self.cfg['DL_Dataset'] == 'PhantomDataset':
            self.train_set = PhantomDataset(self.ds, self.train_list,
                                            transform=self.cfg['train_trans'],
                                            aug=self.cfg['train_aug'],
                                            color_aug=self.cfg['train_color_aug'],
                                            seed=self.cfg['seed'], verbose=self.cfg['dry_run'])
            self.val_set = PhantomDataset(self.ds, self.val_list, transform=self.cfg['val_trans'],
                                          aug=self.cfg['val_aug'],
                                          color_aug=self.cfg['val_color_aug'],
                                          seed=self.cfg['seed'], verbose=self.cfg['dry_run'])
            self.test_set = PhantomDataset(self.ds, self.test_list,
                                           transform=self.cfg['test_trans'],
                                           aug=self.cfg['test_aug'],
                                           color_aug=self.cfg['test_color_aug'],
                                           seed=self.cfg['seed'], verbose=self.cfg['dry_run'])

        elif self.cfg['DL_Dataset'] == 'HyperNetworkDataset':
            self.train_set = HyperNetworkDataset(
                self.ds.sel(subject_id=self.train_list), bounds=self.cfg['bounds'])
            self.val_set = HyperNetworkDataset(
                self.ds.sel(subject_id=self.val_list), bounds=self.cfg['bounds'])
            self.test_set = HyperNetworkDataset(
                self.ds.sel(subject_id=self.test_list), bounds=self.cfg['bounds'])
        else:
            raise ValueError(f'"DL_Dataset" = {self.cfg["DL_Dataset"]} is not recognized')

    def load_main_ds(self):
        '''Load the main xarray Dataset, assign it as a class attribute'''
        input_data = self.cfg['input_data']

        if isinstance(input_data, str):
            input_data = Path(input_data)

        if isinstance(input_data, Path):
            if '*' in input_data.name:
                self.ds = xr.open_mfdataset(input_data.parent.glob(input_data.name)).load()
            elif input_data.suffix != '':
                self.ds = xr.open_dataset(input_data).load()
            else:
                self.ds = xr.open_mfdataset(input_data.glob('*')).load()

        elif isinstance(input_data, xr.Dataset):
            self.ds = input_data.load()
        else:
            raise ValueError(f'"input_data" = {input_data} is not the correct format')

    def init_dataloaders(self):
        '''Initialize the dataloaders using the torch Datasets.'''

        # Set a worker init function for randomizing order of samples in each epoch
        if self.cfg['worker_init_fn'] == 'default':
            worker_init_fn = None
        elif self.cfg['worker_init_fn'] == 'rand_epoch':
            worker_init_fn = rand_worker_init_fn
        else:
            raise ValueError('worker_init_fn specified incorrectly')

        bs = self.cfg['batch_size']
        self.dataloaders = {}
        if self.cfg['train_sample'] == 'shuffle':
            self.dataloaders['train'] = DataLoader(self.train_set, batch_size=bs, shuffle=True,
                                                   num_workers=2, worker_init_fn=worker_init_fn,
                                                   drop_last=False)
        elif self.cfg['train_sample'] == 'resample':
            self.dataloaders['train'] = DataLoader(self.train_set, batch_size=bs, shuffle=False,
                                                   sampler=RandomSampler(
                                                       self.train_set, replacement=True,
                                                       num_samples=self.cfg['train_num_samples']),
                                                   num_workers=2, worker_init_fn=worker_init_fn)
        if self.cfg['val_sample'] == 'shuffle':
            self.dataloaders['val'] = DataLoader(self.val_set, batch_size=bs, shuffle=True,
                                                 num_workers=2, worker_init_fn=worker_init_fn,
                                                 drop_last=False)
        elif self.cfg['val_sample'] == 'resample':
            self.dataloaders['val'] = DataLoader(self.val_set, batch_size=bs, shuffle=False,
                                                 sampler=RandomSampler(
                                                     self.val_set, replacement=True,
                                                     num_samples=self.cfg['val_num_samples']),
                                                 num_workers=2, worker_init_fn=worker_init_fn)
        self.dataloaders['test'] = DataLoader(self.test_set, batch_size=bs,
                                              shuffle=False, num_workers=2,
                                              worker_init_fn=worker_init_fn)

    def load_model_and_device(self):
        '''Initialize the model and send it to the correct device.'''
        # Update this with more model choices
        if self.cfg['model_arch'] == 'DeepLab2D':
            self.model = DeepLab2D(in_channels=1, out_channels=1, output_stride=8, norm='bn')
        elif self.cfg['model_arch'] == 'CNNSirenHypernet':
            self.model = CNNSirenHypernet(in_features=1, out_features=1, image_resolution=256)

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if self.device == 'cpu':
            warnings.warn('Device is running on CPU, not GPU!')

        self.model.to(self.device)

        if self.cfg['pretrain'] is not None:
            print('pretraining')
            model_dict = torch.load(self.cfg['pretrain'], map_location=self.device)
            self.model.load_state_dict(model_dict, strict=True)

    def dry_run_output(self, phase='test'):
        if self.cfg['dry_run']:
            inputs, targets, names = next(iter(self.dataloaders[phase]))
            print(f'{phase} set info:')
            print('inputs', inputs.shape)
            print('targets', targets.shape)
            print('names', names)

            print('Model Summary:')
            # summary(model, input_size=(3, 224, 224))
            dry_size = list(inputs.shape[1:])
            # dry_size[0] = 1
            print(dry_size)
            summary(self.model, input_size=tuple(dry_size))
            return None

    def init_optimizer(self):
        '''Initialize the optimizer and scheduler.'''

        print('initializing optimizer')
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.cfg['lr'])

        # Define optimizer
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer,
                                                   step_size=self.cfg['step_size'],
                                                   gamma=self.cfg['gamma'])

    def init_loss(self):
        if self.cfg['loss'] == 'l2':
            self.loss_func = mse_loss
        elif self.cfg['loss'] == 'hh':
            self.loss_func = hyper_hh_loss

    def init_output(self):
        '''Initialize the output directories for logs and predictions.'''

        # Tensorboardx writer, model, config paths
        print('initializing output directories')
        output_path = self.cfg['output_path']
        self.tb_writer_dir = Path(output_path, 'tb_runs')
        self.config_dir = Path(output_path, 'config')
        self.xr_dir = Path(output_path, 'XR', self.cfg['run_version'])
        self.model_dir = Path(output_path, 'trained_models', self.cfg['test_group_label'])
        self.tb_writer_dir.mkdir(parents=True, exist_ok=True)
        self.config_dir.mkdir(parents=True, exist_ok=True)
        self.xr_dir.mkdir(parents=True, exist_ok=True)
        Path(self.xr_dir, 'test').mkdir(parents=True, exist_ok=True)
        Path(self.xr_dir, 'train').mkdir(parents=True, exist_ok=True)
        Path(self.xr_dir, 'val').mkdir(parents=True, exist_ok=True)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.tb_writer = SummaryWriter(
            str(self.tb_writer_dir) + f'/{self.cfg["run_version"]}_{self.cfg["test_group_label"]}')
        # Model graph is useless without additional tweaks to name layers appropriately
        # writer.add_graph(model, torch.zeros(1, 3, 256, 256).to(device), verbose=True)

    def train_model(self):
        '''Main model training loop.'''

        print('starting model training loop')
        best_model_wts = copy.deepcopy(self.model.state_dict())
        best_loss = 1e16
        phases = ['train', 'val', 'test']

        for epoch in range(self.cfg['num_epochs']):
            # Wrap in try/except block for keyboard interrupt
            try:
                print('Epoch {}/{}'.format(epoch, self.cfg['num_epochs'] - 1))
                print('-' * 10)
                since = time.time()

                # Each epoch has a training, validation, and test phase
                for phase in phases:
                    if phase == 'train':
                        for param_group in self.optimizer.param_groups:
                            print("LR", param_group['lr'])
                        self.model.train()  # Set model to training mode
                    else:
                        self.model.eval()   # Set model to evaluate mode

                    metrics = defaultdict(float)
                    epoch_samples = 0

                    # iterate through batches of data for each epoch in each phase
                    for data in self.dataloaders[phase]:
                        # Switch to dict method a la Siren guys
                        inputs = self.prep_inputs(data)
                        labels = self.prep_targets(data)

                        # zero the parameter gradients
                        self.optimizer.zero_grad()

                        # with torch.set_grad_enabled(phase == 'train'):
                        with torch.set_grad_enabled(True):
                            outputs = self.model(inputs)
                            loss = self.loss_func(outputs, labels, self.cfg, metrics)
                            if phase == 'train':
                                loss.backward()
                                self.optimizer.step()

                        # accrue total number of samples
                        epoch_samples += inputs['image'].size(0)
                    if phase == 'train':
                        self.scheduler.step()

                    print_metrics(metrics, epoch_samples, phase)
                    self.write_to_tb(metrics, epoch_samples, phase, epoch)
                    epoch_loss = metrics['loss'] / epoch_samples

                    # deep copy the model if is it best
                    if phase == 'val' and epoch_loss < best_loss and epoch >= 20:
                        print("saving best model")
                        best_loss = epoch_loss
                        best_model_wts = copy.deepcopy(self.model.state_dict())

                time_elapsed = time.time() - since
                print('{:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

            except KeyboardInterrupt:
                print('Breaking out of training early.')
                break

        self.best_model_wts = best_model_wts
        print('Best val loss: {:4f}'.format(best_loss))

    def eval_model(self):
        '''Evaluate the trained model. Hardcoded right now, will need to improve.
        Generate an xarray object which contains the correct labels and the prediction.
        Need to improve this function a lot to account for different types of outputs.'''

        print('evaluating best model')
        self.model.load_state_dict(self.best_model_wts)
        self.model.eval()   # Set model to evaluate mode
        # iterate through batches of data for each epoch

        self.ds_mem = xr.Dataset(
            {'image': (['subject_id', 'label', 'x', 'y'],
                       np.zeros((self.ds.subject_id.size, 1, self.ds.x.size, self.ds.y.size),
                                dtype=np.float32)),
             'spacing': (['subject_id', 'label', 'img_dims'],
                         np.ones((self.ds.subject_id.size, 1, 2))),
             'origin': (['subject_id', 'label', 'img_dims'],
                        np.zeros((self.ds.subject_id.size, 1, 2))),
             },

            coords={'subject_id': self.ds.subject_id,
                    'label': [self.cfg['eval_label']],
                    'y': self.ds.y,
                    'x': self.ds.x,
                    'img_dims': range(2),
                    }
        )

        phases = ['train', 'val', 'test']
        for phase in phases:
            print(phase)
            for data in self.dataloaders[phase]:
                if self.cfg['eval_label'] == 'wave_pred':
                    # inputs = data['coords'].to(self.device)
                    inputs = self.prep_inputs(data)
                    subject_ids = data['subject_id']
                    for i, name in enumerate(subject_ids):
                        # print('loading pred to mem')
                        prediction = self.model(inputs)['model_out']
                        # prediction = prediction.detach().cpu().numpy().view(256, 256)
                        prediction = prediction.detach().cpu().view(256, 256).numpy()
                        self.ds_mem['image'].loc[
                            {'subject_id': name, 'label': 'wave_pred'}] = prediction.T
                else:
                    inputs = data['coords'].to(self.device)
                    subject_ids = data['subject_id']
                    for i, name in enumerate(subject_ids):
                        # print('loading pred to mem')
                        prediction = self.model(inputs[i:i+1]).data.cpu().numpy()
                        self.ds_mem['image'].loc[{'subject_id': name,
                                                  'label': 'pred'}] = (prediction[0, 0].T*10)

        torch.cuda.empty_cache()

    def write_outputs(self):
        # Write outputs and save model
        test_group_label = self.cfg['test_group_label']
        run_version = self.cfg['run_version']

        config_file = Path(self.config_dir, f'{run_version}_{test_group_label}.pkl')
        with open(config_file, 'wb') as f:
            pkl.dump(self.cfg, f, pkl.HIGHEST_PROTOCOL)

        # self.ds.close()
        ds_test = self.ds_mem.sel(subject_id=self.test_list)
        ds_train = self.ds_mem.sel(subject_id=self.train_list)
        ds_val = self.ds_mem.sel(subject_id=self.val_list)

        ds_val.to_netcdf(Path(self.xr_dir, 'val', f'xarray_pred_{test_group_label}.nc'))
        ds_val.close()
        ds_test.to_netcdf(Path(self.xr_dir, 'test', f'xarray_pred_{test_group_label}.nc'))
        ds_test.close()
        ds_train.to_netcdf(Path(self.xr_dir, 'train', f'xarray_pred_{test_group_label}.nc'))
        ds_train.close()

        self.tb_writer.close()
        torch.save(self.model.state_dict(), str(self.model_dir)+f'/model_{run_version}.pkl')

    def pypat_dataloader(self, phase='train'):
        '''Make a PyPatient of the input images right before they are loaded into the neural
        network.  This is after all augmentations and normalizations are performed.'''
        opts.defaults(
            opts.GridSpace(shared_xaxis=True, shared_yaxis=True),
            opts.Image(cmap='viridis', width=250, height=250, tools=['hover'],
                       xaxis=None, yaxis=None),
            opts.Labels(text_color='white', text_font_size='8pt', text_align='left',
                        text_baseline='bottom'),
            opts.Path(color='white'),
            opts.Spread(width=600),
            opts.Overlay(show_legend=False))

        inputs, targets, names = next(iter(self.dataloaders[phase]))
        inputs = inputs.data.cpu().numpy()
        targets = targets.data.cpu().numpy()
        print(inputs.shape)
        print(targets.shape)
        print(self.ds.image.shape)
        n_missing_labels = self.ds.image.shape[1] - (inputs.shape[1] + targets.shape[1])
        missing_shape = list(inputs.shape)
        missing_shape[1] = n_missing_labels
        print(missing_shape, n_missing_labels)
        missing_img = np.zeros(missing_shape, dtype=inputs.dtype)
        dl_image = np.concatenate([inputs, targets, missing_img], axis=1)
        print(dl_image.shape)

        ds_dl = self.ds.isel(subject_id=slice(0, inputs.shape[0]))
        origin = ds_dl.origin
        spacing = ds_dl.spacing
        ds_dl = ds_dl.copy(data={'image': dl_image, 'origin': origin, 'spacing': spacing})
        print(names)
        ds_dl['subject_id'] = list(names)

        return PyPatient(from_xarray=True, ds=ds_dl)

    def prep_inputs(self, inputs):
        prepped_inputs = {}
        if self.cfg['DL_Dataset'] == 'HyperNetworkDataset':
            prepped_inputs['image'] = inputs['image'].to(self.device)
            prepped_inputs['coords'] = inputs['coords'].to(self.device)

        return prepped_inputs

    def prep_targets(self, targets):
        prepped_targets = {}
        if self.cfg['DL_Dataset'] == 'HyperNetworkDataset':
            prepped_targets['pixels'] = targets['pixels'].to(self.device)

        return prepped_targets

    def write_to_tb(self, metrics, epoch_samples, phase, epoch):
        for loss in metrics:
            self.tb_writer.add_scalar(f'{loss}_{phase}', metrics[loss]/epoch_samples, epoch)


def rand_worker_init_fn(worker_id):
    np.random.seed(torch.random.get_rng_state()[0].item() + worker_id)


def mse_loss(pred, target, metrics):
    pred = pred.contiguous()/1000
    target = target.contiguous()/1000
    mse = (((pred - target)**2)).sum()

    metrics['loss'] += mse.data.cpu().numpy() * target.size(0)
    return mse


def calc_loss(loss_func, pred, target, metrics, lamb=0):

    if loss_func == 'l2':
        loss = mse_loss(pred, target)
        metrics['loss'] += loss.data.cpu().numpy() * target.size(0)
    elif loss_func == 'hh':
        ...
    return loss


def hyper_laplace(y, x, dims=None):
    grad = hyper_gradient(y, x)
    return hyper_divergence(grad, x, dims)


def hyper_divergence(y, x, dims=None):
    if dims is None:
        dims = range(y.shape[-1])
    div = 0.
    for i in dims:
        div += torch.autograd.grad(y[..., i], x, torch.ones_like(y[..., i]),
                                   create_graph=True)[0][..., i:i+1]
    return div


def hyper_gradient(y, x, grad_outputs=None):
    if grad_outputs is None:
        grad_outputs = torch.ones_like(y)
    grad = torch.autograd.grad(y, [x], grad_outputs=grad_outputs, create_graph=True)[0]
    return grad


def latent_loss(model_output):
    return torch.mean(model_output['latent_vec'] ** 2)


def hypo_weight_loss(model_output):
    weight_sum = 0
    total_weights = 0

    for weight in model_output['hypo_params'].values():
        weight_sum += torch.sum(weight ** 2)
        total_weights += weight.numel()

        return weight_sum * (1 / total_weights)


def hyper_hh_loss(model_output, model_target, cfg, metrics):
    '''Compute helmholtz loss and additional losses to regularize hypernet.'''

    # compute gradients on the model
    gt_stiffness = model_target['pixels']
    coords = model_output['coords']
    wave = model_output['model_out']
    laplace_wave = hyper_laplace(wave, coords)
    # print('coords', coords)
    # print('wave', wave)
    # print('gt_stiffness', gt_stiffness)
    # print('laplace_wave', laplace_wave)

    with torch.no_grad():
        zeros1 = torch.zeros_like(wave.detach())
        dirichlet_bc = torch.where(gt_stiffness == zeros1,
                                   zeros1, wave.detach())

        fake_stiff = torch.full_like(gt_stiffness, 0.1)
        gt_stiffness = torch.where(gt_stiffness == zeros1, fake_stiff, gt_stiffness)

    helmholtz_loss = (gt_stiffness*laplace_wave/1000-(-60**2) *
                      wave/1000).pow(2).sum()
    dbc_loss = (wave/1000 - dirichlet_bc/1000).pow(2).sum()
    latent_reg = latent_loss(model_output)
    hypo_reg = hypo_weight_loss(model_output)

    loss = (helmholtz_loss + cfg['dbc_weight']*dbc_loss + cfg['latent_weight']*latent_reg +
            cfg['hypo_weight']*hypo_reg)

    metrics['helmholtz_loss'] += helmholtz_loss.data.cpu().numpy() * gt_stiffness.size(0)
    metrics['dbc_loss'] += dbc_loss.data.cpu().numpy() * gt_stiffness.size(0)
    metrics['latent_reg'] += latent_reg.data.cpu().numpy() * gt_stiffness.size(0)
    metrics['hypo_reg'] += hypo_reg.data.cpu().numpy() * gt_stiffness.size(0)
    metrics['loss'] += loss.data.cpu().numpy() * gt_stiffness.size(0)
    return loss


def print_metrics(metrics, epoch_samples, phase):
    outputs = []
    for k in metrics.keys():
        outputs.append("{}: {:4f}".format(k, metrics[k] / epoch_samples))

    print("{}: {}".format(phase, ", ".join(outputs)))
