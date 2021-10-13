from pathlib import Path
import numpy as np
import torch
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets, models
from collections import defaultdict, OrderedDict
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torch.utils.data.sampler import RandomSampler
import time
import copy
import warnings
from datetime import datetime
from tqdm import tqdm_notebook
from torch.utils.tensorboard import SummaryWriter
import PIL
from skimage import feature, morphology, exposure
import bezier
from mre_ai.pytorch_arch_deeplab_3d import DeepLab
from mre_ai.pytorch_arch_models_genesis import UNet3D

import xarray as xr
import matplotlib.pyplot as plt


# need data to be ordered thusly:
# image_sequence,width,hight,depth


class ChaosDataset(Dataset):
    def __init__(self, xr_ds, set_type='train', clip=False, seed=100,
                 test_subj='001', aug=True, sequence_mode='all', resize=False,
                 val_split=0.2, model_arch='3D', verbose=False, val_subj=None, color_aug=False):

        self.verbose = verbose
        self.model_arch = model_arch
        self.all_sequences = xr_ds.sequence.values
        self.my_sequence = sequence_mode
        if type(test_subj) is not list:
            test_subj = [test_subj]
        self.test_subj = test_subj
        print(self.test_subj)
        if val_subj is None:
            val_subj = ['002', '003', '101', '102']
        xr_ds_test = xr_ds.sel(subject=self.test_subj)
        xr_ds = xr_ds.drop(self.test_subj, dim='subject')

        # default method of training/val split is random shuffling of the main list
        np.random.seed(seed)

        if set_type == 'test':
            input_set = self.test_subj
        elif set_type == 'val':
            input_set = val_subj
        elif set_type == 'train':
            input_set = [i for i in xr_ds.subject.values if i not in val_subj]

        else:
            raise AttributeError('Must choose one of ["train", "val", "test"] for `set_type`.')
        self.my_subjects = input_set

        # pick correct input set, remove test subjects
        if set_type == 'test':
            xr_ds = xr_ds_test
        else:
            print(self.my_subjects)
            xr_ds = xr_ds.sel(subject=self.my_subjects)
            print(xr_ds.subject)

        # assign input and target elements, based on 2d or 3d arch
        if '3D' in self.model_arch:
            if self.my_sequence != 'all':
                if self.my_sequence == 'ct':
                    xr_ds = xr_ds.where(xr_ds.mr_ct_id == 2, drop=True)
                else:
                    xr_ds = xr_ds.where(xr_ds.mr_ct_id == 1, drop=True)

                xr_ds = xr_ds.sel(sequence=self.my_sequence)
                self.input_images = xr_ds['image'].transpose(
                    'subject', 'sequence', 'z', 'y', 'x').values
                self.target_images = xr_ds['mask'].transpose(
                    'subject', 'sequence', 'z', 'y', 'x').values
            else:
                # Stack all the sequences together for appropriate random sampling
                if self.verbose:
                    print('stacking sequences xarray')

                mr_xr_ds = xr_ds.where(xr_ds.mr_ct_id == 1, drop=True)
                mr_xr_ds = mr_xr_ds.sel(sequence=['t1_in', 't1_out', 't2'])
                print(mr_xr_ds)
                n_subj = len(mr_xr_ds.subject)
                n_seq = 3
                z = len(mr_xr_ds.z)
                y = len(mr_xr_ds.y)
                x = len(mr_xr_ds.x)

                self.input_images = mr_xr_ds['image'].transpose(
                    'subject', 'sequence', 'z', 'y', 'x').values.reshape(n_subj*n_seq, 1, z, y, x)
                self.target_images = mr_xr_ds['mask'].transpose(
                    'subject', 'sequence', 'z', 'y', 'x').values.reshape(n_subj*n_seq, 1, z, y, x)
                self.names = ['_'.join((subj, seq)) for subj in mr_xr_ds.subject.values for seq in
                              mr_xr_ds.sequence.values]

                # if set_type != 'test':
                ct_xr_ds = xr_ds.where(xr_ds.mr_ct_id == 2, drop=True)
                ct_xr_ds = ct_xr_ds.sel(sequence=['ct'])
                print(ct_xr_ds)
                n_subj = len(ct_xr_ds.subject)
                n_seq = 1
                z = len(ct_xr_ds.z)
                y = len(ct_xr_ds.y)
                x = len(ct_xr_ds.x)

                ct_input_images = ct_xr_ds['image'].transpose(
                    'subject', 'sequence', 'z', 'y', 'x').values.reshape(
                        n_subj*n_seq, 1, z, y, x)
                ct_target_images = ct_xr_ds['mask'].transpose(
                    'subject', 'sequence', 'z', 'y', 'x').values.reshape(
                        n_subj*n_seq, 1, z, y, x)
                self.input_images = np.concatenate((self.input_images, ct_input_images))
                self.target_images = np.concatenate((self.target_images, ct_target_images))
                self.names += ['_'.join((subj, seq)) for subj in ct_xr_ds.subject.values for seq
                               in ct_xr_ds.sequence.values]

            self.input_images = self.input_images.astype(np.float32)
            self.target_images = self.target_images.astype(np.int32)

        # Additional flags
        self.aug = aug
        self.clip = clip
        self.resize = resize
        self.color_aug = color_aug

    def __len__(self):
        return len(self.input_images)

    def __getitem__(self, idx):

        if self.verbose:
            print(self.names[idx], f'{idx}/{self.__len__()}')
        if '3D' in self.model_arch:
            image, target = self.get_data_aug_3d(idx)
        elif self.model_arch == '2D':
            image, target = self.get_data_aug_2d(idx)

        image = torch.Tensor(image)
        target = torch.Tensor(target)

        return [image, target, self.names[idx]]

    def get_data_aug_3d(self, idx):
        '''get data (image and target), and apply augmentations if indicated'''
        image = self.input_images[idx]
        target = self.target_images[idx]
        target = np.where(target > 0, 1, 0).astype(np.int32)

        if self.clip:
            # t1 image clip
            if (('t1' in self.my_sequence) or
                    (self.my_sequence == 'all' and idx < 2*len(self.my_subjects))):
                image = np.where(image >= 1500, 1500, image)
            # t2 image clip
            else:
                image = np.where(image >= 2000, 2000, image)

        if self.aug:
            rot_angle = np.random.uniform(-5, 5)
            translations = np.random.uniform(-5, 5, 2)
            scale = np.random.uniform(0.90, 1.10)
            restack = np.random.randint(-5, 6)
            # flip = np.random.randint(0, 2)
            flip = 0
        else:
            rot_angle = 0
            translations = (0, 0)
            scale = 1
            restack = 0
            flip = 0
        image = self.input_transform_3d(image, rot_angle,
                                        translations, scale, restack, flip,
                                        resample=TF.InterpolationMode.BILINEAR)
        target = self.affine_transform_3d(target, rot_angle,
                                          translations, scale, restack, flip,
                                          resample=TF.InterpolationMode.NEAREST)
        return image, target

    def get_data_aug_2d(self, idx):
        '''get data (image and target), and apply augmentations if indicated'''
        image = self.input_images[idx]
        target = self.target_images[idx]
        target = np.where(target > 0, 1, 0).astype(np.int32)

        if self.clip:
            for i, seq in enumerate(self.my_sequence):
                if 't1' in seq:
                    image[i, :] = np.where(image[i, :] >= 1500, 1500, image[i, :])
                else:
                    image[i, :] = np.where(image[i, :] >= 2000, 2000, image[i, :])

        if self.aug:
            rot_angle = np.random.uniform(-5, 5, 1)
            translations = np.random.uniform(-5, 5, 2)
            scale = np.random.uniform(0.90, 1.00, 1)
            flip = np.random.randint(0, 2)
        else:
            rot_angle = 0
            translations = (0, 0)
            scale = 1
            flip = 0
        for i in range(len(self.my_sequence)):
            image[i, :] = self.input_transform_2d(image[i, :], rot_angle,
                                                  translations, scale, flip)

            target[i, :] = self.affine_transform_2d(target[i, :], rot_angle,
                                                    translations, scale, flip)
        return image, target

    def input_transform_3d(self, input_image, rot_angle=0, translations=0, scale=1, restack=0,
                           flip=0, resample=None):
        # normalize and offset image
        # image = input_image
        # # image = np.where(input_image <= 1e-9, np.nan, input_image)
        # mean = np.nanmean(image)
        # std = np.nanstd(image)
        # # image = ((image - mean)/std) + 4
        # image = ((image - mean)/std)
        # image = np.where(image != image, 0, image)

        image = input_image*1.0

        v_min, v_max = np.percentile(image, (0.5, 99.5))
        # v_min = max(v_min, 1e-6)
        for i in range(image.shape[0]):
            # v_min, v_max = np.percentile(image[i], (0.5, 99.5))
            image[i, :] = exposure.rescale_intensity(image[i], in_range=(v_min, v_max),
                                                     out_range=(-1.0, 1.0))
            # image[i, :] = exposure.rescale_intensity(image[i], out_range=(-1.0, 1.0))
            # image[i, :] = exposure.equalize_adapthist(image[i], clip_limit=0.03)

            # image = np.where(input_image <= 1e-9, np.nan, input_image)
            # mean = np.nanmean(image, axis=(1, 2, 3))
            # std = np.nanstd(image, axis=(1, 2, 3))
            # image = ((image.T - mean)/std).T
            # image = np.where(image != image, 0, image)
        if self.color_aug:
            nodes = np.asfortranarray([
                [-1, np.random.uniform(-1, 1.0), np.random.uniform(-1, 1.0), 1.0],
                [-1, np.random.uniform(-1, 1.0), np.random.uniform(-1, 1.0), 1.0]])
            curve = bezier.Curve(nodes, degree=3)
            points = curve.evaluate_multi(np.linspace(-1, 1.0, 1000))
            xvals = np.sort(points[0])
            yvals = np.sort(points[1])
            if np.random.rand() > 0.5:
                yvals = yvals[::-1]
            image = np.interp(np.ravel(image), xvals, yvals).reshape(image.shape)
        image = image.astype(np.float32)

        # perform affine transfomrations
        image = self.affine_transform_3d(image, rot_angle, translations, scale, restack, flip,
                                         resample=resample)

        return image

    def input_transform_2d(self, input_image, rot_angle=0, translations=0, scale=1,
                           flip=0, resample=None):
        # normalize and offset image
        image = input_image
        # image = np.where(input_image <= 1e-9, np.nan, input_image)
        mean = np.nanmean(image)
        std = np.nanstd(image)
        # image = ((image - mean)/std) + 4
        image = ((image - mean)/std)
        image = np.where(image != image, 0, image)

        # perform affine transfomrations
        image = self.affine_transform_2d(image, rot_angle, translations, scale, flip,
                                         resample=resample)
        return image

    def affine_transform_3d(self, image, rot_angle=0, translations=0, scale=1, restack=0, flip=0,
                            resample=None):
        output_image = image.copy()
        if self.verbose:
            print(f'rot_angle: {rot_angle}, translations: {translations},'
                  f'scale: {scale}, restack={restack}')

        # for i in range(output_image.shape[0]):
        #     if (i+restack < 0) or (i+restack > output_image.shape[0]):
        #         output_image[0, i] = np.zeros_like(output_image[0, 0])
        #     else:
        #         output_image[0, i] = self.affine_transform_slice(output_image[0, i+restack],
        #                                                          rot_angle, translations, scale)

        for i in range(output_image.shape[1]):
            if (i+restack < 0) or (i+restack > output_image.shape[1]-1):
                output_image[0, i] = np.zeros_like(output_image[0, 0])
            else:
                output_image[0, i] = self.affine_transform_2d(image[0, i+restack],
                                                              rot_angle, translations, scale, flip,
                                                              resample=resample)
        return output_image

    def affine_transform_2d(self, input_slice, rot_angle=0, translations=0, scale=1, flip=0,
                            resample=None):
        input_slice = transforms.ToPILImage()(input_slice)
        input_slice = TF.affine(input_slice, angle=rot_angle,
                                translate=list(translations), scale=scale, shear=0,
                                interpolation=resample)
        if flip:
            input_slice = TF.hflip(input_slice)
        input_slice = transforms.ToTensor()(np.array(input_slice))
        return input_slice


def eval_segmentation(input_xarray, input_subj, input_label,
                      model_name='model_2020-07-24_14-21-43_n3.pkl', overlay_label='new_seg',
                      model_path='/ocean/projects/asc170022p/bpollack/mre_ai/data/CHAOS/',
                      model_arch='3D'):
    '''Evaluate segmentation on an input abdominal image, return in pyviz_med overlay format.'''

    model_path = Path(model_path, 'trained_models', '001', model_name)

    with torch.no_grad():
        if model_arch == '3D':
            model = DeepLab(1, 1, output_stride=8, do_ord=False).to('cuda:0')
            model_dict = torch.load(model_path, map_location='cuda:0')
            # model_dict = OrderedDict([(key[7:], val) for key, val in
            # model_dict.items()])
            model.load_state_dict(model_dict, strict=True)
            model.eval()
        elif model_arch == 'ModelsGenesis3D':
            model = UNet3D().to('cuda:0')

            # Load pre-trained weights
            # weight_dir = ('/pghbio/dbmi/batmanlab/bpollack/intensity_agnostic/' +
            #               'ModelsGenesis/pretrained_weights/Genesis_Chest_CT.pt')
            # checkpoint = torch.load(weight_dir)
            # state_dict = checkpoint['state_dict']
            # unParalled_state_dict = {}
            # for key in state_dict.keys():
            #     unParalled_state_dict[key.replace("module.", "")] = state_dict[key]
            # model.load_state_dict(unParalled_state_dict)

            model_dict = torch.load(model_path, map_location='cuda:0')
            model_dict = OrderedDict([(key[7:], val) for key, val in model_dict.items()])
            model.load_state_dict(model_dict, strict=True)
            model.eval()

    image = input_xarray.sel(
        subject_id=input_subj, label=input_label).transpose('z', 'y', 'x').values.copy()

    image = image*1.0
    v_min, v_max = np.percentile(image, (0.5, 99.5))
    print(v_min, v_max)
    for i in range(image.shape[0]):
        image[i, :] = exposure.rescale_intensity(image[i], in_range=(v_min, v_max),
                                                 out_range=(-1.0, 1.0))

    image = image[np.newaxis, np.newaxis, :]

    # get the model prediction (liver mask)
    with torch.no_grad():
        print('starting torch model loading')
        model_pred = model(torch.Tensor(image).to('cuda:0'))
        print('sigmoid func')
        model_pred = torch.sigmoid(model_pred)
        print('torch done')
        # Convert to binary mask
        # print('ones and zeros')
        # ones = torch.ones_like(model_pred)
        # zeros = torch.zeros_like(model_pred)
        # print('where')
        # model_pred = torch.where(model_pred >
        # 1e-1, ones, zeros)
        output_image_np = np.transpose(model_pred.cpu().numpy()[0, 0, :], (2, 1, 0))
        output_image_np = np.where(output_image_np > 0.5, 1, 0)

    tmp_xr = xr.DataArray(
        output_image_np[np.newaxis, np.newaxis, np.newaxis, :],
        dims=(["subject_id", "label", "overlay_label", "x", "y", "z"]),
        coords={"subject_id": [input_subj], "label": [input_label],
                "overlay_label": [overlay_label],
                "x": input_xarray.x, "y": input_xarray.y, "z": input_xarray.z}, name='overlay')

    return tmp_xr
