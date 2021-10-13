#! usr/bin/env python
import os
import warnings
from pathlib import Path
import re
from collections import OrderedDict
import numpy as np
import pandas as pd
import xarray as xr
import pickle as pkl
import matplotlib.pyplot as plt
from scipy.stats import binned_statistic

import glob
from datetime import datetime
from scipy import ndimage as ndi
from scipy.signal import find_peaks
import SimpleITK as sitk
import skimage as skim
from skimage import feature, morphology, exposure
from skimage.filters import sobel
import PIL
import pdb
from tqdm import tqdm_notebook
from medpy.filter.smoothing import anisotropic_diffusion
import bezier
# import matplotlib.pyplot as plt
# import holoviews as hv


import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.transforms.functional as TF
from scipy.ndimage import gaussian_filter, median_filter

import skimage.draw as draw
import skimage.data as skd
from scipy.stats import mode

from mre_ai.wave_pred import ImageFitting, helmholtz, helmholtz_pml
from mre_ai.pytorch_arch_siren import Siren
from mre_ai.random_shapes import random_shapes


def make_stiff_size_pane(ds, mask_label, bins=100, pred=None, ax=None):
    masks = ds.image.sel(label='mask').values
    true = ds.image.sel(label='stiffness').values
    true_single_label = np.where(masks == mask_label, true, np.nan)
    shape_size = np.count_nonzero(~np.isnan(true_single_label), (1, 2))
    true_mean = np.nanmean(true_single_label, (1, 2))
    df = pd.DataFrame({'shape_size': shape_size, 'true_mean': true_mean})

    if pred is not None:
        pred = ds.image.sel(label='pred').values
        pred_single_label = np.where(masks == mask_label, pred, np.nan)
        pred_mean = np.nanmean(pred_single_label, (1, 2))
        df['pred_mean'] = pred_mean

    df = df.dropna().reset_index()
    df = df.sort_values('shape_size')
    qbins = pd.cut(df.shape_size, bins, duplicates='drop')
    midpoints = [(a.left + a.right)/2 for a in qbins.cat.categories]
    bin_widths = [(a.left - a.right) for a in qbins.cat.categories]
    edges = [a.left for a in qbins.cat.categories]+[qbins.cat.categories[-1].right]

    if ax is None:
        plt.plot(df.shape_size, df.true_mean, 'o-', markersize=3, label='True Stiffness')
        if pred is not None:
            pred_y = binned_statistic(df.shape_size, df.pred_mean, statistic='mean', bins=edges)[0]
            pred_err = binned_statistic(df.shape_size, df.pred_mean, statistic='std', bins=edges)[0]
            plt.errorbar(midpoints, pred_y, xerr=np.asarray(bin_widths)/2, yerr=pred_err, fmt='o',
                         alpha=0.7, label='Predicted Stiffness')
        plt.xlabel('Shape Size')
        plt.ylabel('Shape Stiffness')
    else:
        ax.plot(df.shape_size, df.true_mean, 'o-', markersize=3, label='True Stiffness')
        if pred is not None:
            pred_y = binned_statistic(df.shape_size, df.pred_mean, statistic='mean', bins=edges)[0]
            pred_err = binned_statistic(df.shape_size, df.pred_mean, statistic='std', bins=edges)[0]
            ax.errorbar(midpoints, pred_y, xerr=np.asarray(bin_widths)/2, yerr=pred_err, fmt='o',
                        alpha=0.7, label='Predicted Stiffness')
    return df


def make_stiff_size_grid(ds, bins=100, pred=None):

    shape_mapping = {'Circle': 10, 'Rectangle': 20, 'Triangle': 30, 'Ellipse': 40}
    texture_mapping = {'Grass': 1, 'Gravel': 2, 'Brick': 3, 'Text': 4}

    fig, ax = plt.subplots(4, 4, sharex=True, sharey=True, constrained_layout=True)
    for shape in shape_mapping:
        for texture in texture_mapping:
            shape_id = shape_mapping[shape]
            texture_id = texture_mapping[texture]
            mask_id = shape_id+texture_id
            ax_id = (shape_id//10-1, texture_id-1)
            _ = make_stiff_size_pane(ds, mask_id, bins=bins, pred=pred, ax=ax[ax_id])


def make_hv_stiffness(ds, mask_id):
    masks = ds.image.sel(label='mask').values
    true = ds.image.sel(label='stiffness').values
    true_single_label = np.where(masks == mask_id, true, np.nan)
    shape_size = np.count_nonzero(~np.isnan(true_single_label), (1, 2))
    true_mean = np.nanmean(true_single_label, (1, 2))
    df = pd.DataFrame({'shape_size': shape_size, 'true_mean': true_mean})
    df = df.dropna().sort_values('shape_size')
    df = df.reset_index()
    return df.hvplot.scatter(x='shape_size', y='true_mean', hover_cols=['shape_size', 'true_mean',
                                                                        'index'])
