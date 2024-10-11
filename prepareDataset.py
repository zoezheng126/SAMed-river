import os
import sys
from tqdm import tqdm
import logging
import numpy as np
import argparse
import random
import torch
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
from utils import test_single_volume
from importlib import import_module
from segment_anything import sam_model_registry, SamPredictor
from datasets.river import river_dataset
from torchgeo.datasets import stack_samples
import torch.nn as nn
import matplotlib.pyplot as plt
from scipy.ndimage import zoom
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--path', type=str, default='.dataset/test',help='path to store prepared dataset')
parser.add_argument('--tif_dir', type=str, default='SAMed_input/test_image', help='the path that store test tif images')
parser.add_argument('--shp_dir', type=str, default='SAMed_input/test_mask', help='the path that store test shp masks')
args = parser.parse_args()
img_size = 512  # default should not be changed

# prepare dataset for testing data
tif_dir = os.path.join(os. getcwd(), args.tif_dir)
shp_dir = os.path.join(os. getcwd(), args.shp_dir)
dataset = river_dataset(tif_dir, shp_dir, img_size, save_computed_dataset_path=args.path)

# # prepare dataset for training data
# tif_dir = os.path.join(os. getcwd(), "SAMed_input/train_image")
# shp_dir = os.path.join(os. getcwd(), "SAMed_input/train_mask")
# dataset = river_dataset(tif_dir, shp_dir, img_size, save_computed_dataset_path='.datasets/train')
