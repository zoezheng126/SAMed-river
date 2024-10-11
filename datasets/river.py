import os
from glob import glob

import matplotlib.pyplot as plt
from typing import Any, Dict, Optional

import torch
import numpy as np
import rasterio
import geopandas as gpd
from torch.utils.data import Dataset
from rasterio.features import geometry_mask
import torch.nn.functional as F
from tqdm import tqdm
from torchgeo.datasets import VectorDataset, RasterDataset, stack_samples
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import functional as F

from skimage.morphology import binary_dilation 
from math import ceil
import random

class river_dataset(Dataset):
    def __init__(self, tif_dir, shp_dir, cropped_size, transform=None, pre_computed_dataset_path=None, save_computed_dataset_path=None):
        self.cropped_size = cropped_size

        if pre_computed_dataset_path == None:
            self.tifs = []
            self.masks = []
            self.tif_dir = tif_dir
            self.shp_dir = shp_dir
            self.tif_files = sorted([os.path.join(self.tif_dir, f) for f in os.listdir(tif_dir) if f.endswith('.tif')])
            self.shp_files = sorted([os.path.join(self.shp_dir, f) for f in os.listdir(shp_dir) if f.endswith('.shp')])
            print(f"len of tif {len(self.tif_files)}, len of shp {len(self.shp_files)}")
            set1 = set([os.path.basename(f).split(".")[0] for f in self.tif_files])
            set2 = set([os.path.basename(f).split(".")[0] for f in self.shp_files])
            set_diff = set2.difference(set1)
            for i in set_diff:
                print(i, flush=True)
            assert set([os.path.basename(f).split(".")[0] for f in self.tif_files]) == set([os.path.basename(f).split(".")[0] for f in self.shp_files]), \
                "Mismatch between tif files and shapefiles"
            print(f"Loading all images and masks in the test dataset")
            skipped_tifs = []
            for i in tqdm(range(len(self.tif_files))):
                # load image
                tif_path = self.tif_files[i] 
                src = rasterio.open(tif_path)
                image = src.read()
                image = image[:3,:,:] # comment this for ndwi dataset
                # image = image.astype(np.uint8)
                image = torch.tensor(image, dtype=torch.uint8) 
                _, height, width = F.get_dimensions(image)


                # load mask
                try:
                    shp_path = self.shp_files[i]
                    gdf = gpd.read_file(shp_path)
                    mask = geometry_mask(gdf.geometry, transform=src.transform, invert=True, out_shape=src.shape)
                    mask = torch.tensor(mask, dtype=torch.bool)
                except:
                    continue

                # filter out small images
                if width < int(self.cropped_size * 0.75) or height < int(self.cropped_size * 0.75):
                    file_name = os.path.basename(self.tif_files[i])
                    skipped_tifs.append([file_name, height, width])
                    continue
                # print(f'before padding {tif_path} has shape {image.shape}')
                # padding if needed
                if width < self.cropped_size:
                    padding = [ceil((self.cropped_size - width)/2), 0]
                    image = F.pad(image, padding, 0, "constant")
                    mask = F.pad(mask, padding, 0, "constant")
                if height < self.cropped_size:
                    padding = [0, ceil((self.cropped_size - height)/2)]
                    image = F.pad(image, padding, 0, "constant")
                    mask = F.pad(mask, padding, 0, "constant")
                # get the new image size after padding
                _, height, width = F.get_dimensions(image) 
                print(f'after padding {tif_path} has shape {image.shape}')
                image_size = height * width
                self.masks.append({"mask": mask})
                self.tifs.append({"image": image, # use image_ndwi if creating ndwi dataset
                                "size": image_size,
                                "crop_count": image_size // (self.cropped_size * self.cropped_size),
                                "true_index_in_tif_files": i})  # skipped some index thus different
                src.close()
            print(f"Finish loading all images and masks in the test dataset")
            print(f"length of tifs {len(self.tifs)} length of masks {len(self.masks)}")
            print(f"Below are skipped tifs")
            for skipped_tif, width, height in skipped_tifs:
                print(f"Filename: {skipped_tif}, Width: {width:>4d}, Height: {height:>4d}")

            
            print(f"Getting crop info list")
            self.dataset_meta = {}
            self.crop_infos = []
            for tif_index in tqdm(range(len(self.tifs))):
                image_info = self.tifs[tif_index]
                image = image_info["image"]
                crop_count = image_info["crop_count"]
                true_index_in_tif_files = image_info["true_index_in_tif_files"]
                for _ in range(crop_count):
                    i, j, h, w = transforms.RandomCrop.get_params(image, output_size=(self.cropped_size, self.cropped_size)) 
                    print(f'i: {i}, j: {j}, image shape: {image.shape}')
                    crop_info = {"index": tif_index,
                                "meta": [i, j, h, w],
                                "tif_path": self.tif_files[true_index_in_tif_files]}
                    self.crop_infos.append(crop_info)
            print(f"Total crop data {len(self.crop_infos)}")
            self.check_mask_th = int(0.1 * self.cropped_size * self.cropped_size) 

            # save dataset
            self.dataset_meta["crop_infos"] = self.crop_infos
            self.dataset_meta["check_mask_th"] = self.check_mask_th
            if save_computed_dataset_path != None: 
                print("start saving datasets")
                if not os.path.exists(save_computed_dataset_path):
                    os.mkdir(save_computed_dataset_path)
                np.save(os.path.join(save_computed_dataset_path, "tifs.npy"), self.tifs, allow_pickle=True)
                np.save(os.path.join(save_computed_dataset_path, "masks.npy"), self.masks, allow_pickle=True)
                np.save(os.path.join(save_computed_dataset_path, "dataset_meta.npy"), [self.dataset_meta], allow_pickle=True)
                print("finish saving datasets")
        else:
            print("start loading datasets")
            self.tifs = np.load(os.path.join(pre_computed_dataset_path, "tifs.npy"), allow_pickle=True)
            self.masks = np.load(os.path.join(pre_computed_dataset_path, "masks.npy"), allow_pickle=True)
            self.dataset_meta = np.load(os.path.join(pre_computed_dataset_path, "dataset_meta.npy"), allow_pickle=True)
            self.crop_infos = self.dataset_meta[0]['crop_infos']
            self.check_mask_th = self.dataset_meta[0]['check_mask_th']
            print("finish loading datasets")
        
    def __len__(self):
        return len(self.crop_infos)

    def __getitem__(self, idx):
        if idx >= len(self.crop_infos):
            raise StopIteration
        crop_info = self.crop_infos[idx]
        crop_source_index = crop_info["index"]
        tif_path = crop_info['tif_path']
        # file_name = os.path.basename(self.tif_files[crop_source_index]).split('.')[0]
        i, j, h, w = crop_info["meta"]
        image = self.tifs[crop_source_index]["image"]
        #print(f'inside river.py: tif_path {tif_path}image.shape {image.shape}')
        mask = self.masks[crop_source_index]["mask"]
        mask = transforms.functional.crop(mask, i, j, h, w)
        mask_numpy = mask.numpy()
        valid_pixels_in_mask = mask_numpy.sum()
        if valid_pixels_in_mask < self.check_mask_th:
            del self.crop_infos[idx]
            return self.__getitem__(idx)
        image = transforms.functional.crop(image, i, j, h, w)
        image_numpy = image.numpy() # c h w 
        sample = { 'file_name': tif_path,
                  'image': image_numpy,
                  'mask': mask_numpy,
                  'crop_coord':[i,j,h,w]}
        
        return sample
    
class train_dataset(Dataset):
    def __init__(self, tif_dir, shp_dir, cropped_size=512, transform=None):
        self.cropped_size = cropped_size
        self.dataset_length = 0
        self.tifs = []
        self.masks = []
        self.tif_dir = tif_dir
        self.shp_dir = shp_dir
        self.epoch = 0
        self.tif_files = sorted([os.path.join(self.tif_dir, f) for f in os.listdir(tif_dir) if f.endswith('.tif')])
        self.shp_files = sorted([os.path.join(self.shp_dir, f) for f in os.listdir(shp_dir) if f.endswith('.shp')])
        print(f"len of tif {len(self.tif_files)}, len of shp {len(self.shp_files)}")
        set1 = set([os.path.basename(f).split(".")[0] for f in self.tif_files])
        set2 = set([os.path.basename(f).split(".")[0] for f in self.shp_files])
        set_diff = set2.difference(set1)
        for i in set_diff:
            print(i, flush=True)
        assert set([os.path.basename(f).split(".")[0] for f in self.tif_files]) == set([os.path.basename(f).split(".")[0] for f in self.shp_files]), \
            "Mismatch between tif files and shapefiles"
        print(f"Loading all images and masks in the test dataset")
        skipped_tifs = []
        for i in tqdm(range(len(self.tif_files))):
            # load image
            tif_path = self.tif_files[i] 
            src = rasterio.open(tif_path)
            image = src.read()
            image = image[:3,:,:] # comment this for ndwi dataset
            # image = image.astype(np.uint8)
            image = torch.tensor(image, dtype=torch.uint8) 
            _, height, width = F.get_dimensions(image)

            # load mask
            try:
                shp_path = self.shp_files[i]
                gdf = gpd.read_file(shp_path)
                mask = geometry_mask(gdf.geometry, transform=src.transform, invert=True, out_shape=src.shape)
                mask = torch.tensor(mask, dtype=torch.bool)
            except Exception as e:
                print(e)
                mask = torch.zeros(height,width,dtype=torch.bool)    # if the whole tif has no river

            # filter out small images
            if width < int(self.cropped_size * 0.75) or height < int(self.cropped_size * 0.75):
                file_name = os.path.basename(self.tif_files[i])
                skipped_tifs.append([file_name, height, width])
                continue
            # print(f'before padding {tif_path} has shape {image.shape}')
            # padding if needed
            if width < self.cropped_size:
                padding = [ceil((self.cropped_size - width)/2), 0]
                image = F.pad(image, padding, 0, "constant")
                mask = F.pad(mask, padding, 0, "constant")
            if height < self.cropped_size:
                padding = [0, ceil((self.cropped_size - height)/2)]
                image = F.pad(image, padding, 0, "constant")
                mask = F.pad(mask, padding, 0, "constant")
            # get the new image size after padding
            _, height, width = F.get_dimensions(image) 
            image_size = height * width
            crop_count = image_size // (self.cropped_size * self.cropped_size)
            self.masks.append({"mask": mask})
            self.tifs.append({"image": image, # use image_ndwi if creating ndwi dataset
                            "size": image_size,
                            "crop_count": crop_count,
                            "true_index_in_tif_files": i})  # skipped some index thus different
            src.close()
            self.dataset_length += crop_count
        print(f"Finish loading all images and masks in the test dataset")
        print(f"length of dataset :{self.dataset_length}")
        print(f"Below are skipped tifs")
        for skipped_tif, width, height in skipped_tifs:
            print(f"Filename: {skipped_tif}, Width: {width:>4d}, Height: {height:>4d}")
        
    def __len__(self):
        return self.dataset_length

    def __getitem__(self, idx):
        if (idx == self.dataset_length - 1):
            self.epoch += 1
        threshold = get_threshold(self.epoch)
        water_percentage = -0.01
        while water_percentage < threshold:
            random_idx = random.randint(0, len(self.tifs)-1)
            image_info = self.tifs[random_idx]
            image = image_info["image"]
            image = self.tifs[random_idx]["image"].unsqueeze(0)
            mask = self.masks[random_idx]["mask"]
            true_index_in_tif_files = image_info["true_index_in_tif_files"]
            # i, j, h, w = transforms.RandomCrop.get_params(image, output_size=(self.cropped_size, self.cropped_size)) 
            i, j, h, w = transforms.RandomResizedCrop.get_params(image, scale=(0.08, 1.0), ratio=(3. / 4., 4. / 3.))
            tif_path = self.tif_files[true_index_in_tif_files]
            mask = mask.unsqueeze(0)
            mask = transforms.functional.resized_crop(mask, i, j, h, w, size=(self.cropped_size, self.cropped_size), interpolation=transforms.InterpolationMode.BILINEAR)
            image = transforms.functional.resized_crop(image, i, j, h, w, size=(self.cropped_size, self.cropped_size), interpolation=transforms.InterpolationMode.NEAREST)
            # print(f'percentage of water content is {np.sum(mask_numpy) / (h * w)} at idx {idx}')

            if random.random() > 0.5:
                image = transforms.functional.hflip(image)
                mask = transforms.functional.hflip(mask)
            mask_numpy = mask.squeeze(0).numpy()
            image_numpy = image.squeeze(0).numpy() # c h w
            water_percentage = np.sum(mask_numpy) / (h * w)
            sample = { 'file_name': tif_path,
                    'image': image_numpy,
                    'mask': mask_numpy,
                    'crop_coord':[i,j,h,w]}
        

        return sample
    
def get_threshold(idx):
    if idx < 3:
        threshold = 0.2
    elif idx >= 3 and idx < 8:
        threshold = 0.1
    else:
        threshold = 0
    return threshold    
if __name__ == "__main__":
    pass
