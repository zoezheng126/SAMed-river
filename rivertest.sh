#! /bin/bash

# Inference Mackinaw river test dataset, the following param might need to change
# output_dir = dir of visualization of predicted masks
# pre_computed_dataset_path = dir of precomputed and saved dataset after running prepareDataset.py
# python3 rivertest.py --num_classes 1 --output_dir ./outputs/LoRA_test --img_size 512 --ckpt checkpoints/sam_vit_b_01ec64.pth --lora_ckpt checkpoints/LoRA8.pth --vit_name vit_b --rank 8 --module sam_lora_image_encoder --pre_computed_dataset_path .datasets/test

# Inference Mackinaw river test dataset and also output shapefile and tif, same params need to be changed as above
mkdir -p shp_outputs tif_outputs outputs
python3 rivertest_shp.py \
    --num_classes 1 \
    --output_dir ./outputs \
    --img_size 512 \
    --ckpt checkpoints/sam_vit_b_01ec64.pth \
    --lora_ckpt checkpoints/LoRA8.pth \
    --vit_name vit_b \
    --rank 8 \
    --module sam_lora_image_encoder \
    --pre_computed_dataset_path .datasets/SkySat \
    --path_shp shp_outputs \
    --path_tif tif_outputs 
