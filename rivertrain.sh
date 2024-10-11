#! /bin/bash
# the first one directly load the data from tif images and shp masks
mkdir -p model_output/sam/train-dataset
python3 train.py \
    --output model_output/sam/train-dataset \   
    --dataset River \
    --num_classes 1 \
    --max_epochs 60 \
    --batch_size 4 \
    --n_gpu 1 \
    --img_size 512 \
    --ckpt checkpoints/sam_vit_b_01ec64.pth \
    --module sam_lora_image_encoder_mask_decoder \
    --tif_dir SAMed_input/train_image \
    --shp_dir SAMed_input/train_mask \
    --rank 8

# second one is for loading pre-computed npy dataset, suggest first comment, otherwise uncomment rivertrainer.py line 82 and run script below
# python3 train.py --output model_output/sam/results --dataset River --num_classes 1 --max_epochs 100 --batch_size 8 --n_gpu 1 --img_size 512 --ckpt checkpoints/sam_vit_b_01ec64.pth --module sam_lora_image_encoder --exp test --pre_computed_dataset_path .datasets/train

# mkdir -p model_output/sam/train-dataset
# python3 train.py \
#     --output model_output/sam/train-dataset \
#     --dataset River \
#     --num_classes 1 \
#     --max_epochs 60 \
#     --batch_size 12 \
#     --n_gpu 1 \
#     --img_size 512 \
#     --ckpt checkpoints/sam_vit_b_01ec64.pth \
#     --module sam_lora_image_encoder_mask_decoder \
#     --pre_computed_dataset_path .datasets/train \
#     --tif_dir SAMed_input/train_image \
#     --shp_dir SAMed_input/train_mask