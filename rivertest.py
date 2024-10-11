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
from datasets.river import test_dataset
from torchgeo.datasets import stack_samples
import torch.nn as nn
import matplotlib.pyplot as plt
from scipy.ndimage import zoom

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def inference(net, predictor,img_size=512,batch_size=16,n_gpu=1, output_dir=".", max_test_samples=-1, is_LoRa = False):
    tif_dir = os.path.join(os. getcwd(), "SAMed_input/test_image")
    shp_dir = os.path.join(os. getcwd(), "SAMed_input/test_mask")
    dataset = test_dataset(tif_dir, shp_dir, img_size, pre_computed_dataset_path='.datasets/')
    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    dataloader = DataLoader(dataset, collate_fn=stack_samples, batch_size=batch_size, worker_init_fn=worker_init_fn)
    total_processed_test_samples = 0
    iou_list = []
    num_of_samples = 0
    for k,batch in enumerate(dataloader):
        file_names = batch['file_name']
        images = batch['image']
        gt_masks = batch['mask']
        print(f"Batch = {k+1:>3d}", end='')
        pbar = tqdm(range(len(images)))
        for i in pbar:
            image = images[i]
            image_display = image.copy()
            file_name = file_names[i]
            pbar.set_description(f"Processing {file_name}", refresh=True)
            gt_mask = gt_masks[i]


            if not is_LoRa:
                image = image.transpose(1,2,0)
                image_display = image_display.transpose(1,2,0).astype(np.uint8)

                h, w, c = image.shape
                # we use original SAM
                predictor.set_image(image, image_format="RGB") # image should be H W C
                masks, scores, _ = predictor.predict( # logits These low resolution logits can be passed to a subsequent iteration as mask input.
                    multimask_output=False# False, it seems true is better
                )
                for j, (mask, score) in enumerate(zip(masks, scores)):
                    plt.figure(figsize=(10,10))
                    plt.imshow(image_display)
                    show_mask(mask, plt.gca())
                    plt.title(f"Mask {j+1}, Score: {score:.3f}", fontsize=18)
                    plt.axis('off')
                    gt_mask = gt_mask.copy()
                    gt_mask_diff = np.ones((h, w, 4)) * np.array([[1, 0, 0, 0.2]])
                    gt_mask_diff[~np.logical_and(gt_mask, ~mask)] = 0
                    # gt_mask_diff[~gt_mask] = 0

                    plt.gca().imshow(gt_mask_diff)
                    plt.savefig(os.path.join(output_dir, f'batch{k}_img{i}_mask{j}.png'))
                    np.save(os.path.join(output_dir, f'batch{k}_img{i}_mask{j}.npy'), mask) # save the mask as npy file
                    plt.close()
                
            else:
                # we use LoRa SAM
                # image # c h w 
                inputs = torch.from_numpy(image/255).unsqueeze(0).float().cuda()
                # print(f"image shape {inputs.shape}")
                net.eval()
                with torch.no_grad():
                    outputs = net(inputs, True, img_size)
                    output_masks = outputs['masks']
                    # print(f"output_masks shape {output_masks.shape}")
                    out = torch.argmax(torch.softmax(output_masks, dim=1), dim=1).squeeze(0)
                    out = out.cpu().detach().numpy()
                    out_h, out_w = out.shape
                    if img_size != out_h or img_size != out_w:
                        pred = zoom(out, (img_size / out_h, img_size / out_w), order=0)
                    else:
                        pred = out
                    # print(f"pred shape {pred.shape}")
                    # print(f"pred max {pred.max()}, pred min {pred.min()}")
                    pred_bool = pred.copy().astype(bool)
                    gt_mask_bool = gt_mask.copy().astype(bool)
                    overlap = pred_bool*gt_mask_bool # Logical AND
                    union = pred_bool + gt_mask_bool # Logical OR

                    iou = overlap.sum()/float(union.sum()) 
                    iou_list.append(iou)
                    num_of_samples += 1
                    # plot
                    image_display = image_display.transpose(1,2,0).astype(np.uint8)
                    fig, axs = plt.subplots(1, 3, figsize=(30,10))
                    axs[0].imshow(image_display)
                    axs[0].set_title('Raw Data', fontsize=30)
                    axs[1].imshow(image_display)
                    axs[1].set_title('Ground Truth', fontsize=30)
                    axs[2].imshow(image_display) 
                    axs[2].set_title('Predicted Mask', fontsize=30)
                    fig.suptitle(f"Filename {file_name} - IOU {iou:>5.2f}", fontsize=30)
                    show_mask(pred, axs[2])
                    for ax in axs.ravel():
                        ax.set_axis_off()
                    gt_mask = gt_mask.copy()
                    gt_mask_diff = np.ones((img_size, img_size, 4)) * np.array([[1, 0, 0, 0.2]])
                    # gt_mask_diff[~np.logical_and(gt_mask, ~pred)] = 0
                    gt_mask_diff[~gt_mask] = 0
                    axs[1].imshow(gt_mask_diff)
                    fig.savefig(os.path.join(output_dir, f'batch{k}_img{i}_mask.png'))
                    np.save(os.path.join(output_dir, f'batch{k}_img{i}_mask.npy'), pred) # save the mask as npy file
                    plt.close(fig)
            
            total_processed_test_samples += 1
            if max_test_samples != -1 and total_processed_test_samples >= max_test_samples:
                break
        if max_test_samples != -1 and total_processed_test_samples >= max_test_samples:
            break
    miou = np.array(iou_list).sum() / num_of_samples
    return miou, np.array(iou_list)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default=None, help='The config file provided by the trained model')
    parser.add_argument('--dataset', type=str, default='Synapse', help='Experiment name')
    parser.add_argument('--num_classes', type=int, default=1) # set to 3 is important
    parser.add_argument('--output_dir', type=str, default='./outputs')
    parser.add_argument('--img_size', type=int, default=512, help='Input image size of the network')
    parser.add_argument('--input_size', type=int, default=224, help='The input size for training SAM model')
    parser.add_argument('--seed', type=int,
                        default=1234, help='random seed')
    parser.add_argument('--deterministic', type=int, default=1, help='whether use deterministic training')
    parser.add_argument('--ckpt', type=str, default='checkpoints/sam_vit_b_01ec64.pth',
                        help='Pretrained checkpoint')
    parser.add_argument('--lora_ckpt', type=str, default=None, help='The checkpoint from LoRA')
    parser.add_argument('--vit_name', type=str, default='vit_b', help='Select one vit model')
    parser.add_argument('--rank', type=int, default=4, help='Rank for LoRA adaptation')
    parser.add_argument('--module', type=str, default='sam_lora_image_encoder')

    args = parser.parse_args()

    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    if not os.path.exists(args.output_dir):
       os.makedirs(args.output_dir)

    # register model
    if args.lora_ckpt is None:
        num_classes = 3
    else:
        num_classes = 1

    sam, img_embedding_size = sam_model_registry[args.vit_name](image_size=args.img_size,
                                                                    num_classes=num_classes,
                                                                    checkpoint=args.ckpt, pixel_mean=[0, 0, 0],
                                                                    pixel_std=[1, 1, 1])
    pkg = import_module(args.module)
    net = pkg.LoRA_Sam(sam, args.rank).cuda()
    if args.lora_ckpt is not None:
        net.load_lora_parameters(args.lora_ckpt)
    else:
        pass
    net.sam.eval()
    predictor = SamPredictor(net.sam)

    miou, iou_list = inference(net, predictor, img_size=args.img_size, output_dir=args.output_dir, max_test_samples=-1, is_LoRa = (args.lora_ckpt != None))
    print(f"miou is {miou:>6.2f}")
    np.save(os.path.join((args.output_dir), "iou_list.npy"), np.array(iou_list))




