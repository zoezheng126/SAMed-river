import argparse
import logging
import os
import random
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm import tqdm
from utils import DiceLoss, Focal_loss
from torchvision import transforms

#import in sam-river tune.py
import os
from glob import glob
from typing import Any, Dict
import numpy as np
import torch

import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchgeo.datasets import stack_samples
from scipy.ndimage import zoom
# import segmentation_models_pytorch as smp

from segment_anything import sam_model_registry
from segment_anything.utils.transforms import ResizeLongestSide
from datasets.river import river_dataset, train_dataset

# long loss func
# def calc_loss(outputs, low_res_label_batch, ce_loss, dice_loss, dice_weight:float=0.8):
#     low_res_logits = outputs['low_res_logits']
#     loss_ce = ce_loss(low_res_logits, low_res_label_batch[:].long())
#     loss_dice = dice_loss(low_res_logits, low_res_label_batch, softmax=True)
#     loss = (1 - dice_weight) * loss_ce + dice_weight * loss_dice

#     batch_iou = 0.
#     target = low_res_label_batch.unsqueeze(1)
#     for i in range(low_res_logits.shape[1]):
#         mask = low_res_logits[:, i, :, :]
#         mask = mask.unsqueeze(1)
#         batch_tp, batch_fp, batch_fn, batch_tn = smp.metrics.get_stats(
#             output=mask,
#             target=target,
#             mode='binary',
#             threshold=0.5,
#         )
#         batch_iou += smp.metrics.iou_score(batch_tp, batch_fp, batch_fn, batch_tn) 
#     iou_predictions = outputs['iou_predictions']
#     loss_iou = F.mse_loss(iou_predictions, batch_iou) 

#     return loss, loss_ce, loss_dice, loss_iou

def calc_loss(outputs, low_res_label_batch, ce_loss, dice_loss, dice_weight:float=0.8):
    low_res_logits = outputs['low_res_logits']
    loss_ce = ce_loss(low_res_logits, low_res_label_batch[:].long())
    loss_dice = dice_loss(low_res_logits, low_res_label_batch, softmax=True)
    loss = (1 - dice_weight) * loss_ce + dice_weight * loss_dice
    return loss, loss_ce, loss_dice

def trainer_river(args, model, snapshot_path, multimask_output, low_res):
    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size * args.n_gpu

    # only use precomputed dataset 
    # dataset = river_dataset("", "", args.img_size, pre_computed_dataset_path=args.pre_computed_dataset_path)
    dataset = train_dataset(args.tif_dir, args.shp_dir, args.img_size)
    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    trainloader = DataLoader(dataset, collate_fn=stack_samples, batch_size=batch_size, worker_init_fn=worker_init_fn)

    if args.n_gpu > 1:
        model = nn.DataParallel(model)
    model.train()
    ce_loss = CrossEntropyLoss()
    dice_loss = DiceLoss(num_classes + 1)

    if args.warmup:
        b_lr = base_lr / args.warmup_period
    else:
        b_lr = base_lr
    if args.AdamW:
        optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=b_lr, betas=(0.9, 0.999), weight_decay=0.001)
    else:
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=b_lr, momentum=0.9, weight_decay=0.0001)  # Even pass the model.parameters(), the `requires_grad=False` layers will not update
    writer = SummaryWriter(snapshot_path + '/log')
    iter_num = 0
    max_epoch = args.max_epochs
    stop_epoch = args.stop_epoch
    max_iterations = args.max_epochs * len(trainloader)  # max_epoch = max_iterations // len(trainloader) + 1
    logging.info("{} iterations per epoch. {} max iterations ".format(len(trainloader), max_iterations))
    best_performance = 0.0
    iterator = tqdm(range(max_epoch), ncols=70)
    for epoch_num in iterator:
        for i, batch in tqdm(enumerate(trainloader), total=len(trainloader)):
            # b, c, h, w = batch["image"].shape
            image_batch, label_batch = np.array(batch['image'], dtype=np.float32)/255, np.array(batch['mask'])
            low_res_label_batch = zoom(label_batch,(1,0.25,0.25),order=0) # need to check
            image_batch, label_batch = torch.from_numpy(image_batch).cuda(), torch.from_numpy(label_batch).cuda()
            low_res_label_batch = torch.from_numpy(low_res_label_batch).cuda()
            assert image_batch.max() <= 3, f'image_batch max: {image_batch.max()}'
            outputs = model(image_batch, multimask_output, args.img_size)
            loss, loss_ce, loss_dice = calc_loss(outputs, low_res_label_batch, ce_loss, dice_loss, args.dice_param)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if args.warmup and iter_num < args.warmup_period:
                lr_ = base_lr * ((iter_num + 1) / args.warmup_period)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_
            else:
                if args.warmup:
                    shift_iter = iter_num - args.warmup_period
                    assert shift_iter >= 0, f'Shift iter is {shift_iter}, smaller than zero'
                else:
                    shift_iter = iter_num
                lr_ = base_lr * (1.0 - shift_iter / max_iterations) ** 0.9  # learning rate adjustment depends on the max iterations
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_
            iter_num = iter_num + 1
            writer.add_scalar('info/lr', lr_, iter_num)
            writer.add_scalar('info/total_loss', loss, iter_num)
            writer.add_scalar('info/loss_ce', loss_ce, iter_num)
            writer.add_scalar('info/loss_dice', loss_dice, iter_num)

            logging.info('iteration %d : loss : %f, loss_ce: %f, loss_dice: %f' % (iter_num, loss.item(), loss_ce.item(), loss_dice.item()))

            if iter_num % 20 == 0:
                image = image_batch[0, 0:3, :, :]
                image = (image - image.min()) / (image.max() - image.min())
                writer.add_image('train/Image', image, iter_num)
                output_masks = outputs['masks']
                output_masks = torch.argmax(torch.softmax(output_masks, dim=1), dim=1, keepdim=True)
                writer.add_image('train/Prediction', output_masks[0, ...], iter_num)
                labs = label_batch[0, ...].unsqueeze(0)
                writer.add_image('train/GroundTruth', labs, iter_num)
        save_interval = 1
        if (epoch_num + 1) % save_interval == 0:
            save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
            try:
                model.save_lora_parameters(save_mode_path)
            except:
                # model.module.save_lora_parameters(save_mode_path)
                torch.save(model.state_dict(), save_mode_path)
            logging.info("save model to {}".format(save_mode_path))

        if epoch_num >= max_epoch - 1 or epoch_num >= stop_epoch - 1:
            save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
            try:
                model.save_lora_parameters(save_mode_path)
            except:
                # model.module.save_lora_parameters(save_mode_path)
                torch.save(model.state_dict(), save_mode_path)
            logging.info("save model to {}".format(save_mode_path))
            iterator.close()
            break

    writer.close()
    return "Training Finished!"
