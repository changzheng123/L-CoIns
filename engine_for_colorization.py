# --------------------------------------------------------
# Based on BEiT, timm, DINO and DeiT code bases
# https://github.com/microsoft/unilm/tree/master/beit
# https://github.com/rwightman/pytorch-image-models/tree/master/timm
# https://github.com/facebookresearch/deit
# https://github.com/facebookresearch/dino
# --------------------------------------------------------'
import math
import sys
import os
from time import process_time_ns
from typing import Iterable, Optional

import torch
from torchvision import transforms
from einops.einops import rearrange
from timm.data import Mixup
from timm.utils import accuracy, ModelEma
from torch.serialization import save

import utils
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import random
import cv2
import numpy as np

from modeling_finetune import Sobel_conv, FocalLoss
from skimage.measure import compare_ssim
import lpips
from scipy.optimize import linear_sum_assignment
# import seaborn as sns

def get_loss_scale_for_deepspeed(model):
    optimizer = model.optimizer
    return optimizer.loss_scale if hasattr(optimizer, "loss_scale") else optimizer.cur_scale

def loss_coord_fn(outputs, targets, indices, num_coord):
    """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
        targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
        The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
    """
    # print('indices&num_coord',indices, num_coord)
    idx = _get_src_permutation_idx(indices)
    src_coord = outputs[idx]
    target_coord = torch.cat([torch.as_tensor(t[i]) for t, (_, i) in zip(targets, indices)], dim=0).to(outputs.device)
    # print('loss_target_coord',len(target_coord),target_coord)
    loss_bbox = torch.nn.functional.l1_loss(src_coord/224, target_coord/224, reduction='none')
    # sys.exit()
    loss = loss_bbox.sum() / num_coord

    return loss

def _get_src_permutation_idx(indices):
    # permute predictions following indices
    batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
    src_idx = torch.cat([src for (src, _) in indices])
    return batch_idx, src_idx

@torch.no_grad()
def evaluate(data_loader, model, device, epoch=10000, patch_size=16, save_img_dir=None, istest = False):

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()

    lpips_fn_vgg = lpips.LPIPS(net='vgg').to(device, non_blocking=True)

    for step,(samples, cap, keys, occm_mat) in enumerate(metric_logger.log_every(data_loader, 10, header)):
        images = samples
        images = images.to(device, non_blocking=True)
        # target = target.to(device, non_blocking=True)

        color_data = utils.get_colorization_data(images)
        img_l = color_data['L'] # 取值范围[-1,1]
        img_ab = color_data['ab'] # 取值范围[-1,1]

        # compute output
        with torch.cuda.amp.autocast():
            output, occm_pred, v_feature, l_feature, _ , attn = model(img_l.repeat(1,3,1,1),cap)
        img_ab_fake = output
        # acc1, acc5 = accuracy(output, target, topk=(1, 5))
        
        fake_rgb_tensors = utils.lab2rgb(torch.cat((img_l, img_ab_fake), dim=1))
        real_rgb_tensors = utils.lab2rgb(torch.cat((img_l, img_ab), dim=1))

        fake_rgbs = utils.tensor2im(fake_rgb_tensors)
        real_rgbs = utils.tensor2im(real_rgb_tensors)
        
        assert save_img_dir != None, "save_img_dir == None"

        for i in range(len(fake_rgbs)):
            psnr=utils.calculate_psnr_np(fake_rgbs[i],real_rgbs[i])
            # psnrs_real.append(psnr) 
            ssim = compare_ssim(fake_rgbs[i],real_rgbs[i],multichannel=True)

            metric_logger.update(psnr=psnr)
            metric_logger.update(ssim=ssim)

            if epoch%10 == 0:
                output_path = os.path.join(save_img_dir,'image','epoch_%d'%epoch)
                if not os.path.exists(output_path):
                    try:     
                        os.makedirs(output_path)
                    except:
                        pass
                if istest:
                    noize = str(random.randint(0,999)).zfill(3)
                    output_path_fake = os.path.join(output_path,keys[i].split('.')[0]+ "_" + cap[i][0:150] + noize + '.png')
                    # print("output_path_fake",output_path_fake)
                    save_img_fake = Image.fromarray(fake_rgbs[i])
                    save_img_fake.save(output_path_fake)

                else:
                    output_path_fake = os.path.join(output_path,keys[i].replace('jpg','png'))
                    # print(output_path)
                    save_img_fake = Image.fromarray(fake_rgbs[i])
                    save_img_fake.save(output_path_fake)
        fn_norm = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        lpips_score = lpips_fn_vgg(fn_norm(fake_rgb_tensors),fn_norm(real_rgb_tensors)).mean()
        metric_logger.update(lpips=lpips_score)

    
    metric_logger.synchronize_between_processes()
    print('* psnr {losses.global_avg:.8f}'
          .format(losses=metric_logger.psnr))
    print('* ssim {losses.global_avg:.8f}'
          .format(losses=metric_logger.ssim))
    print('* lpips {losses.global_avg:.8f}'
          .format(losses=metric_logger.lpips))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
