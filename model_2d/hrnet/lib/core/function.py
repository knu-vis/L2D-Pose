# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
 
import time
import logging
import os

import numpy as np
import torch

from core.evaluate import accuracy
from core.inference import get_final_preds
from utils.transforms import flip_back
from utils.vis import save_debug_images
from evaluators.evaluate_3d import compute_joint_distances, get_metrics
from core.inference_tensor import get_final_preds_softargmax

import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, '..', '..', '..', '..'))
sys.path.append(root_dir)
from run.function import h36m2panoptic_heatmap, normalize_pose, unnormalize_pose

logger = logging.getLogger(__name__)


def validate(config, val_loader, val_dataset, model_2d, model_3d, output_dir, device, stats=None, model_3d_name='sb'):
    # switch to evaluate mode
    model_2d.eval()
    
    if config.DATASET.DATASET == 'panoptic':
        num_joints = 15
    elif config.DATASET.DATASET == 'mpi-inf-3dhp':
        num_joints = 14
    elif config.DATASET.DATASET == 'h36m':
        num_joints = 17
    
    num_samples = len(val_dataset)
    all_preds = np.zeros((num_samples, num_joints, 3), dtype=np.float32)
    all_boxes = np.zeros((num_samples, 6))
    image_path = []
    mpjpe_list, p_mpjpe_list = [], []
    idx = 0
    with torch.no_grad():
        end = time.time()
        for i, (input, target, target_weight, meta) in enumerate(val_loader):
            acc = AverageMeter()
            
            input = input.to(device)
            target = target.cuda(non_blocking=True)
            target_weight = target_weight.cuda(non_blocking=True)
                        
            heatmap = model_2d(input)
            # convert heatmap to proper format
            if config.DATASET.DATASET == 'panoptic':
                heatmap = h36m2panoptic_heatmap(heatmap)
            elif config.DATASET.DATASET == 'mpi-inf-3dhp':
                pass
            preds_2d = get_final_preds_softargmax(config, heatmap.clone().cpu(), meta['center'], meta['scale'])

            if model_3d_name == 'sb':   # SimpleBaseline
                normalized_preds_2d = normalize_pose(preds_2d, torch.tensor(stats['mean_2d'], dtype=torch.float32),
                                                  torch.tensor(stats['std_2d'], dtype=torch.float32)).to(device)
                preds_3d = model_3d(normalized_preds_2d.reshape(-1, num_joints*2)).to(device)
                preds_3d = preds_3d.reshape(-1, num_joints, 3)
                gt_3d = normalize_pose(meta['joints_3d'], stats['mean_3d'], stats['std_3d']).to(device)

                p3d = unnormalize_pose(preds_3d.detach().cpu().numpy(), stats['mean_3d'], stats['std_3d'])
                g3d = unnormalize_pose(gt_3d.detach().cpu().numpy(), stats['mean_3d'], stats['std_3d'])
                mpjpe = compute_joint_distances(p3d, g3d, procrustes=False).mean(-1) * 10.0  # cm to mm
                p_mpjpe = compute_joint_distances(p3d, g3d, procrustes=True).mean(-1) * 10.0  # cm to mm
                mpjpe_list.extend(mpjpe);p_mpjpe_list.extend(p_mpjpe)
            elif model_3d_name == 'videopose':
                pass
            elif model_3d_name == 'iganet':
                pass

            num_images = input.size(0)
            # measure accuracy and record loss
            _, avg_acc, cnt, pred = accuracy(heatmap.cpu().numpy(), target.cpu().numpy())
            acc.update(avg_acc, cnt)

            c = meta['center'].numpy()
            s = meta['scale'].numpy()
            score = meta['score'].numpy()

            all_preds[idx:idx + num_images, :, 0:2] = preds_2d[:, :, 0:2]
            # all_preds[idx:idx + num_images, :, 2:3] = maxvals
            # double check this all_boxes parts
            all_boxes[idx:idx + num_images, 0:2] = c[:, 0:2]
            all_boxes[idx:idx + num_images, 2:4] = s[:, 0:2]
            all_boxes[idx:idx + num_images, 4] = np.prod(s*200, 1)
            all_boxes[idx:idx + num_images, 5] = score
            image_path.extend(meta['image'])

            idx += num_images

            if i % config.PRINT_FREQ == 0:
                msg = f'Test: [{i}/{len(val_loader)}]\t Accuracy_2d {acc.avg:.3f}\t MPJPE {np.mean(mpjpe_list):.2f}mm\t P-MPJPE {np.mean(p_mpjpe_list):.2f}mm'
                logger.info(msg)

                prefix = '{}_{}'.format(os.path.join(output_dir, 'val'), i)
                # save_debug_images(config, input, meta, target, pred*4, heatmap, prefix)
                
        _, acc_2d = val_dataset.evaluate(config, all_preds, output_dir, all_boxes, image_path)


    return np.mean(mpjpe_list), np.mean(p_mpjpe_list), acc_2d



# markdown format output
def _print_name_value(name_value, full_arch_name):
    names = name_value.keys()
    values = name_value.values()
    num_values = len(name_value)
    logger.info(
        '| Arch ' +
        ' '.join(['| {}'.format(name) for name in names]) +
        ' |'
    )
    logger.info('|---' * (num_values+1) + '|')

    if len(full_arch_name) > 15:
        full_arch_name = full_arch_name[:8] + '...'
    logger.info(
        '| ' + full_arch_name + ' ' +
        ' '.join(['| {:.3f}'.format(value) for value in values]) +
         ' |'
    )


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count != 0 else 0
