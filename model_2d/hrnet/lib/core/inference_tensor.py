# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import numpy as np
from utils.transforms_tensor import transform_preds
import torch.nn as nn
import torch

def softargmax2d(input, beta=100):
    *_, h, w = input.shape
    input = input.reshape(*_, h * w)
    input = nn.functional.softmax(beta * input, dim=-1)
    indices_c, indices_r = np.meshgrid(
        np.linspace(0, 1, w),
        np.linspace(0, 1, h),
        indexing='xy'
    )
    indices_r = torch.tensor(np.reshape(indices_r, (-1, h * w)), dtype=torch.float32)
    indices_c = torch.tensor(np.reshape(indices_c, (-1, h * w)), dtype=torch.float32)
    
    result_r = torch.sum((h - 1) * input * indices_r, dim=-1)
    result_c = torch.sum((w - 1) * input * indices_c, dim=-1)
    result = torch.stack([result_c, result_r], dim=-1)

    return result


def softargmax1d(input, beta=100):
    # input shape: (N, L)
    *_, length = input.shape
    softmax = nn.functional.softmax(beta * input, dim=-1)
    indices = torch.linspace(0, length - 1, length, device=input.device)
    softargmax = torch.sum(softmax * indices, dim=-1)
    
    return softargmax


def get_final_preds_softargmax(config, batch_heatmaps, center, scale, beta=100):
    coords = softargmax2d(batch_heatmaps, beta=beta)

    heatmap_height = batch_heatmaps.shape[2]
    heatmap_width = batch_heatmaps.shape[3]

    # post-processing
    if config.TEST.POST_PROCESS:
        for n in range(coords.shape[0]):
            for p in range(coords.shape[1]):
                hm = batch_heatmaps[n][p]
                px = int(math.floor(coords[n][p][0] + 0.5))
                py = int(math.floor(coords[n][p][1] + 0.5))
                if 1 < px < heatmap_width-1 and 1 < py < heatmap_height-1:
                    diff = torch.tensor([hm[py][px+1] - hm[py][px-1],
                                        hm[py+1][px]-hm[py-1][px]])
                    coords[n][p] += np.sign(diff) * .25
    preds = coords.clone()

    # Transform back
    for i in range(coords.shape[0]):
        preds[i] = transform_preds(
            coords[i], center[i], scale[i], [heatmap_width, heatmap_height]
        )

    return preds