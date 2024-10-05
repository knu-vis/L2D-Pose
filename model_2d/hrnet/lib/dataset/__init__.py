# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# from .coco import COCODataset as coco
# from .coco_h36m import COCODataset as coco_h36m
from .panoptic import PanopticDataset as panoptic
from .h36m import Human36MDataset as h36m
from .h36m_heatmap import Human36MDataset as h36m_heatmap
from .h36m_t81 import Human36MDataset as h36m_t81