# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import os
import json_tricks as json
from collections import OrderedDict
import glob
import copy
import pickle
import cv2 
import torch

import numpy as np
from scipy.io import loadmat, savemat

from dataset.JointsDataset_3d import JointsDataset
from utils.transforms import projectPoints, get_scale
from utils.transforms_h36m import transform_camera_to_world, transform_world_to_camera, project_to_camera
from .compute_normalization_stats_h36m import move_hip_joint_to_zero
from .compute_normalization_stats_h36m import compute_mean_std


logger = logging.getLogger(__name__)

TRAIN_SUBJECT = ['S1', 'S5', 'S6', 'S7', 'S8']
VAL_SUBJECT =   ['S9', 'S11']

JOINTS_DEF = {
    'mid-hip': 0,
    'r-hip': 1,
    'r-knee': 2,
    'r-ankle': 3,
    'l-hip': 4,
    'l-knee': 5,
    'l-ankle': 6,
    'thorax': 7,
    'neck': 8,
    'nose': 9,
    'head': 10,
    'l-shoulder': 11,
    'l-elbow': 12,
    'l-wrist': 13,
    'r-shoulder': 14,
    'r-elbow': 15,
    'r-wrist': 16,
}

class Human36MDataset(JointsDataset):
    def __init__(self, cfg, root, image_set, is_train, transform=None, temporal=False):
        super().__init__(cfg, root, image_set, is_train, transform)
        self.num_joints = 17
        self.flip_pairs = [[3, 6], [2, 5], [1, 4], [16, 13], [15, 12], [14, 11]]
        self.parent_ids = [0, 0, 1, 2, 0, 4, 5, 0, 7, 8, 9, 8, 11, 12, 8, 14, 15]
        self.root_id = 0
        self.cam_list = ['54138969', '55011271', '58860488', '60457274']
        self.temporal = temporal

        self.upper_body_ids = (7, 8, 9, 10, 11, 12, 13, 14, 15, 16)
        self.lower_body_ids = (0, 1, 2, 3, 4, 5, 6)
        self.remove_joints = np.array([4, 5, 9, 10, 11, 16, 20, 21, 22, 23, 24, 28, 29, 30, 31])
        
        # self._interval = 3
        if self.image_set == 'train':
            self.subject_list = TRAIN_SUBJECT
            self._interval = 40
        elif self.image_set == 'valid':
            self.subject_list = VAL_SUBJECT
            if self.temporal:
                self._interval = 5
            else:
                self._interval = 100

        self.db_file = os.path.join(self.root, f'h36m_{self.image_set}.pkl')
        if os.path.exists(self.db_file):
            info = pickle.load(open(self.db_file, 'rb'))
            assert info['interval'] == self._interval
            self.db = info['db']
        else:
            self.db = self._get_db()
            info = {'interval': self._interval, 'db': self.db}
            pickle.dump(info, open(self.db_file, 'wb'))
        
        if self.image_set == 'train':
            if not os.path.exists(os.path.join(self.root, 'h36m_stats.pkl')):
                self.mean_2d, self.std_2d, self.mean_3d, self.std_3d =  compute_mean_std(self.db)
                stats = {'mean_2d': self.mean_2d, 'std_2d': self.std_2d,
                         'mean_3d': self.mean_3d, 'std_3d': self.std_3d}
                pickle.dump(stats, open(os.path.join(self.root, 'h36m_stats.pkl'), 'wb'))
            else:
                stats = pickle.load(open(os.path.join(self.root, 'h36m_stats.pkl'), 'rb'))
                self.mean_2d, self.std_2d = stats['mean_2d'], stats['std_2d']
                self.mean_3d, self.std_3d = stats['mean_3d'], stats['std_3d']
            self.stats = {'mean_2d': self.mean_2d, 'std_2d': self.std_2d, 
                        'mean_3d': self.mean_3d, 'std_3d': self.std_3d}

        self.db_size = len(self.db)
        logger.info('=> load {} samples'.format(len(self.db)))

    def _get_db(self):
        gt_db = []
        cameras = json.load(open(os.path.join(self.root, 'camera_parameters.json')))
        for subject in self.subject_list:
            for action in sorted(os.listdir(os.path.join(self.root, subject))):
                annot = pickle.load(open(os.path.join(self.root, subject, action, 'annot.pkl'), 'rb'))
                
                total_frame = int(len(annot['frame']) / len(self.cam_list))
                for cam_idx in range(len(self.cam_list)):
                    for frame_idx in range(0, total_frame, self._interval):
                        cur_idx = cam_idx * total_frame + frame_idx
                        camera_name = str(annot['camera'][cur_idx])
                        cur_frame = annot['frame'][cur_idx]
                        image = os.path.join(subject, action, 'imageSequence', camera_name, f'img_{cur_frame:08d}.jpg')
                        
                        width = cameras['intrinsics'][camera_name]['width']
                        height = cameras['intrinsics'][camera_name]['height']
                        # c = np.array([width / 2.0, height / 2.0])
                        # s = get_scale((width, height), self.image_size)
                        
                        pose2d = np.delete(annot['pose/2d'][cur_idx], self.remove_joints, axis=0)
                        pose3d_camera = np.delete(annot['pose/3d'][cur_idx], self.remove_joints, axis=0) / 10    # mm to cm
                        hip_camera = pose3d_camera[self.root_id]
                        pose3d_camera = move_hip_joint_to_zero(pose3d_camera)
                        
                        # joint visibility
                        joints_vis = np.ones((self.num_joints, 2))
                        joints_vis[(pose2d[:,0] < 0) | (pose2d[:,0] >= width) | (pose2d[:,1] < 0) | (pose2d[:,1] >= height)] = 0
                        
                        if np.any(joints_vis[self.root_id] == 0):
                            continue
                        
                        data_numpy = cv2.imread(os.path.join(self.root, image), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
                        if np.all(data_numpy[:,:,0] == 0):  # broken image
                            continue
                        
                        our_cam = {'R': np.array(cameras['extrinsics'][subject][camera_name]['R']),
                                   'T': np.array(cameras['extrinsics'][subject][camera_name]['t']) / 10,    # mm to cm
                                   'K': np.array(cameras['intrinsics'][camera_name]['K']),
                                   'distCoef': np.array(cameras['intrinsics'][camera_name]['distCoef'])}
                        
                        gt_db.append({
                            'image': os.path.join(self.root, image),
                            'joints_3d': pose3d_camera,
                            'joints_2d': pose2d,
                            'joints_2d_vis': joints_vis,
                            'camera': our_cam,
                            'action': action,
                            'width': width,
                            'height': height,
                            'hip_camera': hip_camera,
                        })
                        
        return gt_db
    

    def evaluate(self, cfg, preds, output_dir, *args, **kwargs):
        preds = preds[:, :, 0:2] + 1.0
        if output_dir:
            pred_file = os.path.join(output_dir, 'pred.mat')
            savemat(pred_file, mdict={'preds': preds})
        if 'test' in cfg.DATASET.TEST_SET:
            return {'Null': 0.0}, 0.0
        
        SC_BIAS = 0.6
        threshold = 0.5
        
        ######################
        gt_num = self.db_size
        assert len(preds) == gt_num, 'number mismatch'
        pos_gt_src = np.empty((17, 2, 0))
        jnt_visible = np.empty((17, 0))
        for v in self.db:
            pos_gt_src = np.append(pos_gt_src, v['joints_2d'][:, :2, np.newaxis], axis=2)
            jnt_visible = np.append(jnt_visible, v['joints_2d_vis'][:, 0].astype(int)[:, np.newaxis], axis=1)
        
        pos_pred_src = np.transpose(preds, [1, 2, 0])
        headboxes_src = np.concatenate((pos_gt_src[0, :, :][np.newaxis, :],
                                        pos_gt_src[1, :, :][np.newaxis, :]), axis=0)
        
        uv_error = pos_pred_src - pos_gt_src
        uv_err = np.linalg.norm(uv_error, axis=1)
        headsizes = headboxes_src[1, :, :] - headboxes_src[0, :, :]
        headsizes = np.linalg.norm(headsizes, axis=0)
        scale = np.multiply(headsizes, np.ones((len(uv_err), 1)))
        scaled_uv_err = np.divide(uv_err, scale)
        scaled_uv_err = np.multiply(scaled_uv_err, jnt_visible)
        jnt_count = np.sum(jnt_visible, axis=1)
        less_than_threshold = np.multiply((scaled_uv_err <= threshold),
                                            jnt_visible)
        PCKh = np.divide(100.*np.sum(less_than_threshold, axis=1), jnt_count)
        
        # save
        rng = np.arange(0, 0.5+0.01, 0.01)
        pckAll = np.zeros((len(rng), self.num_joints))

        for r in range(len(rng)):
            threshold = rng[r]
            less_than_threshold = np.multiply(scaled_uv_err <= threshold,
                                              jnt_visible)
            pckAll[r, :] = np.divide(100.*np.sum(less_than_threshold, axis=1),
                                     jnt_count)

        PCKh = np.ma.array(PCKh, mask=False)
        jnt_count = np.ma.array(jnt_count, mask=False)
        jnt_ratio = jnt_count / np.sum(jnt_count).astype(np.float64)

        
        name_value = [
            ('Head', 0.5 * (PCKh[JOINTS_DEF['nose']] + PCKh[JOINTS_DEF['head']])),
            ('Neck', 0.5 * (PCKh[JOINTS_DEF['neck']] + PCKh[JOINTS_DEF['thorax']])),
            ('Shoulder', 0.5 * (PCKh[JOINTS_DEF['l-shoulder']] + PCKh[JOINTS_DEF['r-shoulder']])),
            ('Elbow', 0.5 * (PCKh[JOINTS_DEF['l-elbow']] + PCKh[JOINTS_DEF['r-elbow']])),
            ('Wrist', 0.5 * (PCKh[JOINTS_DEF['l-wrist']] + PCKh[JOINTS_DEF['r-wrist']])),
            ('Hip', (1/3) * (PCKh[JOINTS_DEF['mid-hip']] + PCKh[JOINTS_DEF['l-hip']] + PCKh[JOINTS_DEF['r-hip']])),
            ('Knee', 0.5 * (PCKh[JOINTS_DEF['l-knee']] + PCKh[JOINTS_DEF['r-knee']])),
            ('Ankle', 0.5 * (PCKh[JOINTS_DEF['l-ankle']] + PCKh[JOINTS_DEF['r-ankle']])),
            ('Mean', np.sum(PCKh * jnt_ratio)),
            ('Mean@0.1', np.sum(pckAll[11, :] * jnt_ratio))
        ]
        
        name_value = OrderedDict(name_value)

        return name_value, name_value['Mean']

