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
from utils.camera_utils_panoptic import transform_world_to_camera
from utils.camera_utils_panoptic import transform_camera_to_world
from .compute_normalization_stats_panoptic import move_hip_joint_to_zero
from .compute_normalization_stats_panoptic import compute_mean_std


logger = logging.getLogger(__name__)

TRAIN_LIST = [
    '171204_pose1', '171204_pose2', '171204_pose3', '171026_pose1', '171026_pose2',
    '171026_cello3', '161029_piano3', '161029_piano4', '170407_office2', 
    ]
VAL_LIST =   [
    '171026_pose3', '161029_piano2', '170915_office1',
    ]
 

JOINTS_DEF = {
    'neck': 0,
    'nose': 1,
    'mid-hip': 2,
    'l-shoulder': 3,
    'l-elbow': 4,
    'l-wrist': 5,
    'l-hip': 6,
    'l-knee': 7,
    'l-ankle': 8,
    'r-shoulder': 9,
    'r-elbow': 10,
    'r-wrist': 11,
    'r-hip': 12,
    'r-knee': 13,
    'r-ankle': 14,
    # 'l-eye': 15,
    # 'l-ear': 16,
    # 'r-eye': 17,
    # 'r-ear': 18,
}
M = np.array([[1.0, 0.0, 0.0],
              [0.0, 0.0, -1.0],
              [0.0, 1.0, 0.0]])

class PanopticDataset(JointsDataset):
    def __init__(self, cfg, root, image_set, is_train, transform=None, temporal=False):
        super().__init__(cfg, root, image_set, is_train, transform)
        self.num_joints = 15
        self.flip_pairs = [[14, 8], [13, 7], [12, 6], [11, 5], [10, 4], [9, 3]]
        self.parent_ids = [2, 0, 2, 0, 3, 4, 2, 6, 7, 0, 9, 10, 2, 12, 13]
        self.root_id = 2
        self.cam_list = [(0, 12), (0, 6), (0, 23), (0, 13), (0, 3)]
        self.temporal = temporal

        self.upper_body_ids = (0, 1, 3, 4, 5, 9, 10, 11)
        self.lower_body_ids = (2, 6, 7, 8, 12 ,13, 14)
        
        # self._interval = 3
        if self.image_set == 'train':
            self.sequence_list = TRAIN_LIST
            self._interval = 15
        elif self.image_set == 'valid':
            self.sequence_list = VAL_LIST
            if self.temporal:
                self._interval = 3
            else:   
                self._interval = 15

        # db 미리 저장
        self.db_file = os.path.join(self.root, f'panoptic_{self.image_set}.pkl')
        if os.path.exists(self.db_file):
            info = pickle.load(open(self.db_file, 'rb'))
            assert info['interval'] == self._interval
            self.db = info['db']
        else:  
            self.db = self._get_db()
            info = {'interval': self._interval, 'db': self.db}
            pickle.dump(info, open(self.db_file, 'wb'))
        
        
        if self.image_set == 'train':
            if not os.path.exists(os.path.join(self.root, 'panoptic_stats.pkl')):
                self.mean_2d, self.std_2d, self.mean_3d, self.std_3d =  compute_mean_std(self.db)
                stats = {'mean_2d': self.mean_2d, 'std_2d': self.std_2d,
                         'mean_3d': self.mean_3d, 'std_3d': self.std_3d}
                pickle.dump(stats, open(os.path.join(self.root, 'panoptic_stats.pkl'), 'wb'))
            else:
                stats = pickle.load(open(os.path.join(self.root, 'panoptic_stats.pkl'), 'rb'))
                self.mean_2d, self.std_2d = stats['mean_2d'], stats['std_2d']
                self.mean_3d, self.std_3d = stats['mean_3d'], stats['std_3d']
            self.stats = {'mean_2d': self.mean_2d, 'std_2d': self.std_2d, 
                        'mean_3d': self.mean_3d, 'std_3d': self.std_3d}
           
        self.db_size = len(self.db)
        logger.info('=> load {} samples'.format(len(self.db)))

    def _get_db(self):
        width, height = 1920, 1080
        gt_db = []
        for seq in self.sequence_list:
            cameras = self._get_cam(seq)
            
            cur_anno = os.path.join(self.root, seq, 'hdPose3d_stage1_coco19')
            anno_files = sorted(glob.iglob('{:s}/*.json'.format(cur_anno)))
            for i, file in enumerate(anno_files):
                if i % self._interval == 0:
                    with open(file) as dfile:
                        bodies = json.load(dfile)['bodies']
                    if len(bodies) != 1:   
                        continue

                    pose3d = np.array(bodies[0]['joints19'], dtype=np.float32).reshape((-1, 4))
                    pose3d = pose3d[:self.num_joints]
                    joints_vis = pose3d[:, -1] > 0.1
                    if not joints_vis[self.root_id]:
                        continue
                    
                    pose3d[:, 0:3] = pose3d[:, 0:3].dot(M)
                    # joints_vis_3d = np.repeat(np.reshape(joints_vis, (-1, 1)), 3, axis=1)
                    for k, v in sorted(cameras.items()):
                        postfix = os.path.basename(file).replace('body3DScene', '')
                        prefix = '{:02d}_{:02d}'.format(k[0], k[1])
                        image = os.path.join(seq, 'hdImgs', prefix, prefix + postfix)
                        image = image.replace('json', 'jpg')
                        
                        pose2d = np.zeros((pose3d.shape[0], 2), dtype=np.float32)
                        pose2d[:, :2] = projectPoints(pose3d[:, 0:3].transpose(), v['K'], v['R'],
                                                        v['t'], v['distCoef']).transpose()[:, :2]
                        x_check = np.bitwise_and(pose2d[:, 0] >= 0,pose2d[:, 0] <= width - 1)
                        y_check = np.bitwise_and(pose2d[:, 1] >= 0,pose2d[:, 1] <= height - 1)
                        check = np.bitwise_and(x_check, y_check)
                        joints_vis = pose3d[:, -1] > 0.1
                        joints_vis[np.logical_not(check)] = 0
                        joints_vis_2d = np.repeat(np.reshape(joints_vis, (-1, 1)), 2, axis=1)
                        
                        if not joints_vis[self.root_id]:
                            continue
                        
                        data_numpy = cv2.imread(os.path.join(self.root, image), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
                        if np.all(data_numpy[:,:,0] == 0):  # broken image
                            continue
                        
                        pose3d_camera = transform_world_to_camera(pose3d[:, 0:3], v['R'], v['t'])
                        hip_camera = pose3d_camera[self.root_id]
                        pose3d_camera = move_hip_joint_to_zero(pose3d_camera)
                        our_cam = {}
                        our_cam['R'] = v['R']
                        our_cam['T'] = v['t']
                        our_cam['K'] = v['K']
                        our_cam['distCoef'] = v['distCoef']

                        gt_db.append({
                            'key': "{}_{}{}".format(seq, prefix, postfix.split('.')[0]),
                            'image': os.path.join(self.root, image),
                            'joints_3d': pose3d_camera,
                            # 'joints_3d_vis': joints_vis_3d,
                            'joints_2d': pose2d,
                            'joints_2d_vis': joints_vis_2d,
                            'camera': our_cam,
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
        pos_gt_src = np.empty((15, 2, 0))
        jnt_visible = np.empty((15, 0))
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
            ('Nose', PCKh[JOINTS_DEF['nose']]),
            ('Neck', PCKh[JOINTS_DEF['neck']]),
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


    def _get_cam(self, seq):
        cam_file = os.path.join(self.root, seq, 'calibration_{:s}.json'.format(seq))
        with open(cam_file) as cfile:
            calib = json.load(cfile)
        
        cameras = {}
        for cam in calib['cameras']:
            if (cam['panel'], cam['node']) in self.cam_list:
                sel_cam = {}
                sel_cam['K'] = np.array(cam['K'])
                sel_cam['distCoef'] = np.array(cam['distCoef'])
                sel_cam['R'] = np.array(cam['R']).dot(M)
                sel_cam['t'] = np.array(cam['t']).reshape((3, 1))
                cameras[(cam['panel'], cam['node'])] = sel_cam
        return cameras
