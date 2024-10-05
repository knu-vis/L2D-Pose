# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import os
import numpy as np
import copy
import json 
from glob import glob
from common.skeleton import Skeleton
from common.mocap_dataset import MocapDataset
from common.camera import normalize_screen_coordinates

from common.custom_camera import project_to_camera
from common.custom_camera import transform_world_to_camera as transform_world_to_camera_base
from common.custom_camera import transform_camera_to_world as transform_camera_to_world_base
       
panoptic_skeleton = Skeleton(parents=[2, 0, -1, 0, 3, 4, 2, 6, 7, 0, 9, 10, 2, 12, 13],
                             joints_left=[3, 4, 5, 6, 7, 8],
                             joints_right=[9, 10, 11, 12, 13, 14])

body_edges = np.array([[0,1],[0,2],[0,3],[0,9],[3,4],[9,10],[4,5],
                        [10,11],[2,6],[2,12],[6,7],[12,13],[7,8],[13,14]])
bone_length_limit = [50,80,40,40,50,50,45,45,
                     30,30,65,65,60,60,
                     20,20,25,25]
M = np.array([[1.0, 0.0, 0.0],
              [0.0, 0.0, -1.0],
              [0.0, 1.0, 0.0]])
frame_threshold = 50
width, height = 1920, 1080
NUM_JOINTS = 15
def dist(a, b):  # 좌표 간 거리 구하는 공
    return np.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2 + (a[2] - b[2])**2)

data_dir = '/media/vis/SSD_2TB/3d-guided_2d-HPE/model_3d/iganet/dataset/panoptic_3d'


def load_cameras(calib_path):    
    cam_list = [(0, 12), (0, 6), (0, 23), (0, 13), (0, 3)]
    with open(calib_path) as rf:
        camera_file = json.load(rf)['cameras']
    cameras = []
    for cam in camera_file:
        if (cam['panel'], cam['node']) in cam_list:
            sel_cam = {}
            sel_cam['K'] = np.array(cam['K'])
            sel_cam['distCoef'] = np.array(cam['distCoef'])
            sel_cam['R'] = np.array(cam['R']).dot(M)
            sel_cam['t'] = np.array(cam['t']).reshape((3, 1))
            cameras.append(sel_cam)
    return cameras


def load_data(sequence):
    interval = 3
    data_3d, data_2d, data_camera = [], [], []
    for seq in sequence:
        seq_data = []
        path = os.path.join(data_dir, f"{seq}/hdPose3d_stage1_coco19/*.json")
        calib_path = os.path.join(data_dir, f"{seq}/calibration_{seq}.json")
        cameras = load_cameras(calib_path)

        frames = sorted(glob(path))
        for i, frame in enumerate(frames):
            if i % interval == 0:
                with open(frame, 'r') as rf:
                    poses = json.load(rf)['bodies']
                if len(poses) == 0:
                    continue

                for pose in poses:
                    valid_pose = True
                    temp = np.array(pose['joints19']).reshape((-1, 4))[:NUM_JOINTS, :3].dot(M)
                    for edge, bone_limit in zip(body_edges, bone_length_limit):
                        if dist(temp[edge[0]], temp[edge[1]]) > bone_limit:
                            valid_pose = False
                            break
                    if not valid_pose:
                        continue
                    seq_data.append(temp)
        
        pose3d_world = np.array(seq_data).reshape((-1, 3))
        for idx, cam in enumerate(cameras):
            K, R, T, Kd = cam['K'], cam['R'], cam['t'], cam['distCoef']
            pose2d = project_to_camera(pose3d_world.T, K, R, T, Kd).transpose()[:, :2].reshape(-1, NUM_JOINTS, 2)
            valid_idx = []
            for idx, pose_2d in enumerate(pose2d):
                if all(pose_2d[:,0] < width+frame_threshold) and all(pose_2d[:,0] > 0-frame_threshold) and \
                    all(pose_2d[:,1] < height+frame_threshold) and all(pose_2d[:,1] > 0-frame_threshold):
                    valid_idx.append(idx)
            if len(valid_idx) == 0:
                continue
            pose2d = pose2d[valid_idx]
            
            pose2d[..., :2] = normalize_screen_coordinates(pose2d[..., :2], w=width, h=height)
            pose3d_camera = transform_world_to_camera_base(pose3d_world.reshape((-1, 3)), R, T)
            pose3d_camera = pose3d_camera.reshape((-1, NUM_JOINTS, 3))  # nx(15x3)
            
            root_joints = pose3d_camera[:, 2, :]
            pose3d_camera = pose3d_camera - root_joints[:, np.newaxis, :]
            
            intrinsic = np.array([K[0,0], K[1,1], K[0, 2], K[1, 2], Kd[0], Kd[1], Kd[4], Kd[2], Kd[3]])
        
            data_camera.append(intrinsic)
            data_3d.append(np.array(pose3d_camera)[valid_idx] / 100)   # cm to m
            data_2d.append(pose2d)
            
            # if idx > 4:
            #     break
    
    return data_camera, data_3d, data_2d 

