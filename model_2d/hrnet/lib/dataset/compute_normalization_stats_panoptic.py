import numpy as np
import os

from utils.camera_utils_panoptic import transform_world_to_camera

select_cams = list(range(31))

threshold = 200
width, height = 1920, 1080

def get_data(db):
    data_2d, data_3d = [], []
    for idx in range(len(db)):
        for cam_idx in select_cams:
            R, T = db[idx]['camera'][cam_idx]['R'], db[idx]['camera'][cam_idx]['T']
            pose_cam3d = transform_world_to_camera(db[idx]['joints_3d'], R, T)
            
            data_3d.append(move_hip_joint_to_zero(pose_cam3d))
            
            if all(db[idx]['joints_2d'][cam_idx][:, 0] < width+threshold) and all(db[idx]['joints_2d'][cam_idx][:, 0] > 0-threshold) and \
                    all(db[idx]['joints_2d'][cam_idx][:, 1] < height+threshold) and all(db[idx]['joints_2d'][cam_idx][:, 1] > 0-threshold):
                data_2d.append(db[idx]['joints_2d'][cam_idx])

    return np.array(data_2d), np.array(data_3d)

def compute_mean_std(db):
    data_2d, data_3d = get_data(db)
    
    return np.mean(data_2d, axis=0), np.std(data_2d, axis=0), np.mean(data_3d, axis=0), np.std(data_3d, axis=0)
            

def move_hip_joint_to_zero(pose_3d):
    # pose_3d: 15x3
    hip_joint = pose_3d[2]
    
    return pose_3d - hip_joint