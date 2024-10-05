# references:
# https://github.com/una-dinosauria/3d-pose-baseline/blob/master/src/data_utils.py

import copy
import os
from glob import glob
import json
import pickle

import h5py
import numpy as np

from .camera_utils_panoptic import project_to_camera
from .camera_utils_panoptic import transform_world_to_camera as transform_world_to_camera_base
from .camera_utils_panoptic import transform_camera_to_world as transform_camera_to_world_base

# Panoptic IDs for training and testing.
TRAIN_LIST = [
    '171204_pose1', '171204_pose2', '171204_pose3', '171026_pose1', '171026_pose2',
    '171026_cello3', '161029_piano3', '161029_piano4', '170407_office2', 
    '170407_haggling_a1', '170407_haggling_a2', '160906_pizza1', '160906_band1', '160906_band2',
    '160906_band3', '161029_sports1', '160422_ultimatum1'
    ]
VALID_LIST =   [
    '171026_pose3', '161029_piano2', '170915_office1', '170407_haggling_a3', '160906_band4'
    ]

select_cam = list(range(31))
# select_cam = [3, 6, 12, 13, 23, 30]

# Joints in CMU Panoptic.
PANOPTIC_NAMES = [""] * 15
PANOPTIC_NAMES[0] = "Neck"
PANOPTIC_NAMES[1] = "Nose"
PANOPTIC_NAMES[2] = "Hip"
PANOPTIC_NAMES[3] = "LShoulder"
PANOPTIC_NAMES[4] = "LElbow"
PANOPTIC_NAMES[5] = "LWrist"
PANOPTIC_NAMES[6] = "LHip"
PANOPTIC_NAMES[7] = "LKnee"
PANOPTIC_NAMES[8] = "Lankle"
PANOPTIC_NAMES[9] = "RShoulder"
PANOPTIC_NAMES[10] = "RElbow"
PANOPTIC_NAMES[11] = "RWrist"
PANOPTIC_NAMES[12] = "RHip"
PANOPTIC_NAMES[13] = "RKnee"
PANOPTIC_NAMES[14] = "Rankle"

NUM_JOINTS = len(PANOPTIC_NAMES)


def dist(a, b):
    return np.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2 + (a[2] - b[2])**2)
body_edges_panoptic = np.array([[0,1],[0,2],[0,3],[0,9],[3,4],[9,10],[4,5],
                                [10,11],[2,6],[2,12],[6,7],[12,13],[7,8],[13,14]])
bone_length_limit = [50,80,40,40,50,50,45,45,
                     30,30,65,65,60,60,
                     20,20,25,25]
M = np.array([[1.0, 0.0, 0.0],
              [0.0, 0.0, -1.0],
              [0.0, 1.0, 0.0]])

def load_data(data_dir, sequence):
    interval = 3
    data = {}
    for seq in sequence:
        seq_data = []
        path = os.path.join(data_dir, f"{seq}/hdPose3d_stage1_coco19/*.json")

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
                    for edge, bone_limit in zip(body_edges_panoptic, bone_length_limit):
                        if dist(temp[edge[0]], temp[edge[1]]) > bone_limit:
                            valid_pose = False
                            break
                    if not valid_pose:
                        continue
                    seq_data.append(temp.reshape(-1))
        data[seq] = np.array(seq_data)
    return data


def transform_world_to_camera(pose_set, cams):
    """Transform 3d poses from world coordinate to camera coordinate.

    Args:
        pose_set (dict[tuple, numpy.array]): Dictionary with 3d poses.
        cams (dict[tuple, tuple]): Dictionary with cameras.
        ncams (int, optional): Number of cameras per subject. Defaults to 4.

    Returns:
        t3d_camera (dict[tuple, numpy.array]): Dictionary with 3d poses in camera coordinate.
    """
    width, height, frame_threshold = 1920, 1080, 200
    cam_interval = 1
    t3d_camera = {}
    for i, seq in enumerate(sorted(pose_set.keys())):
        t3d_world = pose_set[seq]  # nx(15x3)
        t3d_world = t3d_world.reshape((-1, 3))  # (nx15)x3

        for cam_idx, cam in enumerate(cams):
            K, R, T, Kd = cam['K'], cam['R'], cam['t'], cam['distCoef']
            K, R, T, Kd = np.array(K), np.array(R), np.array(T), np.array(Kd)
            pts2d = project_to_camera(t3d_world.T, K, R, T, Kd).transpose()[:, :2].reshape(-1, NUM_JOINTS, 2)
            # 2D pose가 frame 밖으로 많이 나가는 경우 제거
            valid_idx = []
            for idx, pose_2d in enumerate(pts2d):
                if all(pose_2d[:,0] < width+frame_threshold) and all(pose_2d[:,0] > 0-frame_threshold) and \
                    all(pose_2d[:,1] < height+frame_threshold) and all(pose_2d[:,1] > 0-frame_threshold):
                    valid_idx.append(idx)
            valid_t3d = t3d_world.reshape((-1, NUM_JOINTS * 3))[valid_idx]
            camera_coord = transform_world_to_camera_base(valid_t3d.reshape((-1, 3)), R, T)
            camera_coord = camera_coord.reshape((-1, NUM_JOINTS * 3))  # nx(15x3)

            t3d_camera[(seq, f"{cam_idx:02d}")] = camera_coord
        

    return t3d_camera
        

def postprocess_3d(pose_set):
    """Centerize 3d joint points around root joint.

    Args:
        pose_set (dict[tuple, numpy.array]): Dictionary with 3d data.

    Returns:
        pose_set (dict[tuple, numpy.array]): Dictionary with 3d data centred around root (center hip) joint.
        root_positions (dict[tuple, numpy.array]): Dictionary with the original 3d position of each pose.
    """
    root_positions = {}
    for k in sorted(pose_set.keys()):
        poses = pose_set[k]  # nx(32x3)

        # Keep track of global position.
        root_begin = PANOPTIC_NAMES.index("Hip") * 3
        root_position = copy.deepcopy(poses[:, root_begin : root_begin + 3])  # nx3

        # Centerize around root.
        poses = poses - np.tile(root_position, [1, NUM_JOINTS])

        pose_set[k] = poses
        root_positions[k] = root_position

    return pose_set, root_positions


def compute_normalization_stats(data, dim, predict_15=True):
    """Compute normalization statistics: mean, std, dimensions to use and ignore.

    Args:
        data (numpy.array): nxd array of poses
        dim (int): Dimensionality of the pose. 2 or 3.
        predict_15 (bool, optional): Whether to use only 14 joints. Defaults to False.

    Returns:
        data_mean (numpy.array): Vector with the mean of the data.
        data_std (numpy.array): Vector with the standard deviation of the data.
        dim_to_ignore (numpy.array): List of dimensions not used in the model.
        dim_to_use (numpy.array): List of dimensions used in the model.
    """
    assert dim in [2, 3], "dim must be 2 or 3."

    data_mean = np.mean(data, axis=0)
    data_std = np.std(data, axis=0)

    if dim == 2:
        # Get dimensions of 16 2d points to use.
        dim_to_ignore = np.where(
            np.array([x in [""] for x in PANOPTIC_NAMES])
        )[0]
        dim_to_ignore = np.sort(np.hstack([dim_to_ignore * 2, dim_to_ignore * 2 + 1]))
        dim_to_use = np.delete(np.arange(NUM_JOINTS * 2), dim_to_ignore)
    else:  # dim == 3
        # Get dimensions of 16 (or 14) 3d points to use.
        if predict_15:
            dim_to_ignore = np.where(
                np.array([x in [""] for x in PANOPTIC_NAMES])
            )[0]
        else:  # predict 16 points
            dim_to_ignore = np.where(np.array([x in [""] for x in PANOPTIC_NAMES]))[
                0
            ]

        dim_to_ignore = np.sort(
            np.hstack([dim_to_ignore * 3, dim_to_ignore * 3 + 1, dim_to_ignore * 3 + 2])
        )
        dim_to_use = np.delete(np.arange(NUM_JOINTS * 3), dim_to_ignore)

    return data_mean, data_std, dim_to_ignore, dim_to_use


def normalize_data(data, data_mean, data_std, dim_to_use):
    """Normalize poses in the dictionary.

    Args:
        data (dict[tuple, numpy.array]): Dictionary with the poses.
        data_mean (numpy.array): Vector with the mean of the data.
        data_std (numpy.array): Vector with the std of the data.
        dim_to_use (numpy.array): Dimensions to keep in the data.

    Returns:
        data_normalized (dict[tuple, numpy.array]): Dictionary with same keys as data, but values have been normalized.
    """
    data_normalized = {}

    for key in sorted(data.keys()):
        data[key] = data[key][:, dim_to_use]  # remove joints to ignore
        mu = data_mean[dim_to_use]
        sigma = data_std[dim_to_use]
        data_normalized[key] = np.divide((data[key] - mu), sigma+1e-8)

    return data_normalized


def read_3d_data(data_dir, cams, camera_frame=True, predict_15=True):
    """Load 3d poses, zero-centred and normalized.

    Args:
        actions (list[str]): Actions to load.
        data_dir (str): Directory where the data can be loaded from.
        cams (dict[tuple, tuple]): Dictionary with camera parameters.
        camera_frame (bool, optional): Whether to convert the data to camera coordinates. Defaults to True.
        predict_15 (bool, optional): Whether to predict only 14 joints. Defaults to False.

    Returns:
        train_set (dict[tuple, numpy.array]): Dictionary with loaded 3d poses for training.
        test_set (dict[tuple, numpy.array]): Dictionary with loaded 3d poses for testing.
        data_mean (numpy.array): Vector with the mean of the 3d training data.
        data_std (numpy.array): Vector with the standard deviation of the 3d training data.
        dim_to_ignore (list[int]): List with the dimensions not to predict.
        dim_to_use (list[int]): List with the dimensions to predict.
        train_root_positions (dict[tuple, numpy.array]): Dictionary with the 3d positions of the root in train set.
        test_root_positions (dict[tuple, numpy.array]: Dictionary with the 3d positions of the root in test set.
    """
    # Load 3d data.
    train_set = load_data(data_dir, TRAIN_LIST)
    test_set = load_data(data_dir, VALID_LIST)

    if camera_frame:
        train_set = transform_world_to_camera(train_set, cams)
        test_set = transform_world_to_camera(test_set, cams)

    # Centering around root (center hip joint).
    train_set, train_root_positions = postprocess_3d(train_set)
    test_set, test_root_positions = postprocess_3d(test_set)

    # Compute normalization statistics.
    train_concat = copy.deepcopy(np.vstack(list(train_set.values())))
    data_mean, data_std, dim_to_ignore, dim_to_use = compute_normalization_stats(
        train_concat, dim=3, predict_15=True
    )

    # Divide every dimension independently.
    train_set = normalize_data(train_set, data_mean, data_std, dim_to_use)
    test_set = normalize_data(test_set, data_mean, data_std, dim_to_use)

    return (
        train_set,
        test_set,
        data_mean,
        data_std,
        dim_to_ignore,
        dim_to_use,
        train_root_positions,
        test_root_positions,
    )


def project_to_cameras(pose_set, cams):
    """Project 3d poses using camera parameters.

    Args:
        pose_set (dict[tuple, numpy.array]): Dictionary with 3d poses.
        cams (dict[tuple, tuple]): Dictionary with cameras.
        ncams (int, optional): Number of cameras per subject. Defaults to 4.

    Returns:
        t2d (dict[tuple, numpy.array]): Dictionary with projected 2d poses.
    """
    width, height, frame_threshold = 1920, 1080, 200
    cam_interval = 1
        
    t2d = {}

    for i, seq in enumerate(sorted(pose_set.keys())):
        t3d = pose_set[seq]  # nx(15x3)
        t3d = t3d.reshape((-1, 3))  # (nx15)x3
        
        for cam_idx, cam in enumerate(cams):
            K, R, T, Kd = cam['K'], cam['R'], cam['t'], cam['distCoef']
            K, R, T, Kd = np.array(K), np.array(R), np.array(T), np.array(Kd)
            pts2d = project_to_camera(t3d.T, K, R, T, Kd).transpose()[:, :2].reshape(-1, NUM_JOINTS, 2)
            
            # 2D pose가 frame 밖으로 많이 나가는 경우 제거
            valid_idx = []
            for idx, pose_2d in enumerate(pts2d):
                if all(pose_2d[:,0] < width+frame_threshold) and all(pose_2d[:,0] > 0-frame_threshold) and \
                    all(pose_2d[:,1] < height+frame_threshold) and all(pose_2d[:,1] > 0-frame_threshold):
                    valid_idx.append(idx)
            pts2d = pts2d.reshape((-1, NUM_JOINTS * 2))[valid_idx].reshape((-1, NUM_JOINTS * 2))
            
            t2d[(seq, f"{cam_idx:02d}")] = pts2d


    return t2d


def create_2d_data(data_dir, cams):
    """Create 2d poses by projecting 3d poses with the corresponding camera parameters,
    and also normalize the 2d poses.

    Args:
        actions (list[str]): Actions to load.
        data_dir (str): Directory where the data can be loaded from.
        cams (dict[tuple, tuple]): Dictionary with camera parameters.

    Returns:
        train_set (dict[tuple, numpy.array]): Dictionary with loaded 2d poses for training.
        test_set (dict[tuple, numpy.array]): Dictionary with loaded 2d poses for testing.
        data_mean (numpy.array): Vector with the mean of the 2d training data.
        data_std (numpy.array): Vector with the standard deviation of the 2d training data.
        dim_to_ignore (list[int]): List with the dimensions not to predict.
        dim_to_use (list[int]): List with the dimensions to predict.
    """
    # Load 3d data.
    train_set = load_data(data_dir, TRAIN_LIST)
    test_set = load_data(data_dir, VALID_LIST)

    train_set = project_to_cameras(train_set, cams)
    test_set = project_to_cameras(test_set, cams)

    # Compute normalization statistics.
    train_concat = copy.deepcopy(np.vstack(list(train_set.values())))
    data_mean, data_std, dim_to_ignore, dim_to_use = compute_normalization_stats(
        train_concat, dim=2, predict_15=True
    )

    # Divide every dimension independently.
    train_set = normalize_data(train_set, data_mean, data_std, dim_to_use)
    test_set = normalize_data(test_set, data_mean, data_std, dim_to_use)

    return train_set, test_set, data_mean, data_std, dim_to_ignore, dim_to_use


def unnormalize_data(data, data_mean, data_std, dim_to_ignore):
    """Un-normalize poses whose mean has been substracted and that has been divided by
    standard deviation. Returned array has mean values at ignored dimensions.

    Args:
        data (numpy.array): nxd array to unnormalize
        data_mean (numpy.array): Vector with the mean of the data.
        data_std (numpy.array): Vector with the std of the data.
        dim_to_ignore (numpy.array): Dimensions that were removed from the original data.

    Returns:
        data_unnormalized (numpy.array): unnormalized array
    """
    N = data.shape[0]  # Batch size.
    D = data_mean.shape[0]  # Dimensionality.
    data_unnormalized = np.zeros((N, D), dtype=np.float32)  # NxD

    dim_to_use = [d for d in range(D) if d not in dim_to_ignore]
    data_unnormalized[:, dim_to_use] = data

    # unnormalize with mean and std
    sigma = data_std.reshape((1, D))  # 1xD
    sigma = np.repeat(sigma, N, axis=0)  # NxD
    mu = data_mean.reshape((1, D))  # 1xD
    mu = np.repeat(mu, N, axis=0)  # NxD
    data_unnormalized = np.multiply(data_unnormalized, sigma) + mu

    return data_unnormalized


def transform_camera_to_world(pose_set, cams, sequence):
    t3d_world = []
    for i in range(len(pose_set)):
        t3d_camera = pose_set[i]
        t3d_camera = t3d_camera.reshape((-1, 3))
        
        cam_idx = int(sequence[1][i])
        R, T = cams[cam_idx]['R'], cams[cam_idx]['t']
        R, T = np.array(R).dot(M), np.array(T)
        
        world_coord = transform_camera_to_world_base(t3d_camera, R, T)
        world_coord = world_coord.reshape((NUM_JOINTS * 3))
        
        t3d_world.append(world_coord)
        
    return np.array(t3d_world)