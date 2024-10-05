import os
import json
import numpy as np
import pickle

from torch.utils.data import DataLoader

from ..utils import camera_utils, data_utils_h36m, camera_utils_panoptic, data_utils_panoptic
from .human36m import Human36M
from .panoptic import PANOPTIC

M = np.array([[1.0, 0.0, 0.0],
              [0.0, 0.0, -1.0],
              [0.0, 1.0, 0.0]])

def get_dataset(config):
    """Get Human3.6M dataset.

    Args:
        config (yacs.config.CfgNode): Configuration.

    Returns:
        (Human36MDatasetHandler): Human3.6M dataset.
    """
    return Human36M_DatasetHandler(config)

def get_dataset_panoptic(config):
    return Panoptic_DatasetHandler(config)

def get_dataset_airsim(config):
    return Airsim_DatasetHandler(config)


class Human36M_DatasetHandler:
    def __init__(self, config):
        """

        Args:
            config (yacs.config.CfgNode): Configuration.
        """
        # Define actions.
        self.actions = self._get_actions(config)

        # Load Human3.6M camera parameters.
        self.cams = camera_utils.load_cameras(
            os.path.join(config.DATA.HM36M_DIR, "cameras.h5")
        )

        # Load Human3.6M 3d poses.
        print("Loading 3d poses...")
        (
            self.poses_3d_train,
            self.poses_3d_test,
            self.mean_3d,
            self.std_3d,
            self.dim_to_ignore_3d,
            self.dim_to_use_3d,
            self.train_root_positions,
            self.test_root_positions,
        ) = data_utils_h36m.read_3d_data(
            self.actions,
            config.DATA.HM36M_DIR,
            self.cams,
            camera_frame=config.DATA.POSE_IN_CAMERA_FRAME,
            predict_17=config.MODEL.PREDICT_17,
        )
        print("Done!")

        # Load Human3.6M 2d poses.
        print("Loading 2d poses...")
        (
            self.poses_2d_train,
            self.poses_2d_test,
            self.mean_2d,
            self.std_2d,
            self.dim_to_ignore_2d,
            self.dim_to_use_2d,
        ) = data_utils_h36m.create_2d_data(self.actions, config.DATA.HM36M_DIR, self.cams)
        print("Done!")
        
        # h36m_stats = {'mean_3d': self.mean_3d, 'std_3d': self.std_3d, 
        #               'mean_2d': self.mean_2d, 'std_2d': self.std_2d}
        # pickle.dump(h36m_stats, open(os.path.join(config.DATA.HM36M_DIR, 'h36m_stats.pkl'), 'wb'))

        # Create pytorch dataloaders for train and test set.s
        self.train_dataloader = self._get_dataloaders(
            config, self.poses_2d_train, self.poses_3d_train, is_train=True
        )

        self.test_dataloader = self._get_dataloaders(
            config, self.poses_2d_test, self.poses_3d_test, is_train=False
        )

    # Private members.

    def _get_actions(self, config):
        actions = config.DATA.ACTIONS
        if len(actions) == 0:
            # If empty, load all actions.
            actions = data_utils_h36m.H36M_ACTIONS
        else:
            # Check if the specified actions are valid.
            for act in actions:
                assert act in data_utils_h36m.H36M_ACTIONS, f"Unrecognized action: {act}."
        return actions

    def _get_dataloaders(self, config, pose_set_2d, pose_set_3d, is_train):
        # Create pytorch dataset.
        dataset = Human36M(
            pose_set_2d, pose_set_3d, camera_frame=config.DATA.POSE_IN_CAMERA_FRAME
        )

        # Create pytorch dataloader.
        if is_train:
            batch_size = config.LOADER.TRAIN_BATCHSIZE
            num_workers = config.LOADER.TRAIN_NUM_WORKERS
            shuffle = True
        else:
            batch_size = config.LOADER.TEST_BATCHSIZE
            num_workers = config.LOADER.TEST_NUM_WORKERS
            shuffle = False

        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True,
        )

        return dataloader


# CMU Panoptic dataset handler
class Panoptic_DatasetHandler:
    def __init__(self, config):
        self.cam_list = [(0, 12), (0, 6), (0, 23), (0, 13), (0, 3)]
        # Load CMU Panoptic camera parameters.
        with open(os.path.join(config.DATA.PANOPTIC_DIR, "171204_pose1", "calibration_171204_pose1.json")) as rf:
            camera_file = json.load(rf)['cameras']
        self.cams = []
        for cam in camera_file:
            if (cam['panel'], cam['node']) in self.cam_list:
                sel_cam = {}
                sel_cam['K'] = np.array(cam['K'])
                sel_cam['distCoef'] = np.array(cam['distCoef'])
                sel_cam['R'] = np.array(cam['R']).dot(M)
                sel_cam['t'] = np.array(cam['t']).reshape((3, 1))
                self.cams.append(sel_cam)
        

        # Load CMU Panoptic 3d poses.
        print("Loading 3d poses...")
        (
            self.poses_3d_train,
            self.poses_3d_test,
            self.mean_3d,
            self.std_3d,
            self.dim_to_ignore_3d,
            self.dim_to_use_3d,
            self.train_root_positions,
            self.test_root_positions,
        ) = data_utils_panoptic.read_3d_data(
            config.DATA.PANOPTIC_DIR,
            self.cams,
            camera_frame=config.DATA.POSE_IN_CAMERA_FRAME,
            predict_15=True,
        )
        print("Done!")

        # Load CMU Panoptic 2d poses.
        print("Loading 2d poses...")
        (
            self.poses_2d_train,
            self.poses_2d_test,
            self.mean_2d,
            self.std_2d,
            self.dim_to_ignore_2d,
            self.dim_to_use_2d,
        ) = data_utils_panoptic.create_2d_data(config.DATA.PANOPTIC_DIR, self.cams)
        print("Done!")
        
        # panoptic_stats = {'mean_3d': self.mean_3d, 'std_3d': self.std_3d, 
        #                   'mean_2d': self.mean_2d, 'std_2d': self.std_2d}
        # pickle.dump(panoptic_stats, open(os.path.join(config.DATA.PANOPTIC_DIR, 'panoptic_stats.pkl'), 'wb'))

        # Create pytorch dataloaders for train and test set.
        self.train_dataloader = self._get_dataloaders(
            config, self.poses_2d_train, self.poses_3d_train, is_train=True
        )

        self.test_dataloader = self._get_dataloaders(
            config, self.poses_2d_test, self.poses_3d_test, is_train=False
        )

    
    def _get_dataloaders(self, config, pose_set_2d, pose_set_3d, is_train):
        # Create pytorch dataset.
        dataset = PANOPTIC(
            pose_set_2d, pose_set_3d, camera_frame=config.DATA.POSE_IN_CAMERA_FRAME
        )

        # Create pytorch dataloader.
        if is_train:
            batch_size = config.LOADER.TRAIN_BATCHSIZE
            num_workers = config.LOADER.TRAIN_NUM_WORKERS
            shuffle = True
        else:
            batch_size = config.LOADER.TEST_BATCHSIZE
            num_workers = config.LOADER.TEST_NUM_WORKERS
            shuffle = False

        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True,
        )

        return dataloader
    
