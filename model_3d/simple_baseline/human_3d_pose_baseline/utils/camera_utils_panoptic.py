# references:
# https://github.com/una-dinosauria/3d-pose-baseline/blob/master/src/cameras.py

import os
import json
import h5py
import numpy as np


def transform_camera_to_world(P, R, T):
    """Transform points from camera to world coordinates.

    Args:
        P (numpy.array): Nx3 3d points in camera coordinates.
        R (numpy.array): Camera rotation matrix.
        T (numpy.array): Camera translation vector.

    Returns:
        X (numpy.array): Nx3 3d points in world coordinates.
    """
    assert len(P.shape) == 2
    assert P.shape[1] == 3

    X = R.T @ (P.T) + T  # rotate and translate
    return X.T


def transform_world_to_camera(P, R, T):
    """Transform points from world to camera coordinates.

    Args:
        P (numpy.array): Nx3 3d points in world coordinates.
        R (numpy.array): Camera rotation matrix.
        T (numpy.array): Camera translation vector.

    Returns:
        X (numpy.array): Nx3 3d points in camera coordinates.
    """
    assert len(P.shape) == 2
    assert P.shape[1] == 3

    X = R @ (P.T - T)  # rotate and translate
    return X.T


def project_to_camera(X, K, R, t, Kd):  # 3D -> 2D projection
    """
    Projects points X (3xN) using camera intrinsics K (3x3),
    extrinsics (R,t) and distortion parameters Kd=[k1,k2,p1,p2,k3].
    Roughly, x = K*(R*X + t) + distortion
    See http://docs.opencv.org/2.4/doc/tutorials/calib3d/camera_calibration/camera_calibration.html
    or cv2.projectPoints
    """
    x = np.dot(R, X) + t

    x[0:2, :] = x[0:2, :] / (x[2, :] + 1e-5)

    r = x[0, :] * x[0, :] + x[1, :] * x[1, :]

    x[0, :] = x[0, :] * (1 + Kd[0] * r + Kd[1] * r * r + Kd[4] * r * r * r
                        ) + 2 * Kd[2] * x[0, :] * x[1, :] + Kd[3] * (
                            r + 2 * x[0, :] * x[0, :])
    x[1, :] = x[1, :] * (1 + Kd[0] * r + Kd[1] * r * r + Kd[4] * r * r * r
                        ) + 2 * Kd[3] * x[0, :] * x[1, :] + Kd[2] * (
                            r + 2 * x[1, :] * x[1, :])

    x[0, :] = K[0, 0] * x[0, :] + K[0, 1] * x[1, :] + K[0, 2]
    x[1, :] = K[1, 0] * x[0, :] + K[1, 1] * x[1, :] + K[1, 2]

    return np.array(x)


# def load_camera_params(hf, base_path):
#     """Load h36m camera parameters.

#     Args:
#         hf (file object): HDF5 open file with h36m cameras data
#         path (str): Path or key inside hf to the camera we are interested in.

#     Returns:
#         R (numpy.array): 3x3 Camera rotation matrix.
#         T (numpy.array): 3x1 Camera translation parameters.
#         f (numpy.array): 2x1 Camera focal length.
#         c (numpy.array): 2x1 Camera center.
#         k (numpy.array): 3x1 Camera radial distortion coefficients.
#         p (numpy.array): 2x1 Camera tangential distortion coefficients.
#         name (str): String with camera id.
#     """
#     R = hf[os.path.join(base_path, "R")][:]
#     R = R.T

#     T = hf[os.path.join(base_path, "T")][:]
#     f = hf[os.path.join(base_path, "f")][:]
#     c = hf[os.path.join(base_path, "c")][:]
#     k = hf[os.path.join(base_path, "k")][:]
#     p = hf[os.path.join(base_path, "p")][:]

#     name = hf[os.path.join(base_path, "Name")][:]
#     name = "".join(chr(int(item)) for item in name)

#     return R, T, f, c, k, p, name



# def load_cameras(self, seq):
#     cam_file = os.path.join(self.dataset_root, seq, 'calibration_{:s}.json'.format(seq))
#     with open(cam_file) as cfile:
#         calib = json.load(cfile)

#     M = np.array([[1.0, 0.0, 0.0],
#                     [0.0, 0.0, -1.0],
#                     [0.0, 1.0, 0.0]])
#     cameras = {}
#     for cam in calib['cameras']:
#         if (cam['panel'], cam['node']) in self.cam_list:
#             sel_cam = {}
#             sel_cam['K'] = np.array(cam['K'])
#             sel_cam['distCoef'] = np.array(cam['distCoef'])
#             sel_cam['R'] = np.array(cam['R']).dot(M)
#             sel_cam['t'] = np.array(cam['t']).reshape((3, 1))
#             cameras[(cam['panel'], cam['node'])] = sel_cam
#     return cameras