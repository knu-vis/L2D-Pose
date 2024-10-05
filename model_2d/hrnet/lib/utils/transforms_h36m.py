# references:
# https://github.com/una-dinosauria/3d-pose-baseline/blob/master/src/cameras.py

import os
import torch
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


def project_to_camera(P, K, distCoef, R, T):
    """Project points from 3d to 2d using camera parameters
    including radial and tangential distortions.

    Args:
        P (numpy.array): Nx3 3d points in world coordinates.
        R (numpy.array): 3x3 Camera rotation matrix.
        T (numpy.array): 3x1 Camera translation parameters.
        K (numpy.array): 3x3 Camera intrinsic parameters.
        distCoef (numpy.array): 5x1 Camera distortion coefficients. (k1, k2, p1, p2, k3)
    Returns:
        2D projection (numpy.array): Nx2 2d points in pixel space.
    """
    f = np.array([K[0][0], K[1][1]]).reshape(2, 1)
    c = np.array([K[0][2], K[1][2]]).reshape(2, 1)
    k = np.array([distCoef[0], distCoef[1], distCoef[4]]).reshape(3, 1)
    p = np.array([distCoef[2], distCoef[3]]).reshape(2, 1)
    
    N = P.shape[0]

    X = transform_world_to_camera(P, R, T)  # Nx3
    X = X.T  # 3xN
    d = X[2, :]  # Depth.
    XX = X[:2, :] / d  # 2xN

    # Radial distorsion term
    r2 = XX[0, :] ** 2 + XX[1, :] ** 2
    radial = 1 + np.einsum(
        "ij,ij->j", np.tile(k, (1, N)), np.array([r2, r2 ** 2, r2 ** 3])
    )
    # Tangential distorsion term.
    tan = p[0] * XX[1, :] + p[1] * XX[0, :]
    # Apply the distorsions.
    XXX = XX * np.tile(radial + tan, (2, 1)) + np.outer(
        np.array([p[1], p[0]]).reshape(-1), r2
    )

    # Project to camera.
    projected = f * XXX + c
    projected = projected.T  # Nx2

    return projected


def project_to_camera_tensor(P, K, distCoef, R, T):
    """Project points from 3d to 2d using camera parameters
    including radial and tangential distortions.

    Args:
        P (numpy.array): Nx3 3d points in world coordinates.
        R (numpy.array): 3x3 Camera rotation matrix.
        T (numpy.array): 3x1 Camera translation parameters.
        K (numpy.array): 3x3 Camera intrinsic parameters.
        distCoef (numpy.array): 5x1 Camera distortion coefficients. (k1, k2, p1, p2, k3)
    Returns:
        2D projection (numpy.array): Nx2 2d points in pixel space.
    """
    f = torch.tensor([K[0][0], K[1][1]]).reshape(2, 1)
    c = torch.tensor([K[0][2], K[1][2]]).reshape(2, 1)
    k = torch.tensor([distCoef[0], distCoef[1], distCoef[4]]).reshape(3, 1)
    p = torch.tensor([distCoef[2], distCoef[3]]).reshape(2, 1)
    
    N = P.shape[0]

    X = transform_world_to_camera(P, R, T)  # Nx3
    X = X.T  # 3xN
    d = X[2, :]  # Depth.
    XX = X[:2, :] / d  # 2xN

    # Radial distorsion term
    r2 = XX[0, :] ** 2 + XX[1, :] ** 2
    radial = 1 + torch.einsum(
        "ij,ij->j", k.repeat(1, N), torch.stack([r2, r2 ** 2, r2 ** 3], dim=0)
    )
    # Tangential distorsion term.
    tan = p[0] * XX[1, :] + p[1] * XX[0, :]
    # Apply the distorsions.
    radial_tan = radial + tan
    radial_tan_tiled = radial_tan.repeat((2, 1))
    outer_product = torch.outer(torch.tensor([p[1], p[0]]), r2)

    # Final calculation
    XXX = XX * radial_tan_tiled + outer_product
    # XXX = XX * np.tile(radial + tan, (2, 1)) + np.outer(
    #     torch.tensor([p[1], p[0]]).reshape(-1), r2
    # )

    # Project to camera.
    projected = f * XXX + c
    projected = projected.T  # Nx2

    return projected