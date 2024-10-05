import numpy as np


def calculate_torso_diameter(ground_truth, left_shoulder_idx, right_hip_idx):
    """
    Calculate the torso diameter for each pose in the ground truth.
    
    Args:
    - ground_truth: numpy array of shape (num_poses, num_joints, 2)
    - left_shoulder_idx: Index of the left shoulder joint
    - right_hip_idx: Index of the right hip joint
    
    Returns:
    - torso_diameters: numpy array of shape (num_poses,)
    """
    left_shoulder = ground_truth[:, left_shoulder_idx]
    right_hip = ground_truth[:, right_hip_idx]
    torso_diameters = np.linalg.norm(left_shoulder - right_hip, axis=1)

    return torso_diameters


def pdj(predictions, ground_truth, left_shoulder_idx, right_hip_idx, threshold_fraction=0.2):
    """
    Calculate the Percentage of Detected Joints (PDJ).
    
    Args:
    - predictions: numpy array of shape (num_poses, num_joints, 2)
    - ground_truth: numpy array of shape (num_poses, num_joints, 2)
    - threshold_fraction: Fraction of the torso diameter to use as the detection threshold
    - left_shoulder_idx: Index of the left shoulder joint
    - right_hip_idx: Index of the right hip joint
    
    Returns:
    - pdj_score: Percentage of detected joints
    """
    assert predictions.shape == ground_truth.shape, "Predictions and ground truth must have the same shape."
    
    # Calculate torso diameters
    torso_diameters = calculate_torso_diameter(ground_truth, left_shoulder_idx, right_hip_idx)
    
    # Calculate detection threshold for each pose
    detection_thresholds = torso_diameters * threshold_fraction
    
    # Calculate distances between predicted joints and ground truth joints
    distances = np.linalg.norm(predictions - ground_truth, axis=-1)
    
    # Determine if joints are detected
    detected_joints = distances < detection_thresholds[:, np.newaxis]
    
    return detected_joints

