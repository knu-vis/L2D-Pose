import numpy as np

from utils.procrustes import compute_similarity_transform

def apply_procrustes_alignment(sources, targets):
    '''
    sources, targets: [batch_size, n_joints, 3]
    
    return: aligned sources [batch_size, n_joints, 3]
    '''
    sources_aligned = []
    num_joints = sources.shape[1]

    batch_size = len(sources)
    for i in range(batch_size):
        target = targets[i].reshape(-1, 3)  # [n_joints, 3]
        source = sources[i].reshape(-1, 3)  # [n_joints, 3]
        _, _, T, b, c = compute_similarity_transform(
            target, source, compute_optimal_scale=True
        )
        aligned = (b * source.dot(T)) + c
        aligned = aligned.reshape((-1, num_joints, 3))  # [1, n_joints, 3]

        sources_aligned.append(aligned)

    return np.vstack(sources_aligned)  # [batch_size, n_joints, 3]

def compute_joint_distances(pred, truth, procrustes=False):
    '''
    pred, truth: [batch_size, n_joints, 3]
    
    return: joint distances [batch_size, n_joints]
    '''
    
    if procrustes:
        pred = apply_procrustes_alignment(pred, truth)
    
    # Compute Euclidean distance error per joint.
    num_joints = pred.shape[1]
    d_squared = (pred - truth) ** 2  # [batch_size, n_joints x 3]
    d_squared = d_squared.reshape(
        (-1, num_joints, 3)
    )  # [batch_size, n_joints, 3]
    d_squared = np.sum(d_squared, axis=2)  # [batch_size, n_joints]
    d = np.sqrt(d_squared)  # [batch_size, n_joints]

    return d

def get_metrics(joint_distances):
    '''
    pred, truth: [batch_size, n_joints, 3]
    
    return: MPJPE (mm), PJPE (mm)
    '''
    mpjpe = np.mean(joint_distances) * 10         # cm to mm
    pjpe = np.mean(joint_distances, axis=0) * 10  # cm to mm
    
    metrics = {
               "MPJPE": mpjpe,
               "PJPE": pjpe.tolist()
               }
    
    return metrics


def compute_joint_distances_2d(pred, truth):
    '''
    pred, truth: [batch_size, n_joints, 2]
    
    return: joint distances [batch_size, n_joints]
    '''
    # Compute Euclidean distance error per joint.
    num_joints = pred.shape[1]
    d_squared = (pred - truth) ** 2  # [batch_size, n_joints x 2]
    d_squared = d_squared.reshape(
        (-1, num_joints, 2)
    )  # [batch_size, n_joints, 2]
    d_squared = np.sum(d_squared, axis=2)  # [batch_size, n_joints]
    d = np.sqrt(d_squared)  # [batch_size, n_joints]

    return d
