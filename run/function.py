import os
import numpy as np
import json
import pickle
import torch
import matplotlib.pyplot as plt
import sys


from model_2d.hrnet.lib.utils.transforms_h36m import transform_camera_to_world as transform_camera_to_world_h36m
from model_2d.hrnet.lib.utils.transforms_h36m import transform_world_to_camera as transform_world_to_camera_h36m
from model_2d.hrnet.lib.utils.camera_utils_panoptic import transform_world_to_camera as transform_world_to_camera_panoptic
from model_2d.hrnet.lib.utils.camera_utils_panoptic import transform_camera_to_world as transform_camera_to_world_panoptic
# from model_2d.hrnet.lib.evaluators.evaluate_3d import compute_joint_distances


h36m_panoptic = [(0,8),(1,9),(2,0),(3,11),(4,12),(5,13),(6,4),(7,5),(8,6),
                 (9,14),(10,15),(11,16),(12,1),(13,2),(14,3)]        
def h36m2panoptic_heatmap(h36m):  
    batch_size, num_keypoints, height, width = h36m.shape
    assert num_keypoints == 17, "Expected input with 17 keypoints"

    panoptic = torch.empty((batch_size, 15, height, width), dtype=h36m.dtype, device=h36m.device)
    
    for i, (p, h) in enumerate(h36m_panoptic):
        panoptic[:, i, :, :] = h36m[:, h, :, :]
    
    return panoptic


def h36m2panoptic_ketpoints(h36m):  
    batch_size, num_keypoints, pos = h36m.shape
    assert num_keypoints == 17, "Expected input with 17 keypoints"

    panoptic = torch.empty((batch_size, 15, pos), dtype=torch.float32, device=h36m.device)
    
    for i, (p, h) in enumerate(h36m_panoptic):
        panoptic[:, i, :] = h36m[:, h, :]
    
    return panoptic


def normalize_pose(pose, mean, std):
    # mean = mean.to(pose.device)
    # std = std.to(pose.device)
    return (pose - mean) / (std + 1e-8)

def unnormalize_pose(pose, mean, std):
    return pose * std + mean


def visualize_pose(preds_2d_batch, gt_2d_batch, preds_3d_batch, gt_3d_batch, image_batch, camera_batch, \
                   model_3d_name, dataset, output_dir, batch_idx, num_samples=1):
    '''
    Visualize 2D and 3D pose predictions and ground truth.
    Args:
    - preds_2d_batch: Tensor of shape (batch_size, num_joints, 2) containing 2D pose predictions
    - gt_2d_batch: Tensor of shape (batch_size, num_joints, 2) containing 2D ground truth
    - preds_3d_batch: Tensor of shape (batch_size, num_joints, 3) containing 3D pose predictions
    - gt_3d_batch: Tensor of shape (batch_size, num_joints, 3) containing 3D ground truth
    - image_batch: image path of the batch
    - dataset: Dataset name ('h36m' or 'panoptic')
    - output_dir: Directory to save visualizations
    - batch_idx: Index of the current batch to visualize
    - num_samples: Number of samples to visualize
    '''
    vis_output_dir = os.path.join(output_dir, 'visualizations', model_3d_name)
    if not os.path.exists(vis_output_dir):
        os.makedirs(vis_output_dir)
        print(f'Created directory: {vis_output_dir}')
    
    # Define connections between joints
    if dataset == 'h36m':
        body_edges = np.array([[0,1],[1,2],[2,3],[0,4],[4,5],[5,6],[0,7],[7,8],[8,9],[9,10],
                               [8,11],[11,12],[12,13],[8,14],[14,15],[15,16]])
    elif dataset == 'panoptic':
        body_edges = np.array([[0,1],[0,2],[0,3],[0,9],[3,4],[9,10],[4,5],
                               [10,11],[2,6],[2,12],[6,7],[12,13],[7,8],[13,14]])
        
    # 2D, 3D subplot
    cnt = 0
    for idx in range(preds_2d_batch.shape[0]):
        plt.figure(figsize=(14, 6))
        ax1 = plt.subplot(1, 2, 1)
        ax2 = plt.subplot(1, 2, 2, projection='3d')
        ax1.imshow(plt.imread(image_batch[idx]))
        
        if dataset == 'h36m':
            gt_3d = transform_camera_to_world_h36m(gt_3d_batch[idx], camera_batch['R'][idx], camera_batch['T'][idx]).numpy()
            preds_3d = transform_camera_to_world_h36m(preds_3d_batch[idx], camera_batch['R'][idx], camera_batch['T'][idx]).numpy()
            gt_3d -= gt_3d[0]  # Center the pose
            preds_3d -= preds_3d[0]
        elif dataset == 'panoptic':
            gt_3d = transform_camera_to_world_panoptic(gt_3d_batch[idx], camera_batch['R'][idx], camera_batch['T'][idx]).numpy()
            preds_3d = transform_camera_to_world_panoptic(preds_3d_batch[idx], camera_batch['R'][idx], camera_batch['T'][idx]).numpy()
            gt_3d -= gt_3d[2]  # Center the pose
            preds_3d -= preds_3d[2]        

        
        labeling = True 
        for edge in body_edges:
            ax1.plot(gt_2d_batch[idx, edge, 0], gt_2d_batch[idx, edge, 1], color='b', label='GT' if labeling else '')
            ax1.plot(preds_2d_batch[idx, edge, 0], preds_2d_batch[idx, edge, 1], color='r', label='Pred' if labeling else '')
            
            ax2.plot(gt_3d[edge, 0], gt_3d[edge, 1], gt_3d[edge, 2], color='b', label='GT' if labeling else '')
            ax2.plot(preds_3d[edge, 0], preds_3d[edge, 1], preds_3d[edge, 2], color='r', label='Pred' if labeling else '')
            labeling = False
        
        ax1.legend();ax2.legend()
        ax2.set_xlim([-700, 700]);ax2.set_ylim([-700, 700]);ax2.set_zlim([-700, 700])
        ax2.set_title(f'MPJPE: {mpjpe:.2f}mm, P-MPJPE: {p_mpjpe:.2f}mm')
        plt.tight_layout()
        plt.suptitle(f'Batch {batch_idx:02d}, Sample {cnt:02d}', fontsize=12)
        plt.savefig(os.path.join(vis_output_dir, f'validation_batch{batch_idx:02d}_{cnt:02d}.jpg'))
        plt.close()  
        
        cnt += 1
        if cnt >= num_samples:
            break


def calculate_dwa_weights(train_losses, T=2):
    K = len(train_losses[0])
    exp_losses = []
    # Calculate exponential of the loss ratios
    for k in range(K):
        exp_losses.append(np.exp((train_losses[-2][k].cpu().detach().numpy() / train_losses[-3][k].cpu().detach().numpy()) / T))
    
    # Calculate the sum of exponential losses
    exp_loss_sum = sum(exp_losses)
    
    # Calculate weights for each loss
    weights = [exp_loss.item() / exp_loss_sum.item() for exp_loss in exp_losses]
    
    return weights
        
        
import math
from torch.optim.lr_scheduler import _LRScheduler

class CosineAnnealingWarmUpRestarts(_LRScheduler):
    def __init__(self, optimizer, T_0, T_mult=1, eta_max=0.1, T_up=0, gamma=1., last_epoch=-1):
        if T_0 <= 0 or not isinstance(T_0, int):
            raise ValueError("Expected positive integer T_0, but got {}".format(T_0))
        if T_mult < 1 or not isinstance(T_mult, int):
            raise ValueError("Expected integer T_mult >= 1, but got {}".format(T_mult))
        if T_up < 0 or not isinstance(T_up, int):
            raise ValueError("Expected positive integer T_up, but got {}".format(T_up))
        self.T_0 = T_0
        self.T_mult = T_mult
        self.base_eta_max = eta_max
        self.eta_max = eta_max
        self.T_up = T_up
        self.T_i = T_0
        self.gamma = gamma
        self.cycle = 0
        self.T_cur = last_epoch
        super(CosineAnnealingWarmUpRestarts, self).__init__(optimizer, last_epoch)
    
    def get_lr(self):
        if self.T_cur == -1:
            return self.base_lrs
        elif self.T_cur < self.T_up:
            return [(self.eta_max - base_lr)*self.T_cur / self.T_up + base_lr for base_lr in self.base_lrs]
        else:
            return [base_lr + (self.eta_max - base_lr) * (1 + math.cos(math.pi * (self.T_cur-self.T_up) / (self.T_i - self.T_up))) / 2
                    for base_lr in self.base_lrs]

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
            self.T_cur = self.T_cur + 1
            if self.T_cur >= self.T_i:
                self.cycle += 1
                self.T_cur = self.T_cur - self.T_i
                self.T_i = (self.T_i - self.T_up) * self.T_mult + self.T_up
        else:
            if epoch >= self.T_0:
                if self.T_mult == 1:
                    self.T_cur = epoch % self.T_0
                    self.cycle = epoch // self.T_0
                else:
                    n = int(math.log((epoch / self.T_0 * (self.T_mult - 1) + 1), self.T_mult))
                    self.cycle = n
                    self.T_cur = epoch - self.T_0 * (self.T_mult ** n - 1) / (self.T_mult - 1)
                    self.T_i = self.T_0 * self.T_mult ** (n)
            else:
                self.T_i = self.T_0
                self.T_cur = epoch
                
        self.eta_max = self.base_eta_max * (self.gamma**self.cycle)
        self.last_epoch = math.floor(epoch)
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr
            
        
def normalize_screen_coordinates(X, w, h):
    assert X.shape[-1] == 2
    return X / w * 2 - [1, h / w]

def normalize_screen_coordinates_tensor(X, w, h):
    assert X.shape[-1] == 2
    if isinstance(w, (int, float)):
        return X / w * 2 - torch.tensor([1, h / w])
    else:  
        hw_ratio = (h / w)[:, None, None]
        hw_tensor = torch.stack([torch.ones_like(hw_ratio), hw_ratio], dim=-1).squeeze(2)
        hw_tensor = hw_tensor.expand_as(X)
        return X / w[:, None, None] * 2 - hw_tensor

def world_to_camera(X, R, t):
    Rt = wrap(qinverse, R) 
    return wrap(qrot, np.tile(Rt, (*X.shape[:-1], 1)), X - t) 

def camera_to_world(X, R, t):
    return wrap(qrot, np.tile(R, (*X.shape[:-1], 1)), X) + t

def wrap(func, *args, unsqueeze=False):
	args = list(args)
	for i, arg in enumerate(args):
	    if type(arg) == np.ndarray:
	        args[i] = torch.from_numpy(arg)
	        if unsqueeze:
	            args[i] = args[i].unsqueeze(0)

	result = func(*args)

	if isinstance(result, tuple):
	    result = list(result)
	    for i, res in enumerate(result):
	        if type(res) == torch.Tensor:
	            if unsqueeze:
	                res = res.squeeze(0)
	            result[i] = res.numpy()
	    return tuple(result)
	elif type(result) == torch.Tensor:
	    if unsqueeze:
	        result = result.squeeze(0)
	    return result.numpy()
	else:
	    return result

def qrot(q, v):
	assert q.shape[-1] == 4
	assert v.shape[-1] == 3
	assert q.shape[:-1] == v.shape[:-1]

	qvec = q[..., 1:]
	uv = torch.cross(qvec, v, dim=len(q.shape) - 1)
	uuv = torch.cross(qvec, uv, dim=len(q.shape) - 1)
	return (v + 2 * (q[..., :1] * uv + uuv))


def qinverse(q, inplace=False):
    if inplace:
        q[..., 1:] *= -1
        return q
    else:
        w = q[..., :1]
        xyz = q[..., 1:]
        return torch.cat((w, -xyz), dim=len(q.shape) - 1)
