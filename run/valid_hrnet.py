import time
import sys
import argparse
import os
import numpy as np
import logging


import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms

# for HR-Net
import _init_paths_hrnet
from config import update_config
import dataset
import models
from config.default import get_default_config
from models.pose_hrnet import PoseHighResolutionNet
from core.inference_tensor import get_final_preds_softargmax
from core.inference import get_final_preds
from evaluators.evaluate_3d import compute_joint_distances
from evaluators.evaluate_2d import pdj

# for 3D lifting model 
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(root_dir)
from run.function import normalize_pose, unnormalize_pose, h36m2panoptic_heatmap,normalize_screen_coordinates_tensor

logger = logging.getLogger(__name__)

H36M_ACTIONS = ['Directions', 'Discussion', 'Eating', 'Greeting', 'Phoning', 'TakingPhoto', 'Posing', 'Purchases',
                'Sitting', 'SittingDown', 'Smoking', 'Waiting', 'WalkingDog', 'Walking', 'WalkingTogether']

def validate(config, val_loader, model_2d, device, model_3d_dict=None, output_dir=None, stats=None,
        gt_2d=False, soft_argmax=True, vis=False, ensemble=False, beta=100, cross_model_3d_dict=None, action_valid=False):

    model_2d.eval()
    
    if config.DATASET.DATASET == 'panoptic':
        num_joints = 15
        left_shoulder_idx, right_hip_idx, root_joint_idx = 3, 12, 2
    elif config.DATASET.DATASET == 'mpi-inf-3dhp':
        num_joints = 14
    elif config.DATASET.DATASET == 'h36m':
        num_joints = 17
        left_shoulder_idx, right_hip_idx, root_joint_idx = 11, 1, 0
        
    
    total_detected, total_joints = 0, 0   # for measuring 2D accuracy
    mpjpe_list, p_mpjpe_list = [], []
    cross_mpjpe_list, cross_p_mpjpe_list = [], []
    if cross_model_3d_dict is not None:   # cross model evaluation 
        model_3d_dict = {**model_3d_dict, **cross_model_3d_dict}
        
    if model_3d_dict is not None:
        mpjpe_dict = {i: [] for i in model_3d_dict.keys()}
    else:
        mpjpe_dict = None
        
    if action_valid:
        action_mpjpe_dict = {i: {action: [] for action in H36M_ACTIONS} for i in model_3d_dict.keys()}
        
    output_dict = {}
    with torch.no_grad():
        end = time.time()
        for i, (input, _, _, meta) in enumerate(val_loader):
            output_dict[i] = {}
            if gt_2d:   # use GT 2D pose
                preds_2d = meta['joints_2d']
            else:
                input = input.to(device)
                heatmap = model_2d(input)
                # convert heatmap to proper format (H36M -> Panoptic or MPI-INF-3DHP)
                if config.DATASET.DATASET == 'panoptic':
                    heatmap = h36m2panoptic_heatmap(heatmap)
                elif config.DATASET.DATASET == 'mpi-inf-3dhp':
                    pass
                
                if soft_argmax:   # use soft argmax
                    preds_2d = get_final_preds_softargmax(config, heatmap.clone().cpu(), meta['center'], meta['scale'], beta=beta)
                else:             # use argmax
                    preds_2d, _ = get_final_preds(config, heatmap.clone().detach().cpu().numpy(), meta['center'].numpy(), meta['scale'].numpy())
                    preds_2d = torch.tensor(preds_2d, dtype=torch.float32)
            
            detected_joints = pdj(preds_2d, meta['joints_2d'], left_shoulder_idx, right_hip_idx, threshold_fraction=0.2)
            total_detected += np.sum(detected_joints)
            total_joints += detected_joints.size
            
            
            if model_3d_dict is not None:
                for model_3d_name, model_3d in model_3d_dict.items():
                    model_3d.eval()
                    
                    if model_3d_name == 'sb':           # SimpleBaseline
                        normalized_preds_2d = normalize_pose(preds_2d, torch.tensor(stats['mean_2d'], dtype=torch.float32),
                                                        torch.tensor(stats['std_2d'], dtype=torch.float32)).to(device)
                        preds_3d = model_3d(normalized_preds_2d.reshape(-1, num_joints*2)).to(device)
                        preds_3d = preds_3d.reshape(-1, num_joints, 3)
                        gt_3d = normalize_pose(meta['joints_3d'], stats['mean_3d'], stats['std_3d']).to(device)
                        p3d = unnormalize_pose(preds_3d.detach().cpu().numpy(), stats['mean_3d'], stats['std_3d']) * 10.0  # cm to mm
                        g3d = unnormalize_pose(gt_3d.detach().cpu().numpy(), stats['mean_3d'], stats['std_3d'])    * 10.0  # cm to mm
                    elif model_3d_name == 'videopose':  # VideoPose
                        preds_2d_videopose = preds_2d.clone()
                        preds_2d_videopose[..., :2] = normalize_screen_coordinates_tensor(preds_2d_videopose[..., :2], w=meta['width'], h=meta['height'])
                        preds_2d_videopose = preds_2d_videopose.unsqueeze(0).to(device)
                        p3d = model_3d(preds_2d_videopose).squeeze(0).detach().cpu().numpy() * 1000.0   # meter to mm
                        g3d = meta['joints_3d'].numpy() * 10.0                                          # cm to mm
                    elif model_3d_name == 'iganet':     # IGANet
                        preds_2d_iganet = preds_2d.clone()
                        preds_2d_iganet[..., :2] = normalize_screen_coordinates_tensor(preds_2d_iganet[..., :2], w=meta['width'], h=meta['height'])
                        preds_2d_iganet = preds_2d_iganet.unsqueeze(0).to(device)
                        p3d = model_3d(preds_2d_iganet).permute(1, 0, 2, 3).squeeze(0).detach().cpu().numpy() * 1000.0  # meter to mm
                        g3d = meta['joints_3d'].numpy() * 10.0     # cm to mm
                    elif model_3d_name == 'mixste':      # MixSTe
                        preds_2d_mixste = preds_2d.clone()
                        preds_2d_mixste[..., :2] = normalize_screen_coordinates_tensor(preds_2d_mixste[..., :2], w=meta['width'], h=meta['height'])
                        preds_2d_mixste = preds_2d_mixste.unsqueeze(1).to(device)
                        p3d = model_3d(preds_2d_mixste).squeeze().detach().cpu().numpy() * 1000.0   # meter to mm
                        # p3d -= p3d[:, root_joint_idx:root_joint_idx+1, :]   # root-relative
                        p3d = p3d.reshape(-1, num_joints, 3)
                        g3d = meta['joints_3d'].numpy() * 10.0

                    mpjpe = compute_joint_distances(p3d, g3d, procrustes=False).mean(-1)
                    p_mpjpe = compute_joint_distances(p3d, g3d, procrustes=True).mean(-1)
                    
                    if cross_model_3d_dict is not None and model_3d_name in cross_model_3d_dict.keys(): # cross model evaulation 따로 저장
                        cross_mpjpe_list.extend(mpjpe); cross_p_mpjpe_list.extend(p_mpjpe)
                    else:
                        mpjpe_list.extend(mpjpe);p_mpjpe_list.extend(p_mpjpe)
                    mpjpe_dict[model_3d_name].extend(mpjpe)
                    
                    
                    if action_valid:    
                        for action in H36M_ACTIONS:
                            mpjpe = compute_joint_distances(p3d[np.array(meta['action']) == action], 
                                                    g3d[np.array(meta['action']) == action], procrustes=False).mean(-1)
                            if len(mpjpe) > 0:
                                action_mpjpe_dict[model_3d_name][action].extend(mpjpe.tolist())
                
                    output_dict[i][model_3d_name] = p3d
                    if 'gt_3d' not in output_dict[i]:
                        output_dict[i]['gt_3d'] = g3d 
            else:
                mpjpe_list, p_mpjpe_list = -1, -1
                

        total_pdj = total_detected / total_joints * 100

    
    if ensemble:    
        ensemble_mpjpe_list, ensemble_p_mpjpe_list = [], []
        for idx in range(len(output_dict)):
            preds_list = np.array([preds_3d for key, preds_3d in output_dict[idx].items() if key in model_3d_dict.keys()])
            median_pred = np.median(preds_list, axis=0)
            mpjpe = compute_joint_distances(median_pred, output_dict[idx]['gt_3d'], procrustes=False).mean(-1)
            p_mpjpe = compute_joint_distances(median_pred, output_dict[idx]['gt_3d'], procrustes=True).mean(-1)
            ensemble_mpjpe_list.extend(mpjpe); ensemble_p_mpjpe_list.extend(p_mpjpe)
        mpjpe_list, p_mpjpe_list = ensemble_mpjpe_list, ensemble_p_mpjpe_list
    
    
    if cross_model_3d_dict is not None:
        return np.mean(cross_mpjpe_list), np.mean(cross_p_mpjpe_list), np.mean(mpjpe_list), np.mean(p_mpjpe_list), total_pdj, mpjpe_dict
    else:
        if action_valid:
            return np.mean(mpjpe_list), np.mean(p_mpjpe_list), total_pdj, mpjpe_dict, action_mpjpe_dict
        else:   
            return np.mean(mpjpe_list), np.mean(p_mpjpe_list), total_pdj, mpjpe_dict



def parse_args():
    parser = argparse.ArgumentParser(description='Train keypoints network')
    parser.add_argument('--cfg', help='experiment configure file name', required=True, type=str)
    parser.add_argument('--model_dir', default='', help='model directory', type=str)
    parser.add_argument('--gpu_id', default=0, help='GPU ID', type=int)
    parser.add_argument('--model_3d', default='sb', help='3D model (sb or videopose or iganet ...)', type=str)
    parser.add_argument('--gt', action='store_true', help='use GT 2D pose (only evaluation)')
    parser.add_argument('--soft_argmax', action='store_true', help='use soft argmax (else argmax)')
    parser.add_argument('--vis', action='store_true', help='visualize 2D and 3D pose (only evaluation)')
    parser.add_argument('--ensemble', action='store_true', help='use ensemble (only for multi-model training)')
    parser.add_argument('--beta', default=100, help='hyperparameter of soft-argmax', type=int)
    parser.add_argument('--cross_valid', action='store_true', help='cross validation (only for multi-model training)')
    parser.add_argument('--action_valid', action='store_true', help='print metirc for each action (only for h36m dataset)')

    
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    args.soft_argmax = True 

    args_hrnet = argparse.Namespace(cfg=args.cfg, opts=None, modelDir='', logDir='', dataDir='', prevModelDir='')
    config_hrnet = get_default_config()
    update_config(config_hrnet, args_hrnet)
    receptive_field = 1     # no temporal model
    
    model_3d_list = sorted(args.model_3d.split(','))
    
    cudnn.benchmark = config_hrnet.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = config_hrnet.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = config_hrnet.CUDNN.ENABLED
    torch.manual_seed(config_hrnet.SEED)
    device = torch.device(f"cuda:{args.gpu_id}" if config_hrnet.USE_CUDA else "cpu")
    
    
    if config_hrnet.DATASET.DATASET == 'panoptic':
        num_joints = 15
    elif config_hrnet.DATASET.DATASET == 'mpi-inf-3dhp':
        num_joints = 14
    elif config_hrnet.DATASET.DATASET == 'h36m':
        num_joints = 17
    
    # Data loading
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transform = transforms.Compose([transforms.ToTensor(), normalize])
    train_dataset = eval('dataset.'+config_hrnet.DATASET.DATASET)(
        config_hrnet, config_hrnet.DATASET.ROOT, config_hrnet.DATASET.TRAIN_SET, True,
        transforms.Compose([transforms.ToTensor(), normalize,]),)
    valid_dataset = eval('dataset.'+config_hrnet.DATASET.DATASET)(
        config_hrnet, config_hrnet.DATASET.ROOT, config_hrnet.DATASET.TEST_SET, False,
        transforms.Compose([transforms.ToTensor(), normalize,]),)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=config_hrnet.TRAIN.BATCH_SIZE_PER_GPU*len(config_hrnet.GPUS),
        shuffle=True, num_workers=config_hrnet.WORKERS, pin_memory=config_hrnet.PIN_MEMORY)
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=config_hrnet.TEST.BATCH_SIZE_PER_GPU*len(config_hrnet.GPUS),
        shuffle=False, num_workers=config_hrnet.WORKERS, pin_memory=config_hrnet.PIN_MEMORY)
    
    # Model loading
    if args.gt:
        model_2d_path = f'output/{config_hrnet.DATASET.DATASET}/hrnet/gt_2d/model_best.pth'
    else:
        if args.cross_valid:    # cross validation (ex. A,B,C train -> D test)
            model_2d_path = f'{args.model_dir}/model_best_cross.pth'
        else:
            model_2d_path = f'{args.model_dir}/model_best.pth'
        
    model_2d = PoseHighResolutionNet(config_hrnet)
    if not args.gt:
        model_2d.load_state_dict(torch.load(model_2d_path, map_location=torch.device("cpu")), strict=False)
    
    
    model_3d_dict = {}
    for model_3d_name in model_3d_list:
        args.model_3d = model_3d_name
    
        if args.model_3d == 'sb':
            if config_hrnet.DATASET.DATASET == 'panoptic':
                from model_3d.simple_baseline.human_3d_pose_baseline.configs.defaults_panoptic import get_default_config as get_default_config_sb
            elif config_hrnet.DATASET.DATASET == 'h36m':
                from model_3d.simple_baseline.human_3d_pose_baseline.configs.defaults_h36m import get_default_config as get_default_config_sb
            from model_3d.simple_baseline.human_3d_pose_baseline.models import get_model as get_model_sb
            config_sb = get_default_config_sb()
            config_sb.merge_from_file(os.path.join(root_dir, f'model_3d/simple_baseline/experiments/{config_hrnet.DATASET.DATASET}.yaml'))
            model_3d = get_model_sb(config_sb, num_joints=num_joints, 
                            model_weight_path=f'model_3d/simple_baseline/weights/model_best_{config_hrnet.DATASET.DATASET}.pth')
        elif args.model_3d == 'videopose':
            from model_3d.videopose.common.model import TemporalModel
            args_videopose = argparse.Namespace(causal=False, dropout=0.25, channels=1024, dense=False, arc='1,1,1')
            filter_widths = [int(x) for x in args_videopose.arc.split(',')]
            model_3d = TemporalModel(num_joints, 2, num_joints, filter_widths=filter_widths, causal=args_videopose.causal, 
                            dropout=args_videopose.dropout, channels=args_videopose.channels, dense=args_videopose.dense)
            model_3d.load_state_dict(torch.load(f'model_3d/videopose/weights/model_best_{config_hrnet.DATASET.DATASET}_t{receptive_field}.bin', 
                                                map_location=torch.device("cpu")))
            pass
        elif args.model_3d == 'iganet':
            if config_hrnet.DATASET.DATASET == 'panoptic':
                from model_3d.iganet.model.model_IGANet_panoptic import Model as IGANet
            elif config_hrnet.DATASET.DATASET == 'h36m':
                from model_3d.iganet.model.model_IGANet_h36m import Model as IGANet
            args_iganet = argparse.Namespace(layers=3, channel=512, d_hid=1024, n_joints=num_joints)
            model_3d = IGANet(args_iganet, device=device)
            model_3d.load_state_dict(torch.load(f'model_3d/iganet/weights/model_best_{config_hrnet.DATASET.DATASET}.pth', map_location=torch.device("cpu")))
        elif args.model_3d == 'mixste':      # MixSTe
            from model_3d.mixste.common.model_cross import MixSTE2
            args_mixste = argparse.Namespace(in_chans=2, cs=512, dep=8, num_heads=8, mlp_ratio=2., qkv_bias=True, drop_path_rate=0)
            model_3d = MixSTE2(num_frame=receptive_field, num_joints=num_joints, in_chans=2, embed_dim_ratio=args_mixste.cs, depth=args_mixste.dep,
                        num_heads=args_mixste.num_heads, mlp_ratio=args_mixste.mlp_ratio, qkv_bias=args_mixste.qkv_bias, qk_scale=None, drop_path_rate=0)
            model_3d = nn.DataParallel(model_3d)
            model_checkpoint = torch.load(f'model_3d/mixste/weights/model_best_{config_hrnet.DATASET.DATASET}_t{receptive_field}.bin',
                                        map_location=lambda storage, loc: storage)
            model_3d.load_state_dict(model_checkpoint, strict=False)
            model_3d = model_3d.module
        
        model_2d, model_3d = model_2d.to(device), model_3d.to(device)
        model_3d.eval()
        model_3d_dict[model_3d_name] = model_3d
        
    
    # validation
    if args.action_valid:
        test_mpjpe, test_p_mpjpe, test_acc_2d, test_mpjpe_dict, test_action_mpjpe_dict = validate(config_hrnet, valid_loader, model_2d, device, 
                        model_3d_dict, output_dir=os.path.dirname(model_2d_path), stats=train_dataset.stats, gt_2d=args.gt, 
                        soft_argmax=args.soft_argmax, vis=args.vis, ensemble=args.ensemble, beta=args.beta, action_valid=args.action_valid)
    else:
        test_mpjpe, test_p_mpjpe, test_acc_2d, test_mpjpe_dict = validate(config_hrnet, valid_loader, model_2d, device, 
                model_3d_dict, output_dir=os.path.dirname(model_2d_path), stats=train_dataset.stats, gt_2d=args.gt, 
                soft_argmax=args.soft_argmax, vis=args.vis, ensemble=args.ensemble, beta=args.beta, action_valid=args.action_valid)
    
    print(f"model dir: {os.path.dirname(model_2d_path)}")
    print(f"[{config_hrnet.DATASET.DATASET}] HRNet + {'+'.join(model_3d_list)} (GT 2D: {args.gt}, soft argmax: {args.soft_argmax}, beta:{args.beta})")
    print(f"[{config_hrnet.DATASET.DATASET}] Test MPJPE: {test_mpjpe:.2f}mm, Test P-MPJPE: {test_p_mpjpe:.2f}mm, Test PDJ@0.2: {test_acc_2d:.2f}%")
    if len(test_mpjpe_dict) > 0:
        for model_3d_name, mpjpe_list in test_mpjpe_dict.items():
            print(f"[{config_hrnet.DATASET.DATASET}] {model_3d_name} Test MPJPE: {np.mean(mpjpe_list):.2f}mm")
        print()
    if args.action_valid:
        for model_3d_name, action_mpjpe_dict in test_action_mpjpe_dict.items():
            print(f"[{config_hrnet.DATASET.DATASET}] {model_3d_name} Dir.{np.mean(action_mpjpe_dict['Directions']):.1f} | Disc.{np.mean(action_mpjpe_dict['Discussion']):.1f} |\
 Eat.{np.mean(action_mpjpe_dict['Eating']):.1f} | Greet.{np.mean(action_mpjpe_dict['Greeting']):.1f} | Phone.{np.mean(action_mpjpe_dict['Phoning']):.1f} |\
 Photo.{np.mean(action_mpjpe_dict['TakingPhoto']):.1f} | Pose.{np.mean(action_mpjpe_dict['Posing']):.1f} | Purch.{np.mean(action_mpjpe_dict['Purchases']):.1f} |\
 Sit.{np.mean(action_mpjpe_dict['Sitting']):.1f} | SitD.{np.mean(action_mpjpe_dict['SittingDown']):.1f} | Smoke.{np.mean(action_mpjpe_dict['Smoking']):.1f} |\
 Wait.{np.mean(action_mpjpe_dict['Waiting']):.1f} | WalkD.{np.mean(action_mpjpe_dict['WalkingDog']):.1f} | Walk.{np.mean(action_mpjpe_dict['Walking']):.1f} |\
 WalkT.{np.mean(action_mpjpe_dict['WalkingTogether']):.1f}")
            print()
    print()