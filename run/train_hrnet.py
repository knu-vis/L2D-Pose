import time
import sys
import argparse
import os
import pprint
import numpy as np
import pickle

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
from collections import deque

# for HR-Net
import _init_paths_hrnet
from config import update_config
from core.loss import JointsMSELoss, JointsMSELoss_ohklm, ohklm
import dataset
import models
from config.default import get_default_config
from utils.utils import get_optimizer, create_logger, save_checkpoint
from core.evaluate import accuracy as accuracy_2d
from models.pose_hrnet import PoseHighResolutionNet
from core.inference_tensor import get_final_preds_softargmax

# for 3D lifting model 
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(root_dir)
from run.function import normalize_pose, unnormalize_pose, h36m2panoptic_heatmap,calculate_dwa_weights,normalize_screen_coordinates_tensor
from run.valid_hrnet import validate



def parse_args():
    parser = argparse.ArgumentParser(description='Train keypoints network')
    
    parser.add_argument('--cfg', help='experiment configure file name', required=True, type=str)
    parser.add_argument('--gpu_id', default=0, help='GPU ID', type=int)
    parser.add_argument('--model_3d', default='sb', help='3D model (sb or videopose or iganet or mixste', type=str)
    parser.add_argument('--loss_3d', action='store_true', help='Use loss_3d for train 2D model')
    parser.add_argument('--loss_3d_z', action='store_true', help='Use z loss_3d for train 2D model')
    
    parser.add_argument('--resume', action='store_true', help='resume training')
    parser.add_argument('--ensemble', action='store_true', help='use ensemble (only for multi-model training)')
    parser.add_argument('--finetune', action='store_true', help='use finetune 2d')
    parser.add_argument('--dropout', action='store_true', help='use model dropout')
    parser.add_argument('--ohklm', action='store_true', help='online hard keypoint lifting mining (OHKLM)')
    parser.add_argument('--dwa', action='store_true', help='use dynamic weight average (DWA)')
    parser.add_argument('--loss_3d_cons', action='store_true', help='use consistency loss_3d')
    parser.add_argument('--loss_3d_avg', action='store_true', help='use average loss_3d')
    
    parser.add_argument('--loss_2d_w', default=0.5, help='weight of loss_2d', type=float)
    parser.add_argument('--ohklm_w', default=0.3, help='weight of loss_2d (OHKLM)', type=float)
    parser.add_argument('--loss_3d_xy_w', default=0.02, help='weight of loss_3d (xy)', type=float)
    parser.add_argument('--loss_3d_z_w', default=0.04, help='weight of loss_3d (z)', type=float)
    parser.add_argument('--loss_3d_w', default=0.06, help='weight of loss_3d (not use z loss)', type=float)
    parser.add_argument('--loss_3d_cons_w', default=0.1, help='weight of loss_3d (z)', type=float)
    parser.add_argument('--loss_3d_avg_w', default=0.05, help='weight of loss_3d (z)', type=float)
    parser.add_argument('--loss_single_w', default=1.0, help='weight of single-model loss', type=float)
    
    parser.add_argument('--epochs', default=40, help='Epochs for training', type=int)
    parser.add_argument('--top_k', default=10, help='OHKLM top-k number', type=int)
    parser.add_argument('--version', default=0, help='training version (default=0)', type=int)
    
    args = parser.parse_args()
    return args

def generate_output_string(var_name, var_value):  
    if type(var_value) == bool:
        return f"{var_name}{'1' if var_value else '0'}"
    else:
        return f"{var_name}{var_value}"

def main():
    args = parse_args()
    
    args_hrnet = argparse.Namespace(cfg=args.cfg, opts=None, modelDir='', logDir='', dataDir='', prevModelDir='')
    config_hrnet = get_default_config()
    update_config(config_hrnet, args_hrnet)
    receptive_field = 1   # no temporal model
    
    model_3d_list = sorted(args.model_3d.split(','))
    args.dwa = True
    
    if not args.loss_3d:
        loss_type = 'only_loss_2d'
    else:   # ohklm, loss_3d_z, top_k에 따라 ablation study 이름 변경
        # ohklm_str = generate_output_string('ohklm', args.ohklm)
        # top_k_str = f"topk{args.top_k}"
        loss_3d_z_str = generate_output_string('loss3dz', args.loss_3d_z)
        dwa_str = generate_output_string('dwa', args.dwa)
        loss_3d_xy_w_str = generate_output_string('3xyw', args.loss_3d_xy_w)
        loss_3d_z_w_str = generate_output_string('3zw', args.loss_3d_z_w)
        ohklm_w = f"ohklmw{args.ohklm_w}"
        dropout_str = generate_output_string('dropout', args.dropout)
        loss_3d_avg_str = generate_output_string('loss3davg', args.loss_3d_avg)
        loss_3d_cons_str = generate_output_string('loss3dcons', args.loss_3d_cons)
        loss_3d_avg_w_str = generate_output_string('avgw', args.loss_3d_avg_w)
        loss_3d_cons_w_str = generate_output_string('consw', args.loss_3d_cons_w)
        loss_single_w_str = generate_output_string('singlew', args.loss_single_w)
        
        if(len(model_3d_list) > 1): 
            loss_type = f'{dropout_str}_{loss_3d_avg_str}_{loss_3d_cons_str}_{loss_3d_avg_w_str}_{loss_3d_cons_w_str}_{loss_single_w_str}_multi'
        else:
            loss_type = f'{loss_3d_z_str}_{dwa_str}_{loss_3d_xy_w_str}_{loss_3d_z_w_str}_{ohklm_w}'
        
    
    if len(model_3d_list) > 1:
        available_models = ['sb', 'videopose', 'iganet', 'mixste']
        logger, final_output_dir = create_logger(config_hrnet, args.cfg, args.model_3d, loss_type, 'train',
                                                 multi_model_list=model_3d_list)
    else:
        available_models = [args.model_3d]
        logger, final_output_dir = create_logger(config_hrnet, args.cfg, args.model_3d, loss_type, 'train')
    logger.info(pprint.pformat(args))
    
    if not args.resume and os.path.exists(os.path.join(final_output_dir, 'model_best.pth')):
        raise ValueError('Final output directory exists: {}'.format(final_output_dir))
    
    # cudnn related setting
    cudnn.benchmark = config_hrnet.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = config_hrnet.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = config_hrnet.CUDNN.ENABLED
    torch.manual_seed(config_hrnet.SEED)
    np.random.seed(config_hrnet.SEED)
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
    
    # Load 2D Model (HR-Net)
    model_2d = PoseHighResolutionNet(config_hrnet)
    if args.finetune:
        model_2d.load_state_dict(torch.load('/output/h36m/hrnet/only_loss_2d_cosine_lr/model_best.pth',
                                            map_location=torch.device("cpu")), strict=False)
    else:
        model_2d.load_state_dict(torch.load(config_hrnet.MODEL.PRETRAINED, map_location=torch.device("cpu")), strict=False)
    model_2d = model_2d.to(device)
    
    # Load 3D model if it uses loss_3d
    if args.loss_3d:
        model_3d_dict = {}  # 3D model dictionary
        valid_model_3d_dict = {}  # 3D model dictionary for validation (only for multi-model training)
        for model_3d_name in available_models:
            args.model_3d = model_3d_name
            if args.model_3d == 'sb':           # SimpleBaseline
                if config_hrnet.DATASET.DATASET == 'panoptic':
                    from model_3d.simple_baseline.human_3d_pose_baseline.configs.defaults_panoptic import get_default_config as get_default_config_sb
                elif config_hrnet.DATASET.DATASET == 'h36m':
                    from model_3d.simple_baseline.human_3d_pose_baseline.configs.defaults_h36m import get_default_config as get_default_config_sb
                from model_3d.simple_baseline.human_3d_pose_baseline.models import get_model as get_model_sb
                config_sb = get_default_config_sb()
                config_sb.merge_from_file(os.path.join(root_dir, f'model_3d/simple_baseline/experiments/{config_hrnet.DATASET.DATASET}.yaml'))
                model_3d = get_model_sb(config_sb, num_joints=num_joints,  
                                model_weight_path=f'model_3d/simple_baseline/weights/model_best_{config_hrnet.DATASET.DATASET}.pth')
            elif args.model_3d == 'videopose':  # VideoPose
                from model_3d.videopose.common.model import TemporalModel
                args_videopose = argparse.Namespace(causal=False, dropout=0.25, channels=1024, dense=False, arc='1,1,1')
                filter_widths = [int(x) for x in args_videopose.arc.split(',')]
                model_3d = TemporalModel(num_joints, 2, num_joints, filter_widths=filter_widths, causal=args_videopose.causal, 
                                dropout=args_videopose.dropout, channels=args_videopose.channels, dense=args_videopose.dense)
                model_3d.load_state_dict(torch.load(f'model_3d/videopose/weights/model_best_{config_hrnet.DATASET.DATASET}_t{receptive_field}.bin', 
                                                    map_location=torch.device("cpu")))
            elif args.model_3d == 'iganet':     # IGANet
                if config_hrnet.DATASET.DATASET == 'panoptic':
                    from model_3d.iganet.model.model_IGANet_panoptic import Model as IGANet
                elif config_hrnet.DATASET.DATASET == 'h36m':
                    from model_3d.iganet.model.model_IGANet_h36m import Model as IGANet
                args_iganet = argparse.Namespace(layers=3, channel=512, d_hid=1024, n_joints=num_joints)
                model_3d = IGANet(args_iganet, device=device)
                model_3d.load_state_dict(torch.load(f'model_3d/iganet/weights/model_best_{config_hrnet.DATASET.DATASET}.pth', map_location=torch.device("cpu")))
            elif args.model_3d == 'mixste':     # MixStE
                from model_3d.mixste.common.model_cross import MixSTE2
                args_mixste = argparse.Namespace(in_chans=2, cs=512, dep=8, num_heads=8, mlp_ratio=2., qkv_bias=True, drop_path_rate=0)
                model_3d = MixSTE2(num_frame=receptive_field, num_joints=num_joints, in_chans=2, embed_dim_ratio=args_mixste.cs, depth=args_mixste.dep,
                            num_heads=args_mixste.num_heads, mlp_ratio=args_mixste.mlp_ratio, qkv_bias=args_mixste.qkv_bias, qk_scale=None, drop_path_rate=0)
                model_3d = nn.DataParallel(model_3d)
                model_checkpoint = torch.load(f'model_3d/mixste/weights/model_best_{config_hrnet.DATASET.DATASET}_t{receptive_field}.bin',
                                        map_location=lambda storage, loc: storage)
                model_3d.load_state_dict(model_checkpoint, strict=False)
                model_3d = model_3d.module
                
            model_3d = model_3d.to(device)
            model_3d.eval()
            if model_3d_name in model_3d_list:
                model_3d_dict[model_3d_name] = model_3d
            else:
                if len(model_3d_list) > 1:
                    valid_model_3d_dict[model_3d_name] = model_3d
            
        criterion_3d = nn.MSELoss(reduction="mean").to(device)
        criterion_3d_ohklm = nn.MSELoss(reduction="none").to(device)
    
    criterion_2d = JointsMSELoss(use_target_weight=config_hrnet.LOSS.USE_TARGET_WEIGHT).to(device)
    criterion_2d_ohklm = JointsMSELoss_ohklm(use_target_weight=config_hrnet.LOSS.USE_TARGET_WEIGHT).to(device)
           
    last_epoch = -1
    best_test_perf = 10000 if args.loss_3d else -1
    best_test_perf_cross = 10000 if args.loss_3d else -1
    best_model = False
    best_model_cross = False
    begin_epoch = 0
    checkpoint_file = os.path.join(final_output_dir, 'checkpoint.pth')
    
    train_losses = {'2d': [], '3d': [], 'total': []}
    test_metrics = {'mpjpe': [], 'p_mpjpe': [], 'acc_2d': []}
    
    optimizer_2d = torch.optim.Adam(model_2d.parameters(), lr=config_hrnet.TRAIN.LR)
    lr_scheduler_2d = torch.optim.lr_scheduler.MultiStepLR(
        optimizer_2d, config_hrnet.TRAIN.LR_STEP, config_hrnet.TRAIN.LR_FACTOR, last_epoch=last_epoch)
    
    
    losses_3d_dwa = deque(maxlen=3)
    temp = []
    for _ in model_3d_dict.keys():
        temp.append(torch.tensor(1.0))
    losses_3d_dwa.append(temp)
    losses_3d_dwa.append(temp)
    

    if args.resume and os.path.exists(checkpoint_file):
        logger.info(f"=> loading checkpoint '{checkpoint_file}'")
        checkpoint = torch.load(checkpoint_file)
        begin_epoch = checkpoint['epoch']
        best_test_perf = checkpoint['test_perf']
        last_epoch = checkpoint['epoch']
        model_2d.load_state_dict(checkpoint['state_dict'])
        optimizer_2d.load_state_dict(checkpoint['optimizer'])
        logger.info(f"=> loaded checkpoint '{checkpoint_file}' (epoch {checkpoint['epoch']})")
        
        train_losses = pickle.load(open(os.path.join(final_output_dir, 'train_losses.pkl'), 'rb'))
        test_metrics = pickle.load(open(os.path.join(final_output_dir, 'test_metrics.pkl'), 'rb'))
    

    # Training ...
    for epoch in range(begin_epoch, args.epochs+1):
        batch_time, data_time, losses, acc = AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()
        losses_2d, losses_3d = AverageMeter(), AverageMeter()
        model_2d.train()
        end = time.time()

        for i, (input, target, target_weight, meta) in enumerate(train_loader):
            data_time.update(time.time() - end)
            input, target, target_weight = input.to(device), target.to(device), target_weight.to(device)
            heatmap = model_2d(input)
            
            # convert heatmap to proper format
            if config_hrnet.DATASET.DATASET == 'panoptic':
                heatmap = h36m2panoptic_heatmap(heatmap)
            elif config_hrnet.DATASET.DATASET == 'mpi-inf-3dhp':
                raise NotImplementedError
                pass
            
            if args.loss_3d:
                loss_2d_ohklm_list, loss_3d_list, loss_3d_xy_list, loss_3d_z_list = [], [], [], []
                loss_3d_avg_list, loss_3d_cons_list = [], []
                preds_3d_list = []
                for model_3d_name, model_3d in model_3d_dict.items():
                    
                    preds_2d = get_final_preds_softargmax(config_hrnet, heatmap.clone().cpu(), meta['center'], meta['scale'])
                    args.model_3d = model_3d_name
                    
                    if args.model_3d == 'sb':   # SimpleBaseline
                        if config_hrnet.DATASET.DATASET == 'panoptic':
                            preds_2d_rtm = preds_2d_rtm.reshape(-1,num_joints* 2)
                        normalized_preds_2d = normalize_pose(preds_2d, torch.tensor(train_dataset.mean_2d, dtype=torch.float32),
                                                        torch.tensor(train_dataset.std_2d, dtype=torch.float32)).to(device)
                        preds_3d = model_3d(normalized_preds_2d.reshape(-1, num_joints*2)).to(device)
                        preds_3d = preds_3d.reshape(-1, num_joints, 3)
                        # gt_3d = normalize_pose(meta['joints_3d'], train_dataset.mean_3d, train_dataset.std_3d).to(device)
                        gt_3d = meta['joints_3d'] / 100.0 # cm to meter
                        gt_3d = gt_3d.to(device)
                        if config_hrnet.DATASET.DATASET == 'panoptic':
                            preds_3d = preds_3d.reshape(-1,num_joints* 3)
                        preds_3d = unnormalize_pose(preds_3d, torch.tensor(train_dataset.mean_3d, dtype=torch.float32).to(device), torch.tensor(train_dataset.std_3d, dtype=torch.float32).to(device)) / 100.0  # meter
                        if config_hrnet.DATASET.DATASET == 'panoptic':
                            preds_3d = preds_3d.reshape(-1,num_joints, 3)
                    elif args.model_3d == 'videopose':  # VideoPose
                        preds_2d[..., :2] = normalize_screen_coordinates_tensor(preds_2d[..., :2], w=meta['width'], h=meta['height'])
                        preds_2d = preds_2d.unsqueeze(0).to(device)
                        preds_3d = model_3d(preds_2d).squeeze(0)        # meter
                        gt_3d = meta['joints_3d'].to(device) / 100.0    # cm to meter
                    elif args.model_3d == 'iganet':     # IGANet
                        preds_2d[..., :2] = normalize_screen_coordinates_tensor(preds_2d[..., :2], w=meta['width'], h=meta['height'])
                        preds_2d = preds_2d.unsqueeze(0).to(device)
                        preds_3d = model_3d(preds_2d).permute(1, 0, 2, 3).squeeze(0)  # meter
                        gt_3d = meta['joints_3d'].to(device) / 100.0                  # cm to meter
                    elif args.model_3d == 'mixste':     # MixStE
                        preds_2d[..., :2] = normalize_screen_coordinates_tensor(preds_2d[..., :2], w=meta['width'], h=meta['height'])
                        preds_2d = preds_2d.unsqueeze(1).to(device)
                        preds_3d = model_3d(preds_2d).squeeze()        # meter
                        gt_3d = meta['joints_3d'].to(device) / 100.0    # cm to meter
                    
                    preds_3d_list.append(preds_3d)  # multi-model training을 위한 preds_3d_list
                    
                    loss_3d_list.append(criterion_3d(preds_3d, gt_3d))
                    loss_3d_xy_list.append(criterion_3d(preds_3d[:,:,:2], gt_3d[:,:,:2]))
                    loss_3d_z_list.append(criterion_3d(preds_3d[:,:,2], gt_3d[:,:,2]))
                    if args.ohklm:
                        ohklm_loss_2d, ohklm_loss_3d = ohklm(criterion_2d_ohklm(heatmap, target, target_weight), 
                                                    criterion_3d_ohklm(preds_3d, gt_3d).mean(dim=2), top_k=args.top_k)
                        loss_2d_ohklm_list.append(ohklm_loss_2d)
                
                if args.loss_3d_avg:    # average loss_3d
                    loss_3d_avg_list.append(criterion_3d(torch.stack(preds_3d_list).mean(dim=0), gt_3d))
                if args.loss_3d_cons:   # consistency loss_3d
                    temp_loss = 0.0
                    for a in range(len(preds_3d_list)):
                        for b in range(a+1, len(loss_3d_list)):
                            temp_loss += criterion_3d(preds_3d_list[a], preds_3d_list[b])
                    loss_3d_cons_list.append(temp_loss)
                
                losses_3d_dwa.append(loss_3d_list)
            loss_2d = criterion_2d(heatmap, target, target_weight)
        
            
            if args.loss_3d:    # 2D loss + 3D loss
                if args.dwa:
                    dwa_w = calculate_dwa_weights(losses_3d_dwa)
                else:
                    dwa_w = [1.0/len(losses_3d_dwa[-1]) for _ in losses_3d_dwa[-1]]
                    
                dropout_idx = np.random.randint(0, len(model_3d_list)+1)
                if args.dropout and len(model_3d_list) > 1 and dropout_idx < len(model_3d_list):
                    dwa_w[dropout_idx] = 0.0

                # compute weighted 2D loss 
                if args.ohklm:   # Online Hard Keypoint Lifting Mining
                    ohklm_loss_2d = sum(w * l for w, l in zip(dwa_w, loss_2d_ohklm_list))
                    # weighted_loss_2d = args.loss_2d_w/2 * loss_2d + args.loss_2d_w/2 * ohklm_loss_2d
                    weighted_loss_2d = args.loss_2d_w * loss_2d + args.ohklm_w * ohklm_loss_2d
                else:
                    weighted_loss_2d = args.loss_2d_w * loss_2d
                
                # compute weighted 3D loss 
                if args.loss_3d_z:
                    weighted_loss_xy_3d = sum(w * l for w, l in zip(dwa_w, loss_3d_xy_list))
                    weighted_loss_z_3d = sum(w * l for w, l in zip(dwa_w, loss_3d_z_list))
                    weighted_loss_3d = args.loss_3d_xy_w * weighted_loss_xy_3d + args.loss_3d_z_w * weighted_loss_z_3d
                else:
                    weighted_loss_3d = sum(w * l for w, l in zip(dwa_w, loss_3d_list)) * args.loss_3d_w
                
                loss = weighted_loss_2d + weighted_loss_3d   # final loss
                loss *= args.loss_single_w
                
                if args.loss_3d_avg:
                    loss_3d_avg = sum(loss_3d_avg_list) * args.loss_3d_avg_w
                    loss += loss_3d_avg
                if args.loss_3d_cons:
                    loss_3d_cons = sum(loss_3d_cons_list) * args.loss_3d_cons_w
                    loss += loss_3d_cons
                
                
            else:   # only 2D loss
                args.loss_2d_w, args.loss_3d_xy_w, args.loss_3d_z_w = 1, 0, 0
                weighted_loss_2d = loss_2d
                weighted_loss_3d = torch.tensor(-1)
                loss = weighted_loss_2d
        
                
            optimizer_2d.zero_grad()
            loss.backward()
            optimizer_2d.step()
                
            losses.update(loss.item(), input.size(0))
            losses_2d.update(weighted_loss_2d.item(), input.size(0))
            losses_3d.update(weighted_loss_3d.item(), input.size(0))
            _, avg_acc_2d, cnt, pred_2d = accuracy_2d(heatmap.detach().cpu().numpy(), target.detach().cpu().numpy())
            acc.update(avg_acc_2d, cnt)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            
            if i % config_hrnet.PRINT_FREQ == 0:
                if not args.loss_3d_cons:
                    loss_3d_cons = torch.tensor(-1)
                if not args.loss_3d_avg:
                    loss_3d_avg = torch.tensor(-1)
                msg = f'Epoch: [{epoch}][{i}/{len(train_loader)}]\t' \
                      f'Speed {input.size(0)/batch_time.val:.1f} samples/s\t' \
                      f'Loss_2d {losses_2d.avg:.5f}\t' \
                      f'Loss_3d {losses_3d.avg:.5f}\t' \
                      f'Loss {losses.avg:.5f}\t' \
                      f'Accuracy_2d {acc.avg:.3f}\t' \
                      f'LR {optimizer_2d.param_groups[0]["lr"]:.5f}\t' \
                      f'loss2d_w {args.loss_2d_w:.3f}  loss3d_xy_w {args.loss_3d_xy_w:.3f}  loss3d_z_w {args.loss_3d_z_w:.3f}\t' \
                      f'loss3d_avg {loss_3d_avg:.6f}  loss3d_cons {loss_3d_cons:.6f}' 
                          
                logger.info(msg)
                
            # break
            
        lr_scheduler_2d.step()

        if epoch == 30:
            for param_group in optimizer_2d.param_groups:
                param_group['lr'] *= 5
            logger.info(f"LR is multiplied by 10: before {optimizer_2d.param_groups[0]['lr']/5:.5f}, after {optimizer_2d.param_groups[0]['lr']:.5f}")
            
        train_losses['2d'].append(losses_2d.avg)
        train_losses['3d'].append(losses_3d.avg)
        train_losses['total'].append(losses.avg)
        pickle.dump(train_losses, open(os.path.join(final_output_dir, 'train_losses.pkl'), 'wb'))
        
        
        # validation
        if args.loss_3d:
            if len(model_3d_list) > 1 and len(valid_model_3d_dict) > 0: 
                test_mpjpe, test_p_mpjpe, cross_test_mpjpe, cross_test_p_mpjpe, test_acc_2d, test_mpjpe_dict =  \
                    validate(config_hrnet, valid_loader, model_2d, device, model_3d_dict=model_3d_dict, cross_model_3d_dict=valid_model_3d_dict,
                             output_dir=final_output_dir, stats=train_dataset.stats, ensemble=args.ensemble)
            else:
                test_mpjpe, test_p_mpjpe, test_acc_2d, test_mpjpe_dict = validate(config_hrnet, valid_loader, model_2d, device,
                            model_3d_dict=model_3d_dict, output_dir=final_output_dir, stats=train_dataset.stats, ensemble=args.ensemble)
            
            test_metrics['mpjpe'].append(test_mpjpe);test_metrics['p_mpjpe'].append(test_p_mpjpe);test_metrics['acc_2d'].append(test_acc_2d)
            pickle.dump(test_metrics, open(os.path.join(final_output_dir, 'test_metrics.pkl'), 'wb'))
            
            if test_mpjpe < best_test_perf:
                best_test_perf = test_mpjpe
                best_model = True
            else:
                best_model = False
            
            if len(model_3d_list) > 1 and len(valid_model_3d_dict) > 0 and cross_test_mpjpe < best_test_perf_cross:
                best_test_perf_cross = cross_test_mpjpe
                best_model_cross = True
            else:
                best_model_cross = False
                
            logger.info(f"=> saving checkpoint to {final_output_dir} --> best: {best_model} (PDJ@0.2: {test_acc_2d:.2f}, MPJPE: {test_mpjpe:.2f}mm, P-MPJPE: {test_p_mpjpe:.2f}mm)")
            
            if len(test_mpjpe_dict) > 1:
                for model_3d_name, mpjpe_list in test_mpjpe_dict.items():
                    logger.info(f"{model_3d_name}\t Test MPJPE: {np.mean(mpjpe_list):.2f}mm")
            
            save_checkpoint({
                'epoch': epoch + 1, 'model': f'hrnet_{config_hrnet.DATASET.DATASET}',
                'state_dict': model_2d.state_dict(), 'best_state_dict': model_2d.state_dict(),
                'test_perf': test_mpjpe, 'optimizer': optimizer_2d.state_dict(),
            }, best_model, final_output_dir)
            
            if len(model_3d_list) > 1 and len(valid_model_3d_dict) > 0:
                save_checkpoint({
                    'epoch': epoch + 1, 'model': f'hrnet_{config_hrnet.DATASET.DATASET}',
                    'state_dict': model_2d.state_dict(), 'best_state_dict': model_2d.state_dict(),
                    'test_perf': test_mpjpe, 'optimizer': optimizer_2d.state_dict(),
                }, best_model_cross, final_output_dir, cross_valid=True)
            
            
            if best_model and len(test_mpjpe_dict.keys()) > 1:
                pickle.dump(test_mpjpe_dict, open(os.path.join(final_output_dir, 'test_mpjpe_dict.pkl'), 'wb'))
        else:
            _, _, test_acc_2d, _ = validate(config_hrnet, valid_loader, model_2d, device, 
                                model_3d_dict=None, output_dir=final_output_dir, stats=train_dataset.stats)
            test_metrics['acc_2d'].append(test_acc_2d)
            pickle.dump(test_metrics, open(os.path.join(final_output_dir, 'test_metrics.pkl'), 'wb'))
            if test_acc_2d > best_test_perf:
                best_test_perf = test_acc_2d
                best_model = True
            else:
                best_model = False
            logger.info(f"=> saving checkpoint to {final_output_dir} --> best: {best_model} (PDJ@0.2: {test_acc_2d:.2f})")
            save_checkpoint({
                'epoch': epoch + 1, 'model': f'hrnet_{config_hrnet.DATASET.DATASET}',
                'state_dict': model_2d.state_dict(), 'best_state_dict': model_2d.state_dict(),
                'test_perf': test_acc_2d, 'optimizer': optimizer_2d.state_dict(),
            }, best_model, final_output_dir)
            
        torch.cuda.empty_cache() 
            
    final_model_state_file = os.path.join(final_output_dir, 'final_state.pth')
    logger.info(f'=> saving final model state to {final_model_state_file}')
    torch.save(model_2d.state_dict(), final_model_state_file)
    
    torch.cuda.empty_cache()

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count != 0 else 0
        
        
if __name__ == '__main__':
    main()
    