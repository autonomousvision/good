"""This file contains exemplar code to extract depth and normal maps from pretrained Omnidata models.

Reference repository:
      https://github.com/EPFL-VILAB/omnidata/tree/main/omnidata_tools/torch
"""
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree

import argparse
import os.path
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

import torch
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
from torchvision import transforms
from mmcv import Config
from mmdet.datasets import (build_dataloader, build_dataset)
import torch.nn.functional as F
from modules.midas.dpt_depth import DPTDepthModel



parser = argparse.ArgumentParser(description='Visualize output for a single Task')

parser.add_argument('--output_path', default='dataset/coco/', dest='output_path', help="path to where output image should be stored")
parser.add_argument('--max_size', default=576, type=int)
parser.add_argument('--ori', action='store_true', help="whether to use original shape as input to the depth/normal extractor")
parser.add_argument('--dataset', default='train', type=str, choices=['train', 'val'])
parser.add_argument('--model', default='hybrid', type=str, choices=['hybrid', 'large'])


args = parser.parse_args()

if args.ori:
    cfg = Config.fromfile(f'mmdet_config_ori.py')
else:
    cfg = Config.fromfile(f'mmdet_config_{args.max_size}.py')
cfg.data_root = 'dataset/coco/'
cfg.data.train.ann_file = cfg.data_root + 'annotations/instances_train2017.json'
cfg.data.train.img_prefix = cfg.data_root  + 'train2017/'
cfg.data.val.ann_file = cfg.data_root + 'annotations/instances_val2017.json'
cfg.data.val.img_prefix = cfg.data_root  + 'val2017/'
cfg.data.test.ann_file = cfg.data_root + 'annotations/instances_val2017.json'
cfg.data.test.img_prefix = cfg.data_root  + 'val2017/'

if args.dataset == 'train':
    cfg.data.train.pipeline = cfg.data.test.pipeline
    dataset = build_dataset(cfg.data.train, dict(test_mode=True))
else:
    dataset =  build_dataset(cfg.data.test, dict(test_mode=True))
data_loader = build_dataloader(
            dataset,
            samples_per_gpu=1,
            workers_per_gpu=cfg.data.workers_per_gpu,
            dist=False,
            shuffle=False)
target_tasks = ['depth', 'normal']
root_dir = './pretrained_models/'

trans_topil = transforms.ToPILImage()

os.system(f"mkdir -p {args.output_path}")
map_location = (lambda storage, loc: storage.cuda()) if torch.cuda.is_available() else torch.device('cpu')
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

patch_size = 32

def standardize_depth_map_np(img, mask_valid=None, trunc_value=0.1):
    if mask_valid is not None:
        img[~mask_valid] = np.nan
    sorted_img = np.sort((img.flatten()))
    # Remove nan, nan at the end of sort
    num_nan = np.isnan(sorted_img).sum()
    if num_nan > 0:
        sorted_img = sorted_img[:-num_nan]
    # Remove outliers
    trunc_img = sorted_img[int(trunc_value * len(sorted_img)): int((1 - trunc_value) * len(sorted_img))]
    trunc_mean = trunc_img.mean()
    trunc_var = trunc_img.var()
    eps = 1e-6
    # Replace nan by mean
    img = np.nan_to_num(img, nan=trunc_mean)
    # Standardize
    img = (img - trunc_mean) / np.sqrt(trunc_var + eps)
    return img

for task in target_tasks:
    if task == "normal":
        pretrained_weights_path = root_dir + 'omnidata_dpt_normal_v2.ckpt'
        model = DPTDepthModel(backbone='vitb_rn50_384', num_channels=3) 
    else:
        if args.model == "large":
            assert len(target_tasks) == 1, "No DPT-Large for normal"
            pretrained_weights_path = root_dir + 'dpt_large-midas-2f21e586.pt'  # 'omnidata_dpt_depth_v1.ckpt'
            model = DPTDepthModel(backbone='vitl16_384') # DPT Large
        else:
            pretrained_weights_path = root_dir + 'omnidata_dpt_depth_v2.ckpt'
            model = DPTDepthModel(backbone='vitb_rn50_384') # DPT Hybrid
    checkpoint = torch.load(pretrained_weights_path, map_location=map_location)
    if 'state_dict' in checkpoint:
        state_dict = {}
        for k, v in checkpoint['state_dict'].items():
            state_dict[k[6:]] = v
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)
    model.to(device)

    for i, data in tqdm(enumerate(data_loader), desc=f'Extracting {task}'):
        if args.ori:
            output_path = args.output_path + f'{args.dataset}_' + task + f'_ori_omni' if args.model=="hybrid" else args.output_path + f'{args.dataset}_' + task + f'_ori_omni_large'
        else:
            output_path = args.output_path + f'{args.dataset}_' + task + f'{args.max_size}_omni' if args.model=="hybrid" else args.output_path + f'{args.dataset}_' + task + f'{args.max_size}_omni_large'
        img_path = data['img_metas'][0].data[0][0]['filename']
        imgname = os.path.splitext(os.path.basename(img_path))[0]
        save_path = os.path.join(output_path, f'{imgname}.jpg')
        with torch.no_grad():
            img_ori_shape = data['img_metas'][0].data[0][0]['ori_shape']
            img_shape = data['img_metas'][0].data[0][0]['img_shape']
            ori_height, ori_width = img_ori_shape[0], img_ori_shape[1]
            height, width = img_shape[0], img_shape[1]
            img_bgr = data['img'][0].float().div(255)
            img = img_bgr[:, [2,1,0], :, :] # convert bgr to rgb
            if args.ori:
                size_im = (
                    1,
                    img.shape[1],
                    int(np.ceil(img.shape[2] / patch_size) * patch_size),
                    int(np.ceil(img.shape[3] / patch_size) * patch_size),
                )
                paded = torch.zeros(size_im)
                paded[0, :, : img.shape[2], : img.shape[3]] = img
                img = paded
            img = img.to(device)

            # Normalize
            if task == "depth":
                mean = torch.tensor([0.5, 0.5, 0.5]).reshape((1,3,1,1)).to(device)
                std = torch.tensor([0.5, 0.5, 0.5]).reshape((1,3,1,1)).to(device)
                img = (img - mean)/std
            if args.model == 'large':
                output = model(img)
                if args.ori:
                        output_to_save = output[:, :height, :width]
                else:
                    output_to_save = F.interpolate(output[:, :height, :width].unsqueeze(0), (ori_height, ori_width), mode='bicubic').squeeze(0)
                output_to_save = standardize_depth_map_np(output_to_save.detach().cpu().numpy())
                plt.imsave(save_path, output_to_save.squeeze(),cmap='viridis')
                plt.close()
            else:
                output = model(img).clamp(min=0, max=1) 
                if task == "depth":
                    if args.ori:
                        output_to_save = output[:, :height, :width]
                    else:
                        output_to_save = F.interpolate(output[:, :height, :width].unsqueeze(0), (ori_height, ori_width), mode='bicubic').squeeze(0)
                    output_to_save = output_to_save.clamp(0,1)
                    output_to_save = 1 - output_to_save
                    plt.imsave(save_path, output_to_save.detach().cpu().squeeze(),cmap='viridis')
                    plt.close()
                else:
                    if args.ori:
                        output_to_save = output[0][:, :height, :width]
                    else:
                        output_to_save = F.interpolate(output[0][:, :height, :width].unsqueeze(0), (ori_height, ori_width), mode='bicubic').squeeze(0)
                    img_to_save = trans_topil(output_to_save)
                    img_to_save.save(save_path) 
                    img_to_save.close()
