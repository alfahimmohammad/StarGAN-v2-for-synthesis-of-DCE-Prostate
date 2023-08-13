"""
StarGAN v2
Copyright (c) 2020-present NAVER Corp.

This work is licensed under the Creative Commons Attribution-NonCommercial
4.0 International License. To view a copy of this license, visit
http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.
"""

import os
import shutil
from collections import OrderedDict, defaultdict
from tqdm import tqdm

import numpy as np
import torch

from metrics.fid import calculate_fid_given_paths
from metrics.lpips import calculate_lpips_given_images
from core.data_loader import get_eval_loader
from core import utils
from dataset_prostate import *
import torch.nn.functional as F
# from main_1 import create_data_loaders

def def_value():
    return []


def create_data_loaders(args):
        
    acc_factors = args.acceleration_factor#.split(',')
    mask_types = args.mask_type#.split(',')
    dataset_type_1 = args.dataset_type_1#.split(',')
    loader = torch.utils.data.DataLoader(SliceDataProstate(args.train_path,mode='train'),batch_size=1,shuffle=False)
    
    test_loader = torch.utils.data.DataLoader(SliceDataProstate(args.validation_path,mode='test'),batch_size=1,shuffle=False)

    return loader , test_loader

@torch.no_grad()
def calculate_metrics(nets, args, step, mode):
    acc_factors = args.acceleration_factor#.split(',')
    mask_types = args.mask_type#.split(',')
    print('Calculating evaluation metrics...')
    assert mode in ['latent', 'reference']
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

#     domains = os.listdir(args.validation_path)
    domains = ['mrbrain_t1', 'mrbrain_flair', 'mrbrain_ir']
    domains.sort()
    num_domains = len(domains)
#     print('Number of domains: %d' % num_domains)

    lpips_dict = OrderedDict()
    loader = torch.utils.data.DataLoader(SliceDataProstate(args.validation_path,mode='test'),batch_size=1,shuffle=False)
    task_preds = defaultdict(def_value)
    lpips_values = defaultdict(def_value)
    
    for i, (x_src, y_src, x_trg, y_trg, src_domain, trg_domain) in enumerate(tqdm(loader, total=len(loader))):

        x_src, y_src, x_trg, y_trg = x_src.to(device), y_src.to(device), x_trg.to(device), y_trg.to(device)
#         print(f'x_src: {x_src.shape}, x_trg: {x_trg.shape}, y_src: {y_src.shape}, y_trg: {y_trg.shape}')
        path_src = os.path.join(args.validation_path , src_domain[0])
        task = '%s2%s_%s' % (src_domain[0], trg_domain[0],step)
        path_fake = os.path.join(args.eval_dir, task)
#             print('path fake is ', path_fake)
#         shutil.rmtree(path_fake, ignore_errors=True)
        os.makedirs(path_fake,exist_ok = True)

        

        x_src = F.pad(input=x_src, pad=(8, 8, 8, 8), mode='replicate')
        x_src = x_src.unsqueeze_(1).float()
        masks = nets.fan.get_heatmap(x_src) if args.w_hpf > 0 else None
        N = x_src.size(0)

        group_of_images = []
        if mode == 'latent':
            z_trg = torch.randn(N, args.latent_dim).to(device)
            s_trg = nets.mapping_network(z_trg, y_trg)
        else:
            x_trg = F.pad(input=x_trg, pad=(8, 8, 8, 8), mode='replicate')
            x_trg = x_trg.unsqueeze_(1).float()

            if x_trg.size(0) > N:
                x_trg = x_trg[:N]
            s_trg = nets.style_encoder(x_trg, y_trg)
        
#         print(f'eval, x_src: {x_src.shape}, s_trg: {s_trg.shape}, masks: {masks[0].shape, masks[1].shape}')
#         s_trg = s_trg.squeeze(1)
        x_fake = nets.generator(x_src, s_trg, masks=masks)
        group_of_images.append(x_fake)

        # save generated images to calculate FID later
        for k in range(N):
            filename = os.path.join(
                path_fake,
                '%.4i_%.2i.png' % (i*args.val_batch_size+(k+1), 1))
            utils.save_image(x_fake[k], ncol=1, filename=filename)

#         lpips_value = calculate_lpips_given_images(group_of_images)
        task_preds[task].append(x_fake)
        lpips_value = calculate_lpips_given_images([x_fake])
        lpips_values[task].append(lpips_value)

    for key, item in lpips_values.items():
        lpips_dict['LPIPS_%s/%s' % (mode, key)] = np.array(lpips_values[key]).mean()
        
#     lpips_dict['LPIPS_%s/%s' % (mode, task)] = lpips_mean
        
    lpips_mean = 0
    for _, value in lpips_dict.items():
        lpips_mean += value / len(lpips_dict)
    lpips_dict['LPIPS_%s/mean' % mode] = lpips_mean

    # report LPIPS values
    filename = os.path.join(args.eval_dir, 'LPIPS_%.5i_%s.json' % (step, mode))
    utils.save_json(lpips_dict, filename)

    # calculate and report fid values
    calculate_fid_for_all_tasks(args, domains, step=step, mode=mode)

def calculate_fid_for_all_tasks(args, domains, step, mode):
    acc_factors = args.acceleration_factor#.split(',')
    mask_types = args.mask_type#.split(',')
    print('Calculating FID for all tasks..yy.')
    fid_values = OrderedDict()
    for trg_domain in domains:
        src_domains = [x for x in domains if x != trg_domain]

        for src_domain in src_domains:
            task = '%s2%s' % (src_domain, trg_domain)
            loader_ref = torch.utils.data.DataLoader(SliceDataref2(args.validation_path,acc_factors,[trg_domain],mask_types,'validation'),batch_size=4,shuffle=False)
            loader_src = torch.utils.data.DataLoader(SliceDataref2(args.validation_path,acc_factors,[src_domain],mask_types,'validation'),batch_size=4,shuffle=False)
#             path_real = "home/lekhasri/stargan-v2/directories/real_dirhome/lekhasri/stargan-v2/directories/real_dir"
            path_fake = os.path.join(args.eval_dir, task)
            print('Calculating FID for %s...' % task)
            fid_value = calculate_fid_given_paths(
#                 paths=[path_real, path_fake],
                paths = [loader_ref, loader_src],
                img_size=args.img_size,
                batch_size=args.val_batch_size)
            fid_values['FID_%s/%s' % (mode, task)] = fid_value

    # calculate the average FID for all tasks
    fid_mean = 0
    for _, value in fid_values.items():
        fid_mean += value / len(fid_values)
    fid_values['FID_%s/mean' % mode] = fid_mean

    # report FID values
    filename = os.path.join(args.eval_dir, 'FID_%.5i_%s.json' % (step, mode))
    utils.save_json(fid_values, filename)
