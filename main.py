from builtins import breakpoint
import os
import random
import argparse
import numpy as np
import torch
import torch.cuda
from torch.utils.data.dataloader import DataLoader 
from dataset import dataset_dict
from utils_funcs import * 
from funcs import mask_generate

import warnings
warnings.filterwarnings('ignore')

def get_argparser():
    parser = argparse.ArgumentParser()                      
    parser.add_argument('--dataset', type=str, default='ChangeSim_Multi',
                        choices=['VL_CMU_CD', 'TSUNAMI', 'GSV', 'ChangeSim_Multi'],
                        help='Name of dataset to use')
    parser.add_argument('--data_cv', type=int, default=0,
                        help='CV only for PCD')      
    parser.add_argument('--changesim_subset', type=str, default='normal',
                        choices=['normal', 'dark', 'dust'],
                        help='ChangeSim Data Subset')      
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Batch size for label')        
    parser.add_argument('--num_workers', type=int, default=4,
                        help='num workers')                                            
    parser.add_argument('--gpu_id', type=str, default='0',
                        help='ID of GPU to use')                        
    parser.add_argument('--random_seed', type=int, default=1,
                        help='Random seed for reproducibility')
    parser.add_argument('--save_path', type=str, default='./output',
                        help='Path to save files')                                                                                                  
    parser.add_argument('--mask_model', type=str, default='sam',
                        choices=['sam'],
                        help='Name of model to generate mask')       
    return parser

def get_dataset(opts, 
                train,
                exp_name='experiments',
                ):
    ''' 
    Dataset
    '''
    opts.randomflip = False
    if opts.dataset == 'ChangeSim_Multi':
        opts.input_size = 480
    elif opts.dataset == 'VL_CMU_CD':
        opts.input_size = 512
    elif opts.dataset == 'TSUNAMI' or opts.dataset == 'GSV':
        opts.input_size = 256

    dataset = dataset_dict[opts.dataset](opts, train=train, exp_name=exp_name)

    return dataset

if __name__ == '__main__':
    # init
    args = get_argparser()
    opts = args.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = opts.gpu_id
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device: %s' % device)
    torch.manual_seed(opts.random_seed)
    np.random.seed(opts.random_seed)
    random.seed(opts.random_seed)


    ################################################ 

    # main functions
    run_mask_generate = True
    run_mask_track = True
    run_aggregate_pred = True

    # result
    run_eval = True

    # output folder 
    exp_name = 'experiment'  

    # tracking hyperparams
    clip_len : int = 60 
    detection_every : int = 5
    num_voting_frames : int = 3

    ################################################ 

    
    dataset = get_dataset(opts=opts,train=False, exp_name=exp_name)
    data_loader = DataLoader(dataset=dataset, 
                              batch_size=opts.batch_size, 
                              shuffle=False, 
                              num_workers=opts.num_workers)
    print('Dataset: %s, len: %d' %  (opts.dataset, len(data_loader)))    
        
    os.makedirs(opts.save_path, exist_ok=True)
    print("run_mask_generate:", run_mask_generate)
    if run_mask_generate:
        mask_generate(data_loader=data_loader, 
                      device=device, 
                      model=opts.mask_model)

    print("run_mask_track:", run_mask_track)
    if run_mask_track:
        from deva.deva_main import mask_track_clip
        from deva.inference.eval_args import add_deva_args
        with torch.no_grad():
            args = add_deva_args(args)
            mask_track_clip(meta_loader=data_loader, cross='qr', clip_len=clip_len, 
                           detection_every=detection_every, num_voting_frames=num_voting_frames, args=args)
            mask_track_clip(meta_loader=data_loader, cross='rq', clip_len=clip_len, 
                           detection_every=detection_every, num_voting_frames=num_voting_frames, args=args)

    print('run_aggregate_pred', run_aggregate_pred)
    if run_aggregate_pred: 
        aggregate_clip_prediction(data_loader, cross = ['qr', 'rq'], clip_len=clip_len, args=args)

    print('run_eval', run_eval)
    if run_eval:
      if opts.dataset == 'ChangeSim_Multi':
        compute_miou(data_loader)
        
      else:
        compute_f1(data_loader)
