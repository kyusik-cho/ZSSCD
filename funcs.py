import os
import numpy as np
import torch
import torch.cuda
from torch.utils.data.dataloader import DataLoader 
from PIL import Image
from tqdm import tqdm
#from utils import *
from model.model import load_mask_model
import warnings
warnings.filterwarnings('ignore')
  
def mask_generate(data_loader: DataLoader,
                  device: torch.device,
                  model = 'sam'):
    model = load_mask_model(model,device)

    for img_t0, img_t1, mask, ref_rgb_path, test_rgb_path in tqdm(data_loader, desc='generating mask...'):
        # img: [C, H, W], uint8
        # lbl: [1, H, W], (0, 1, ...20)
        # img_name: (str,)
        # lbl_name: (str,)
        
        subsetname = data_loader.dataset.get_subset_name(ref_rgb_path[0]) 
        rgb_idx = ref_rgb_path[0].split('/')[-1].split('.')[0]
    
        for idx, image in enumerate([img_t0, img_t1]):
            tidx = 'Query' if idx else 'Ref'
            image = np.array(image).squeeze(0)
            masks = model.generate(image)

            masks = sorted(masks, key=lambda x: np.sum(x['segmentation']), reverse=False)
            label_counter = 1
            label_map = np.zeros_like(masks[0]['segmentation'], dtype=np.uint16)
            os.makedirs(os.path.join(data_loader.dataset.mask_dir, subsetname), exist_ok=True)
            mask_path = os.path.join(data_loader.dataset.mask_dir, subsetname, tidx, rgb_idx + '.npy')
            for i in range(len(masks)):
                cur_mask = masks[i]['segmentation']
                nonzero_pixels = np.where(cur_mask > 0)
                if 0 not in np.unique(label_map[nonzero_pixels]):
                    continue
                else:
                    label_map[nonzero_pixels] = label_counter
                    label_counter = label_counter + 1
                    
            #### If a mask is obscured too much by another mask, it is merged with that mask.
            label_map = process_small_masks(label_map, masks, prop_th=0.5)

            if data_loader.dataset.dataset_name == 'VL_CMU_CD':
                # remove the remove_region
                label_map = remove_ignore_region(label_map, masks, image, prop_th=0.5)
                    
            os.makedirs(os.path.dirname(mask_path),exist_ok=True)
            np.save(mask_path, label_map)

""" process small masks with overlapping proportion threshold"""
def process_small_masks(label_map, masks, prop_th=0.5):
    indices = np.unique(label_map)
    indices = indices[1:] if 0 in indices else indices

    for i in indices:
        mask_pos = masks[i-1]['segmentation'] # note that label_counter starts from 1, masks index starts from 0.
        _label, _count = np.unique(label_map[mask_pos], return_counts=True)
        if len(_label) < 2: # Does not overlap with any mask
            continue 
        count = _count[_label != i]

        overlap_prop = count.sum() / _count.sum()
        if overlap_prop > prop_th:
            label = _label[_label != i]
            new_label = label[np.argmax(count[label != i])]
            label_map[mask_pos] = new_label

    return label_map

def remove_ignore_region(label_map, masks, image, prop_th=0.5):
    indices = np.unique(label_map)
    indices = indices[1:] if 0 in indices else indices

    for i in indices:
        mask_pos = masks[i-1]['segmentation'] # note that label_counter starts from 1, masks index starts from 0.
        ignore_pixels = (image[mask_pos] == np.array([[0,0,0]])).all(1)
        if ignore_pixels.sum() / ignore_pixels.size > prop_th:
            label_map[mask_pos] = 0

    return label_map





