# https://github.com/DoctorKey/C-3PO/blob/main/src/dataset/vl_cmu_cd.py
import os
import glob

from os.path import join as pjoin, splitext as spt

import dataset.transforms as T 
from dataset.dataset import CDDataset, get_transforms

import dataset.path_config as Data_path
from deva.data.detection_video_reader import DetectionVideoReader

from PIL import Image
import cv2
import torch
import numpy as np


class VL_CMU_CD(CDDataset):
    # all images are 512x512
    def __init__(self, root, split = 'test', rotation=True, transforms=None, revert_transforms=None, input_size=None, exp_name = 'exp'):
        super(VL_CMU_CD, self).__init__(root, transforms)
        self.dataset_name = 'VL_CMU_CD'
        self.root = root
        self.rotation = rotation
        self.gt, self.t0, self.t1 = self._init_data_list()
        self._transforms = transforms
        self._revert_transforms = revert_transforms

        self.mask_dir = os.path.join(self.root, 'sam_mask')
        self.masktrack_dir = os.path.join(self.mask_dir, 'track_' + exp_name)
        self.pred_dir = './output/VL_CMU_CD/%s/'%exp_name

        self.query_image_path = self.root

        size_dict = {
            512: (512, 512),
            768: (768, 1024)
        }
        self.input_size = size_dict[input_size]

    def _init_data_list(self):
        gt = []
        t0 = []
        t1 = []
        self.image_total_files = sorted(glob.glob(os.path.join(self.root, 'mask', '*.png')))
        for file in self.image_total_files:
            if self._check_validness(file):
                idx = int(file.split('.')[0].split('_')[-1])
                if self.rotation or idx == 0:
                    gt.append(pjoin(self.root, 'mask', os.path.basename(file)))
                    t0.append(pjoin(self.root, 't0', os.path.basename(file)))
                    t1.append(pjoin(self.root, 't1', os.path.basename(file)))
        return gt, t0, t1

    def get_raw(self, index):
        fn_t0 = self.t0[index]
        fn_t1 = self.t1[index]
        fn_mask = self.gt[index]
        img_t0 = self._pil_loader(fn_t0)
        img_t1 = self._pil_loader(fn_t1)

        mask = self._pil_loader(fn_mask).convert("L")
        return img_t0, img_t1, mask, fn_t0, fn_t1
    
    def file_name_comp(self, seqname): return lambda idx: seqname + '_' + idx.zfill(2) + '_0.png'
    def get_frameidx(self, path): return int(os.path.basename(path).split('_')[2])
    def pth2gtpth(self, path): return path
    def pth2predpth(self, path): return path.replace(self.root, self.pred_dir).replace('/mask/', '/%s/'%self.get_subset_name(path))
    
    def load_label(self, gt_pth):
        label = np.asarray(Image.open(gt_pth)).copy()
        return torch.LongTensor(label)

    def get_subset_name(self, path, concat=True):
        return os.path.basename(path)[:5] if concat else ['', os.path.basename(path)[:5]]

    def preprocess_sequence_frames(self, clip_len):
        self.clip_len = clip_len
        self.all_data = {}
        cnt = 0

        for frame in self.image_total_files:
            seqname = os.path.basename(frame)[:5]
            if not seqname in self.all_data : self.all_data[seqname] = 0 
            self.all_data[seqname] += 1

        for seqname in self.all_data:
            seq_len = self.all_data[seqname]
            start = 0
            while start < seq_len:
                start += clip_len
                cnt += 1

        return cnt
    
    def get_datasets_clip(self, cross, **kwargs):
        assert all([i in ['q', 'r'] for i in cross]), '"Cross" must consist of only q (image from query seq) and r (image from reference seq).'
        is_query = list(map(lambda x: 1 if x=='q' else 0, cross)) # 'qr' to 10

        for seqname in self.all_data:
            seq_len = self.all_data[seqname]
            start = 0
            while start < seq_len:
                tc = self.clip_len if start + self.clip_len < seq_len else seq_len - start

                yield DetectionVideoReader(
                    vid_name = os.path.join(seqname, cross, str(start)),
                    ref_dir=os.path.join(self.root, 't0/'), 
                    query_dir=os.path.join(self.root, 't1/'),
                    mask_dir=os.path.join(self.mask_dir, seqname),
                    start = start, #'0.png',
                    clip_len=tc,
                    is_query=is_query * tc, # [1,0] to [1,0,1,0, ... , 1,0]
                    size=[512,512], # following C-3PO

                    file_name_comp = self.file_name_comp(seqname),
                )

                start += self.clip_len

def get_VL_CMU_CD(args, train=True, exp_name='experiments'):
    mode = 'train' if train else 'test'
    raw_root = Data_path.get_dataset_path('CMU_binary')
    size_dict = {
        512: (512, 512),
        768: (768, 1024)
    }
    transforms, revert_transforms = get_transforms(args, train, size_dict)
    dataset = VL_CMU_CD(os.path.join(raw_root, mode), 
                        input_size=args.input_size,        
                        transforms=transforms, revert_transforms=revert_transforms, exp_name=exp_name)
    print("VL_CMU_CD {}: {}".format(mode, len(dataset)))
    return dataset
        