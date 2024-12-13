# https://github.com/DoctorKey/C-3PO/blob/main/src/dataset/pcd.py

import os, glob
import torch
import numpy as np
import PIL
from PIL import Image

from os.path import join as pjoin, splitext as spt

from dataset.dataset import CDDataset, get_transforms
import dataset.transforms as T

import dataset.path_config as Data_path
from deva.data.detection_video_reader import DetectionVideoReader


class PCD_Raw(CDDataset):
    # all images are 224x1024
    # object: black(0)  ->   white(255)  ->  True
    #                 invert           toTensor  
    def __init__(self, root, num=0, train=True, args=None, size_dict=None, name=None, transforms=None, revert_transforms=None, exp_name = 'exp'):
        super(PCD_Raw, self).__init__(root, transforms)
        assert num in [0, 1, 2, 3, 4]
        assert name in ['GSV', 'TSUNAMI']
        self.root = root
        self.num = num
        self.istrain = train
        self.gt, self.t0, self.t1 = self._init_data_list()
        self._transforms = transforms
        self._revert_transforms = revert_transforms
        self.input_size = size_dict[args.input_size][::-1] # [height, width] -> [width, height]

        self.dataset_name = name
        self.mask_dir = os.path.join(self.root, 'sam_mask')
        self.masktrack_dir = os.path.join(self.mask_dir, 'track_' + exp_name)
        self.pred_dir = './output/PCD/%s/'%exp_name
        self.query_image_path = self.root

    def _init_data_list(self):
        gt = []
        t0 = []
        t1 = []
        self.image_total_files = sorted(glob.glob(os.path.join(self.root, 'mask', '*.bmp')))
        
        for _file in self.image_total_files:
            file = os.path.basename(_file)
            if self._check_validness(file):
                # idx = int(file.split('.')[0])
                # img_is_test = self.num * 2 <= (idx % 10) < (self.num + 1) * 2
                # if (self.istrain and not img_is_test) or (not self.istrain and img_is_test):
                    gt.append(pjoin(self.root, 'mask', file))
                    t0.append(pjoin(self.root, 't0', file.replace('bmp', 'jpg')))
                    t1.append(pjoin(self.root, 't1', file.replace('bmp', 'jpg')))
        return gt, t0, t1

    def get_raw(self, index):
        fn_t0 = self.t0[index]
        fn_t1 = self.t1[index]
        fn_mask = self.gt[index]
        img_t0 = self._pil_loader(fn_t0)
        img_t1 = self._pil_loader(fn_t1)

        mask = self._pil_loader(fn_mask).convert("L")
        mask = PIL.ImageOps.invert(mask)

        # if self.input_size is not None:
        mask = mask.resize(self.input_size, resample=Image.NEAREST)
        img_t0 = img_t0.resize(self.input_size, resample=Image.BICUBIC)
        img_t1 = img_t1.resize(self.input_size, resample=Image.BICUBIC)

        return img_t0, img_t1, mask, fn_t0, fn_t1


    def file_name_comp(self, seqname): return lambda idx: seqname.zfill(8) + '.jpg' # Regardless of the index within the sequence, only the sequence name is used.
    def get_frameidx(self, path): return 0 #int(os.path.basename(path)[:-4]) # interact with clip_len. 
    def pth2gtpth(self, path): return path
    def pth2predpth(self, path): #return path.replace(self.root, self.pred_dir).replace('/mask/', '%s/'%self.get_subset_name(path))
        return  os.path.join(
                self.pred_dir, 
                self.get_subset_name(path), 
                self.file_name_comp(os.path.basename(path)[:5])(str(self.get_frameidx(path))).replace('.jpg', '.png'))
            

    def load_label(self, gt_pth):
        label = Image.open(gt_pth).convert('L')
        label = PIL.ImageOps.invert(label)
        return torch.LongTensor(np.array(label))


    def get_subset_name(self, path, concat=True):
        return os.path.basename(path)[:-4] if concat else ['', os.path.basename(path)[:-4]]

    def preprocess_sequence_frames(self, clip_len):
        self.clip_len = clip_len
        self.all_data = {}
        cnt = 0

        for frame in self.image_total_files:
            # maybe,  
            seqname = self.get_subset_name(frame)
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
                    size=self.input_size[::-1], 
                    file_name_comp = self.file_name_comp(seqname),
                )
                start += self.clip_len


def get_pcd_raw(args, sub, num=0, train=True, exp_name='experiments'):
    assert sub in ['GSV', 'TSUNAMI']
    assert num in [0, 1, 2, 3, 4]
    root = os.path.join(Data_path.get_dataset_path('PCD_raw'), sub)
    input_size = args.input_size
    size_dict = {
        224: (224, 1024),
        256: (256, 1024),
        448: (448, 2048)
    }
    assert input_size in size_dict, "input_size: {}".format(size_dict.keys())
    transforms, revert_transforms = get_transforms(args, train, size_dict)
    dataset = PCD_Raw(root, num, train, args, size_dict, name=sub, transforms=transforms, revert_transforms=revert_transforms, exp_name=exp_name)
    dataset.name = sub
    mode = "Train" if train else "Test"
    print("PCD_Raw_{}_{} {}: {}".format(sub, num, mode, len(dataset)))
    return dataset

def get_GSV(args, train=True, exp_name='experiments'):
    return get_pcd_raw(args, 'GSV', args.data_cv, train=train, exp_name=exp_name)

def get_TSUNAMI(args, train=True, exp_name='experiments'):
    return get_pcd_raw(args, 'TSUNAMI', args.data_cv, train=train, exp_name=exp_name)

