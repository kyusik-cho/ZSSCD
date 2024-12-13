import os
from os import path

from torch.utils.data.dataset import Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode
import torch.nn.functional as F
from PIL import Image
import numpy as np
import cv2
from deva.dataset.utils import im_normalization

import pdb
class DetectionVideoReader(Dataset):
    """
    This class is used to read a video, one frame at a time
    """
    def __init__(self,
                 vid_name,
                 image_dir,
                 mask_dir,
                 size=-1,
                 to_save=None,
                 size_dir=None,
                 start=-1,
                 end=-1,
                 reverse=False,
                 image_suffix='_leftImg8bit',
                 source_dtype='.npy'):
        """
        image_dir - points to a directory of jpg images
        mask_dir - points to a directory of png masks
        size - resize min. side to size. Does nothing if <0.
        to_save - optionally contains a list of file names without extensions 
            where the segmentation mask is required
        """
        self.image_suffix = image_suffix
        self.source_dtype = source_dtype
        self.vid_name = vid_name
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.to_save = to_save
        if size_dir is None:
            self.size_dir = self.image_dir
        else:
            self.size_dir = size_dir

        self.frames = sorted(os.listdir(self.image_dir)) 
        print(self.frames)
        if start > 0:
            self.frames = self.frames[start:]
        if end > 0:
            self.frames = self.frames[:end]
        if reverse:
            self.frames = reversed(self.frames)
        #pdb.set_trace()
        if self.source_dtype == '.png':
            self.palette = Image.open(path.join(mask_dir, self.frames[0].replace('%s'%self.image_suffix,'').replace('.jpg',
                                                                             '.png'))).getpalette()
        elif self.source_dtype == '.npy':
            self.palette = None
        self.first_gt_path = path.join(self.mask_dir, self.frames[0].replace('%s'%self.image_suffix,'').replace('.jpg', self.source_dtype))
        
        if size < 0:
            self.im_transform = transforms.Compose([
                transforms.ToTensor(),
                im_normalization,
            ])
            self.mask_transform = transforms.Compose([])
        else:
            self.im_transform = transforms.Compose([
                transforms.ToTensor(),
                im_normalization,
                transforms.Resize(size, interpolation=InterpolationMode.BILINEAR, antialias=True),
            ])
            self.mask_transform = transforms.Compose([
                transforms.Resize(size, interpolation=InterpolationMode.NEAREST),
            ])
        self.size = size
        self.is_rgb = None

    def __getitem__(self, idx):
        frame = self.frames[idx]
        info = {}
        data = {}
        info['frame'] = frame
        info['save'] = (self.to_save is None) or (frame.replace('%s'%self.image_suffix,'')[:-4] in self.to_save)

        im_path = path.join(self.image_dir, frame)
        img = Image.open(im_path).convert('RGB')

        if self.image_dir == self.size_dir:
            shape = np.array(img).shape[:2]
        else:
            size_path = path.join(self.size_dir, frame)
            size_im = Image.open(size_path).convert('RGB')
            shape = np.array(size_im).shape[:2]

        img = self.im_transform(img)

        mask_path = path.join(self.mask_dir, frame.replace('%s'%self.image_suffix,'')[:-4] + self.source_dtype)

        if path.exists(mask_path):
          if self.source_dtype == '.png':
            mask = Image.open(mask_path)
            mask = self.mask_transform(mask)
            if mask.mode == 'RGB':
                mask = np.array(mask, dtype=np.int32)
                mask = mask[:, :, 0] + mask[:, :, 1] * 256 + mask[:, :, 2] * 256 * 256
                self.is_rgb = True
            else:
                mask = mask.convert('P')
                mask = np.array(mask, dtype=np.int32)
                self.is_rgb = False
            data['mask'] = mask

          elif self.source_dtype == '.npy':
            mask = np.load(mask_path).astype(np.int32)
            mask = cv2.resize(mask, dsize=(round(self.size*mask.shape[1]/mask.shape[0]),self.size), interpolation=cv2.INTER_NEAREST)
            self.is_rgb = False
            data['mask'] = mask

        # defer json loading to the model 
        json_path = path.join(self.mask_dir, frame[:-4] + '.json')
        if path.exists(json_path):
            info['json'] = json_path

        info['is_rgb'] = self.is_rgb
        info['shape'] = shape
        info['need_resize'] = not (self.size < 0)
        info['path_to_image'] = im_path
        data['rgb'] = img
        data['info'] = info

        return data

    def get_palette(self):
        return self.palette

    def __len__(self):
        return len(self.frames)
