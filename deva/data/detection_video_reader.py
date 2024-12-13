import os
from os import path

from torch.utils.data.dataset import Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from PIL import Image
import numpy as np


im_normalization = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

class DetectionVideoReader(Dataset):
    """
    This class is used to read a video, one frame at a time
    """
    def __init__(self,
                 vid_name,
                 ref_dir, 
                 query_dir, 
                 mask_dir,
                 to_save=None,
                 start = 0, # mask_dir + frame number
                 clip_len=3,
                 size=-1,
                 is_query:list=[1,0,1,0,1,0],
                 file_name_comp = None):

        self.vid_name = vid_name
        self.query_path = query_dir
        self.ref_path = ref_dir
        self.mask_dir = mask_dir
        self.to_save = to_save
        self.clip_len = clip_len
        self.is_query = is_query

        self.file_name_comp = file_name_comp

        assert self.clip_len == len(is_query) // 2

        self.frames = []
        self.start_idx = start
        change= 0
        for i, iQ in enumerate(is_query):
            change += (_now := is_query[0] ^ iQ) 
            self.frames.append(
                self.start_idx + i - change
            )

        try:
            self.palette = Image.open(path.join(mask_dir, 'Query', str(start)+'.png')).getpalette()
        except:
            self.palette = None

        if size is None:
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
        frame = self.file_name_comp(str(self.frames[idx]))
        info = {}
        data = {}
        info['frame'] = ('Q' if self.is_query[idx] else 'R') + frame     

        im_path = os.path.join(self.query_path, frame) if self.is_query[idx] else os.path.join(self.ref_path, frame)   
        img = Image.open(im_path).convert('RGB')

        shape = np.array(img).shape[:2]
        img = self.im_transform(img)
        mask_path = path.join(self.mask_dir, 
                                'Query' if self.is_query[idx] else 'Ref',
                                frame[:-4] + '.npy') 
        
        info['save'] = (self.to_save is None) or (path.basename(frame)[:-4] in self.to_save) 
        if path.exists(mask_path):
            mask = np.load(mask_path).astype(np.int32)
            self.is_rgb = False
            data['mask'] = mask

        # defer json loading to the model 
        json_path = path.join(self.mask_dir, path.basename(frame)[:-4] + '.json')
        if path.exists(json_path):
            info['json'] = json_path

        info['is_rgb'] = self.is_rgb
        info['shape'] = self.size if self.size != None else shape
        info['need_resize'] = self.size != None
        info['path_to_image'] = im_path
        data['rgb'] = img
        data['info'] = info

        return data

    def get_palette(self):
        return self.palette

    def __len__(self):
        return len(self.frames)

