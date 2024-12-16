import os
import numpy as np
import torch
from PIL import Image
import glob
from torchvision.transforms import functional as F
import cv2

import dataset.transforms as T 
from dataset.dataset import CDDataset
import dataset.path_config as Data_path

from deva.data.detection_video_reader import DetectionVideoReader
from os import path

def Dict_indexing():
    Dict = {}
    Dict['background']={'index':0,'subnames':[]}
    Dict['column']={'index':1,'subnames':['pillar','pilar']} #sero
    Dict['pipe']={'index':2,'subnames':['tube']}
    Dict['wall']={'index':3,'subnames':['tunnel']}
    Dict['beam']={'index':4,'subnames':[]} # garo
    Dict['floor']={'index':5,'subnames':['slam','ground','road','walk','floorpanel']}
    Dict['frame']={'index':6,'subnames':['scafolding','scaffolding','scaffold','formwork','pole','support']}
    Dict['fence']={'index':7,'subnames':['fencning']}

    Dict['wire']={'index':8,'subnames':['wirecylinder']}
    Dict['cable']={'index':9,'subnames':[]}
    Dict['window']={'index':10,'subnames':['glass_panel']}
    Dict['railing']={'index':11,'subnames':[]}
    Dict['rail']={'index':12,'subnames':[]}
    Dict['ceiling']={'index':13,'subnames':['roof']}
    Dict['stair']={'index':14,'subnames':[]}
    Dict['duct']={'index':15,'subnames':['vent','ventilation']}
    Dict['gril']={'index':16,'subnames':['grid']}  # bunker's platform

    Dict['lamp']={'index':17,'subnames':['light']} # GOOD
    Dict['trash']={'index':18,'subnames':['debris','book','paper']} # GOOD
    Dict['shelf']={'index':19,'subnames':['drawer','rack','locker','cabinet']}
    Dict['door']={'index':20,'subnames':['gate']} #GOOD
    Dict['barrel']={'index':21,'subnames':['barel','drum','tank']} # GOOD
    Dict['sign']={'index':22,'subnames':['signcver']} # GOOD
    Dict['box']={'index':23,'subnames':['paperbox','bin','cube','crateplastic']} # Good
    Dict['bag']={'index':24,'subnames':[]} # GOOD
    Dict['electric_box']={'index':25,'subnames':['fusebox','switchboard','electricalsupply',
                                             'electric_panel','powerbox','control_panel']} # GOOD
    Dict['vehicle']={'index':26,'subnames':['truck','trailer','transporter','forklift']}
    Dict['ladder']={'index':27,'subnames':[]} # GOOD
    Dict['canister']={'index':28,'subnames':['can','bottle','cylinder','keg']}
    Dict['extinguisher']={'index':29,'subnames':['fire_ex']} # GOOD
    Dict['pallet'] = {'index': 30, 'subnames': ['palete', 'palette']}  # GOOD
    Dict['hand_truck'] = {'index': 31, 'subnames': ['pumptruck','pallet_jack']}  # GOOD

    return Dict


class SegHelper:
    def __init__(self,opt=None,idx2color_path='../../backup/idx2color.txt',num_class=32):
        self.opt = opt
        self.num_classes = num_class
        self.idx2color_path = idx2color_path
        f = open(self.idx2color_path, 'r')
        self.idx2color = {k:[] for k in range(self.num_classes)}
        for j in range(256):
            line = f.readline()
            line = line.strip(' \n').strip('[').strip(']').strip(' ').split()
            line = [int(l) for l in line if l.isdigit()]
            self.idx2color[j] = line # color in rgb order

        self.color2idx = {tuple(v):k for k,v in self.idx2color.items()}
        name2idx = Dict_indexing()
        self.name2idx = {k: name2idx[k]['index'] for k in name2idx.keys()}
        self.idx2name = {v:k for k,v in self.name2idx.items()}
        self.idx2name_padding = {v:'BG' for v in range(self.num_classes,256)}
        self.idx2name.update(self.idx2name_padding)

    def unique(self,array):
        uniq, index = np.unique(array, return_index=True, axis=0)
        return uniq[index.argsort()]

    def extract_color_from_seg(self,img_seg):
        colors = img_seg.reshape(-1, img_seg.shape[-1]) # (H*W,3) # color channel in rgb order
        unique_colors = self.unique(colors) # (num_class_in_img,3)
        return unique_colors

    def extract_class_from_seg(self,img_seg):
        unique_colors = self.extract_color_from_seg(img_seg) # (num_class_in_img,3) # color channel in rgb order
        classes_idx = [self.color2idx[tuple(color.tolist())]for color in unique_colors]
        classes_str = [self.idx2name[idx] for idx in classes_idx]
        return classes_idx, classes_str

    def colormap2classmap(self, seg_array):
        seg_array_flattened = torch.LongTensor(seg_array.reshape(-1,3))
        seg_map_class_flattened = torch.zeros((seg_array.shape[0],seg_array.shape[1],1)).view(-1,1)
        for color, cls in self.color2idx.items():
            matching_indices = (seg_array_flattened == torch.LongTensor(color))
            matching_indices = (matching_indices.sum(dim=1)==3)
            seg_map_class_flattened[matching_indices] = cls
        seg_map_class = seg_map_class_flattened.view(seg_array.shape[0],seg_array.shape[1],1)
        # return CPU
        seg_map_class = seg_map_class.squeeze().long()
        return seg_map_class
    
    def classmap2colormap(self,seg_map_class):
        seg_map_class_flattened = seg_map_class.view(-1,1)
        seg_map_color_flattened = torch.zeros(seg_map_class.shape[0]*seg_map_class.shape[1],3).cuda().long()
        for cls, color in self.idx2color.items():
            matching_indices = (seg_map_class_flattened == torch.LongTensor([cls]).cuda())
            seg_map_color_flattened[matching_indices.view(-1)] = torch.LongTensor(color).cuda()
        seg_map_color_flattened = seg_map_color_flattened.view(seg_map_class.shape[0],seg_map_class.shape[1],3)
        seg_map_color_flattened = seg_map_color_flattened.cpu().permute(2,0,1)
        return seg_map_color_flattened

    def split_SemAndChange(self,seg_map_class):
        seg_map_change_class = seg_map_class//50
        seg_map_semantic_class = torch.fmod(seg_map_class,50)
        return seg_map_semantic_class, seg_map_change_class


class ChangeSim(CDDataset):
    def __init__(self, ROOT='', split='test', num_classes=2, subset = None, seg=None, transforms=None, revert_transforms=None, input_size=None,exp_name = 'exp'):
        """
        ChangeSim Dataloader
        Please download ChangeSim Dataset in https://github.com/SAMMiCA/ChangeSim
        Args:
            num_classes (int): Number of target change detection class
                               5 for multi-class change detection
                               2 for binary change detection (default: 5)
            set (str): 'train' or 'test' (defalut: 'train')
        """
        super(ChangeSim, self).__init__(ROOT, transforms)
        self.dataset_name = 'ChangeSim'
        assert subset in ['normal', 'dark', 'dust']
        self.subset = subset
        self.num_classes = num_classes
        self.set = split
        train_list = ['Warehouse_0', 'Warehouse_1', 'Warehouse_2', 'Warehouse_3', 'Warehouse_4', 'Warehouse_5']
        test_list = ['Warehouse_6', 'Warehouse_7', 'Warehouse_8', 'Warehouse_9']
        self.image_total_files = []
        if split == 'train':
            ...
            # for map in train_list:
            #     self.image_total_files += sorted(glob.glob(ROOT + '/Query/Query_Seq_Train/' + map + '/Seq_0/rgb/*.png'))
            #     self.image_total_files += sorted(glob.glob(ROOT + '/Query/Query_Seq_Train/' + map + '/Seq_1/rgb/*.png'))
        elif split == 'test':
            for map in test_list:
                if subset == 'normal':
                    self.image_total_files += sorted(glob.glob(ROOT + '/Query/Query_Seq_Test/' + map + '/Seq_0/rgb/*.png'))
                    self.image_total_files += sorted(glob.glob(ROOT + '/Query/Query_Seq_Test/' + map + '/Seq_1/rgb/*.png'))
                else:
                    self.image_total_files += sorted(glob.glob(ROOT + '/Query/Query_Seq_Test/' + map + '/Seq_0_%s/rgb/*.png'%subset))
                    self.image_total_files += sorted(glob.glob(ROOT + '/Query/Query_Seq_Test/' + map + '/Seq_1_%s/rgb/*.png'%subset))

        if split == 'train':
            self.query_image_path = ROOT + '/Query/Query_Seq_Train/'
        elif split == 'test':
            self.query_image_path = ROOT + '/Query/Query_Seq_Test/'
            
        self.mask_dir = os.path.join(self.root, 'changesim_mask_%s'%subset)

        self.seg = seg
        self._transforms = transforms
        self._revert_transforms = revert_transforms
 
        size_dict = {
            256: (256, 256),
            480: None,
            }

        self.input_size = size_dict[input_size]        

        # Masktrack
        self.masktrack_dir = path.join(self.mask_dir, 'track_' + exp_name)

        # prediciton
        self.pred_dir = './output/%s/'%exp_name

    def __len__(self):
        return len(self.image_total_files)

    def get_raw(self, index):
        test_rgb_path = self.image_total_files[index]
        ref_rgb_path = test_rgb_path.replace('rgb', 't0/rgb')
        change_segmentation_path = test_rgb_path.replace('rgb', 'change_segmentation')

        img_t0 = self._pil_loader(ref_rgb_path)
        img_t1 = self._pil_loader(test_rgb_path)

        mask = self._pil_loader(change_segmentation_path)
        if self.num_classes == 2:
            mask = mask.convert("L")
        # return imgs, mask, ref_rgb_path, test_rgb_path
                    
        if self.input_size is not None:
            mask = mask.resize(self.input_size, resample=Image.NEAREST)
            img_t0 = img_t0.resize(self.input_size, resample=Image.BICUBIC)
            img_t1 = img_t1.resize(self.input_size, resample=Image.BICUBIC)
        return img_t0, img_t1, mask, ref_rgb_path, test_rgb_path

    def get_mask_ratio(self):
        if self.num_classes == 2:
            return [0.0846, 0.9154]

    def get_pil(self, imgs, mask, pred=None):
        assert self._revert_transforms is not None
        t0, t1 = self._revert_transforms(imgs.cpu())
        w, h = t0.size
        output = Image.new('RGB', (w * 2, h * 2))
        output.paste(t0)
        output.paste(t1, (w, 0))
        if self.num_classes == 5:
            mask = self.seg.classmap2colormap(mask.cuda())
            pred = self.seg.classmap2colormap(pred.cuda())

        mask = Image.fromarray(np.array(mask.permute(1,2,0)).astype(np.uint8))
        pred = Image.fromarray(np.array(pred.permute(1,2,0)).astype(np.uint8))
        output.paste(mask, (0, h))
        output.paste(pred, (w, h))
        return output

    def file_name_comp(self, frame): return frame + '.png'
    def get_frameidx(self, path): return int(os.path.basename(path)[:-4])
    def pth2gtpth(self, path): return path.replace('/rgb/', '/change_segmentation/')
    def pth2predpth(self, path): return path.replace(self.query_image_path, self.pred_dir).replace('/rgb/', '/')


    def load_label(self, gt_pth):
        # load label function for ChangeSim dataset
        change_label_mapping = np.asarray(Image.open(gt_pth)).copy()
        if self.input_size is not None:
            change_label_mapping = cv2.resize(change_label_mapping,dsize=self.input_size, interpolation=cv2.INTER_NEAREST)
        change_mapping = self.seg.colormap2classmap(change_label_mapping)
        label = change_mapping.permute(0,1).long()
        return label

    def get_subset_name(self, path, concat=True):
        _mi = path.rindex('Warehouse_')
        warehousename = path[_mi:_mi+11]
        _si = path.rindex('Seq_')
        if path[_si+5] == '/':
            seqname = path[_si:_si+5] 
        else:
            seqname = path[_si:_si+10]
        return os.path.join(warehousename, seqname) if concat else [warehousename, seqname]

    def preprocess_sequence_frames(self, clip_len):
        self.clip_len = clip_len
        self.all_data = {}
        cnt = 0

        for frame in self.image_total_files:
            warehousename, seqname = self.get_subset_name(frame, concat=False)
            if not warehousename in self.all_data : self.all_data[warehousename] = {}  
            if not seqname in self.all_data[warehousename] : self.all_data[warehousename][seqname] = 0 
            self.all_data[warehousename][seqname] += 1

        for warehousename in self.all_data: 
            for seqname in self.all_data[warehousename]:
                seq_len = self.all_data[warehousename][seqname]
                start = 0
                while start < seq_len:
                    start += clip_len
                    cnt += 1
        return cnt

    def get_datasets_clip(self, cross):
        assert all([i in ['q', 'r'] for i in cross]), '"Cross" must consist of only q (image from query seq) and r (image from reference seq).'
        is_query = list(map(lambda x: 1 if x=='q' else 0, cross)) # 'qr' to 10
        
        for warehousename in self.all_data:
            for seqname in self.all_data[warehousename]:
                seq_len = self.all_data[warehousename][seqname]
                start = 0
                while start < seq_len:
                    tc = self.clip_len if start + self.clip_len < seq_len else seq_len - start

                    # print("processing:", path.join(warehousename, seqname, cross, str(start)), 
                    #         "\n  data from: frame",start, "and clip length:", tc)
                    yield DetectionVideoReader(
                        vid_name = path.join(warehousename, seqname, cross, str(start)),
                        ref_dir=path.join(self.query_image_path, warehousename, seqname, 't0/rgb/'), 
                        query_dir=path.join(self.query_image_path, warehousename, seqname, 'rgb/'),
                        mask_dir=path.join(self.mask_dir, warehousename, seqname),
                        start = start, #'0.png',
                        clip_len=tc,
                        is_query=is_query * tc, # [1,0] to [1,0,1,0, ... , 1,0]
                        size=self.input_size,
                        file_name_comp = self.file_name_comp 
                    )

                    start += self.clip_len 

def get_ChangeSim(args, train=True, num_class=2):
    input_size = args.input_size
    raw_root = Data_path.get_dataset_path('ChangeSim')
    size_dict = {
        256: (256, 256),
        480: None,
    }
    assert input_size in size_dict, "input_size: {}".format(size_dict.keys())
    input_size = size_dict[input_size]
    mode = 'train' if train else 'test'

    if num_class == 2 or num_class == 5:
        seg_class_num = 5
    else:
        seg_class_num = 32
    seg = SegHelper(idx2color_path=path.join('./dataset/idx2color.txt'), num_class=seg_class_num)

    mean=(0.485, 0.456, 0.406)
    std=(0.229, 0.224, 0.225)

    print("{} Aug:".format(mode))
    augs = []
    if input_size is not None:
        augs.append(T.Resize(input_size))

    if train:
        augs.append(T.RandomHorizontalFlip(args.randomflip))
        #augs.append(T.ColorJitter(0.4, 0.4, 0.4, 0.25))

    if num_class == 2:
        augs.append(T.ToTensor())
        augs.append(T.Normalize(mean=mean, std=std))
        augs.append(T.ConcatImages())
    elif num_class == 5:
        augs.append(T.PILToTensor(seg.colormap2classmap))
        augs.append(T.Normalize(mean=mean, std=std))
        augs.append(T.ConcatImages())
    else:
        augs.append(T.PILToTensor(seg.colormap2classmap))
        augs.append(T.Normalize(mean=mean, std=std))

    transforms = T.Compose(augs)

    if num_class == 2 or num_class == 5:
        revert_transforms = T.Compose([
            T.SplitImages(),
            T.RevertNormalize(mean=mean, std=std),
            T.ToPILImage()
        ])
    else:
        revert_transforms = T.Compose([
            T.RevertNormalize(mean=mean, std=std),
            T.ToPILImage()
        ])
    return raw_root, mode, seg, transforms, revert_transforms

# def get_ChangeSim_Binary(args, train=True,exp_name='experiments'):
#     raw_root, mode, seg, transforms, revert_transforms = get_ChangeSim(args, train, 2)
#     dataset = ChangeSim(raw_root, mode, num_classes=2, seg=seg,
#         transforms=transforms, revert_transforms=revert_transforms, exp_name=exp_name)
#     print("ChangeSim Binary {}: {}".format(mode, len(dataset)))
#     return dataset

def get_ChangeSim_Multi(args, train=True, exp_name='experiments'):
    raw_root, mode, seg, transforms, revert_transforms = get_ChangeSim(args, train, 5)
    print('subset:', args.changesim_subset)
    dataset = ChangeSim(raw_root, mode, num_classes=5, seg=seg, 
                        input_size=args.input_size, subset=args.changesim_subset,  
                        transforms=transforms, revert_transforms=revert_transforms, exp_name=exp_name)
    print("ChangeSim Multi {}: {}".format(mode, len(dataset)))
    return dataset
