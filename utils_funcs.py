from builtins import breakpoint
import cv2
import os
import numpy as np
import torch
from tqdm import tqdm
from PIL import Image
import eval_utils
import torch.nn.functional as F

changesim_multi_palette = [0, 0, 0, 
81, 38, 0, # New
41, 36, 132, # Miss
25, 48, 16,  # Rot
131, 192, 13, # Rep
]

def load_keyframes(qr_dir, rq_dir, tframe, cross_or_stright):
    qr_frame_pth = os.path.join(qr_dir, ('Q' if cross_or_stright=='straight' else 'R') \
                                                        + os.path.basename(tframe)[:-4] + '.npy')
    rq_frame_pth = os.path.join(rq_dir, ('R' if cross_or_stright=='straight' else 'Q') \
                                                        + os.path.basename(tframe)[:-4] + '.npy')
    qr_frame = np.load(qr_frame_pth)
    rq_frame = np.load(rq_frame_pth)
    return qr_frame, rq_frame

def compute_miou(data_loader, mode='multi'):
    assert mode in ['multi', 'binary']
    num_classes = data_loader.dataset.num_classes if mode =='multi' else 2
    # num_classes = 8 if mode =='multi' else 2
    confmat = eval_utils.ConfusionMatrix(num_classes-1)
    metric_logger = eval_utils.MetricLogger(delimiter="  ")
    count = 0
    for keyframe in data_loader.dataset.image_total_files:     
        
        gt_pth = data_loader.dataset.pth2gtpth(keyframe)
        pred_pth = data_loader.dataset.pth2predpth(keyframe)
        
        target = data_loader.dataset.load_label(gt_pth)

        target[target==3] = 0
        target[target==4] = 3

        pred = np.array(Image.open(pred_pth))
        pred[pred==3] = 0
        pred[pred==4] = 3

        if mode == 'binary':
            target = target>0
            pred = (pred>0).astype(np.uint8)
        
        confmat.update(target.flatten(), pred.flatten())

        confmat.reduce_from_all_processes()
    print("{} confmat: {}".format(
        data_loader.dataset.name,
        confmat
    ))
    acc_global, acc, iu = confmat.compute()
    mIoU = confmat.mIoU(iu)

    return mIoU


    
def compute_f1(data_loader):
    metric_logger = eval_utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('Prec', eval_utils.SmoothedValue(window_size=1, fmt='{value:.3f} ({global_avg:.3f})'))
    metric_logger.add_meter('Rec', eval_utils.SmoothedValue(window_size=1, fmt='{value:.3f} ({global_avg:.3f})'))
    metric_logger.add_meter('Acc', eval_utils.SmoothedValue(window_size=1, fmt='{value:.3f} ({global_avg:.3f})'))
    metric_logger.add_meter('F1score', eval_utils.SmoothedValue(window_size=1, fmt='{value:.4f} ({global_avg:.4f})'))
    count = 0
    
    for keyframe in data_loader.dataset.image_total_files:     
        gt_pth = data_loader.dataset.pth2gtpth(keyframe)
        pred_pth = data_loader.dataset.pth2predpth(keyframe)

        target = data_loader.dataset.load_label(gt_pth)
        target = F.interpolate(target.unsqueeze(0).unsqueeze(0).float(), size=data_loader.dataset.input_size[::-1], mode='nearest').squeeze(0).squeeze(0).long()
        pred = np.array(Image.open(pred_pth))

        mask_gt = (target > 0).unsqueeze(0)#[:, 0]
        mask_pred = (torch.Tensor(pred) > 0 ).unsqueeze(0) #[:, 0]

        precision, recall, accuracy, f1score = eval_utils.CD_metric_torch(mask_pred, mask_gt)
        metric_logger.Prec.update(precision.mean(), n=len(precision))
        metric_logger.Rec.update(recall.mean(), n=len(precision))
        metric_logger.Acc.update(accuracy.mean(), n=len(precision))
        metric_logger.F1score.update(f1score.mean(), n=len(f1score))

    metric_logger.synchronize_between_processes()

    print("Total: {} Metric Prec: {:.4f} Recall: {:.4f} F1: {:.4f}".format(
        metric_logger.F1score.count,
        metric_logger.Prec.global_avg,
        metric_logger.Rec.global_avg,
        metric_logger.F1score.global_avg
    ))
    return metric_logger.F1score.global_avg


def aggregate_clip_prediction(data_loader, 
                                       cross = ['qr', 'rq'], 
                                       clip_len=20, 
                                       args=None):

    from collections import defaultdict
    qr_ref_all_labeldict = defaultdict(lambda: {}) 
    rq_query_all_labeldict = defaultdict(lambda: {}) 

    for keyframe in tqdm(data_loader.dataset.image_total_files, desc='aggregating predicition...'):   
        subsetname = data_loader.dataset.get_subset_name(keyframe)         
        folder_idx = data_loader.dataset.get_frameidx(keyframe) // clip_len * clip_len

        qr_dir = os.path.join(data_loader.dataset.masktrack_dir, subsetname, cross[0], str(folder_idx))
        rq_dir = os.path.join(data_loader.dataset.masktrack_dir, subsetname, cross[1], str(folder_idx))

        qr_ref_frame, rq_query_frame = load_keyframes(qr_dir, rq_dir, keyframe, 'cross')   

        # Accumulate
        if not folder_idx in qr_ref_all_labeldict[subsetname]: qr_ref_all_labeldict[subsetname][folder_idx] = defaultdict(lambda:0) # init

        label_ids, counts = np.unique(qr_ref_frame, return_counts=True)
        for label_id, count in zip(label_ids, counts):     
            # compare & update       
            qr_ref_all_labeldict[subsetname][folder_idx][label_id] = max(qr_ref_all_labeldict[subsetname][folder_idx][label_id], count)
        qr_ref_all_labeldict[subsetname][folder_idx]['count'] += 1

        if not folder_idx in rq_query_all_labeldict[subsetname]: rq_query_all_labeldict[subsetname][folder_idx] = defaultdict(lambda:0) # init

        label_ids, counts = np.unique(rq_query_frame, return_counts=True)
        for label_id, count in zip(label_ids, counts):   
            rq_query_all_labeldict[subsetname][folder_idx][label_id] = max(rq_query_all_labeldict[subsetname][folder_idx][label_id], count)
        rq_query_all_labeldict[subsetname][folder_idx]['count'] += 1

    for keyframe in tqdm(data_loader.dataset.image_total_files, desc='comparing predicition...'):  
        subsetname = data_loader.dataset.get_subset_name(keyframe) 
        
        folder_idx = data_loader.dataset.get_frameidx(keyframe) // clip_len * clip_len

        qr_dir = os.path.join(data_loader.dataset.masktrack_dir, subsetname, cross[0], str(folder_idx))
        rq_dir = os.path.join(data_loader.dataset.masktrack_dir, subsetname, cross[1], str(folder_idx))
        qr_query_frame, rq_ref_frame = load_keyframes(qr_dir, rq_dir, keyframe, 'straight')
        
        qr_query_only = qr_query_frame.copy().astype(int)            
        label_ids, counts = np.unique(qr_query_frame, return_counts=True)
        for label_id, count in zip(label_ids, counts):   
            count_in_ref = qr_ref_all_labeldict[subsetname][folder_idx][label_id]  
            mask_threshold = get_adaptive_maskth(qr_ref_all_labeldict[subsetname][folder_idx]['count'])

            if count_in_ref / count > mask_threshold: # the instance is static if the object is large  
                qr_query_only[qr_query_only == label_id] = -1
               
        rq_ref_only = rq_ref_frame.copy().astype(int) 
        label_ids, counts = np.unique(rq_ref_frame, return_counts=True)
        for label_id, count in zip(label_ids, counts):   
            count_in_query = rq_query_all_labeldict[subsetname][folder_idx][label_id]  
            mask_threshold = get_adaptive_maskth(rq_query_all_labeldict[subsetname][folder_idx]['count'])

            if count_in_query / count > mask_threshold: # the instance is static if the object is large  
                rq_ref_only[rq_ref_only == label_id] = -1

        # Prediction: New + Missing         
        pred = np.zeros_like(qr_query_only).astype(np.uint8)

        if data_loader.dataset.dataset_name == 'VL_CMU_CD':  
            pred += (rq_ref_only>0).astype(np.uint8) * 2 # There is no 'New' class VL_CMU_CD (Only Missing)
        else:   
            pred += (qr_query_only>0).astype(np.uint8) * 1 # New 
            pred += (rq_ref_only>0).astype(np.uint8) * 2 # Missing
            pred[pred==3] = 4 # Replaced

        predsave = Image.fromarray(pred)
        predsave.putpalette(changesim_multi_palette)
        os.makedirs(os.path.join(data_loader.dataset.pred_dir, subsetname), exist_ok=True)
        if data_loader.dataset.dataset_name == 'ChangeSim':
            predsave.save(os.path.join(
                data_loader.dataset.pred_dir, 
                subsetname, 
                data_loader.dataset.file_name_comp(str(data_loader.dataset.get_frameidx(keyframe))))
            )

        elif data_loader.dataset.dataset_name == 'VL_CMU_CD':           
            predsave.save(os.path.join(
                data_loader.dataset.pred_dir, 
                subsetname, 
                data_loader.dataset.file_name_comp(os.path.basename(keyframe)[:5])(str(data_loader.dataset.get_frameidx(keyframe))))
            )
        elif (data_loader.dataset.dataset_name == 'GSV') or (data_loader.dataset.dataset_name == 'TSUNAMI'):           
            predsave.save(os.path.join(
                data_loader.dataset.pred_dir, 
                subsetname, 
                data_loader.dataset.file_name_comp(os.path.basename(keyframe)[:5])(str(data_loader.dataset.get_frameidx(keyframe))).replace('.jpg', '.png')) 
            )

def get_adaptive_maskth(length):
    return 0.5 - 0.9 / (np.sqrt(length) + 1)
