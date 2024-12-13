import os

def load_mask_model(model_name,device):
    model_list = ['sam']
    if model_name not in model_list:
        raise ValueError("Model Error")
    print("load " + model_name) 
    from model.sam.segment_anything  import SamAutomaticMaskGenerator, sam_model_registry
    cwd = os.getcwd()
    sam_path = os.path.join(cwd, 'model_weights/sam_vit_h_4b8939.pth')
    sam = sam_model_registry['vit_h'](checkpoint=sam_path)
    sam.to(device=device)
    for m in sam.parameters():
        m.requires_grad = False
    model = SamAutomaticMaskGenerator(model=sam,
        points_per_side=32,
        pred_iou_thresh=0.86,
        stability_score_thresh=0.92,
        crop_n_layers=1,
        crop_n_points_downscale_factor=2,
        min_mask_region_area=100)
    return model