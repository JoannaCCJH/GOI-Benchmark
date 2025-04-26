import torch
import os
from tqdm import tqdm
from os import makedirs
import torchvision
from argparse import ArgumentParser
from torch.nn.functional import softmax
from sklearn.decomposition import PCA
import numpy as np
import json
import glob
import random
import colormaps
from collections import defaultdict
from utils import smooth, colormap_saving, vis_mask_save, polygon_to_mask, stack_mask, show_result
import clip
from pathlib import Path
import cv2
from metric import calculate_iou, calculate_mean_pixel_accuracy
from clipModel import OpenClipModel
    

def compute_similarity(normed_feature, out_bg_mask=None, LUT=None, MLP=None, vlm=None, text=None):
   
    vlm.encode_texts(text)
    sim = vlm.compute_relevancy_map(normed_feature)

    return sim

def eval_gt_scannetpp(gt_npy_folder=None, feat_dir=None, ouput_path=None, label_name_list=None, suffix=''):
    """
    organise lerf's gt annotations
    gt format:
        file name: frame_xxxxx.json
        file content: labelme format
    return:
        gt_ann: dict()
            keys: str(int(idx))
            values: dict()
                keys: str(label)
                values: dict() which contain 'bboxes' and 'mask'
    """
    to_eval_npy_paths = sorted(os.listdir(feat_dir)) # choose one level to get name
    to_eval_npy_paths = [path_j for path_j in to_eval_npy_paths if path_j.endswith('.npy')]
    img_names = [path_j.split('.npy')[0] for path_j in to_eval_npy_paths]
    
    base_folder = os.path.basename(gt_npy_folder)
    render_img_root = gt_npy_folder.replace(base_folder, 'undistorted_images')
    img_paths = [os.path.join(render_img_root, f'{img_name}.JPG') for img_name in img_names]

    gt_ann = {}
    for idx, to_eval_npy_path in enumerate(tqdm(to_eval_npy_paths)):
        img_ann = defaultdict(dict)
        gt_ann_i = os.path.join(gt_npy_folder, to_eval_npy_path)
        gt_ann_i = np.load(gt_ann_i)
        h, w = gt_ann_i.shape 
        labels = np.unique(gt_ann_i)
        for label in labels:
            if label == -1:
                continue
            label_name = label_name_list[label]
            mask = (gt_ann_i == label).astype(np.uint8)
            if img_ann.get(label_name, None) is not None:
                mask = stack_mask(img_ann[label_name]['mask'], mask)
                img_ann[label_name]['bboxes'] = np.concatenate(
                    [img_ann[label_name]['bboxes'].reshape(-1, 4), np.array([0, 0, w, h]).reshape(-1, 4)], axis=0)
            else:
                img_ann[label_name] = {}
                img_ann[label_name]['bboxes'] = np.array([0, 0, w, h])
                img_ann[label_name]['mask'] = mask
            # save for visulsization
            save_path = Path(ouput_path) / 'gt' / to_eval_npy_path.split('.npy')[0] / f'{label_name}.jpg'
            save_path.parent.mkdir(exist_ok=True, parents=True)
            vis_mask_save(mask, save_path)
        gt_ann[f'{idx}'] = img_ann
    
    return gt_ann, (h, w), img_paths

def evaluate(label_name_path, iteration, gt_npy_dir, logit_npy_dir, output_dir, lut_model_path):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    colormap_options = colormaps.ColormapOptions(
        colormap="turbo",
        normalize=True,
        colormap_min=-1.0,
        colormap_max=1.0,
    )
    
    label_name_txt = np.loadtxt(
        label_name_path,
        dtype=str,
        delimiter=".",  # dummy delimiter to replace " "
    )
    
    gt_ann, image_shape, image_paths = eval_gt_scannetpp(gt_npy_dir, logit_npy_dir, output_dir, label_name_txt)
    
    eval_index_list = [int(idx) for idx in list(gt_ann.keys())]
    feat_paths_lvl = sorted(glob.glob(os.path.join(logit_npy_dir, '*.npy')))
    
    # instantiate autoencoder and openclip
    clip_model = OpenClipModel()
    LUT = torch.load(lut_model_path)
    
    ignore_classes = ["wall", "floor", "ceiling"]
    mIoU_list = []
    fg_mIoU_list = []
    class_IoUs = {} 
    
    print(f"Evaluting {len(eval_index_list)} images...")
    
    for j, idx in enumerate(eval_index_list):
        sem_logits = np.load(feat_paths_lvl[idx])
        sem_feature = LUT[sem_logits]
        normed_feature = sem_feature / sem_feature.norm(dim=-1, keepdim=True)
        
        rgb_img = cv2.imread(image_paths[idx])[..., ::-1]
        rgb_img = (rgb_img / 255.0).astype(np.float32)
        rgb_img = torch.from_numpy(rgb_img)
        
        img_ann = gt_ann[str(idx)]
        labels = img_ann.keys()
        relevancy_map = compute_similarity(normed_feature, vlm=clip_model, text=labels)
        relevancy_map = relevancy_map.cpu().detach().reshape(image_shape[0], image_shape[1], -1)
        for i, label in enumerate(img_ann.keys()):
            sim_map = relevancy_map[..., i:i+1]
            
            # heatmap_save_path = Path(output_dir) / 'heatmaps' / os.path.basename(image_paths[idx]).split('.')[0] / f'{label}_heatmap.jpg'
            # heatmap_save_path.parent.mkdir(exist_ok=True, parents=True)
            # colormap_saving(sim_map, colormap_options, heatmap_save_path, original_image=rgb_img, alpha=0.5)
            
            
            pred_mask = sim_map > 0.5
            pred_mask = pred_mask.cpu()
            pred_mask = pred_mask.reshape(image_shape[0], image_shape[1], -1)
            
            # save_path = Path(output_dir) / 'pred' / os.path.basename(image_paths[idx]).split('.')[0] / f'{label}.jpg'
            # save_path.parent.mkdir(exist_ok=True, parents=True)
            # vis_mask_save(pred_mask.squeeze(-1).numpy().astype(np.uint8) * 255, save_path)
            
            gt_mask = torch.from_numpy(img_ann[label]['mask']).unsqueeze(-1) # 
            
            iou = calculate_iou(gt_mask, pred_mask)
            mIoU_list.append(iou)
            if label not in ignore_classes: # foreground objects
                fg_mIoU_list.append(iou)
            # Update class-specific IoUs - corrected check
            if label not in class_IoUs.keys():
                class_IoUs[label] = []
            class_IoUs[label].append(iou)
        
        
    # Calculate the mean IoU values
    mIoU = sum(mIoU_list) / len(mIoU_list) if mIoU_list else 0
    fg_mIoU = sum(fg_mIoU_list) / len(fg_mIoU_list) if fg_mIoU_list else 0
    
    # Print or return the results
    print(f"Mean IoU (mIoU): {mIoU:.4f}")
    print(f"Foreground Mean IoU (fg_mIoU): {fg_mIoU:.4f}")
    
    # Create a dictionary to store the metrics
    metrics = {
        "mIoU": float(mIoU),
        "fg_mIoU": float(fg_mIoU),
        "class_mIoU": {label: float(sum(ious) / len(ious)) for label, ious in class_IoUs.items()},
        "class_counts": {label: len(ious) for label, ious in class_IoUs.items()}
    }

    
    # Save metrics as JSON
    json_path = Path(output_dir) / "segmentation_metrics.json"
    with open(json_path, "w") as f:
        json.dump(metrics, f, indent=4)

    print(f"Metrics saved to {json_path}")

def seed_everything(seed_value):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    
    if torch.cuda.is_available(): 
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True
        
if __name__ == "__main__":
    seed_num = 42
    seed_everything(seed_num)
    
    parser = ArgumentParser(description="prompt any label")
    parser.add_argument("--label_name_path", type=str, default=None)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--gt_npy_dir", type=str, default=None)
    parser.add_argument("--logit_npy_dir", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--lut_model_path", type=str, default=None)
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    evaluate(args.label_name_path, args.iteration, args.gt_npy_dir, args.logit_npy_dir, args.output_dir, args.lut_model_path)
    
    