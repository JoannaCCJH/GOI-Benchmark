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
from finetuneRESModel import FinetuneResModel
from networks import LinearSVM

def compute_similarity(normed_feature, out_bg_mask=None, LUT=None, MLP=None, vlm=None, text=None):
    vlm.encode_texts(text)
    sim = vlm.compute_relevancy_map(normed_feature)
    return sim

def eval_gt_scannetpp(gt_npy_folder=None, feat_dir=None, ouput_path=None, label_name_list=None):
    """
    Processes ground truth annotations from ScanNetPP dataset, extracting label masks and image paths.

    Returns:
        - Annotation dictionary with label masks
            gt_ann (Dict): 
                Structure: {
                    'image_index' (str): {
                        'label_name' (str): {
                            'mask': numpy_array (binary mask)
                        },
                        'label_name' (str): {
                            'mask': numpy_array (binary mask)
                        }
                    }
                }
        - Image shape (height, width)
        - List of image file paths
    """
    
    # Filter and sort .npy files from feature directory
    to_eval_npy_paths = sorted([path for path in os.listdir(feat_dir) if path.endswith('.npy')])
    
    # Extract image names and corresponding paths
    img_names = [path.split('.npy')[0] for path in to_eval_npy_paths]
    render_img_root = gt_npy_folder.replace(os.path.basename(gt_npy_folder), 'undistorted_images')
    img_paths = [os.path.join(render_img_root, f'{img_name}.JPG') for img_name in img_names]

    # Process ground truth annotations
    gt_ann = {}
    for idx, to_eval_npy_path in enumerate(to_eval_npy_paths):
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
            else:
                img_ann[label_name] = {}
            img_ann[label_name]['mask'] = mask
            
            # save for visulsization
            save_path = Path(ouput_path) / 'gt' / to_eval_npy_path.split('.npy')[0] / f'{label_name}.jpg'
            save_path.parent.mkdir(exist_ok=True, parents=True)
            vis_mask_save(mask, save_path)
            
        gt_ann[f'{idx}'] = img_ann
    
    return gt_ann, (h, w), img_paths

def evaluate(label_name_path, iteration, gt_npy_dir, logit_npy_dir, output_dir, lut_model_path, linearSVM_dir):
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Colormap configuration for visualization with normalization
    colormap_options = colormaps.ColormapOptions(
        colormap="turbo",
        normalize=True,
        colormap_min=0.0,
        colormap_max=1.0,
    )
    
    # Load label names from text file
    label_name_txt = np.loadtxt(
        label_name_path,
        dtype=str,
        delimiter=".",  # dummy delimiter to replace " "
    )
    
    # Load GT annotations 
    gt_ann, image_shape, image_paths = eval_gt_scannetpp(gt_npy_dir, logit_npy_dir, output_dir, label_name_txt)
    
    eval_index_list = [int(idx) for idx in list(gt_ann.keys())]
    feat_paths_lvl = sorted(glob.glob(os.path.join(logit_npy_dir, '*.npy')))
    
    # Initialize models and evaluation metrics
    clip_model = OpenClipModel() # CLIP feature extractor
    LUT = torch.load(lut_model_path) # lookup table
    resMLPs = {}

    ignore_classes = ["wall", "floor", "ceiling"]
    mIoU_list = []
    fg_mIoU_list = []
    class_IoUs = {}
    mAP_list = []
    fg_mAP_list = []
    class_mAPs = {}
    
    print(f"Evaluting {len(eval_index_list)} images...")
    
    for j, idx in enumerate(tqdm(eval_index_list)):
        # Load 2D pred feature map, convert back to 512dim using LUT
        sem_logits = np.load(feat_paths_lvl[idx])
        sem_feature = LUT[sem_logits]
        normed_feature = sem_feature / sem_feature.norm(dim=-1, keepdim=True)
        
        # Read image
        img = cv2.imread(image_paths[idx])
        rgb_img = cv2.resize(img, (image_shape[1], image_shape[0]))[..., ::-1] # (BGR)
        rgb_img = (rgb_img / 255.0).astype(np.float32)
        rgb_img = torch.from_numpy(rgb_img).permute(2, 0, 1) # (3, H, W)
        
        img_ann = gt_ann[str(idx)]
        labels = img_ann.keys()

        for i, label in enumerate(labels):
            
            with torch.no_grad():
                if resMLPs.get(label, None) is None:
                    
                    linearSVM_path = linearSVM_dir + f'/{label}_svm.pth'
                    if os.path.exists(linearSVM_path):
                        resMLPs[label] = LinearSVM.load(linearSVM_path).to(device)
                
                if resMLPs.get(label, None) is None:
                    sim_map = clip_model.compute_relevancy_map_single_text(normed_feature, label)
                    sim_map = sim_map.cpu().detach().reshape(image_shape[0], image_shape[1], -1)
                else:
                    logit = resMLPs[label](normed_feature.cuda()).squeeze()
                    sim_map = (logit).sigmoid().squeeze(-1).reshape(image_shape[0], image_shape[1], 1) # (h, w, 1)
            
            heatmap_save_path = Path(output_dir) / 'heatmaps' / os.path.basename(image_paths[idx]).split('.')[0] / f'{label}_heatmap.jpg'
            heatmap_save_path.parent.mkdir(exist_ok=True, parents=True)
            colormap_saving(sim_map, colormap_options, heatmap_save_path, original_image=img, alpha=0.5)            
            
            pred_mask = sim_map > 0.5
            pred_mask = pred_mask.cpu()
            pred_mask = pred_mask.reshape(image_shape[0], image_shape[1], -1)
            
            save_path = Path(output_dir) / 'pred' / os.path.basename(image_paths[idx]).split('.')[0] / f'{label}.jpg'
            save_path.parent.mkdir(exist_ok=True, parents=True)
            vis_mask_save(pred_mask.squeeze(-1).numpy().astype(np.uint8) * 255, save_path)
            
            gt_mask = torch.from_numpy(img_ann[label]['mask']).unsqueeze(-1) # (h, w, 1)
            
            iou = calculate_iou(gt_mask, pred_mask)
            ap = calculate_mean_pixel_accuracy(gt_mask, pred_mask)
            mIoU_list.append(iou)
            mAP_list.append(ap)
            if label not in ignore_classes: # foreground objects
                fg_mIoU_list.append(iou)
                fg_mAP_list.append(ap)
            # Update class-specific IoUs - corrected check
            if label not in class_IoUs.keys():
                class_IoUs[label] = []
                class_mAPs[label] = []
            class_IoUs[label].append(iou)
            class_mAPs[label].append(ap)
        
    # Calculate the mean IoU values
    mIoU = sum(mIoU_list) / len(mIoU_list) if mIoU_list else 0
    fg_mIoU = sum(fg_mIoU_list) / len(fg_mIoU_list) if fg_mIoU_list else 0
    mAP= sum(mAP_list) / len(mAP_list) if mAP_list else 0
    fg_mAP = sum(fg_mAP_list) / len(fg_mAP_list) if fg_mAP_list else 0
    
    # Print or return the results
    print(f"Mean IoU (mIoU): {mIoU:.4f}")
    print(f"Foreground Mean IoU (fg_mIoU): {fg_mIoU:.4f}")
    print(f"Mean AP (mAP): {mAP:.4f}")
    print(f"Foreground Mean AP (fg_AP): {fg_mAP:.4f}")
    
    # Create a dictionary to store the metrics
    metrics = {
        "mIoU": float(mIoU),
        "fg_mIoU": float(fg_mIoU),
        "mAP": float(mAP),
        "fg_mAP": float(fg_mAP),
        "class_mIoU": {label: float(sum(ious) / len(ious)) for label, ious in class_IoUs.items()},
        "class_mAP": {label: float(sum(aps) / len(aps)) for label, aps in class_mAPs.items()},
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
    parser.add_argument("--linearSVM_dir", type=str, default=None)
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    evaluate(args.label_name_path, args.iteration, args.gt_npy_dir, args.logit_npy_dir, args.output_dir, args.lut_model_path, args.linearSVM_dir)
    
    