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
from metric import calculate_iou, calculate_mean_pixel_accuracy, calculate_mean_precision
from clipModel import OpenClipModel

def compute_similarity(normed_feature, out_bg_mask=None, LUT=None, MLP=None, vlm=None, text=None):
   
    # if self.res_finetuned:
    #     # 使用res finetune的MLP作为分割效果
    #     logit = self.resMLP(normed_feature.cuda()).squeeze()
    #     # print(logit.max(), logit.min())
    #     sim = (logit).sigmoid().squeeze(-1)
    #     thresh = 0.5
    # else:
    vlm.encode_texts(text)
    sim = vlm.compute_relevancy_map(normed_feature)
    # thresh = 0.86
    # _bg_mask = sim < thresh
    # if out_bg_mask is not None:
    #     out_bg_mask[:] = _bg_mask
    # sim[_bg_mask] = 0
    return sim

def eval_gt_lerfdata(json_folder= None, feat_dir=None, ouput_path=None):
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
    gt_json_paths = sorted(glob.glob(os.path.join(str(json_folder), 'frame_*.json')))
    img_paths = sorted(glob.glob(os.path.join(str(json_folder), 'frame_*.jpg')))
    # print(f"gt_json_paths: {gt_json_paths}")
    # gt_json_paths: ['/scratch/joanna_cheng/lerf_ovs/label/ramen/frame_00006.json', '/scratch/joanna_cheng/lerf_ovs/label/ramen/frame_00024.json', ...]
    # print(f"img_paths: {img_paths}")
    # img_paths: ['/scratch/joanna_cheng/lerf_ovs/label/ramen/frame_00006.jpg',...]
    gt_ann = {}
    for idx, js_path in enumerate(gt_json_paths):
        img_ann = defaultdict(dict)
        with open(js_path, 'r') as f:
            gt_data = json.load(f)
        
        h, w = gt_data['info']['height'], gt_data['info']['width']
        # idx = int(gt_data['info']['name'].split('_')[-1].split('.jpg')[0]) - 1 
        for prompt_data in gt_data["objects"]:
            label = prompt_data['category']
            box = np.asarray(prompt_data['bbox']).reshape(-1)           # x1y1x2y2
            mask = polygon_to_mask((h, w), prompt_data['segmentation'])
            if img_ann[label].get('mask', None) is not None:
                mask = stack_mask(img_ann[label]['mask'], mask)
                img_ann[label]['bboxes'] = np.concatenate(
                    [img_ann[label]['bboxes'].reshape(-1, 4), box.reshape(-1, 4)], axis=0)
            else:
                img_ann[label]['bboxes'] = box
            img_ann[label]['mask'] = mask
            
            # # save for visulsization
            save_path = ouput_path / 'gt' / gt_data['info']['name'].split('.jpg')[0] / f'{label}.jpg'
            save_path.parent.mkdir(exist_ok=True, parents=True)
            vis_mask_save(mask, save_path)
        gt_ann[f'{idx}'] = img_ann

    return gt_ann, (h, w), img_paths

def evaluate(iteration, json_dir, logit_npy_dir, output_dir, lut_model_path):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    colormap_options = colormaps.ColormapOptions(
        colormap="turbo",
        normalize=True,
        colormap_min=-1.0,
        colormap_max=1.0,
    )
    
    gt_ann, image_shape, image_paths = eval_gt_lerfdata(Path(json_dir), logit_npy_dir, Path(output_dir))
    
    eval_index_list = [int(idx) for idx in list(gt_ann.keys())]
    feat_paths_lvl = sorted(glob.glob(os.path.join(logit_npy_dir, '*.npy')))
    
    # instantiate autoencoder and openclip
    clip_model = OpenClipModel()
    LUT = torch.load(lut_model_path)
    
    print(f"Evaluting {len(eval_index_list)} images...")
    
    for j, idx in enumerate(eval_index_list):
        sem_logits = np.load(feat_paths_lvl[idx])
        sem_feature = LUT[sem_logits]
        normed_feature = sem_feature / sem_feature.norm(dim=-1, keepdim=True)
        print(feat_paths_lvl[idx])
        
        rgb_img = cv2.imread(image_paths[idx])[..., ::-1]
        rgb_img = (rgb_img / 255.0).astype(np.float32)
        rgb_img = torch.from_numpy(rgb_img)
        
        img_ann = gt_ann[str(idx)]
        labels = img_ann.keys()
        relevancy_map = compute_similarity(normed_feature, vlm=clip_model, text=labels)
        relevancy_map = relevancy_map.cpu().detach().reshape(image_shape[0], image_shape[1], -1)
        for i, label in enumerate(img_ann.keys()):
            sim_map = relevancy_map[..., i:i+1]
            
            heatmap_save_path = Path(output_dir) / 'heatmaps' / os.path.basename(image_paths[idx]).split('.')[0] / f'{label}_heatmap.jpg'
            heatmap_save_path.parent.mkdir(exist_ok=True, parents=True)
            colormap_saving(sim_map, colormap_options, heatmap_save_path, original_image=rgb_img, alpha=0.5)
            
            
            pred_mask = sim_map > 0.5
            pred_mask = pred_mask.cpu()
            pred_mask = pred_mask.reshape(image_shape[0], image_shape[1], -1)
            
            save_path = Path(output_dir) / 'pred' / os.path.basename(image_paths[idx]).split('.')[0] / f'{label}.jpg'
            save_path.parent.mkdir(exist_ok=True, parents=True)
            vis_mask_save(pred_mask.squeeze(-1).numpy().astype(np.uint8) * 255, save_path)
            
            gt_mask = torch.from_numpy(img_ann[label]['mask']).unsqueeze(-1)
            
            # iou = calculate_iou(gt_mask, pred_mask)
            # mpa = calculate_mean_pixel_accuracy(gt_mask, pred_mask)
            # mp = calculate_mean_precision(gt_mask, pred_mask)
            
            # print(f"label: {label}, iou: {iou}, mpa: {mpa}, mp: {mp}")
        
        # img_ann = gt_ann[str(idx)]
        # print(img_ann.keys())
        # print(f"Image path: {image_paths[idx]}")
        # print(img_ann['wall']['mask'].shape) # (584, 876)

        # iou = calculate_iou(gt_masks[i], pred_mask)
        # mpa = calculate_mean_pixel_accuracy(gt_masks[i], pred_mask)
        # mp = calculate_mean_precision(gt_masks[i], pred_mask)

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
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--logit_npy_dir", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--lut_model_path", type=str, default=None)
    parser.add_argument("--json_dir", type=str, default=None)
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    evaluate(args.iteration, args.json_dir, args.logit_npy_dir, args.output_dir, args.lut_model_path)
    
    