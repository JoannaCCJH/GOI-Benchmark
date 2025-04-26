from res_model import RES_MODEL
import torch
import cv2
import PIL
from PIL import Image, ImageDraw, ImageFont, ImageOps
from torchvision.utils import save_image
import sys, os
from argparse import ArgumentParser

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

from networks import LinearSVM
import numpy as np
from collections import defaultdict
from tqdm import tqdm
import random
from clipModel import OpenClipModel

def collect_label_image_paths(gt_npy_folder=None, feat_dir=None, label_name_list=None):
    """
    Collects image paths for each unique label in ground truth annotations.

    Args:
        gt_npy_folder (str): Directory containing ground truth numpy files
        feat_dir (str): Directory containing feature files
        label_name_list (list): List of label names

    Returns:
        label_to_img_paths (Dict[str, List[str]]): Mapping of labels to their image paths
        image_shape (Tuple[int, int]): Height and width of images
    """
    
    # Filter and sort .npy files from feature directory
    to_eval_npy_paths = sorted([path for path in os.listdir(feat_dir) if path.endswith('.npy')])
    
    # Extract image names and corresponding paths
    img_names = [path.split('.npy')[0] for path in to_eval_npy_paths]
    render_img_root = gt_npy_folder.replace(os.path.basename(gt_npy_folder), 'undistorted_images')
    img_paths = [os.path.join(render_img_root, f'{img_name}.JPG') for img_name in img_names]
    
    # Create a dictionary to track image paths for each unique label
    label_to_img_paths = defaultdict(list)

    for idx, to_eval_npy_path in enumerate(to_eval_npy_paths):

        gt_ann_i = os.path.join(gt_npy_folder, to_eval_npy_path)
        gt_ann_i = np.load(gt_ann_i)
        h, w = gt_ann_i.shape 
        labels = np.unique(gt_ann_i)
        for label in labels:
            if label == -1:
                continue
            label_name = label_name_list[label]
            
            # Add image path to the list for this label
            label_to_img_paths[label_name].append(img_paths[idx])
            
    
    return label_to_img_paths, (h, w)

def get_res_mask(res_prompt, input_image, guidance_res):
    """
    Generate a response mask based on a text prompt and an input image.
    
    The input image should be read using cv2.imread() and properly resized before calling.
    
    input_image: (3, H, W)
    """
    
    res_mask, pred_image = guidance_res.predict_res_mask(input_image.cpu(), res_prompt)
    
    if pred_image is None:
        return None
    
    return res_mask.permute(1,2,0) # torch.Size([H, W, 1])
    
def finetune_prompt_with_res(normed_feature, res_mask, prompt, linearSVM, save_path, img_name):
    
    h, w, _ = res_mask.shape
    gt = res_mask.reshape(-1, 1)
    epoch = 0
    max_epoch = 6000
    target_iou_thresh = 0.9
    iou = 0
    
    # Make a detached copy of normed_feature
    normed_feature = normed_feature.cuda().detach()
    gt = gt.detach()

    while (epoch < max_epoch and iou < target_iou_thresh):
        
        x = normed_feature.clone()  # This creates a new tensor
        y = gt.clone()  # This creates a new tensor
        loss, iou = linearSVM.step(x, y)

        if epoch % 500 == 0:
            print("epoch:", epoch, "loss:", loss.item(), "iou:", iou)

        epoch += 1
        
    with torch.no_grad():
        logit = (linearSVM(normed_feature.cuda())).squeeze()
        logit = (logit).sigmoid().squeeze(-1)
        print(logit.min(), logit.max())
        
        clip_thresh = 0.5
        logit_mask = logit > clip_thresh
        logit_mask = logit_mask.cpu().numpy()

        logit_mask = logit_mask.reshape(h, w)
        gt_mask = gt.bool().squeeze(-1).cpu().numpy().reshape(h, w)
        
        logit_mask_uint8 = logit_mask.astype(np.uint8) * 255
        gt_mask_uint8 = gt_mask.astype(np.uint8) * 255
                    
    mask_image = Image.fromarray(logit_mask_uint8)
    gt_mask_image = Image.fromarray(gt_mask_uint8)
    
    # mask_out_expand = res_mask.expand_as(input_image)
    # viz_images = torch.cat([pred_image.unsqueeze(0) ,mask_out_expand.unsqueeze(0)],dim=0)                
    # save_image(viz_images, "masktest.png")

    img_name = img_name.split('.')[0]
    mask_image.save(f'{save_path}/Finetune_mask_image-epoch'+str(epoch)+'-'+prompt+'-'+img_name+'-'+f"{iou:.2f}"+'.png')
    gt_mask_image.save(f'{save_path}/Finetune_gt_mask_image-epoch'+str(epoch)+'-'+prompt+'-'+img_name+'-'+f"{iou:.2f}"+'.png')
    
    return linearSVM
        
        

if __name__ == "__main__":
    
    # Argument parser
    parser = ArgumentParser(description="prompt any label")
    parser.add_argument("--scene", type=str, default=None)
    parser.add_argument("--gt_npy_dir", type=str, default=None)
    parser.add_argument("--logit_npy_dir", type=str, default=None)
    parser.add_argument("--model_output_dir", type=str, default=None)
    parser.add_argument("--label_name_path", type=str, default=None)
    parser.add_argument("--lut_model_path", type=str, default=None)
    parser.add_argument("--feature_dim", type=int, default=512)
    args = parser.parse_args()
    os.makedirs(args.model_output_dir, exist_ok=True)
    
    # Load label names from text file
    label_name_txt = np.loadtxt(
        args.label_name_path,
        dtype=str,
        delimiter=".",  # dummy delimiter to replace " "
    )
    # Collects image paths for each unique label
    label_to_img_paths, image_shape = collect_label_image_paths(args.gt_npy_dir, args.logit_npy_dir, label_name_txt)
    
    # Define LinearSVM params
    clip_feature_thresh = 0.8
    feature_dim = args.feature_dim
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    clip_model = OpenClipModel() # CLIP feature extractor
    guidance_res = RES_MODEL(device)
    LUT = torch.load(args.lut_model_path) # lookup table
    
    print(f"Training {len(label_to_img_paths.keys())} linear SVMs...")
    
    # For each label, randomly select an image
    for idx, label in enumerate(tqdm(label_to_img_paths.keys())):
        print(f"Label {label}...")
        
        img_paths = label_to_img_paths[label]
        # Randomly select an image path for the current label
        while img_paths:
            # Randomly select an image path for the current label
            random.seed(42) 
            selected_img_path = random.choice(img_paths)
            
            # Read the selected image
            image = cv2.imread(selected_img_path)
            image = cv2.resize(image, (image_shape[1], image_shape[0]))[..., ::-1] # (RGB)
            rgb_img = (image / 255.0).astype(np.float32)
            input_image = torch.from_numpy(rgb_img).permute(2, 0, 1) # (3, H, W)
            res_mask = get_res_mask(label, input_image, guidance_res)
            
            if res_mask is None:
                # Remove the problematic image path
                img_paths.remove(selected_img_path)
                print(f"Skipping image for label {label}")
                continue
            
            res_mask = res_mask.to(device)
            break  # Successfully processed an image
        
        if res_mask is None:
            print(f"Skipping linearSVM for label {label}")
            continue
        
        # Load 2D pred feature map
        sem_logits = np.load(args.logit_npy_dir + '/' + os.path.basename(selected_img_path).replace('.JPG', '.npy'))
        sem_feature = LUT[sem_logits]
        normed_feature = sem_feature / sem_feature.norm(dim=-1, keepdim=True)
        
        # Train a SVM for this label using this selected image
        prompt_embed = clip_model.encode_text_for_resMLP(label)
        linearSVM = LinearSVM(set_bias=clip_feature_thresh, 
                                    input_dim=feature_dim).to(device)
        linearSVM.weight_set(prompt_embed)
        
        save_path = args.model_output_dir + f"/visualizations"
        os.makedirs(save_path, exist_ok=True)
        linearSVM = finetune_prompt_with_res(normed_feature, res_mask, label, linearSVM, save_path, os.path.basename(selected_img_path))
        
        # Save the model
        linearSVM.save(args.model_output_dir + f'/{label}_svm.pth')
        
        # new_model = LinearSVM()
        # loaded_model = new_model.load(args.model_output_dir + f'/{label}_svm.pth')
        
       
    
    # Saving the model
    # model = LinearSVM()
    # model.save('linear_svm_model.pth')

    # # Loading the model
    # loaded_model = LinearSVM.load('linear_svm_model.pth')
    
    # pass