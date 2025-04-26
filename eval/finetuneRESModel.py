
from res_model import RES_MODEL
import torch
import cv2
import PIL
from PIL import Image, ImageDraw, ImageFont, ImageOps
from torchvision.utils import save_image
import sys, os

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

from networks import LinearSVM
import numpy as np

class FinetuneResModel():
    def __init__(self, clip_feature_thresh=0.8, feature_dim=512, device="cuda", height=None, width=None):
        
        self.device = device
        
        # Initialize RES model
        self.guidance_res = RES_MODEL(self.device)
        self.resMLP = {}
        self.feature_dim = feature_dim
        self.clip_feature_thresh = clip_feature_thresh
    
        
    def finetune_prompt_with_res(self, normed_feature, res_mask, prompt):     
        
        print("Training SVM for prompt: ", prompt)
        h, w, _ = res_mask.shape
        gt = res_mask.reshape(-1, 1) # (h*w, 1)
        epoch = 0
        max_epoch = 5000
        target_iou_thresh = 0.8
        iou = 0
        
        # Make a detached copy of normed_feature
        normed_feature = normed_feature.cuda().detach()
        gt = gt.detach()

        while (epoch < max_epoch and iou < target_iou_thresh):

            # if epoch % 500 == 0:

            #     # Detach tensors for evaluation to avoid grad issues
            #     with torch.no_grad():
            #         logit = (self.resMLP[prompt](normed_feature.cuda())).squeeze()
            #         logit = (logit).sigmoid().squeeze(-1)
            #         clip_thresh = 0.5
            #         logit_mask = logit > clip_thresh
            #         logit_mask = logit_mask.cpu().numpy()

            #         logit_mask = logit_mask.reshape(h, w)
            #         gt_mask = gt.bool().squeeze(-1).cpu().numpy().reshape(h, w)
                    
            #         logit_mask_uint8 = logit_mask.astype(np.uint8) * 255
            #         gt_mask_uint8 = gt_mask.astype(np.uint8) * 255
                    
            #     mask_image = Image.fromarray(logit_mask_uint8)
            #     gt_mask_image = Image.fromarray(gt_mask_uint8)

            #     mask_image.save('./testfolder/Finetune_mask_image-epoch'+str(epoch)+'-'+prompt+'.png')
            #     gt_mask_image.save('./testfolder/Finetune_gt_mask_image-epoch'+str(epoch)+'-'+prompt+'.png')

            # Make new tensors for each step to avoid graph reuse
            x = normed_feature.clone()  # This creates a new tensor
            y = gt.clone()  # This creates a new tensor
            loss, iou = self.resMLP[prompt].step(x, y)

            if epoch % 500 == 0:
                print("epoch:", epoch, "loss:", loss.item(), "iou:", iou)

            epoch += 1
            
        # with torch.no_grad():
        #     logit = (self.resMLP[prompt](normed_feature.cuda())).squeeze()
        #     logit = (logit).sigmoid().squeeze(-1)
            
        #     clip_thresh = 0.5
        #     logit_mask = logit > clip_thresh
        #     logit_mask = logit_mask.cpu().numpy()

        #     logit_mask = logit_mask.reshape(h, w)
        #     gt_mask = gt.bool().squeeze(-1).cpu().numpy().reshape(h, w)
            
        #     logit_mask_uint8 = logit_mask.astype(np.uint8) * 255
        #     gt_mask_uint8 = gt_mask.astype(np.uint8) * 255
                    
        # mask_image = Image.fromarray(logit_mask_uint8)
        # gt_mask_image = Image.fromarray(gt_mask_uint8)

        # mask_image.save('./testfolder/Finetune_mask_image-epoch'+str(epoch)+'-'+prompt+'.png')
        # gt_mask_image.save('./testfolder/Finetune_gt_mask_image-epoch'+str(epoch)+'-'+prompt+'.png')

            
        return self.resMLP[prompt]
    
    def get_linearSVL(self, prompt):
        
        if self.resMLP.get(prompt, None) is not None:
            return self.resMLP[prompt]
        else:
            return None
    
    def create_LinearSVM(self, prompt, prompt_embed):
        
        if self.resMLP.get(prompt, None) is None:
            linearSVM = LinearSVM(set_bias=self.clip_feature_thresh, 
                                    input_dim=self.feature_dim).to(self.device)
            linearSVM.weight_set(prompt_embed)
            self.resMLP[prompt] = linearSVM
            
        return
        
    def get_res_mask(self, res_prompt, input_image):
        """
        Generate a response mask based on a text prompt and an input image.
        
        The input image should be read using cv2.imread() and properly resized before calling.
        
        input_image: (3, H, W)
        """
        
        # image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # input_image = torch.from_numpy(image_rgb).float().permute(2, 0, 1) / 255.0
        
        res_mask, pred_image = self.guidance_res.predict_res_mask(input_image.cpu(), res_prompt)
        
        if pred_image is None:
            return None
        
        # mask_out_expand = res_mask.expand_as(input_image)
        # viz_images = torch.cat([pred_image.unsqueeze(0) ,mask_out_expand.unsqueeze(0)],dim=0)                
        # save_image(viz_images, "masktest.png")
        
        res_mask = res_mask.to(self.device) # torch.Size([1, H, W])
        
        return res_mask.permute(1,2,0) # torch.Size([H, W, 1])
    

if __name__=="__main__":
    
    model = RES_MODEL(device='cuda')
    
    image_path = "/scratch/joanna_cheng/scannetpp_v1_val_subset/3db0a1c8f3/dslr/undistorted_images/DSC00090.JPG"
    image = cv2.imread(image_path)
    image = cv2.resize(image, (800, 600))
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Convert to tensor and normalize to [0, 1]
    # OpenCV images are in uint8 format (0-255), so we divide by 255
    output_image = torch.from_numpy(image_rgb).float().permute(2, 0, 1) / 255.0
    prompt = "window"
    
    mask_out, output_image = model.predict_res_mask(output_image, prompt)
    
    print(mask_out.shape)
    
    mask_out_expand = mask_out.expand_as(output_image) 
    viz_images = torch.cat([output_image.unsqueeze(0) ,mask_out_expand.unsqueeze(0)],dim=0)                
    save_image(viz_images, "masktest.png")
    