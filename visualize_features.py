import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.decomposition import PCA
import os
from PIL import Image

def visualize_langsplat_format(feature_path, output_dir):
    
    seg_map = torch.from_numpy(np.load(feature_path + '_s.npy'))  # seg_map: torch.Size([1, 730, 988])
    feature_map = torch.from_numpy(np.load(feature_path + '_f.npy')) # feature_map: torch.Size([281, 512])
    
    
    channels, height, width =seg_map.shape[0], seg_map.shape[1], seg_map.shape[2]
    
    print(seg_map.shape, feature_map.shape) # torch.Size([1, 584, 876]) torch.Size([51, 768])
    
    # point_feature = feature_map[seg_map[0].squeeze(-1).long()].squeeze(0)
    # print(point_feature.shape) #torch.Size([730, 988, 256])
    
    # # run pca
    # feature_np = feature_map.cpu().numpy()
    # pca = PCA(n_components=3)
    # reduced_features = pca.fit_transform(feature_np)
    
    # rgb_features = reduced_features[seg_map[0].squeeze(-1).long()]
    # print(rgb_features.shape) #torch.Size([730, 988, 3])
    
    # # Normalize each channel to 0-1 range for visualization
    # for i in range(3):
    #     channel = rgb_features[:, :, i]
    #     min_val = channel.min()
    #     max_val = channel.max()
    #     rgb_features[:, :, i] = (channel - min_val) / (max_val - min_val)
        
    # # # Create and save the visualization
    # plt.figure(figsize=(12, 10))
    # plt.imshow(rgb_features)
    # plt.title("PCA Visualization of Feature Map (RGB = PC1, PC2, PC3)")
    # plt.axis('off')
    # plt.tight_layout()
    
    # # # Save PCA visualization
    # output_path = os.path.join(output_dir, f"{Path(feature_path).stem}_pca_rgb_ape.png")
    # plt.savefig(output_path, dpi=300)
    # print(f"PCA RGB visualization saved to: {output_path}")
    
    # point_feature = point_feature.reshape(height, width, -1).permute(2, 0, 1)
    
def visualize_ape_format(feature_path, output_dir):
    
    feature_map = torch.load(feature_path)
    
    # Print basic information about the tensor
    print("Feature map loaded successfully")
    print(f"Type: {type(feature_map)}")
    if hasattr(feature_map, 'shape'):
        print(f"Shape: {feature_map.shape}")
    if hasattr(feature_map, 'dtype'):
        print(f"Data type: {feature_map.dtype}")
        
    output_dir = "/home/joanna_cheng/workspace/GOI-Hyperplane/visualizations"
    
    feature_np = feature_map.cpu().numpy()
    channels, height, width = feature_np.shape
    # Reshape to have features as columns
    # Transpose the tensor to [height, width, channels]
    feature_np = np.transpose(feature_np, (1, 2, 0))
    # Reshape to [height*width, channels]
    reshaped_features = feature_np.reshape(-1, channels)
    
    pca = PCA(n_components=3)
    reduced_features = pca.fit_transform(reshaped_features)
    rgb_features = reduced_features.reshape(height, width, 3)
    
     # Normalize each channel to 0-1 range for visualization
    for i in range(3):
        channel = rgb_features[:, :, i]
        min_val = channel.min()
        max_val = channel.max()
        rgb_features[:, :, i] = (channel - min_val) / (max_val - min_val)
        
    # Create and save the visualization
    plt.figure(figsize=(12, 10))
    plt.imshow(rgb_features)
    plt.title("PCA Visualization of Feature Map (RGB = PC1, PC2, PC3)")
    plt.axis('off')
    plt.tight_layout()
    
    # Save PCA visualization
    output_path = os.path.join(output_dir, f"{Path(feature_path).stem}_pca_rgb.png")
    plt.savefig(output_path, dpi=300)
    print(f"PCA RGB visualization saved to: {output_path}")

def visualize_scannet_langfeat(feature_path, output_dir, full_image_path):
    
    # Read image
    image = Image.open(full_image_path)
    original_size = image.size # (1296, 968)
    
    resize = [640,480] # (640, 480)
    crop_edge = 12
    width, height = original_size
    image = image.resize((resize[0], resize[1]), Image.LANCZOS)  # Resize to (640, 480)
    image = image.crop((crop_edge, crop_edge, resize[0] - crop_edge, resize[1] - crop_edge))  # Crop to (616, 456)
    
    # Convert PIL image to numpy array for matplotlib
    image_np = np.array(image) # (480, 640, 3)
    
    # Read langfeat
    seg_map = torch.from_numpy(np.load(feature_path + '_s.npy'))  # seg_map: torch.Size([1, 456, 616])
    feature_map = torch.from_numpy(np.load(feature_path + '_f.npy')) # feature_map: torch.Size([281, 512])
    print(seg_map.shape)
    print(feature_map.shape)
    
    channels, height, width =seg_map.shape[0], seg_map.shape[1], seg_map.shape[2]
    # point_feature = feature_map[seg_map[0].squeeze(-1).long()].squeeze(0)
    
    # run pca
    feature_np = feature_map.cpu().numpy()
    pca = PCA(n_components=3)
    reduced_features = pca.fit_transform(feature_np)
    
    rgb_features = reduced_features[seg_map[0].squeeze(-1).long()] # (456, 616, 3)
    
    # Normalize each channel to 0-1 range for visualization
    rgb_features_normalized = np.zeros_like(rgb_features, dtype=np.float32)
    for i in range(3):
        channel = rgb_features[:, :, i]
        min_val = channel.min()
        max_val = channel.max()
        rgb_features_normalized[:, :, i] = (channel - min_val) / (max_val - min_val)
        
    # 1. Stack the original image and feature map side by side
    plt.figure(figsize=(20, 10))
    
    plt.subplot(1, 3, 1)
    plt.imshow(image_np)
    plt.title("Original Image")
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.imshow(rgb_features_normalized)
    plt.title("PCA Visualization (RGB = PC1, PC2, PC3)")
    plt.axis('off')
    
    # 2. Create an overlay of the image with the feature map
    plt.subplot(1, 3, 3)
    plt.imshow(image_np)
    plt.imshow(rgb_features_normalized, alpha=0.5)  # Overlay with 50% transparency
    plt.title("Image with Feature Map Overlay")
    plt.axis('off')
    
    plt.tight_layout()
    
    # Save the visualization
    output_path = os.path.join(output_dir, os.path.basename(full_image_path).split('.')[0] + '_visualization.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    

if __name__ == "__main__":
    
    output_dir = "/home/joanna_cheng/workspace/GOI-Hyperplane/visualizations"
    
    # feature_path = os.path.join("/scratch/joanna_cheng/lerf_ovs/teatime/clip_feat/frame_00005.pt")
    # visualize_ape_format(feature_path, output_dir)
    
    # language_feature_name = "/scratch/joanna_cheng/lerf_ovs/teatime/ape_feat/frame_00001"
    language_feature_name = "/scratch/joanna_cheng/scannet_2d_language_features/scene0011_00/00000"
    image_path = "/scratch/joanna_cheng/scannet_interval/scans/scene0011_00/color_interval/00000.jpg"
    # visualize_langsplat_format(language_feature_name, output_dir)
    visualize_scannet_langfeat(language_feature_name, output_dir, image_path)