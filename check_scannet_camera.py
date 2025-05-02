import torch
import numpy as np

if __name__ == "__main__":
    
    # Path to the camera poses file
    file_path = '/scratch/joanna_cheng/scannet_val_subset/scannet_subset_sequences/subset_sequences/scene0011_01/pose/camera_poses.npy'
    
    camera_poses = np.load(file_path)
    
    # Print the shape to understand the data structure
    print(f"Camera poses shape: {camera_poses.shape}")

    # Print the first camera pose to see the format
    print("\nFirst camera pose:")
    print(camera_poses[0])

    