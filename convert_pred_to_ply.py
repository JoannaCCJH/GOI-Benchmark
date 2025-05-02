import numpy as np
from plyfile import PlyData, PlyElement
import matplotlib.pyplot as plt
import torch
import open_clip
from tqdm import tqdm
from scipy.spatial import cKDTree as KDTree
import sys
from scene import SemanticModel
from torch.nn.functional import softmax
import os
from sklearn.decomposition import PCA

import numpy as np
from plyfile import PlyData, PlyElement

def convert_to_ply_with_rgb(xyz, lang_features, output_path):
    """
    Convert point cloud coordinates and language features to PLY format with RGB color visualization
    and save to file.
    
    Parameters:
    -----------
    xyz : numpy.ndarray
        Point cloud coordinates of shape (N, 3)
    lang_features : numpy.ndarray
        Language features of shape (N, 3) - assumed to be PCA-reduced to 3 dimensions
    output_path : str
        Path to save the output PLY file
    """
    # Get the number of points
    num_points = xyz.shape[0]
    
    # Ensure language features are 3-dimensional for RGB mapping
    assert lang_features.shape[1] == 3, "Language features must be 3-dimensional for RGB mapping"
    
    # Normalize features to 0-255 range for RGB
    # First find min and max for each feature dimension
    min_vals = np.min(lang_features, axis=0)
    max_vals = np.max(lang_features, axis=0)
    
    # Normalize to 0-1 range
    normalized_features = (lang_features - min_vals) / (max_vals - min_vals + 1e-10)
    
    # Scale to 0-255 and convert to uint8
    rgb_values = (normalized_features * 255).astype(np.uint8)
    
    # Create a structured array for the vertices
    dtype_list = [
        ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
        ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')
    ]
    
    vertex_array = np.zeros(num_points, dtype=dtype_list)
    
    # Fill in the coordinate values
    vertex_array['x'] = xyz[:, 0]
    vertex_array['y'] = xyz[:, 1]
    vertex_array['z'] = xyz[:, 2]
    
    # Fill in the RGB values
    vertex_array['red'] = rgb_values[:, 0]
    vertex_array['green'] = rgb_values[:, 1]
    vertex_array['blue'] = rgb_values[:, 2]
    
    # Create the PlyElement
    vertex_element = PlyElement.describe(vertex_array, 'vertex')
    
    # Create the PlyData object and write to file
    ply_data = PlyData([vertex_element], text=True)
    ply_data.write(output_path)
    
    print(f"PLY file with RGB colors saved to {output_path}")
    return output_path

def load_ply(path):
        semantic_dim=10
        
        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)


        sem_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("sem_")]
        sem_names = sorted(sem_names, key=lambda x: int(x.split('_')[-1]))
        sems = np.zeros((xyz.shape[0], len(sem_names) or semantic_dim))
        if len(sem_names) == semantic_dim:
            for idx, attr_name in enumerate(sem_names):
                sems[:, idx] = np.asarray(plydata.elements[0][attr_name])
        
        return xyz, sems
    
@torch.no_grad()
def compute_relevancy_scores(
    lang_feat: torch.Tensor,      # shape: (N, 512)
    text_feat: torch.Tensor,      # shape: (C, 512)
    device: torch.device,
    use_dot_similarity: bool = False,
):
    # Move to device
    lang_feat = lang_feat.to(device, non_blocking=True)
    text_feat = text_feat.to(device, non_blocking=True)
    
    # Fast path: plain dot‑product similarity
    if use_dot_similarity:
        dot_lang_text = torch.matmul(lang_feat, text_feat.t())  # (N, C)
        pred_label = torch.argmax(dot_lang_text, dim=1)
        return pred_label.cpu().numpy()


if __name__ == "__main__":
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    
    coord_path = "/scratch/joanna_cheng/scannetpp_gt/val/0d2ee665be/coord.npy"
    segment_path = "/scratch/joanna_cheng/scannetpp_gt/val/0d2ee665be/segment.npy"
    output_path = "./0d2ee665be_pred_point_cloud.ply"
    scene_3dgs_folder = "/scratch/joanna_cheng/scannetpp_3dgs/0d2ee665be"
    ply_path = os.path.join(scene_3dgs_folder, "point_cloud/iteration_1500_lvl_3", "point_cloud.ply")
    label_path = "/scratch/joanna_cheng/scannetpp_v1_val_subset/scannetpp_semseg_top100.txt"
    
    # Load labels
    with open(label_path, "r") as f:
        label_names = [line.strip() for line in f if len(line.strip()) > 0]
    prompt_list = ["this is a " + name for name in label_names]
    
    # Load GT
    coord = np.load(coord_path)               # (N, 3)
    segment = np.load(segment_path)           # (N, 3) first col => label index
    labeled_gt = segment[:, 0]    # scannetpp save top-3 lables as some points have multiple labels             
    valid_mask = labeled_gt >= 0  # in the preprocessing, -1 is used for ignore labels
    if valid_mask.sum() == 0:
            print("All labels are -1!")
    xyz_val = coord[valid_mask]
    gt_val = labeled_gt[valid_mask].astype(np.int64)
    
    # Create Index Mapping
    gt_unique = np.unique(gt_val)    
    # Create a mapping from original indices to new sequential indices (0-32)
    orig_to_new_map = {original: new for new, original in enumerate(gt_unique)}
    # Create a mapping from new sequential indices to original text labels
    # Each original index in gt_unique corresponds to a label in prompt_list
    new_to_text_map = {new_idx: prompt_list[orig_idx] for orig_idx, new_idx in orig_to_new_map.items()}
    # Also create reverse mappings if needed
    new_to_orig_map = {new: original for original, new in orig_to_new_map.items()}
    text_to_new_map = {text: new for new, text in new_to_text_map.items()}
    
    # Load Model
    model, _, _ = open_clip.create_model_and_transforms("ViT-B-16", pretrained="laion2b_s34b_b88k")
    model = model.eval().to(device)
    tokenizer = open_clip.get_tokenizer("ViT-B-16")
    
    lut_model_path = os.path.join(scene_3dgs_folder, "point_cloud/iteration_1500_lvl_3", "LUT.pt")
    LUT = torch.load(lut_model_path).to(device)
    mlp_model_path = os.path.join(scene_3dgs_folder, "point_cloud/iteration_1500_lvl_3", "semantic_MLP.pt")
    MLP = SemanticModel.load(mlp_model_path).to(device)
    
    # Load pred
    gauss_xyz, gauss_lang_feat = load_ply(ply_path)
    gauss_lang_feat = torch.from_numpy(gauss_lang_feat).float().to(device) # (G, 10)
    # Process Pred
    gauss_lang_label = MLP(gauss_lang_feat) # (G, 500)
    
    # PCA
    pca = PCA(n_components=3)
    reduced_lang_feat = pca.fit_transform(gauss_lang_feat.cpu().numpy())
    convert_to_ply_with_rgb(gauss_xyz, reduced_lang_feat, output_path)
    
    # sem_logit = softmax(gauss_lang_label*10, dim=-1) # torch.Size([G, 500])
    # sem_logit = sem_logit.argmax(dim=-1) # (G)
    # gauss_lang_feat = LUT[sem_logit] # (G, 512)
    
    # norms = gauss_lang_feat.norm(dim=1)
    # gauss_lang_feat = gauss_lang_feat / gauss_lang_feat.norm(dim=-1, keepdim=True)

    # keep_mask_gs = (norms > 0)
    # gauss_xyz = gauss_xyz[keep_mask_gs.cpu().numpy()]
    # gauss_lang_feat = gauss_lang_feat[keep_mask_gs]
    
    # # ---- 3.2.c KD‑Tree NN search ----
    # kd_tree = KDTree(gauss_xyz)
    # _, nn_idx = kd_tree.query(xyz_val) # (N,) Find the nearest neighbor for each point in xyz_val
    # nn_lang_feat = gauss_lang_feat[nn_idx] # (N, 512)
    
    # prompt_list  = [new_to_text_map[i] for i in range(len(gt_unique))]
    # # Encode text
    # with torch.no_grad():
    #     text_tokens = tokenizer(prompt_list)
    #     text_feat = model.encode_text(text_tokens.to(device))  # (C, 512)
    #     text_feat /= text_feat.norm(dim=-1, keepdim=True)
    #     text_feat = text_feat.cpu()
    
    # print(nn_lang_feat.shape)
    
    # # ---- 3.2.d Predict labels ----
    # pred_label = compute_relevancy_scores(
    #     nn_lang_feat,
    #     text_feat,
    #     device=device,
    #     use_dot_similarity=True,
    # )
    
    # pred_label_unique = np.unique(pred_label)
    # print("Number of unique pred labels: ", len(pred_label_unique))
    # print(pred_label_unique)
   
    # print(new_to_text_map[pred_label_unique[0]])
    # # numpy_to_ply(coord, segment, output_path)