from plyfile import PlyData, PlyElement
import torch
import numpy as np
import os
from argparse import ArgumentParser
import open_clip
from tqdm import tqdm
from scipy.spatial import cKDTree as KDTree
import sys
from scene import SemanticModel
from torch.nn.functional import softmax

def load_scene_list(gt_scene_dir):
    # Check if the directory exists
    if not os.path.isdir(gt_scene_dir):
        raise ValueError(f"Directory {gt_scene_dir} does not exist")
    
    # Get all entries in the directory
    all_entries = os.listdir(gt_scene_dir)
    
    # Filter only directories (folders)
    folder_names = [entry for entry in all_entries 
                   if os.path.isdir(os.path.join(gt_scene_dir, entry))]
    
    return folder_names
    
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
    canon_feat: torch.Tensor,     # shape: (K, 512)
    device: torch.device,
    use_dot_similarity: bool = False,
    use_res_finetune: bool = False,
):
    """
    Computes predicted labels for each of the N language features using one of two
    methods:

    1. Ratio method (default, identical to original code):
       Score_c = min_i [ exp(lang . text_c) / (exp(lang . canon_i) + exp(lang . text_c)) ]
       where i indexes the canonical phrases.

    2. Dot‑similarity method (if ``use_dot_similarity`` is ``True``):
       Simply picks the text embedding with the highest CLIP dot similarity.

    Returns
    -------
    pred_label : ndarray, shape (N,)
        The predicted label indices in [0..C‑1].
    """

    # Move to device
    lang_feat = lang_feat.to(device, non_blocking=True)
    text_feat = text_feat.to(device, non_blocking=True)
    canon_feat = canon_feat.to(device, non_blocking=True)

    # Fast path: plain dot‑product similarity
    if use_dot_similarity:
        dot_lang_text = torch.matmul(lang_feat, text_feat.t())  # (N, C)
        pred_label = torch.argmax(dot_lang_text, dim=1)
        return pred_label.cpu().numpy()

    # Original ratio‑based relevancy score
    dot_lang_text = torch.matmul(lang_feat, text_feat.t())    # (N, C)
    dot_lang_canon = torch.matmul(lang_feat, canon_feat.t())  # (N, K)

    exp_lang_text = dot_lang_text.exp()    # (N, C)
    exp_lang_canon = dot_lang_canon.exp()  # (N, K)

    N, C = dot_lang_text.shape

    relevancy_scores = []
    for c_idx in range(C):
        text_c_exp = exp_lang_text[:, c_idx].unsqueeze(-1)  # (N,1)
        ratio_c = text_c_exp / (exp_lang_canon + text_c_exp)  # (N, K)
        score_c = torch.min(ratio_c, dim=1).values  # (N,)
        relevancy_scores.append(score_c)

    relevancy_matrix = torch.stack(relevancy_scores, dim=0).t()  # (N, C)
    pred_label = torch.argmax(relevancy_matrix, dim=1)           # (N,)
    return pred_label.cpu().numpy()

def parse_args():
    parser = ArgumentParser(description="Open‑Vocal 3DGS semantic evaluation")
    parser.add_argument("--gt_scene_dir", type=str, default="/scratch/joanna_cheng/scannetpp_gt/val")
    parser.add_argument("--preprocessed_root", type=str, default="/home/yli7/scratch2/datasets/ptv3_preprocessed/scannetpp_v2_preprocessed")
    parser.add_argument("--gs_root", type=str, default="/scratch/joanna_cheng/scannetpp_3dgs")
    parser.add_argument("--langfeat_root", type=str, default="/home/yli7/scratch2/outputs/ludvig_inference/scannetpp")
    parser.add_argument("--label_path", type=str, default="/scratch/joanna_cheng/scannetpp_v1_val_subset/scannetpp_semseg_top100.txt")
    parser.add_argument("--use_dot_similarity", action="store_true", help="If set, use plain CLIP dot similarity instead of ratio scoring.")
    return parser.parse_args()


def main():
    args = parse_args()
    args.use_dot_similarity = True  

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # ------------------------------
    # 1) Load validation scenes
    # ------------------------------
    scene_ids = load_scene_list(args.gt_scene_dir)
    print(f"Found {len(scene_ids)} validation scenes.")

    # ------------------------------
    # 2) Load CLIP model (once)
    # ------------------------------
    model, _, _ = open_clip.create_model_and_transforms("ViT-B-16", pretrained="laion2b_s34b_b88k")
    model = model.eval().to(device)
    tokenizer = open_clip.get_tokenizer("ViT-B-16")

    # Canonical phrases stay fixed
    canonical_phrases = ["object", "things", "stuff", "texture"]
    with torch.no_grad():
        canon_tokens = tokenizer(canonical_phrases)
        canon_feat = model.encode_text(canon_tokens.to(device))
        canon_feat /= canon_feat.norm(dim=-1, keepdim=True)
        canon_feat = canon_feat.cpu()

    # ------------------------------
    # 3) Evaluate 
    # ------------------------------

    ignore_classes = ["wall", "floor", "ceiling"]  # for foreground metrics

    # ---- 3.1 Load label names & encode text ----
    with open(args.label_path, "r") as f:
        label_names = [line.strip() for line in f if len(line.strip()) > 0]
    prompt_list = ["this is a " + name for name in label_names]

    with torch.no_grad():
        text_tokens = tokenizer(prompt_list)
        text_feat = model.encode_text(text_tokens.to(device))  # (C, 512)
        text_feat /= text_feat.norm(dim=-1, keepdim=True)
        text_feat = text_feat.cpu()

    num_classes = len(label_names)
    confusion_mat = np.zeros((num_classes, num_classes), dtype=np.int64)

    # Build ignore mask once
    ignore_mask = np.array([name in ignore_classes for name in label_names], dtype=bool)

    # ---- 3.2 Loop over scenes ----
    for scene_id in tqdm(scene_ids, desc="Scene", dynamic_ncols=True):
        # Paths to data
        scene_preproc_folder = os.path.join(args.gt_scene_dir, scene_id)
        if not os.path.isdir(scene_preproc_folder):
            print(f"[Warning] Preprocessed folder not found: {scene_preproc_folder}")
            continue

        coord_path = os.path.join(scene_preproc_folder, "coord.npy")
        segment_path = os.path.join(scene_preproc_folder, "segment.npy")
        if not (os.path.isfile(coord_path) and os.path.isfile(segment_path)):
            print(f"[Warning] Missing coord.npy or segment.npy at {scene_preproc_folder}")
            continue

        coord = np.load(coord_path)               # (N, 3)
        segment = np.load(segment_path)           # (N, 3) first col => label index
        labeled_gt = segment[:, 0]    # scannetpp save top-3 lables as some points have multiple labels             
        valid_mask = labeled_gt >= 0  # in the preprocessing, -1 is used for ignore labels
        if valid_mask.sum() == 0:
            continue

        xyz_val = coord[valid_mask]
        gt_val = labeled_gt[valid_mask].astype(np.int64)

        # ---- 3.2.b Load 3DGS & CLIP feats ----
        scene_3dgs_folder = os.path.join(args.gs_root, scene_id)
        ply_path = os.path.join(scene_3dgs_folder, "point_cloud/iteration_1500_lvl_3", "point_cloud.ply")
        if not os.path.isfile(ply_path):
            print(f"[Warning] 3DGS .ply not found for scene {scene_id}")
            continue
        gauss_xyz, gauss_lang_feat = load_ply(ply_path)
        gauss_lang_feat = torch.from_numpy(gauss_lang_feat).float().to(device) # (G, 10)
        
        lut_model_path = os.path.join(scene_3dgs_folder, "point_cloud/iteration_1500_lvl_3", "LUT.pt")
        LUT = torch.load(lut_model_path).to(device)
        
        mlp_model_path = os.path.join(scene_3dgs_folder, "point_cloud/iteration_1500_lvl_3", "semantic_MLP.pt")
        MLP = SemanticModel.load(mlp_model_path).to(device)
        
        gauss_lang_label = MLP(gauss_lang_feat) # (G, 512)
        sem_logit = softmax(gauss_lang_label*10, dim=-1).argmax(dim=-1) # (G, 512)
        gauss_lang_feat = LUT[sem_logit] # (G, 512)
        
        norms = gauss_lang_feat.norm(dim=1)
        gauss_lang_feat = gauss_lang_feat / gauss_lang_feat.norm(dim=-1, keepdim=True)

        keep_mask_gs = (norms > 0)
        gauss_xyz = gauss_xyz[keep_mask_gs.cpu().numpy()]
        gauss_lang_feat = gauss_lang_feat[keep_mask_gs]
        if gauss_xyz.shape[0] == 0:
            print(f"[Warning] All 3DGS zero feats in {scene_id}")
            continue

        # ---- 3.2.c KD‑Tree NN search ----
        kd_tree = KDTree(gauss_xyz)
        _, nn_idx = kd_tree.query(xyz_val) # (N,) Find the nearest neighbor for each point in xyz_val
        nn_lang_feat = gauss_lang_feat[nn_idx] # (N, 512)

        # ---- 3.2.d Predict labels ----
        pred_label = compute_relevancy_scores(
            nn_lang_feat,
            text_feat,
            canon_feat,
            device=device,
            use_dot_similarity=args.use_dot_similarity,
        )

        # ---- 3.2.e Accumulate confusion ----
        for gt_c, pr_c in zip(gt_val, pred_label):
            if gt_c < num_classes and pr_c < num_classes:  # guard against idx mismatch
                confusion_mat[gt_c, pr_c] += 1

    # ------------------------------
    # 4) Compute metrics
    # ------------------------------
    ious = []
    per_class_acc = []
    gt_class_counts = np.sum(confusion_mat, axis=1)

    for c in range(num_classes):
        tp = confusion_mat[c, c]
        fn = gt_class_counts[c] - tp
        fp = np.sum(confusion_mat[:, c]) - tp
        denom = tp + fp + fn
        iou_c = tp / denom if denom > 0 else 0.0
        ious.append(iou_c)

        acc_c = tp / gt_class_counts[c] if gt_class_counts[c] > 0 else 0.0
        per_class_acc.append(acc_c)

    valid_mask = gt_class_counts > 0
    mean_iou = np.mean(np.array(ious)[valid_mask]) if valid_mask.any() else 0.0
    mean_class_acc = np.mean(np.array(per_class_acc)[valid_mask]) if valid_mask.any() else 0.0

    total_correct = np.trace(confusion_mat)
    total_count = confusion_mat.sum()
    global_acc = total_correct / (total_count + 1e-12)

    # Foreground metrics (exclude ignore classes)
    fg_mask = valid_mask & (~ignore_mask)
    fg_miou = np.mean(np.array(ious)[fg_mask]) if fg_mask.any() else 0.0
    fg_macc = np.mean(np.array(per_class_acc)[fg_mask]) if fg_mask.any() else 0.0

    # ------------------------------
    # 5) Print final results
    # ------------------------------
    print("\n======== RESULTS ========")
    print("Per‑class IoU:")
    for c, name in enumerate(label_names):
        print(f"  {name:24s}: {ious[c]:.4f}")
    print(f"Mean IoU: {mean_iou:.4f}")
    print(f"Global Accuracy: {global_acc:.4f}")
    print(f"Mean Class Accuracy: {mean_class_acc:.4f}")
    print(f"Foreground mIoU: {fg_miou:.4f}")
    print(f"Foreground mAcc: {fg_macc:.4f}")

if __name__ == "__main__":
    main()