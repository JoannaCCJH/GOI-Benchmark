import torch
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
# from gaussian_renderer import GaussianModel
from scene import SemanticModel, GaussianModel
from torch.nn.functional import softmax
from sklearn.decomposition import PCA
import numpy as np

def render_set(model_path, name, iteration, views, gaussians, pipeline, background, semantic_MLP, lut, sem_dim, language_feature_dir, feature_level):
    render_path = os.path.join(model_path, name, "ours_{}_lvl_{}".format(iteration, feature_level), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}_lvl_{}".format(iteration, feature_level), "gt")
    logit_path = os.path.join(model_path, name, "ours_{}_lvl_{}".format(iteration, feature_level), "logit")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)
    makedirs(logit_path, exist_ok=True)

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        render_pkg = render(view, gaussians, pipeline, background)
        # gt = view.original_image[0:3, :, :]
        # torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        # torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))
        
        _, sem_feature, _, _, _ = render_pkg["render"], render_pkg[
            "semantics"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
        
        sem_feature = sem_feature.permute(1, 2, 0).reshape(-1, sem_dim)
        sem_label = semantic_MLP(sem_feature)
        sem_logit = softmax(sem_label*10, dim=-1).argmax(dim=-1)
        # print(sem_logit.shape) # torch.Size([511584])
        saved_sem_logit = sem_logit.cpu().numpy()
        np.save(os.path.join(logit_path, view.image_name + ".npy"), saved_sem_logit)
        
        sem_feature = lut[sem_logit]
        normed_feature = sem_feature / sem_feature.norm(dim=-1, keepdim=True)
        normed_feature = normed_feature.cpu().numpy()
        
        gtl, mask = view.get_language_feature(language_feature_dir, feature_level)
        # print("normed_feature.shape", normed_feature.shape) 
        # print("gtl.shape", gtl.shape)
        # normed_feature.shape torch.Size([511584, 512])
        # gtl.shape torch.Size([512, 584, 876])
        
        C, H, W = gtl.shape
        gt_reshaped = gtl.reshape(C, -1).T.cpu().numpy()
        combined = np.concatenate((normed_feature, gt_reshaped), axis=0)
        
        pca = PCA(n_components=3)
        reduced_combined = pca.fit_transform(combined)
        # Normalize the entire PCA result at once
        reduced_min = reduced_combined.min(axis=0, keepdims=True)
        reduced_max = reduced_combined.max(axis=0, keepdims=True)
        reduced_combined_norm = (reduced_combined - reduced_min) / (reduced_max - reduced_min + 1e-8)
        
        reduced_semantic = reduced_combined_norm[:H*W, :]
        reduced_gt = reduced_combined_norm[H*W:, :]
        
        rendering = torch.tensor(reduced_semantic.T.reshape(3, H, W))
        gt = torch.tensor(reduced_gt.T.reshape(3, H, W))
        torchvision.utils.save_image(rendering, os.path.join(render_path, view.image_name + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, view.image_name + ".png"))
         

def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool):
    
    language_feature_dir=f"{dataset.source_path}/dslr/language_features"
    # language_feature_dir=f"{dataset.source_path}/language_features"
    
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree, dataset.sem_dim)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False, is_render_sem=True)
        
        psem = os.path.join(dataset.model_path, "point_cloud","iteration_" + str(iteration) + "_lvl_" + str(dataset.feature_level), "semantic_MLP.pt")      
        plut = os.path.join(dataset.model_path, "point_cloud","iteration_" + str(iteration) + "_lvl_" + str(dataset.feature_level), "LUT.pt")      
        MLP = SemanticModel.load(psem)
        LUT = torch.load(plut)

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        
        print("Number of training views: ", len(scene.getTrainCameras()))
        print("Number of testing views: ", len(scene.getTestCameras()))

        if not skip_train:
             render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background, MLP, LUT, dataset.sem_dim, language_feature_dir, dataset.feature_level)

        if not skip_test:
             render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background, MLP, LUT, dataset.sem_dim, language_feature_dir, dataset.feature_level)

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test)