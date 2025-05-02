import os
import random
import json
from utils.system_utils import searchForMaxIteration
from scene.dataset_readers import sceneLoadTypeCallbacks
from scene.gaussian_model import GaussianModel
from arguments import ModelParams
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON
from scene.semantic_model import SemanticModel

class Scene:

    gaussians : GaussianModel

    def __init__(self, args : ModelParams, gaussians : GaussianModel, load_iteration=None, shuffle=True, resolution_scales=[1.0], is_render_sem=False, dataset_type=None):
        """b
        :param path: Path to colmap scene main folder.
        """
        self.model_path = args.model_path
        self.loaded_iter = None
        self.gaussians = gaussians

        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        self.train_cameras = {}
        self.test_cameras = {}

        if os.path.exists(os.path.join(args.source_path, "sparse")):
            scene_info = sceneLoadTypeCallbacks["Colmap"](args.source_path, args.images, args.eval)
        # elif os.path.exists(os.path.join(args.source_path, "transforms_train.json")):
        #     print("Found transforms_train.json file, assuming Blender data set!")
        #     scene_info = sceneLoadTypeCallbacks["Blender"](args.source_path, args.white_background, args.eval)
        elif os.path.exists(os.path.join(args.source_path, "dslr/nerfstudio")):
            print("Using Scannetpp data")
            scene_info = sceneLoadTypeCallbacks["ScanNetpp"](os.path.join(args.source_path, "dslr/nerfstudio"), args.white_background, os.path.join(args.source_path, "dslr/undistorted_depths"), args.eval, os.path.join(args.source_path, "dslr/nerfstudio/", "lang_feat_selected_imgs.json"))
        elif dataset_type == "scannet":
            print("Using Scannet data")
            scene_name = os.path.basename(args.source_path)
            src_path = os.path.join("/scratch/joanna_cheng/scannet_interval/scans", scene_name)
            camera_path = os.path.join(src_path, "lang_feat_selected_imgs.json")
            scene_info = sceneLoadTypeCallbacks["ScanNet"](src_path, args.white_background, eval=args.eval, lang_path=camera_path)
        elif dataset_type == "matterport3d":
            print("Using matterport3d data")
            camera_path = os.path.join(args.source_path, "lang_feat_selected_imgs.json")
            scene_info = sceneLoadTypeCallbacks["MatterPort3D"](args.source_path, args.white_background, eval=args.eval, lang_path=camera_path)

        if not self.loaded_iter:
            with open(scene_info.ply_path, 'rb') as src_file, open(os.path.join(self.model_path, "input.ply") , 'wb') as dest_file:
                dest_file.write(src_file.read())
            json_cams = []
            camlist = []
            if scene_info.test_cameras:
                camlist.extend(scene_info.test_cameras)
            if scene_info.train_cameras:
                camlist.extend(scene_info.train_cameras)
            for id, cam in enumerate(camlist):
                json_cams.append(camera_to_JSON(id, cam))
            with open(os.path.join(self.model_path, "cameras.json"), 'w') as file:
                json.dump(json_cams, file)

        if shuffle:
            random.shuffle(scene_info.train_cameras)  # Multi-res consistent random shuffling
            random.shuffle(scene_info.test_cameras)  # Multi-res consistent random shuffling

        self.cameras_extent = scene_info.nerf_normalization["radius"]

        for resolution_scale in resolution_scales:
            print("Loading Training Cameras")
            self.train_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.train_cameras, resolution_scale, args)
            print("Loading Test Cameras")
            self.test_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.test_cameras, resolution_scale, args)

        if self.loaded_iter:
            if is_render_sem:
                self.gaussians.load_ply(os.path.join(self.model_path,
                                                        "point_cloud",
                                                        "iteration_" + str(self.loaded_iter) + "_lvl_" + str(args.feature_level),
                                                        "point_cloud.ply"))
            else:
                if dataset_type == "scannet" or dataset_type == "matterport3d":
                    print("Loading Scannet point cloud")
                    self.gaussians.load_ply(os.path.join(self.model_path,
                                                         "ckpts",
                                                         "point_cloud_30000.ply"))
                else:
                    self.gaussians.load_ply(os.path.join(self.model_path,
                                                                "point_cloud",
                                                                "iteration_" + str(self.loaded_iter),
                                                                "point_cloud.ply"))
        else:
            self.gaussians.create_from_pcd(scene_info.point_cloud, self.cameras_extent)

    def save(self, iteration, feature_level):
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}_lvl_{}".format(iteration, feature_level))
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))

    def getTrainCameras(self, scale=1.0):
        return self.train_cameras[scale]

    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]