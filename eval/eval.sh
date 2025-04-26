feature_level=3
SCENE=cc5237fd77

python eval_semantics_scannetpp.py \
    --label_name_path /scratch/joanna_cheng/scannetpp_v1_val_subset/scannetpp_semseg_top100.txt \
    --iteration 1500 \
    --gt_npy_dir /scratch/joanna_cheng/scannetpp_v1_val_subset/$SCENE/dslr/segmentation_2d \
    --logit_npy_dir /scratch/joanna_cheng/scannetpp_3dgs/$SCENE/test/ours_1500_lvl_$feature_level/logit \
    --output_dir /scratch/joanna_cheng/scannetpp_3dgs/$SCENE/test/ours_1500_lvl_$feature_level/eval \
    --lut_model_path /scratch/joanna_cheng/scannetpp_3dgs/$SCENE/point_cloud/iteration_1500_lvl_$feature_level/LUT.pt

# python eval_semantics_lerf.py \
#     --iteration 1500 \
#     --logit_npy_dir /home/joanna_cheng/workspace/gaussian-splatting-gsplats/output/teatime/test/ours_1500_lvl_$feature_level/logit \
#     --output_dir /home/joanna_cheng/workspace/gaussian-splatting-gsplats/output/teatime/test/ours_1500_lvl_$feature_level/eval \
#     --lut_model_path /home/joanna_cheng/workspace/gaussian-splatting-gsplats/output/teatime/point_cloud/iteration_1500_lvl_$feature_level/LUT.pt \
#     --json_dir /scratch/joanna_cheng/lerf_ovs/label/teatime
