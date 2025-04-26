feature_level=3
SCENE=38d58a7a31

python eval_semantics_scannetpp_res.py \
    --label_name_path /scratch/joanna_cheng/scannetpp_v1_val_subset/scannetpp_semseg_top100.txt \
    --iteration 1500 \
    --gt_npy_dir /scratch/joanna_cheng/scannetpp_v1_val_subset/$SCENE/dslr/segmentation_2d \
    --logit_npy_dir /scratch/joanna_cheng/scannetpp_3dgs/$SCENE/test/ours_1500_lvl_$feature_level/logit \
    --output_dir /scratch/joanna_cheng/scannetpp_3dgs/$SCENE/test/ours_1500_lvl_$feature_level/eval_res \
    --lut_model_path /scratch/joanna_cheng/scannetpp_3dgs/$SCENE/point_cloud/iteration_1500_lvl_$feature_level/LUT.pt \
    --linearSVM_dir /scratch/joanna_cheng/GOI_linearSVM/$SCENE