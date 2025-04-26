#!/bin/bash

# Base directories
BASE_DIR="/scratch/joanna_cheng/scannetpp_3dgs"
GT_BASE_DIR="/scratch/joanna_cheng/scannetpp_v1_val_subset"
LABEL_NAME_PATH="${GT_BASE_DIR}/scannetpp_semseg_top100.txt"
FEATURE_LEVEL=3
ITERATION=1500

# Check if directories exist
if [ ! -d "$BASE_DIR" ]; then
    echo "Error: Base directory $BASE_DIR does not exist"
    exit 1
fi

if [ ! -d "$GT_BASE_DIR" ]; then
    echo "Error: GT directory $GT_BASE_DIR does not exist"
    exit 1
fi

# Check if label file exists
if [ ! -f "$LABEL_NAME_PATH" ]; then
    echo "Error: Label name file $LABEL_NAME_PATH does not exist"
    exit 1
fi

# Specify the scene to remove
REMOVED_SCENE="38d58a7a31"

# Find all scene directories
for SCENE_DIR in "$BASE_DIR"/*/; do
    # Extract just the scene name
    SCENE=$(basename "$SCENE_DIR")

    # Check if the current scene is in the list of removed scenes
    for REMOVED in $REMOVED_SCENES; do
        if [ "$SCENE" = "$REMOVED" ]; then
            echo "Skipping manually removed scene: $SCENE" >> "$LOG_FILE"
            continue 2  # Skip to the next SCENE_DIR in outer loop
        fi
    done
    
    echo "==============================================="
    echo "Evaluating scene: $SCENE"
    echo "==============================================="
    
    # Check if required directories and files exist
    GT_NPY_DIR="${GT_BASE_DIR}/${SCENE}/dslr/segmentation_2d"
    LOGIT_NPY_DIR="${BASE_DIR}/${SCENE}/test/ours_${ITERATION}_lvl_${FEATURE_LEVEL}/logit"
    OUTPUT_DIR="${BASE_DIR}/${SCENE}/test/ours_${ITERATION}_lvl_${FEATURE_LEVEL}/eval_res"
    LUT_MODEL_PATH="${BASE_DIR}/${SCENE}/point_cloud/iteration_${ITERATION}_lvl_${FEATURE_LEVEL}/LUT.pt"
    LINEAR_SVM_DIR="/scratch/joanna_cheng/GOI_linearSVM/${SCENE}"
    
    # Create output directory if it doesn't exist
    mkdir -p "$OUTPUT_DIR"
    
    # Run evaluation
    echo "Running evaluation for scene $SCENE"
    python eval_semantics_scannetpp_res.py \
        --label_name_path "$LABEL_NAME_PATH" \
        --iteration "$ITERATION" \
        --gt_npy_dir "$GT_NPY_DIR" \
        --logit_npy_dir "$LOGIT_NPY_DIR" \
        --output_dir "$OUTPUT_DIR" \
        --lut_model_path "$LUT_MODEL_PATH" \
        --linearSVM_dir "$LINEAR_SVM_DIR"
    
    echo "Evaluation completed for scene $SCENE"
done

echo "All evaluations completed!"