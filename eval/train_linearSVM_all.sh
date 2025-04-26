#!/bin/bash

# Base directories
BASE_DIR="/scratch/joanna_cheng/scannetpp_3dgs"
GT_BASE_DIR="/scratch/joanna_cheng/scannetpp_v1_val_subset"
LABEL_NAME_PATH="${GT_BASE_DIR}/scannetpp_semseg_top100.txt"
FEATURE_LEVEL=3
ITERATION=1500
OUTPUT_BASE_DIR="/scratch/joanna_cheng/GOI_linearSVM"

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

# Ensure output base directory exists
mkdir -p "$OUTPUT_BASE_DIR"

# Initialize log file
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="${OUTPUT_BASE_DIR}/linearSVM_training_${TIMESTAMP}.log"
touch "$LOG_FILE"

# Counter for successful and failed scenes
TOTAL_SCENES=0
SUCCESS_SCENES=0
FAILED_SCENES=0

# List of removed scenes
REMOVED_SCENES="38d58a7a31 0d2ee665be 09c1414f1b"

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
    
    # Skip if not a valid scene directory
    GT_NPY_DIR="${GT_BASE_DIR}/${SCENE}/dslr/segmentation_2d"
    if [ ! -d "$GT_NPY_DIR" ]; then
        continue
    fi
    
    TOTAL_SCENES=$((TOTAL_SCENES + 1))
    
    # Construct paths
    LOGIT_NPY_DIR="${BASE_DIR}/${SCENE}/test/ours_${ITERATION}_lvl_${FEATURE_LEVEL}/logit"
    OUTPUT_DIR="${OUTPUT_BASE_DIR}/${SCENE}"
    LUT_MODEL_PATH="${BASE_DIR}/${SCENE}/point_cloud/iteration_${ITERATION}_lvl_${FEATURE_LEVEL}/LUT.pt"
    
    # Create output directory for this scene
    mkdir -p "$OUTPUT_DIR"
    
    echo "===============================================" | tee -a "$LOG_FILE"
    echo "Training LinearSVM for scene: $SCENE" | tee -a "$LOG_FILE"
    echo "===============================================" | tee -a "$LOG_FILE"
    
    # Run training
    python train_linearSVM.py \
        --scene "$SCENE" \
        --label_name_path "$LABEL_NAME_PATH" \
        --gt_npy_dir "$GT_NPY_DIR" \
        --logit_npy_dir "$LOGIT_NPY_DIR" \
        --model_output_dir "$OUTPUT_DIR" \
        --lut_model_path "$LUT_MODEL_PATH" \
        2>&1 | tee -a "$LOG_FILE"
    
    # Check training status
    if [ $? -eq 0 ]; then
        SUCCESS_SCENES=$((SUCCESS_SCENES + 1))
        echo "Training completed successfully for scene $SCENE" | tee -a "$LOG_FILE"
    else
        FAILED_SCENES=$((FAILED_SCENES + 1))
        echo "Training FAILED for scene $SCENE" | tee -a "$LOG_FILE"
    fi
done