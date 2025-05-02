#!/bin/bash

# Directory containing all scenes
BASE_DIR="/scratch/joanna_cheng/matterport3d_region_mini_test_set_suite/mcmc_3dgs"
DATA_DIR="/scratch/joanna_cheng/matterport3d_region_mini_test_set_suite/original_data"

# Check if the base directory exists
if [ ! -d "$BASE_DIR" ]; then
    echo "Error: Base directory $BASE_DIR does not exist"
    exit 1
fi

# Check if the data directory exists
if [ ! -d "$DATA_DIR" ]; then
    echo "Error: Data directory $DATA_DIR does not exist"
    exit 1
fi

# REMOVED_SCENES="scene0011_01 scene0011_00"

# Find all directories in the base directory
# This assumes each directory in BASE_DIR is a scene
for SCENE_DIR in "$BASE_DIR"/*/; do
    # Extract just the scene name (the directory name)
    SCENE=$(basename "$SCENE_DIR")
    
    echo "Processing scene: $SCENE"

    # Check if the corresponding data directory exists
    if [ ! -d "$DATA_DIR/$SCENE" ]; then
        echo "Skipping scene $SCENE: Data directory $DATA_DIR/$SCENE does not exist"
        echo "----------------------------------------"
        continue
    fi

    # Check if the current scene is in the list of removed scenes
    # for REMOVED in $REMOVED_SCENES; do
    #     if [ "$SCENE" = "$REMOVED" ]; then
    #         echo "Skipping manually removed scene: $SCENE" >> "$LOG_FILE"
    #         continue 2  # Skip to the next SCENE_DIR in outer loop
    #     fi
    # done
    
    # Train semantics
    echo "Training scene: $SCENE"
    python train.py --iteration 1500 -m "$BASE_DIR/$SCENE" -s "$DATA_DIR/$SCENE" --dataset_type matterport3d --feature_level 0 --eval
    
    echo "Completed processing scene: $SCENE"
    echo "----------------------------------------"
done

echo "All scenes have been processed"

# SCENE=scene0011_00
# render semantics
# python train.py --iteration 1500 -m /scratch/joanna_cheng/scannet_3dgs/scene0651_02 -s /scratch/joanna_cheng/scannet_language_feature/scene0651_02 --feature_level 3 --eval
# python render_semantics.py --iteration 1500 --skip_train -m /scratch/joanna_cheng/scannet_3dgs/scene0011_00 --feature_level 3 --eval