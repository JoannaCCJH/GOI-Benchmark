#!/bin/bash

# Directory containing all scenes
BASE_DIR="/scratch/joanna_cheng/scannet_3dgs"
DATA_DIR="/scratch/joanna_cheng/scannet_2d_language_features"

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

# Find all directories in the base directory
# This assumes each directory in BASE_DIR is a scene
for SCENE_DIR in "$BASE_DIR"/*/; do
    # Extract just the scene name (the directory name)
    SCENE=$(basename "$SCENE_DIR")
    
    echo "Processing scene: $SCENE"
    
    # Train semantics
    echo "Training scene: $SCENE"
    python train.py --iteration 1500 -m "$BASE_DIR/$SCENE" -s "$DATA_DIR/$SCENE" --feature_level 3 --eval
    
    echo "Completed processing scene: $SCENE"
    echo "----------------------------------------"
done

echo "All scenes have been processed"

