#!/bin/bash

# Directory containing all scenes
BASE_DIR="/scratch/joanna_cheng/scannetpp_3dgs"

# Check if the base directory exists
if [ ! -d "$BASE_DIR" ]; then
    echo "Error: Base directory $BASE_DIR does not exist"
    exit 1
fi

# Find all directories in the base directory
# This assumes each directory in BASE_DIR is a scene
for SCENE_DIR in "$BASE_DIR"/*/; do
    # Extract just the scene name (the directory name)
    SCENE=$(basename "$SCENE_DIR")
    
    echo "Processing scene: $SCENE"
    
    # Run the render command
    python render_semantics.py --iteration 1500 --skip_train -m "$BASE_DIR/$SCENE" --feature_level 3 --eval
    
    echo "Completed rendering scene: $SCENE"
    echo "----------------------------------------"
done

echo "All scenes have been rendered"