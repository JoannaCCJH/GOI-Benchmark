#!/bin/bash

# Directory containing all scenes
BASE_DIR="/scratch/joanna_cheng/scannetpp_3dgs"
DATA_DIR="/scratch/joanna_cheng/scannetpp_v1_val_subset"

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
    
    # Render semantics (commented out as per your update)
    # echo "Rendering scene: $SCENE"
    # python render_semantics.py --iteration 1500 --skip_train -m "$BASE_DIR/$SCENE" --feature_level 3 --eval
    
    echo "Completed processing scene: $SCENE"
    echo "----------------------------------------"
done

echo "All scenes have been processed"






# SCENE=38d58a7a31
# render semantics
# python render_semantics.py --iteration 1500 --skip_train -m /scratch/joanna_cheng/scannetpp_3dgs/$SCENE --feature_level 3 --eval
# python train.py --iteration 1500 -m /scratch/joanna_cheng/scannetpp_3dgs/$SCENE -s /scratch/joanna_cheng/scannetpp_v1_val_subset/$SCENE --feature_level 3 --eval


# script for langsplat scenes
# python train.py --iteration 1500 -m /home/joanna_cheng/workspace/gaussian-splatting-gsplats/output/teatime -s /scratch/joanna_cheng/lerf_ovs/teatime --feature_level 3
# python render_semantics.py --iteration 1500 --skip_train -m /home/joanna_cheng/workspace/gaussian-splatting-gsplats/output/teatime --feature_level 3 --eval

