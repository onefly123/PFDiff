#!/bin/bash


# Define the algorithm list
# ALGORITHMS=("DPM_Solver" "PFDiff_1" "PFDiff_2_1" "PFDiff_2_2" "PFDiff_3_1" "PFDiff_3_2" "PFDiff_3_3")  # "origin" "PLMS"
ALGORITHMS=("DPM_Solver")
# Define the step size list
STEPS=(15)
# STEPS=(3 4 5 7 9 14 19)
GPU_INDEX=0

# Iterate through the algorithms and step sizes
for ALGORITHM in "${ALGORITHMS[@]}"; do
    for STEP in "${STEPS[@]}"; do
        if [ "$ALGORITHM"  =  "PLMS" ]; then
            STEP=$(($STEP-1))
        fi
        if [ "$ALGORITHM" = "origin" ]; then
            T_STEP=$STEP
        elif [ "$ALGORITHM" = "PFDiff_1" ]; then
            T_STEP=$(($STEP * 2 - 1))
        elif [ "$ALGORITHM" = "PFDiff_2_1" ] || [ "$ALGORITHM" = "PFDiff_2_2" ]; then
            T_STEP=$(($STEP * 3 - 2))
        elif [ "$ALGORITHM" = "PFDiff_3_1" ] || [ "$ALGORITHM" = "PFDiff_3_2" ] || [ "$ALGORITHM" = "PFDiff_3_3" ]; then
            T_STEP=$(($STEP * 4 - 3))
        else
            T_STEP=$STEP  # For algorithms not on the list, the default T_STEP is set to STEP.
        fi

        # Execute the command, note that the example paths and filenames are used here, please adjust according to the actual situation.
        CUDA_VISIBLE_DEVICES=$GPU_INDEX \
        python scripts/txt2img_save_img.py  \
        --ckpt ./models/ldm/stable-diffusion-v1/sd-v1-4.ckpt \
        --outdir outputs/new/${ALGORITHM}-${STEP}-uniform_2m/ \
        --config ./configs/stable-diffusion/v1-inference_coco.yaml \
        --n_samples 6 \
        --num_sample 10000 \
        --ddim_steps $T_STEP \
        --algorithm $ALGORITHM \
        --timesteps "" \
        --scale 1.5
    done
done




