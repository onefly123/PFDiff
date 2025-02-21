#!/bin/bash

DATASET=Church
METHOD=DDIM    # DDIM or DDPM or DDPM_SIGMA

GPU_INDEX=0
SKIP_TYPE=uniform    # uniform or quad

ETA=0.0
if [ "$METHOD" = "DDIM" ]; then
    ETA=0.0
    SAMPLE_TYPE=generalized
elif [ "$METHOD" = "DDPM" ]; then
    ETA=1.0
    SAMPLE_TYPE=generalized
elif [ "$METHOD" = "DDPM_SIGMA" ]; then
    SAMPLE_TYPE=ddpm_noisy
fi

# Array of steps
steps=(5 10)
# Array of algorithms "PFDiff_2_2" "PFDiff_3_1" "PFDiff_3_2" "PFDiff_3_3"
algorithms=("origin")

for STEP in "${steps[@]}"; do
    for ALGORITHM in "${algorithms[@]}"; do
        if [ "$ALGORITHM" = "origin" ]; then
            T_STEP=$STEP
        elif [ "$ALGORITHM" = "PFDiff_1" ]; then
            T_STEP=$(($STEP * 2 - 1))
        elif [[ "$ALGORITHM" == PFDiff_2* ]]; then
            T_STEP=$(($STEP * 3 - 2))
        elif [[ "$ALGORITHM" == PFDiff_3* ]]; then
            T_STEP=$(($STEP * 4 - 3))
        fi

        CUDA_VISIBLE_DEVICES=$GPU_INDEX python main.py --config church.yml --exp ./exp/$DATASET-$METHOD-$STEP-$ALGORITHM-$SKIP_TYPE-get_imgs/ --use_pretrained --sample --fid --timesteps $T_STEP --eta $ETA --ni --doc logs \
        --skip_type $SKIP_TYPE \
        --sample_type $SAMPLE_TYPE \
        --algorithm $ALGORITHM

        # python -m pytorch_fid ./exp/$DATASET-$METHOD-$STEP-$ALGORITHM-$SKIP_TYPE-get_imgs/image_samples/images ./exp/npz/church/church_fid.npz --device cuda:$GPU_INDEX

    done
done

