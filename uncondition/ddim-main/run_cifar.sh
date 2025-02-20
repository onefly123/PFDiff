#!/bin/bash

DATASET=Cifar
METHOD=DDIM    # DDIM or DDPM or DDPM_SIGMA

GPU_INDEX=0
SKIP_TYPE=quad    # uniform or quad

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
steps=(4 6 8 10 15 20)
# Array of algorithms
algorithms=("origin" "PFDiff_1" "PFDiff_2_1" "PFDiff_2_2" "PFDiff_3_1" "PFDiff_3_2" "PFDiff_3_3")

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

        CUDA_VISIBLE_DEVICES=$GPU_INDEX python main.py --config cifar10.yml --exp ./exp/$DATASET-$METHOD-$STEP-$ALGORITHM-$SKIP_TYPE/ --use_pretrained --sample --fid --timesteps $T_STEP --eta $ETA --ni --doc logs \
        --skip_type $SKIP_TYPE \
        --sample_type $SAMPLE_TYPE \
        --algorithm $ALGORITHM

        # python -m pytorch_fid ./exp/$DATASET-$METHOD-$STEP-$ALGORITHM-$SKIP_TYPE/image_samples/images ./exp/npz/cifar10/cifar10.train.npz --device cuda:$GPU_INDEX

    done
done

