#!/bin/bash


CUDA_INDEX=2
STEPS=(100)
# ALGORITHMS=("origin" "PFDiff_1" "PFDiff_2_1" "PFDiff_2_2" "PFDiff_3_1" "PFDiff_3_2" "PFDiff_3_3")
ALGORITHMS=("origin")
BATCH_SIZE=64

# Outer loop iterates over STEP values
for STEP in "${STEPS[@]}"; do
  #  Inner loop iterates over ALGORITHM values
  for ALGORITHM in "${ALGORITHMS[@]}"; do
    # Calculate T_STEP based on ALGORITHM and STEP values
    if [ "$ALGORITHM" = "origin" ]; then
        T_STEP=$STEP
    elif [ "$ALGORITHM" = "PFDiff_1" ]; then
        T_STEP=$(($STEP * 2 - 1))
    elif [ "$ALGORITHM" = "PFDiff_2_1" ] || [ "$ALGORITHM" = "PFDiff_2_2" ]; then
        T_STEP=$(($STEP * 3 - 2))
    elif [ "$ALGORITHM" = "PFDiff_3_1" ] || [ "$ALGORITHM" = "PFDiff_3_2" ] || [ "$ALGORITHM" = "PFDiff_3_3" ]; then
        T_STEP=$(($STEP * 4 - 3))
    fi

    MODEL_FLAGS="--attention_resolutions 32,16,8 --class_cond True --diffusion_steps 1000 --dropout 0.1 --image_size 64 --learn_sigma True --noise_schedule cosine --num_channels 192 --num_head_channels 64 --num_res_blocks 3 --resblock_updown True --use_new_attention_order True --use_fp16 True --use_scale_shift_norm True"
    SAMPLE_FLAGS="--batch_size $BATCH_SIZE --num_samples 50000 --timestep_respacing ddim$T_STEP --use_ddim True"

    PYTHONPATH="" \
    CUDA_VISIBLE_DEVICES=$CUDA_INDEX \
    python ./scripts/classifier_sample.py $MODEL_FLAGS --classifier_scale 1.0 \
    --classifier_path ./models/64x64_classifier.pt --classifier_depth 4 \
    --model_path ./models/64x64_diffusion.pt $SAMPLE_FLAGS \
    --save_dir ./exps/$ALGORITHM-$STEP \
    --MASTER_PORT "1236" \
    --classifier_scale 1.0 \
    --algorithm $ALGORITHM

    CUDA_VISIBLE_DEVICES=$CUDA_INDEX \
    python ./evaluations/evaluator.py ./fid_stats/VIRTUAL_imagenet64_labeled.npz \
    ./exps/$ALGORITHM-$STEP/samples_50000x64x64x3.npz
  done
done


