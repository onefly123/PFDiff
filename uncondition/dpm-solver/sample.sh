#!/bin/bash

devices="0"
eps="1e-3"
skip="time_quadratic"
method="singlestep_fixed"
order="1" 
dir="experiments/cifar10_ddpmpp_deep_continuous_steps"

# Define the list of STEP values
steps="5 6 7 8 9"

# Define the list of ALGORITHM values
algorithms="PFDiff_1 PFDiff_2_1 PFDiff_2_2 PFDiff_3_1 PFDiff_3_2 PFDiff_3_3"

# Loop over STEP values
for STEP in $steps; do
    # Loop over ALGORITHM values
    for ALGORITHM in $algorithms; do
        # Calculate T_STEP based on ALGORITHM and STEP
        if [ "$ALGORITHM" = "origin" ]; then
            T_STEP=$STEP
        elif [ "$ALGORITHM" = "PFDiff_1" ]; then
            T_STEP=$(($STEP * 2 - 1 * $order))
        elif [ "$ALGORITHM" = "PFDiff_2_1" ] || [ "$ALGORITHM" = "PFDiff_2_2" ]; then
            T_STEP=$(($STEP * 3 - 2 * $order))
        elif [ "$ALGORITHM" = "PFDiff_3_1" ] || [ "$ALGORITHM" = "PFDiff_3_2" ] || [ "$ALGORITHM" = "PFDiff_3_3" ]; then
            T_STEP=$(($STEP * 4 - 3 * $order))
        fi

        # Execute the command with the calculated T_STEP and current ALGORITHM
        CUDA_VISIBLE_DEVICES=$devices python main.py --config "configs/vp/cifar10_ddpmpp_deep_continuous.py" \
        --mode "eval" \
        --workdir $dir \
        --config.sampling.eps=$eps --config.sampling.method="dpm_solver" \
        --config.sampling.steps=$T_STEP --config.sampling.skip_type=$skip \
        --config.sampling.dpm_solver_order=$order \
        --config.sampling.dpm_solver_method=$method \
        --config.eval.batch_size=256  \
        --config.eval.enable_sampling \
        --config.model.algorithm=$ALGORITHM \
        --config.model.mydir="$STEP-$ALGORITHM-$order-pp"
    done
done

