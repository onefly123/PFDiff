
#!/bin/bash



DEVICES=0
# "PFDiff_3_3"  "origin" "PFDiff_1" "PFDiff_2_1" "PFDiff_2_2" "PFDiff_3_1" "PFDiff_3_2"
ALGORITHMS=("PFDiff_3_3")

for ALGORITHM in "${ALGORITHMS[@]}"; do

# echo ./exps/ddim4-$ALGORITHM-256_2.0
CUDA_VISIBLE_DEVICES=$DEVICES python -m pytorch_fid ./fid_stats/VIRTUAL_imagenet256_labeled.npz  ./exps/ddim4-$ALGORITHM-256_2.0/imgs --device cuda:0
CUDA_VISIBLE_DEVICES=$DEVICES python inception_score.py --image_path ./exps/ddim4-$ALGORITHM-256_2.0/imgs --device cuda:0

done

