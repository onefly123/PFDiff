#!/bin/bash


CUDA_VISIBLE_DEVICES=1 python evaluator.py ../fid_stats/VIRTUAL_imagenet64_labeled.npz ../exps/origin-250/samples_50000x64x64x3.npz
