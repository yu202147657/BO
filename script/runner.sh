#!/bin/bash
set -e
export PYTHONPATH=.
export CUDA_VISIBLE_DEVICES=0,1,2,3

for dataset in amazon wiki
do
  for sparsity in 500 2000 5000
  do
    for class_imbalance in 0.5 0.75 0.85
    do
        python main.py \
        -c taa/confs/bert_${dataset}_example.yaml \
        --abspath '/home/jovyan/examples/examples/pytorch/MPhil_BO/taa/Data' \
        --class_imbalance $class_imbalance \
        --sparsity $sparsity \
        --n_aug 8 \
        --alpha 0.05
    done
  done
done