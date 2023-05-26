#!/bin/bash
set -e
export PYTHONPATH=.
export CUDA_VISIBLE_DEVICES=0,1,2,3

for dataset in amazon
do
  for sparsity in 500 
  do
    for class_imbalance in 0.5
    do
        python /home/yfn21/ondemand/data/sys/myjobs/projects/default/3/MPhil_BO/main.py \
        -c taa/confs/bert_${dataset}_example.yaml \
        --abspath '/home/yfn21/ondemand/data/sys/myjobs/projects/default/3/MPhil_BO' \
        --class_imbalance $class_imbalance \
        --sparsity $sparsity \
        --n_aug 8 \
        --alpha 0.05 \
        --num-search 1
    done
  done
done