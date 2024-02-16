#!/bin/bash

# 各ドロップアウト率に対して実行
echo "Running with dropout value: $dropout_value"
python train.py --dataset kitti --log_dir log_kitti --batch_size 16 --max_epoch 50 --dropout 0.0 --overwrite
python eval.py --dataset kitti --dump_dir /home/kalzansundaram/research/eval/eval_kitti --cluster_sampling seed_fps --use_3d_nms --use_cls_nms --per_class_proposal --batch_size 16 --dropout_rate 0.0
echo "Finished running with dropout value: $dropout_value"