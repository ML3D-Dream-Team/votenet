#!/bin/bash

# ドロップアウト率の値のリスト
dropout_values=(0.3 0.4 0.5)

# 各ドロップアウト率に対して実行
for dropout_value in "${dropout_values[@]}"; do
    echo "Running with dropout value: $dropout_value"
    python eval.py --dataset kitti --dump_dir /home/kalzansundaram/research/eval/eval_kitti --cluster_sampling seed_fps --use_3d_nms --use_cls_nms --per_class_proposal --batch_size 16 --dropout_rate $dropout_value
    echo "Finished running with dropout value: $dropout_value"
done