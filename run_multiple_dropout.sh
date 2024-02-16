#!/bin/bash

# ドロップアウト率の値のリスト
dropout_values=(0.2 0.3 0.4 0.5 0.0)

# 各ドロップアウト率に対して実行
for dropout_value in "${dropout_values[@]}"; do
    echo "Running with dropout value: $dropout_value"
    python train.py --dataset kitti --log_dir log_kitti --batch_size 16 --max_epoch 50 --dropout $dropout_value
    echo "Finished running with dropout value: $dropout_value"
done