#!/bin/bash

python3 ./unimof/infer.py /workspace/Uni-MOF/data \
    --user-dir ./unimof \
    --task-name hmof \
    --valid-subset test \
    --task unimof_v2 --loss mof_v2_mse --arch unimof_v2 \
    --num-classes 1 --batch-size 10 \
    --path ./logs_finetune/qmof_finetune_v2/checkpoint_last.pt \
    --results-path /workspace/Uni-MOF/дообучение/infer_out_v2 \
    --fp16 --remove-hydrogen --log-format simple \
    --num-workers 0