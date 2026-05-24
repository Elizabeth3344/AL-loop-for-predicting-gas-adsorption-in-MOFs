#!/bin/bash

python3 -m torch.distributed.launch \
    --nproc_per_node=1 \
    --master_port=10086 \
    $(which unicore-train) ./data/finetune_v2 \
    --user-dir ./unimof \
    --task-name hmof \
    --train-subset train \
    --valid-subset valid \
    --task unimof_v2 --loss mof_v2_mse --arch unimof_v2 \
    --optimizer adam --adam-betas '(0.9, 0.99)' --adam-eps 1e-6 --clip-norm 1.0 \
    --lr-scheduler polynomial_decay --lr 1e-5 --warmup-ratio 0.1 \
    --max-epoch 30 --batch-size 8 \
    --update-freq 1 --seed 1 \
    --fp16 --fp16-init-scale 4 --fp16-scale-window 256 \
    --num-classes 1 --pooler-dropout 0.1 \
    --finetune-mol-model ./weights/unimof_hMOF.pt \
    --log-interval 10 --log-format simple \
    --validate-interval 1 --remove-hydrogen \
    --save-dir ./logs_finetune/qmof_finetune_v2 \
    --save-interval 1 \
    --num-workers 0