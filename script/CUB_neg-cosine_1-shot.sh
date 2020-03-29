#!/bin/bash

set -x
set -e

python main.py --config configs/CUB/exp_neg-cosine.yml \
    --supp 1-shot \
    train.adam_params.weight_decay 5e-4 \
    method.metric_params.margin -0.01 \
    train.drop_rate 0.1 \
    train.dropblock_size 5
