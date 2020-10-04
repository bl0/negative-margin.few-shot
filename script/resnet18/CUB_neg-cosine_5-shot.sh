#!/bin/bash

set -x
set -e

python main.py --config configs/CUB/exp_neg-cosine.yml \
    --supp 5-shot \
    train.adam_params.weight_decay 1e-3 \
    method.metric_params.margin -0.01 \
    train.drop_rate 0.1 \
    train.dropblock_size 5
