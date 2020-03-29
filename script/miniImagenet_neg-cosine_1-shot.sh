#!/bin/bash

set -x
set -e

python main.py --config configs/miniImagenet/exp_neg-cosine.yml \
    --supp 1-shot \
    train.adam_params.weight_decay 5e-5 \
    method.metric_params.margin -0.05 \
    train.drop_rate 0.2 \
    train.dropblock_size 3