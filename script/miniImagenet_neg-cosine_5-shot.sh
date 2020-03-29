#!/bin/bash

set -x
set -e

python main.py --config configs/miniImagenet/exp_neg-cosine.yml \
    --supp 5-shot \
    train.adam_params.weight_decay 1e-4 \
    method.metric_params.margin -0.02 \
    train.drop_rate 0.2 \
    train.dropblock_size 3