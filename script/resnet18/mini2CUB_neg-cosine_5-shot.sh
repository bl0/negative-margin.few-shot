#!/bin/bash

set -x
set -e

python main.py --config configs/mini2CUB/exp_neg-cosine.yml \
    --supp 5-shot \
    train.adam_params.weight_decay 5e-5 \
    method.metric_params.margin -0.05
