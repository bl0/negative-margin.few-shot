#!/bin/bash

set -x
set -e

python main.py --config configs/mini2CUB/exp_neg-softmax.yml \
    --supp 5-shot \
    train.adam_params.weight_decay 1e-6 \
    method.metric_params.margin -0.005
