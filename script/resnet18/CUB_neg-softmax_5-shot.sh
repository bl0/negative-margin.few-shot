#!/bin/bash

set -x
set -e

python main.py --config configs/CUB/exp_neg-softmax.yml \
    --supp 5-shot \
    train.adam_params.weight_decay 3e-3 \
    method.metric_params.margin -0.005
