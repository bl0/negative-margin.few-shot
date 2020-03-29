import argparse
from os.path import basename, join, splitext

from yacs.config import CfgNode

from .utils import check_dir

# define default configs
cfg = CfgNode(dict(
    misc=dict(
        rng_seed=0,
        evaluate=False,  # only evaluate
        # resume from checkpoint for train and test(extract features)
        # if empty string, no resume
        # if 'last', resume from last checkpoint dir
        # if 'best', resume from best checkpoint dir
        # if 'epoch_n', resume from the n epoch checkpoint
        # if valid path, resume from the specified checkpoint
        resume="",
        num_workers=8,
    ),
    dataset=dict(
        type="CUB",
        base_file="data/CUB/base.json",  # training data on the base classes
        base_val_file="",  # validataion data on the base classes, if is None, not used
        val_file="data/CUB/val.json",
        novel_file="data/CUB/novel.json",
        num_class=200
    ),
    train=dict(
        optim="Adam",
        sgd_params=dict(
            lr=1.0,
            momentum=0.9,
            weight_decay=0.0005,
            nesterov=True,
        ),
        adam_params=dict(
            lr=3e-3,
            weight_decay=0.0
        ),
        lr_scheduler="warmup_cosine",
        warmup_params=dict(
            multiplier=16,
            epoch=50,
        ),
        batch_size=256,
        print_freq=10,
        stop_epoch=200,
        # for dropblock
        drop_rate=0.0,
        dropblock_size=5,
    ),
    val=dict(
        num_episode=100,  # number of task to validate
        freq=20,  # valid frequence, if <= 0, no val
        n_support=5,  # validate 5 shot accuracy
    ),
    test=dict(
        finetune_params=dict(
            iter=100,  # iter number of each finetune for each task
            optim="SGD",
            sgd_params=dict(
                lr=0.01,
                momentum=0.9,
                dampening=0.9,
                weight_decay=0.001,
            ),
        ),
        split="novel",  # 'base/val/novel'
        num_episode=600,  # number of task to run for test
        n_way=5,  # number of classes to test
        n_support=[1, 5],  # both test one shot and five shot
        n_query=16,  # number of queries to test per task
        batch_size=1024,  # used when extracting feature
    ),
    method=dict(
        metric='cosineface',
        metric_params=dict(
            scale_factor=30.0,
            margin=0.0,
        ),
        metric_params_test=dict(
            scale_factor=5,
            margin=0.0,
        ),
        backbone="resnet18",  # Conv4 / resnet12 / resnet18 / WideResNet28_10,
        image_size=224,  # 84 / 84 / 224 / 80, corresponding to the backbone above
    ),
))

# load from file and overrided by command line arguments
parser = argparse.ArgumentParser("few-shot script")
parser.add_argument('--config', type=str, required=True,
                    help='to set the parameters', )
parser.add_argument('--supp', type=str, default='',
                    help='supplement string for output dir')
args, unknown = parser.parse_known_args()
cfg.merge_from_file(args.config)
cfg.merge_from_list(unknown)

# inference some folder dir
exp_name = splitext(basename(args.config))[0]
if len(args.supp):
    exp_name = f"{exp_name}_{args.supp}"
# used to log results
cfg.misc.output_dir = check_dir(join('./output', cfg.dataset.type, exp_name))
cfg.misc.checkpoint_dir = check_dir(join(cfg.misc.output_dir, 'checkpoints'))
cfg.misc.log_dir = check_dir(join(cfg.misc.output_dir, 'tensorboard'))

cfg.freeze()
