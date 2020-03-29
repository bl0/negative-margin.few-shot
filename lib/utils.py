import glob
import os
import sys

import numpy as np
import torch
# noinspection PyProtectedMember
from torch.optim.lr_scheduler import _LRScheduler


def check_dir(folder):
    if not os.path.isdir(folder):
        os.makedirs(folder)
    return folder


# noinspection PyAttributeOutsideInit
class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target):
    """Computes the precision@k for the specified values of k"""
    with torch.no_grad():
        pred = output.argmax(dim=1)
        acc = (pred == target).float().mean()
        return acc


def one_hot(y, num_class):
    return torch.zeros((len(y), num_class)).to(y.device).scatter_(1, y.unsqueeze(1), 1)


def get_few_shot_label(n_way, n_data_per_way):
    return torch.from_numpy(np.repeat(range(n_way), n_data_per_way))


# noinspection PyAttributeOutsideInit
class GradualWarmupScheduler(_LRScheduler):
    """ Gradually warm-up(increasing) learning rate in optimizer.
      Proposed in 'Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour'.
      Args:
          optimizer (Optimizer): Wrapped optimizer.
          multiplier: target learning rate = base lr * multiplier
          total_epoch: target learning rate is reached at total_epoch, gradually
          after_scheduler: after target_epoch, use this scheduler(eg. ReduceLROnPlateau)
      """

    def __init__(self, optimizer, multiplier, total_epoch, after_scheduler):
        self.multiplier = multiplier
        if self.multiplier <= 1.:
            raise ValueError('multiplier should be greater than 1.')
        self.total_epoch = total_epoch
        self.after_scheduler = after_scheduler
        self.finished = False
        super().__init__(optimizer)

    def get_lr(self):
        return [base_lr / self.multiplier * ((self.multiplier - 1.) * self.last_epoch / self.total_epoch + 1.)
                for base_lr in self.base_lrs]

    def step(self, epoch):
        if epoch > self.total_epoch:
            self.after_scheduler.step(epoch - self.total_epoch)
        else:
            super(GradualWarmupScheduler, self).step(epoch)


def get_assigned_file(checkpoint_dir, epoch):
    assign_file = os.path.join(checkpoint_dir, f'{epoch}.tar')
    return assign_file


def get_resume_file(checkpoint_dir):
    filelist = glob.glob(os.path.join(checkpoint_dir, '*.tar'))
    if len(filelist) == 0:
        return None

    filelist = [x for x in filelist if os.path.basename(x) != 'best_model.tar']
    epochs = np.array(
        [int(os.path.splitext(os.path.basename(x))[0]) for x in filelist])
    max_epoch = np.max(epochs)
    resume_file = os.path.join(checkpoint_dir, f'{max_epoch}.tar')
    return resume_file


def get_best_file(checkpoint_dir):
    best_file = os.path.join(checkpoint_dir, 'best_model.tar')
    if os.path.isfile(best_file):
        return best_file
    else:
        return get_resume_file(checkpoint_dir)


class Logger(object):
    def __init__(self, log_file):
        self.terminal = sys.stdout
        if os.path.exists(log_file):
            print(f"warning: {log_file} already exists and will be overwritten")
        self.log = open(log_file, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def close(self):
        self.log.close()
