import numpy as np
import torch
import torch.nn as nn

from lib import metric
from lib.utils import get_few_shot_label


def get_linear_clf(metric_type, feature_dimension, num_classes, scale_factor=None, margin=None):
    if metric_type == 'softmax':
        classifier = nn.Linear(feature_dimension, num_classes)
    elif metric_type == 'cosine':
        classifier = metric.CosineSimilarity(feature_dimension, num_classes, scale_factor=scale_factor)
    elif metric_type == 'cosineface':
        classifier = metric.AddMarginProduct(feature_dimension, num_classes, scale_factor=scale_factor, margin=margin)
    elif metric_type == 'neg-softmax':
        classifier = metric.SoftmaxMargin(feature_dimension, num_classes, scale_factor=scale_factor, margin=margin)
    else:
        raise ValueError(f'Unknown metric type: "{metric_type}"')
    return classifier


class BaselineTrain(nn.Module):

    def __init__(self, model_func, num_class, metric_type, metric_params):
        super(BaselineTrain, self).__init__()
        self.feature = model_func()
        self.metric_type = metric_type

        self.classifier = get_linear_clf(metric_type, self.feature.final_feat_dim, num_class, **metric_params)

    def forward(self, x, y=None):
        feature = self.feature.forward(x)
        if self.metric_type in ['cosineface', 'neg-softmax']:
            scores = self.classifier.forward(feature, y)
        else:
            scores = self.classifier.forward(feature)
        return scores


class BaselineFinetune(nn.Module):
    def __init__(self, n_way, n_support, metric_type, metric_params, finetune_params):
        super(BaselineFinetune, self).__init__()
        self.n_way = n_way
        self.n_support = n_support
        self.metric_type = metric_type
        self.metric_params = metric_params
        self.finetune_params = finetune_params

    def forward(self, z_all):
        z_all = z_all.cuda()
        z_support = z_all[:, :self.n_support, :]
        z_query = z_all[:, self.n_support:, :]

        feature_dim = z_support.shape[-1]
        z_support = z_support.contiguous().view(-1, feature_dim)
        z_query = z_query.contiguous().view(-1, feature_dim)
        y_support = get_few_shot_label(self.n_way, self.n_support).cuda()
        linear_clf = get_linear_clf(self.metric_type, feature_dim, self.n_way, **self.metric_params).cuda()

        if self.finetune_params.optim == "SGD":
            finetune_optimizer = torch.optim.SGD(linear_clf.parameters(), **self.finetune_params.sgd_params)
        else:
            raise ValueError(f"finetune optimzation not supported: {self.finetune_params.optim}")

        loss_function = nn.CrossEntropyLoss().cuda()

        batch_size = 4
        support_size = self.n_way * self.n_support
        for _ in range(self.finetune_params.iter):
            rand_id = np.random.permutation(support_size)
            for i in range(0, support_size, batch_size):

                selected_id = torch.from_numpy(rand_id[i: min(i+batch_size, support_size)]).cuda()
                z_batch = z_support[selected_id]
                y_batch = y_support[selected_id]
                if self.metric_type in ['cosineface', 'neg-softmax']:
                    scores = linear_clf(z_batch, y_batch)
                else:
                    scores = linear_clf(z_batch)
                loss = loss_function(scores, y_batch)

                finetune_optimizer.zero_grad()
                loss.backward()
                finetune_optimizer.step()

        scores = linear_clf(z_query)
        return scores
