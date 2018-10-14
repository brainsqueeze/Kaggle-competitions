import torch


class SvmLoss(torch.nn.Module):

    def __init__(self):
        super(SvmLoss, self).__init__()

    def forward(self, decisions, targets):
        targets = targets.float() * 2 - 1
        projection_dist = 1 - targets * decisions
        margin = torch.max(torch.zeros_like(projection_dist), projection_dist)
        return margin.mean()


class SvmProbsLoss(torch.nn.Module):

    def __init__(self):
        super(SvmProbsLoss, self).__init__()

    def forward(self, decisions, logits, targets, multi_label=False):
        y = targets.float()

        # SVM soft-margin loss function
        svm_targets = y * 2 - 1
        projection_dist = 1 - svm_targets * decisions
        margin = torch.max(torch.zeros_like(projection_dist), projection_dist)
        svm_loss = margin.mean()

        # loss function for the Platt posterior probabilities
        n_plus = torch.sum(y, dim=0)
        n_minus = torch.sum(1. - y, dim=0)

        n_plus_rate = (n_plus + 1.) / (n_plus + 2.)
        n_minus_rate = 1. / (n_minus + 2.)

        y_cv = n_plus_rate * y + n_minus_rate * (1 - y)
        y_hat = torch.sigmoid(logits) if multi_label else torch.softmax(logits, dim=-1)
        platt_loss = (-1) * torch.mean(y_cv * torch.log(y_hat) + (1 - y_cv) * torch.log(1 - y_hat))
        return svm_loss + platt_loss
