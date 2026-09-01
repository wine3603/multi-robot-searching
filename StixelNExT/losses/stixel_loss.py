from torch import nn
import torch


class StixelLoss(nn.Module):
    # threshold means the threshold when (probab) a border is detected
    def __init__(self, alpha=False, beta=False, gamma=False, pos_weight=50.0):
        """Intersection over Union"""
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        # pos_weight: give higher weight to positive class (ground) to handle class imbalance
        self.pos_weight = pos_weight
        self.bce_loss = nn.BCELoss(reduction='mean', weight=None)

    def forward(self, inputs, targets):
        loss_bce_occ = 0.0
        loss_maximum_cuts_occ = 0.0
        loss_bce_cut = 0.0
        single_monitoring_dicts = []
        if self.alpha:
            # Weight positive pixels (ground) more heavily to handle class imbalance
            # pos_weight means background = 1, ground = pos_weight
            weight = torch.ones_like(targets[:, 0, :, :])
            weight[targets[:, 0, :, :] > 0.5] = self.pos_weight
            loss_bce_occ = (weight * self.bce_loss(inputs[:, 0, :, :], targets[:, 0, :, :])).mean()
            loss_bce_occ = self.alpha * loss_bce_occ
            single_monitoring_dicts.append({'name': "bce_occupancy", 'value': loss_bce_occ})
        if self.beta:
            # This term penalizes high mean occupancy - we don't need it for ground dataset where ground is already minority
            # Keeping the code but effectively disabled by not contributing
            # loss_maximum_cuts_occ = torch.mean(inputs[:, 0, :, :])
            # loss_maximum_cuts_occ = self.beta * loss_maximum_cuts_occ
            # single_monitoring_dicts.append({'name': "sum_occupancy", 'value': loss_maximum_cuts_occ})
            pass
        if self.gamma:
            # Boundary pixels are also very sparse, so need strong positive weighting
            weight = torch.ones_like(targets[:, 1, :, :])
            weight[targets[:, 1, :, :] > 0.5] = self.pos_weight
            loss_bce_cut = (weight * self.bce_loss(inputs[:, 1, :, :], targets[:, 1, :, :])).mean()
            loss_bce_cut = self.gamma * loss_bce_cut
            single_monitoring_dicts.append({'name': "bce_edges", 'value': loss_bce_cut})
        return loss_bce_occ + loss_maximum_cuts_occ + loss_bce_cut, single_monitoring_dicts
