import torch
import torch.nn as nn


class TripletCosineMarginLoss(nn.Module):
    def __init__(self, margin):
        super(TripletCosineMarginLoss, self).__init__()
        self.margin = float(margin)

    def forward(self, anchor, positive, negative):
        positive_sims = torch.cosine_similarity(anchor, positive, dim=1)
        negative_sims = torch.cosine_similarity(anchor, negative, dim=1)

        losses = torch.clamp_min(negative_sims - positive_sims + self.margin, min=0)
        loss = torch.mean(losses)
        return loss
