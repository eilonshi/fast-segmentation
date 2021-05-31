import torch
import torch.nn as nn


class OHEMCrossEntropyLoss(nn.Module):

    def __init__(self, thresh, ignore_label=255):
        super(OHEMCrossEntropyLoss, self).__init__()

        self.thresh_percent = thresh
        self.thresh = -torch.log(torch.tensor(thresh, requires_grad=False, dtype=torch.float)).cuda()
        self.ignore_lb = ignore_label
        self.criteria = nn.CrossEntropyLoss(ignore_index=ignore_label, reduction='mean')

    def forward(self, logits, labels):
        # loss = self.criteria(logits, labels).view(-1)
        # n_min = int(loss.numel() * self.thresh_percent)
        # loss_hard, _ = loss.topk(n_min)
        #
        # return torch.mean(loss_hard)
        return self.criteria(logits, labels)


if __name__ == '__main__':
    pass
