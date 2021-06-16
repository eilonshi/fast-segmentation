import torch
import torch.nn as nn

from src.core.consts import NUM_CLASSES


class SoftDiceLoss(nn.Module):
    """
    Soft dice loss calculation for arbitrary batch size, number of classes, and number of spatial dimensions.
    Assumes the `channels_last` format.

    # Arguments
        y_true: b x X x Y( x Z...) x c One hot encoding of ground truth
        y_pred: b x X x Y( x Z...) x c Network output, must sum to 1 over c channel (such as after softmax)
        epsilon: Used for numerical stability to avoid divide by zero errors
    """

    def __init__(self, ignore_label=255):
        super(SoftDiceLoss, self).__init__()

        self.ignore_label = ignore_label

    def forward(self, y_pred, y_true, epsilon=1e-6):
        # skip the batch and class axis for calculating Dice score
        y_true_one_hot = (torch.arange(NUM_CLASSES).cuda() == y_true[..., None] - 0).permute([0, 3, 1, 2]).type(
            torch.uint8)

        one_hot_shape = y_pred.shape
        y_pred = y_pred[y_true_one_hot != self.ignore_label].reshape([one_hot_shape[0], one_hot_shape[1], -1])
        y_true_one_hot = y_true_one_hot[y_true_one_hot != self.ignore_label].reshape(
            [one_hot_shape[0], one_hot_shape[1], -1])

        numerator = 2. * torch.sum(y_pred * y_true_one_hot, dim=[0, 2])
        denominator = torch.sum(torch.square(y_pred) + torch.square(y_true_one_hot), dim=[0, 2])

        return 1 - torch.mean((numerator + epsilon) / (denominator + epsilon))  # average over classes and batch
