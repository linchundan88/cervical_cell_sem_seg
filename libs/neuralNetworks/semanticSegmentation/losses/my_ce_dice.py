
import torch
import torch.nn as nn
import torch.nn.functional as F
from segmentation_models_pytorch.losses.dice import DiceLoss
from torch.nn import BCEWithLogitsLoss
# from segmentation_models_pytorch.losses.soft_bce import SoftBCEWithLogitsLoss


class CE_Dice_combine_Loss(nn.Module):
    def __init__(self, weight_bce=1, weight_dice=1, pos_weight=1.0):
        super(CE_Dice_combine_Loss, self).__init__()
        self.weight_bce = weight_bce
        self.weight_dice = weight_dice
        self.dice = DiceLoss()
        self.cross_entropy = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    def forward(self, output, target):
        CE_loss = self.weight_bce * self.cross_entropy(output, target)
        dice_loss = self.weight_dice * self.dice(output, target)
        total_loss = CE_loss + dice_loss / (self.weight_bce + self.weight_dice)
        return total_loss

