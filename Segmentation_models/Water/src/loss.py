import torch
import torch.nn as nn

class CombinedLoss1(nn.Module):
    def __init__(self, alpha=0.5):
        super(CombinedLoss1, self).__init__()
        self.alpha = alpha
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, pred, target,modle):
        bce_loss = self.bce(pred, target)
        pred = torch.sigmoid(pred)
        dice_loss = 1 - (2 * (pred * target).sum() + 1) / (pred.sum() + target.sum() + 1)
        return self.alpha * bce_loss + (1 - self.alpha) * dice_loss

class CombinedLoss11(nn.Module):
    def __init__(self, alpha=0.5, gamma=2, beta=1):
        super(CombinedLoss1, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.beta = beta
        self.bce = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, pred, target,modle):
        bce_loss = self.bce(pred, target)
        pred_prob = torch.sigmoid(pred)
        # Focal loss component for hard-to-classify examples
        focal_weight = self.alpha * (1 - pred_prob).pow(self.gamma)
        focal_loss = focal_weight * bce_loss
        # Dice loss component for localization
        dice_loss = 1 - (2 * (pred_prob * target).sum() + 1) / (pred_prob.sum() + target.sum() + 1)
        # Combined loss with balancing factor
        loss = focal_loss.mean() + self.beta * dice_loss
        return loss
class CombinedLoss_with_w(nn.Module):
    def __init__(self, alpha=0.5, beta=0.005):
        super(CombinedLoss_with_w, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, pred, target, model):
        bce_loss = self.bce(pred, target)
        pred = torch.sigmoid(pred)
        dice_loss = 1 - (2 * (pred * target).sum() + 1) / (pred.sum() + target.sum() + 1)

        # Adding L2 regularization
        l2_penalty = 0.0
        for param in model.parameters():
            l2_penalty += torch.norm(param)**2

        return self.alpha * bce_loss + (1 - self.alpha) * dice_loss + self.beta * l2_penalty


class CombinedLoss2(nn.Module):
    def __init__(self, alpha=0.7, beta=0.3, gamma=2, lambda1=0.5, lambda2=0.25):
        super(CombinedLoss2, self).__init__()
        self.focal = FocalLoss(alpha=alpha, gamma=gamma)
        self.bce = nn.BCEWithLogitsLoss()
        self.alpha = alpha
        self.beta = beta
        self.lambda1 = lambda1
        self.lambda2 = lambda2

    def tversky_loss(self, y_true, y_pred):
        smooth = 1e-5
        y_true_pos = y_true
        y_pred_pos = y_pred
        true_pos = torch.sum(y_true_pos * y_pred_pos)
        false_neg = torch.sum(y_true_pos * (1 - y_pred_pos))
        false_pos = torch.sum((1 - y_true_pos) * y_pred_pos)
        return 1 - (true_pos + smooth) / (true_pos + self.alpha * false_pos + self.beta * false_neg + smooth)

    def forward(self, pred, target, model):
        focal_loss = self.focal(pred, target)
        bce_loss = self.bce(pred, target)
        tversky_loss_value = self.tversky_loss(target, torch.sigmoid(pred))

        # L2 regularization
        l2_penalty = 0.0
        for param in model.parameters():
            l2_penalty += torch.norm(param, 2)**2

        # Combine the losses
        combined_loss = self.lambda1 * (focal_loss + bce_loss) + self.lambda2 * tversky_loss_value + self.beta * l2_penalty

        return combined_loss


class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss
        return F_loss.mean()

def tversky_loss(y_true, y_pred, alpha=0.7, beta=0.3):
    numerator = torch.sum(y_true * y_pred)
    denominator = y_true * y_pred + alpha * (y_true - y_pred * y_true) + beta * (y_pred - y_pred * y_true)
    return 1 - (numerator + 1) / (torch.sum(denominator) + 1)



class CombinedLoss(nn.Module):
    def __init__(self, alpha=0.5, gamma=2.0, beta=0.5):
        super(CombinedLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.beta = beta
        self.bce = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, pred, target,model):
        # BCE loss
        bce_loss = self.bce(pred, target)

        # Focal loss
        pred_prob = torch.sigmoid(pred)
        focal_loss = self.alpha * ((1 - pred_prob) ** self.gamma) * bce_loss
        focal_loss = focal_loss.mean()

        # Dice loss
        intersection = (pred_prob * target).sum()
        dice_loss = 1 - (2 * intersection + 1) / (pred_prob.sum() + target.sum() + 1)

        # Combined loss
        return self.beta * focal_loss + (1 - self.beta) * dice_loss



