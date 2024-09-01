import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossEntropyLoss(nn.Module):
    def __init__(self, ):
        super(CrossEntropyLoss, self).__init__()
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, inputs, labels):
        loss = self.ce_loss(inputs, labels)
        return loss
        
class CrossEntropyLabelSmooth(nn.Module):
    def __init__(self, num_classes = 7, epsilon=0.1):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon     = epsilon
        self.logsoftmax  = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets):
        log_probs = self.logsoftmax(inputs)
        
        targets = torch.zeros(log_probs.size(), device=targets.device).scatter_(1, targets.unsqueeze(1).data, 1)
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        
        loss = (- targets * log_probs).mean(0).sum()
        return loss


class FocalLoss(nn.Module):
    def __init__(self, gamma = 2, eps = 1e-7):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.eps   = eps
        self.ce    = nn.CrossEntropyLoss()

    def forward(self, input, target):
        logp = self.ce(input, target)
        p    = torch.exp(-logp)
        loss = (1 - p) ** self.gamma * logp
        return loss.mean()
        

class ArcFaceLoss(nn.modules.Module):
    def __init__(self, num_class=7, s=30.0, m=0.5, label_smooth=0.0):
        super().__init__()
        self.num_class    = num_class
        self.label_smooth = label_smooth
        
        self.s = s
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def smooth_label(self, labels):
        nY  = F.one_hot(labels, self.num_class).float()
        nY += self.label_smooth / (self.num_class - 1)
        nY[range(labels.size(0)), labels] -= self.label_smooth / (self.num_class - 1) + self.label_smooth
        
        return nY

    def forward(self, logits, labels):
        labels = self.smooth_label(labels)
        cosine = logits.float()
        sine   = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi    = cosine * self.cos_m - sine * self.sin_m
        phi    = torch.where(cosine > self.th, phi, cosine - self.mm)

        cosine  = (labels * phi) + ((1.0 - labels) * cosine)
        cosine *= self.s

        logprobs = F.log_softmax(cosine, dim=-1)
        loss = -logprobs * labels
        loss = loss.sum(-1)
        return loss.mean()


class BCEWithLogits(nn.Module):
    def __init__(self, num_class = 7):
        super(BCEWithLogits, self).__init__()
        self.num_class = num_class
        self.criterion = nn.BCEWithLogitsLoss()

    def forward(self, inputs, labels, weights = None):
        one_hot = F.one_hot(labels, self.num_class).float()
        loss = self.criterion(inputs, one_hot)
        return loss
        
