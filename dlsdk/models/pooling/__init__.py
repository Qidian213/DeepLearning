

import torch
import torch.nn as nn
import torch.nn.functional as F


def build_pooling(cfgs):

    pooling = nn.AdaptiveAvgPool2d((1, 1))

    return pooling