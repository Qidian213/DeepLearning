import torch 
import torch.nn as nn 
import torch.nn.functional as F 


class BaseNeck(nn.Module):
    def __init__(self, cfgs, drop_out=0.1):
        super(BaseNeck, self).__init__()

        self.in_features = cfgs.Model_Set['Out_Features']

        self.bnneck = nn.BatchNorm1d(self.in_features)
        self.dropout = nn.Dropout(drop_out)

    def forward(self, x):
        x = torch.flatten(x, 1)
        
        x = self.bnneck(x)
        x = F.normalize(x, dim=-1)
        x = self.dropout(x)
        
        return x
        