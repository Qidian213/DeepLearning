import torch 
import torch.nn as nn 
import torch.nn.functional as F 


class BaseHead(nn.Module):
    def __init__(self, cfgs):
        super(BaseHead, self).__init__()
        self.in_features = cfgs.Model_Set['Out_Features']
        self.num_class   = cfgs.Model_Set['Num_class']

        self.classifier = nn.Linear(self.in_features, self.num_class)
        self.classifier.bias.requires_grad_(False)
        
    def forward(self, cls_feat): 
        cls_out = self.classifier(cls_feat)

        return cls_out 