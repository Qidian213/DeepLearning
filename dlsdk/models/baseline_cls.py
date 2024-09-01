import torch 
import torch.nn as nn
from collections import OrderedDict

from .backbones import build_backbone
from .pooling import build_pooling
from .necks import build_neck
from .heads import build_head


class BaselineCls(nn.Module):
    def __init__(self, cfgs):
        super(BaselineCls, self).__init__()
        self.cfgs = cfgs
        
        self.backbone = build_backbone(self.cfgs)
        self.pooling  = build_pooling(self.cfgs)
        self.neck     = build_neck(self.cfgs)
        self.head     = build_head(self.cfgs)

    def forward(self, x, labels=None):
        x = self.pooling(self.backbone(x))

        cls_feat = self.neck(x)
        cls_out = self.head(cls_feat)
        
        return cls_out

    def load_param(self, file):
        checkpoint = torch.load(file, map_location='cpu')
        
        if('state_dict' in checkpoint.keys()):
            checkpoint = checkpoint['state_dict']
        
        model_state_dict = self.state_dict()
        new_state_dict   = OrderedDict()
        for k, v in checkpoint.items():
            name = k.replace('module.', '')
            new_state_dict[name] = v
            if name in model_state_dict:
                if v.shape != model_state_dict[name].shape:
                    print('Skip loading parameter {}, required shape{}, '\
                          'loaded shape{}.'.format(name, model_state_dict[name].shape, v.shape))
                    new_state_dict[name] = model_state_dict[name]
            else:
                print('Drop parameter {}.'.format(name))

        for key in model_state_dict.keys():
            if(key not in new_state_dict.keys()):
                print('No param {}.'.format(key))
                new_state_dict[key] = model_state_dict[key]
            
        self.load_state_dict(new_state_dict, strict=False)