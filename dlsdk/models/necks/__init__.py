from .baseneck import BaseNeck


def build_neck(cfgs):
    
    neck = BaseNeck(cfgs)

    return neck