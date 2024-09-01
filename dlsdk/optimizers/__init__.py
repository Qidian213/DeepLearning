import torch 

from .schedulers import WarmUpLR, WarmupMultiStepLR, WarmupMultiEpochLR


def add_weight_decay(model, weight_decay=1e-4, skip_list=()):
    decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        if len(param.shape) == 1 or name.endswith(".bias") or name in skip_list:
            no_decay.append(param)
        else:
            decay.append(param)
    return [{'params': no_decay, 'weight_decay': 0.}, {'params': decay, 'weight_decay': weight_decay}]


def get_optimizer(cfgs, model, iter_per_epoch):
    optim_scheduler = {}
    
    if(cfgs.Optimizer['Decay_BN']):
        parameters = model.parameters
        weight_decay = 1e-4
    else:
        parameters = add_weight_decay(model, weight_decay=1e-4)
        weight_decay = 0
        
    ### optimizer
    if(cfgs.Optimizer['Optim_Type'] == 'SGD'):
        optimizer = torch.optim.SGD(parameters, lr=cfgs.Optimizer['Lr_Base'], weight_decay=weight_decay)
    elif(cfgs.Optimizer['Optim_Type'] == 'AdamW'):
        optimizer = torch.optim.AdamW(parameters, lr=cfgs.Optimizer['Lr_Base'], betas=(0.9, 0.99), weight_decay=weight_decay)
    else:
        raise "not support Optim_Type"

    ### lr_scheduler
    if(cfgs.Optimizer['Sche_Type'] == "WarmupMultiStepLR"):
        scheduler = WarmupMultiStepLR(optimizer, iter_per_epoch, cfgs.Optimizer['Warmup_epoch'], cfgs.Optimizer['Lr_Adjust'])
    if(cfgs.Optimizer['Sche_Type'] == "WarmupMultiEpochLR"):
        scheduler = WarmupMultiEpochLR(optimizer, iter_per_epoch, cfgs.Optimizer['Warmup_epoch'], cfgs.Optimizer['Lr_Adjust'])
    else:
        raise "not support Sche_Type"

    optim_scheduler['optimizer'] = optimizer
    optim_scheduler['scheduler'] = scheduler

    return optim_scheduler
