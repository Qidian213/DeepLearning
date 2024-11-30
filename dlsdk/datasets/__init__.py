
import torch 
import torchvision.transforms as T
from torch.utils.data import DataLoader
from torch.utils.data.sampler import RandomSampler

from .animals_data import AminalDataset


def get_transform(cfgs, is_train=True):
    Normalize_Transform = T.Normalize(mean=cfgs.Data_Transform['RGB_mean'], std=cfgs.Data_Transform['RGB_std'])
    
    if(is_train):
        Transforms = [
                    T.Resize(cfgs.Data_Transform['Resize']),
                    T.RandomHorizontalFlip(cfgs.Data_Transform['RandomHFlip']),
                    T.CenterCrop(cfgs.Data_Transform['Cropsize']),
                ]

        if(cfgs.Data_Transform['ColorJitter']['Use']):
            Transforms.append(T.RandomApply([
                T.ColorJitter(
                    brightness=cfgs.Data_Transform['ColorJitter']['Brightness'], 
                    contrast=cfgs.Data_Transform['ColorJitter']['Contrast'], 
                    saturation=cfgs.Data_Transform['ColorJitter']['Saturation'], 
                    hue=cfgs.Data_Transform['ColorJitter']['Hue'])], 
                    p=cfgs.Data_Transform['ColorJitter']['Probs']
                )
            )

        if(cfgs.Data_Transform['Rotation']['Use']):
            Transforms.append(T.RandomRotation(cfgs.Data_Transform['Rotation']['Angle']))

        Transforms.append(T.ToTensor())
        Transforms.append(Normalize_Transform)

    else:
        Transforms = [
                    T.Resize(cfgs.Data_Transform['Resize']),
                    T.CenterCrop(cfgs.Data_Transform['Cropsize']),
                    T.ToTensor(),
                    Normalize_Transform
                ]

    Transforms = T.Compose(Transforms)
    
    return Transforms


def get_sampler(cfgs, dataset):
    if(cfgs.Data_Loader['Data_sampler'] == 'RandomSampler'):
        sampler = RandomSampler(dataset)

    return sampler


def get_dataloader(cfgs):
    Data_dict = {}

    ### get transforms
    train_transform = get_transform(cfgs, is_train=True)
    val_transform = get_transform(cfgs, is_train=False)

    ### get dataset
    train_data = AminalDataset(cfgs, train_transform)
    val_data = AminalDataset(cfgs, val_transform)

    # ### get sampler && dataloader
    data_sampler = get_sampler(cfgs, train_data)

    train_loader = DataLoader(dataset=train_data, batch_size=cfgs.Data_Loader['Batch_size'], shuffle=False, sampler=data_sampler, num_workers =cfgs.Data_Loader['Num_worker'], pin_memory=True)
    val_loader = DataLoader(dataset=val_data, batch_size=cfgs.Data_Loader['Batch_size'], shuffle=False, num_workers=cfgs.Data_Loader['Num_worker'], pin_memory=True)

    Data_dict['Train_loader'] = train_loader
    Data_dict['Val_loader']   = val_loader
    
    return Data_dict

