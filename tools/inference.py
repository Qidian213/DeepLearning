import os 
import cv2
import random
import numpy as np
from PIL import Image
from loguru import logger 

import torch 
import torch.nn as nn
import torch.nn.functional as F

from dlsdk.models import get_model
from dlsdk.data import get_transform

from configs import Cfg_Opts

Class_MAP = {
    "dog" : 0, 
    "horse": 1, 
    "elephant": 2, 
    "butterfly": 3,
    "chicken": 4, 
    "cat": 5, 
    "cow": 6, 
    "sheep": 7, 
    "spider": 8, 
    "squirrel": 9
}
Class_MAP_Index = {label:name for name, label in Class_MAP.items()}


class Mainer(object):
    def __init__(self, cfgs, checkpoint=""):
        self.cfgs = cfgs
        
        ### model 
        self.model = get_model(self.cfgs)
        self.model.load_param(checkpoint)
        self.model.cuda().eval()

        self.transform = get_transform(self.cfgs, is_train=False)

    def inference(self, file_path):
        with torch.no_grad():
            item_data = cv2.imread(file_path)
            item_data = Image.fromarray(item_data).convert('RGB')
            item_data = self.transform(item_data)

            item_data = item_data.unsqueeze(0).cuda()

            cls_out = self.model(item_data)
            softmax_output = F.softmax(cls_out, dim=1)

            class_label = torch.argmax(softmax_output)
            class_score = softmax_output[0, class_label]

            return int(class_label.item()), class_score.item()


if __name__ == '__main__':

    cfgs = Cfg_Opts()

    checkpoint_file = "./work_dirs/ResNet18_20240831235957/epoch_best.pth"
    mainer = Mainer(cfgs, checkpoint=checkpoint_file)

    file_list = [
        "data/Animals-10/cane/OIF-e2bexWrojgtQnAPPcUfOWQ.jpeg", # dog
        "data/Animals-10/cavallo/OIP-_-5DDAGLz1A9FyrB0FLdgwHaFw.jpeg", ## horse
        "data/Animals-10/gatto/1514.jpeg", # cat
        "data/Animals-10/elefante/eb33b20d2dfc043ed1584d05fb1d4e9fe777ead218ac104497f5c978a4eebdbd_640.jpg", ## elephant
        "data/Animals-10/scoiattolo/OIP-tB1j3IubY822FuTgmb1jGgHaFI.jpeg", # squirrel
        "data/Animals-10/cane/OIP-cQCmLMua8s0ykw8RjdB4rQHaGh.jpeg", ## dog
        "data/Animals-10/pecora/OIP-hzEA1m71cBe0zbm8zJ_63AAAAA.jpeg", # sheep
        "data/Animals-10/ragno/OIP-KulGH0KjHFysu67WcgVIpwHaFH.jpeg", ## spider
    ]

    for file_path in file_list:
        class_label, class_score =  mainer.inference(file_path)
        print(file_path, Class_MAP_Index[class_label], class_score)
