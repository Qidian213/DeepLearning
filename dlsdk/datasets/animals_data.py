import os 
import torch
import cv2
from PIL import Image
from torch.utils.data import Dataset

from dlsdk.utils import read_json, suppress_stdout_stderr


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

class AminalDataset(Dataset):
    def __init__(self, cfgs, transform, is_train=True):
        self.cfgs        = cfgs
        self.data_dir    = cfgs.DataSet['Data_dir']
        self.data_file   = cfgs.DataSet['Train_file'] if is_train else cfgs.DataSet['Eval_file']
        self.num_classes = cfgs.DataSet['Num_class']
        self.transform   = transform
        self.is_train    = is_train

        self.data_list = read_json(self.data_file)

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        info_item = self.data_list[index]

        file_path = os.path.join(self.data_dir, info_item["file_path"])
        label = Class_MAP[info_item["class_name"]]

        with suppress_stdout_stderr():
            item_data = cv2.imread(file_path)
            item_data = Image.fromarray(item_data).convert('RGB')
        item_data = self.transform(item_data)

        return item_data, label