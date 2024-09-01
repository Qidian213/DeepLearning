import os
from loguru import logger 


class Cfg_Opts(object):
    def __init__(self,):
    ### Data setting
        self.DataSet                    = {}
        self.DataSet['Data_dir']        = './data'
        self.DataSet['Train_file']      = 'data/Animals_Train.json'
        self.DataSet['Eval_file']       = 'data/Animals_Eval.json'
        self.DataSet['Num_class']       = 10

    ### Data augmentation
        self.Data_Transform                   = {}
        self.Data_Transform['RGB_mean']       = [0.5,0.5,0.5]
        self.Data_Transform['RGB_std']        = [0.5,0.5,0.5]
        self.Data_Transform['Resize']         = 256
        self.Data_Transform['Cropsize']       = 224
        self.Data_Transform['RandomHFlip']    = 0.5
        self.Data_Transform['ColorJitter']    = {'Use': True, 'Probs': 1.0, 'Brightness': 0.3, 'Contrast': 0.3, 'Saturation': 0.3, 'Hue': 0}
        self.Data_Transform['Rotation']       = {'Use': True, 'Angle': 15}
        self.Data_Transform['RandomAffine']   = {'Use': True, 'Probs': 0.5}

    ### Data augmentation
        self.Data_Loader                    = {}
        self.Data_Loader['Batch_size']      = 64
        self.Data_Loader['Num_worker']      = 4
        self.Data_Loader['Data_sampler']    = 'RandomSampler'

    ### Model Setting
        self.Model_Set                  = {}
        self.Model_Set['Model_name']    = 'ResNet18'
        self.Model_Set['Num_class']     = self.DataSet['Num_class']
        self.Model_Set['Pre_Trained']   = 'data/resnet18-f37072fd.pth' #Model_pretrained[self.Model_Set['Model_name']] 
        self.Model_Set['Resume']        = ''

    ### Optimizer Setting
        self.Optimizer                  = {}
        self.Optimizer['Optim_Type']    = 'AdamW'
        self.Optimizer['Decay_BN']      = False
        self.Optimizer['Sche_Type']     = 'WarmupMultiEpochLR'
        self.Optimizer['Lr_Base']       = 1e-4
        self.Optimizer['Lr_Adjust']     = [60,80,100]
        self.Optimizer['Warmup_epoch']  = 1
        self.Optimizer['Start_epoch']   = 0
        self.Optimizer['Max_epoch']     = 120

    ### Loss Setting
        self.Loss_set                   = {}
        self.Loss_set['Cls_type']       = 'CrossEntropyLabelSmooth'

    ### Work space setting
        self.Work_Space                 = {}
        self.Work_Space['Save_dir']     = './work_dirs'
        self.Work_Space['Log_Inter']    = 50
        self.Work_Space['Cuda_env']     = '0'

    ### work mode 
        self.Work_Mode                  = 'Train'
    #   self.Work_Mode                  = 'Val'

    def List_Setting(self, ):
        for name, value in vars(self).items():
            if(isinstance(value, dict)):
                logger.info(f"{name}\r")
                for key in value.keys():
                    logger.info(f"{key}={value[key]}\r")
            else:
                logger.info(f"{name}={value}\r")