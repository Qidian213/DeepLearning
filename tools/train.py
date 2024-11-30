import os 
import random
import numpy as np
from loguru import logger 

import torch 
import torch.nn as nn

from dlsdk.datasets import get_dataloader
from dlsdk.models import get_model
from dlsdk.optimizers import get_optimizer
from dlsdk.losses import get_loss
from dlsdk.utils import make_dir, get_timestamp, AverageMeter, Accuracy

from configs import Cfg_Opts


def Setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True 
    torch.backends.cudnn.benchmark = True


class Mainer(object):
    def __init__(self, cfgs):
        self.cfgs = cfgs
        
        ### dataset
        self.data_dict = get_dataloader(self.cfgs)
        self.epoch_batchs = len(self.data_dict['Train_loader'])

        ### model 
        self.model = get_model(self.cfgs)

        ### optimizer and scheduler
        self.optim_schedulers = get_optimizer(self.cfgs, self.model, self.epoch_batchs)

        ## loss 
        self.loss_meter = get_loss(self.cfgs)

        ## 
        self.model.cuda()
        self.model = nn.DataParallel(self.model)
        
        # ## logger 
        # self.cfgs.List_Setting()


    def train(self):
        best_acc = 0.0
        best_epoch = 0
        for epoch in range(self.cfgs.Optimizer['Start_epoch'], self.cfgs.Optimizer['Max_epoch']+1):
            self.train_epoch(epoch)
            val_acc = self.val_epoch(epoch)
            
            if(val_acc > best_acc):
                best_acc   = val_acc
                best_epoch = epoch
                save_file  = os.path.join(self.cfgs.Work_Space['Save_dir'], 'epoch_best.pth')
                torch.save(self.model.state_dict(), save_file)
                
            save_file = os.path.join(self.cfgs.Work_Space['Save_dir'], 'epoch_ck.pth')
            torch.save(self.model.state_dict(),save_file)

            logger.info(f"----------------------------------------------------------------\r")
            logger.info(f" Best_Epoch: {best_epoch},  Best_acc: {best_acc:.5f}\r")
            logger.info(f"----------------------------------------------------------------\r")
            

    def train_epoch(self, epoch):
        self.model.train()
        
        log_top1 = AverageMeter()
        log_loss = AverageMeter()
        for step,(images, labels) in enumerate(self.data_dict['Train_loader']):
            self.optim_schedulers['optimizer'].zero_grad()

            images = images.cuda()
            labels = labels.cuda()
            
            cls_out = self.model(images)

            loss = self.loss_meter["cls_loss"](cls_out, labels)
            loss.backward()

            self.optim_schedulers['optimizer'].step()
            self.optim_schedulers['scheduler'].step()

            acc1 = Accuracy(cls_out, labels, topk=(1, ))
            log_top1.update(acc1[0].item(),len(labels))
            log_loss.update(loss.item(), len(labels))

            if(step % self.cfgs.Work_Space['Log_Inter'] == 0):
                logger.info(f"iter: {epoch}:{self.epoch_batchs}:{step}, loss: {log_loss.avg:.5f}, top1: {log_top1.avg:.5f}, LR: {'%e'%self.optim_schedulers['optimizer'].param_groups[0]['lr']}\r")


    def val_epoch(self, epoch):
        self.model.eval()
        with torch.no_grad():
            log_top1  = AverageMeter()
            acc_dicts = [AverageMeter() for i in range(self.cfgs.DataSet['Num_class'])]

            for step,(images, labels) in enumerate(self.data_dict['Val_loader']):
                images = images.cuda()
                labels = labels.cuda()

                cls_out = self.model(images)
                
                acc1 = Accuracy(cls_out, labels, topk=(1, ))
                log_top1.update(acc1[0].item(), len(labels))

                preds = torch.argmax(cls_out.detach(), 1)
                for pd, label in zip(preds, labels):
                    if(pd == label):
                        acc_dicts[label].update(1, 1)
                    else:
                        acc_dicts[label].update(0, 1)

            for label, log_loger in enumerate(acc_dicts):
                logger.info(f"Val: {epoch}, {label}_acc: {log_loger.avg:.5f} \r")
            logger.info(f"Val: {epoch}, top1: {log_top1.avg:.5f} \r")

            return log_top1.avg


if __name__ == '__main__':
    Setup_seed(233)

    cfgs = Cfg_Opts()
    # cfgs.List_Setting()

    # os.environ['CUDA_VISIBLE_DEVICES'] = cfgs.Work_Space['Cuda_env']

    cfgs.Work_Space['Save_dir'] = os.path.join(cfgs.Work_Space['Save_dir'],  f"{cfgs.Model_Set['Model_name']}_{get_timestamp()}")
    cfgs.Work_Space['Save_dir'] = make_dir(cfgs.Work_Space['Save_dir'])

    logger.info("startup... \r")
    logger.info("Train mode ... \r")

    mainer = Mainer(cfgs)
    mainer.train()


## export LD_LIBRARY_PATH="./"