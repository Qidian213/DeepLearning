
import os 
from loguru import logger 

from .baseline_cls import BaselineCls


def get_model(cfgs):
    model = BaselineCls(cfgs)
    
    if(cfgs.Model_Set['Resume'] != ''):
        logger.info(f"Resume from {cfgs.Model_Set['Resume']}")
        model.load_param(cfgs.Model_Set['Resume'])
        
    return model

