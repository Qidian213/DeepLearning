from loguru import logger 

from .resnet import resnet18


def build_backbone(cfgs):
    if(cfgs.Model_Set['Model_name'] == 'ResNet18'):
        model = resnet18()
        logger.info(f"Model name = {cfgs.Model_Set['Model_name']}")

    if((cfgs.Work_Mode == 'Train') and (cfgs.Model_Set['Pre_Trained'] != '')):
        model.init_weights()
        model.load_param(cfgs.Model_Set['Pre_Trained'])
        logger.info(f"Using pretrained model: {cfgs.Model_Set['Pre_Trained']}")
    else:
        model.init_weights()
    
    cfgs.Model_Set['Out_Features'] = model.out_features
    return model
    
