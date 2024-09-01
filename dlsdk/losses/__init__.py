from .loss_fun import *


def get_loss(cfgs):
    """ return given loss function
    """
    loss_fun = {}
    
    if(cfgs.Loss_set['Cls_type'] == 'CrossEntropyLoss'):
        function = CrossEntropyLoss()
        loss_fun['cls_loss'] = function
    elif(cfgs.Loss_set['Cls_type'] == 'CrossEntropyLabelSmooth'):
        function = CrossEntropyLabelSmooth()
        loss_fun['cls_loss'] = function
    elif(cfgs.Loss_set['Cls_type'] == 'ArcFaceLoss'):
        function = ArcFaceLoss()
        loss_fun['cls_loss'] = function
    elif(cfgs.Loss_set['Cls_type'] == 'BCEWithLogits'):
        function = BCEWithLogits()
        loss_fun['cls_loss'] = function
    else:
        raise 'the function name you have entered is not supported yet'

    return loss_fun