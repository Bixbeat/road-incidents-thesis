import os
from numpy import random
from collections import OrderedDict
import pickle

import torch
import torch.nn as nn

def set_dropout_probability(model, p=0):
    ## In PyTorch 0.3 .eval() on model doesn't set dropout to zero in specific cases. This is a bandaid solution.
    for module in model.modules():
        if type(module) is torch.nn.modules.dropout.Dropout2d:
            module.p = p

def fix_state_dict(state_dict):
    """README: This is a cheap hack to fix model saving in PyTorch with the Densenet implementation.
    After updating to PyTorch 0.3 model saving was done with explicit 'module.-' prefixes
    because the model contains nn.dataparalel. This caused problems with PyTorch loading.
    The following snippet is a workaround
    Reference: https://stackoverflow.com/questions/44230907/keyerror-unexpected-key-module-encoder-embedding-weight-in-state-dict
    """
    corrected_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] # remove `module.`
        corrected_state_dict[name] = v
    return corrected_state_dict

def write_normalize_values(norm_params, output_file_path):
    """Stores the normalization parameters from the training dataset"""
    norm_path = open(output_file_path, 'wb')
    pickle.dump(norm_params, norm_path)

def exp_lr_scheduler(optimizer, epoch, lr_decay=0.1, lr_decay_epoch=7):
    """Decay learning rate by a factor of lr_decay every lr_decay_epoch epochs
    https://discuss.pytorch.org/t/adaptive-learning-rate/320/25"""
    
    if epoch in lr_decay_epoch:
        return optimizer
    
    for param_group in optimizer.param_groups:
        param_group['lr'] *= lr_decay
    return optimizer

def weights_init(m):
    """For all convolutional layers in a model,
    initialises weights using He-initialisation"""
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_uniform(m.weight) #He-initialisation
        m.bias.data.zero_()

def get_predictions(output_batch):
    """For a given input batch, retrieves the most likely class (argmax along channel)
    [SRC]"""
    bs,c,h,w = output_batch.size()
    tensor = output_batch.data
    # Argmax along channel axis (softmax probabilities)
    values, indices = tensor.cpu().max(1)
    indices = indices.view(bs,h,w)
    return indices

def write_loss(file, run_name, loss, epoch=0):
    if not os.path.isfile(file):
        with open(file, 'w') as f:
            f.write('Run_name, loss, epoch\n')
            
    with open(file, 'a') as f:
        f.write("{},{},{}\n".format(run_name, loss, epoch))