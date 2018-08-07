import os
from copy import deepcopy
from collections import OrderedDict
import pickle
import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable

from data_management import data_utils

class LossRecorder():
    def __init__(self, output_dir='outputs/'):
        self.output_dir = output_dir
        self.run_name = None
        
        self.store_models = False
        self.store_loss = False
        self.loss_files = {'train':'', 'val':''}
        self.all_loss = {'train':np.array([]), 'val':np.array([])}
        self.accuracy = {'train':np.array([]), 'val':np.array([])}
        self.conf_matrix = {'train':[], 'val':[]}
        
    def setup_output_storage(self, run_name, store_models=True, store_loss=True):
        self.run_name = run_name
        if store_loss is True:
            self.store_loss = True
            data_utils.create_dir_if_not_exist(os.path.join(self.output_dir, 'loss'))
        if store_models is True:
            self.store_models = True
            data_utils.create_dir_if_not_exist(os.path.join(self.output_dir, 'models'))

    def store_epoch_loss(self, split, epoch, avg_epoch_loss, epoch_accuracy):
        self.save_loss_if_enabled(self.loss_files[split], avg_epoch_loss, epoch)
        self.all_loss[split] = np.append(self.all_loss[split], avg_epoch_loss)
        self.accuracy[split] = np.append(self.accuracy[split], epoch_accuracy)

    def set_loss_file(self, split):
        data_utils.create_dir_if_not_exist(f'{self.output_dir}/loss/{self.run_name}')
        self.loss_files[split] = f'{self.output_dir}/loss/{self.run_name}/{split}.csv' 
    
    def save_loss_if_enabled(self, loss_file, epoch_loss, epoch):
        if self.store_loss is not None:
            write_loss(loss_file, self.run_name, epoch_loss, epoch)    
            
    def is_loss_at_plateau(self, epochs_until_decay):
        if len(self.all_loss['train']) >= epochs_until_decay:
            lowest_loss = min(self.all_loss['train'])
            current_loss = self.all_loss['train'][-1] 
            is_lowest_loss = current_loss > lowest_loss
             
            best_loss_epoch = int(np.where(self.all_loss['train']==lowest_loss)[-1])
            epochs_since_best = len(self.all_loss['train']) - best_loss_epoch
            epoch_patience_expired = epochs_since_best >= epochs_until_decay

            return is_lowest_loss and epoch_patience_expired

    def save_model(self, model, file_name):
        torch.save(model.state_dict(), os.path.join(self.output_dir, 'models', f'{file_name}.pkl'))

def imgs_labels_to_variables(images, labels):
    if torch.cuda.is_available():
        return(Variable(images.cuda()), Variable(labels.cuda()))
    else:
        return(Variable(images), Variable(labels))

def set_dropout_probability(model, p=0):
    ## In PyTorch 0.3 .eval() on model doesn't set dropout to zero in specific cases. This is a bandaid solution.
    for module in model.modules():
        if type(module) is torch.nn.modules.dropout.Dropout2d:
            module.p = p

def var_to_cpu(var):
    if var.is_cuda:
        var = var.cpu()
    return var

def var_to_cuda(var):
    if torch.cuda.is_available():
        var = var.cuda()
    return var

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

def add_accuracy(accuracy_list, preds, labels):
    batch_length = len(preds)
    total_correct_in_batch = int(torch.sum(preds == labels))
    correct_ratio_in_batch = float(total_correct_in_batch/batch_length)
    accuracy_list.append(correct_ratio_in_batch)
    return(accuracy_list)  

class ConfusionMatrix():
    def __init__(self, n_classes):
        self.n_classes = n_classes
        self.matrix = np.ndarray([n_classes, n_classes], dtype='uint16')
        self.matrix.fill(0)
    
    def update(self, label, predicted):
        self.matrix[:,label][predicted] += 1

    def get_labelled_matrix(self):
        class_array = np.ndarray([self.n_classes,1], dtype='uint16')
        class_array[:,0] = np.arange(self.n_classes)
        out_matrix = np.hstack((class_array, self.matrix))
        return out_matrix

def get_f1_scores(matrix):
    f1_scores = []
    for class_index, row in enumerate(matrix):
        true_pos = row[class_index]
        
        false_pos = deepcopy(row)
        false_pos_col = np.delete(false_pos, class_index)
        false_pos = np.sum(false_pos_col)
        
        false_negs_col = deepcopy(matrix[:,class_index]) 
        false_negs_col = np.delete(false_negs_col, class_index)
        false_neg = np.sum(false_negs_col) 

        precision = true_pos / (true_pos + false_pos)
        recall = true_pos / (true_pos + false_neg)

        f1_scores.append((2 / ( (1 / recall) + (1 / precision) )))
    return f1_scores

def weights_init(m):
    """For all convolutional layers in a model,
    initialises weights using He-initialisation"""
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_uniform(m.weight) #He-initialisation
        m.bias.data.zero_()

def get_predictions(output_batch):
    """For a given input batch, retrieves the most likely class (argmax along channel)
    """
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

def get_relative_class_weights(dir_in, inverse_weights=True):
    entries_per_class = {}
    for root, _, files in os.walk(dir_in):
        if files:
            class_name = os.path.split(root)[1]
            if not class_name in entries_per_class.keys():
                entries_per_class[class_name] = len(files)
            else:
                entries_per_class[class_name] += len(files)
    # PyTorch classes are loaded alphabetically, so we return the dict indexwise
    indexwise_counts = [count for key, count in entries_per_class.items()]
    total_n_images = sum(indexwise_counts)
    for key, count in entries_per_class.items():
        if inverse_weights:
            entries_per_class[key] = 1-(count/total_n_images)
        else:
            entries_per_class[key] = count/total_n_images
    class_weights = [count for key, count in sorted(entries_per_class.items())]
    return class_weights
