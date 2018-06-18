from model_utils import visualise
from model_utils import analysis_utils
from model_utils.analysis_utils import var_to_cpu, var_to_cuda

import os
from os import path
import numpy as np
import datetime as dt

import scipy.misc as misc
import torch
import torch.nn as nn
from torch.optim import SGD, Adam, RMSprop
from torchvision.transforms import Compose, Normalize, ToTensor
from torch.autograd import Variable
from tensorboardX import SummaryWriter
from PIL import Image

from data_management import data_utils


class ImageAnalysis(object):
    def __init__(self, model, classes, train_loader=None, val_loader=None, means=None, sdevs=None):
        ## Model components
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader

        ## Norm-parameters
        self.means = means
        self.sdevs = sdevs
        self.classes = classes

        # Timekeeping
        self.start_time = dt.datetime.now()
            
        # Tracking loss
        self.all_train_loss = np.array([])
        self.all_val_loss = np.array([])          
        
        self.loss_tracker = analysis_utils.LossRecorder()
        self.writer = None

    def initialize_visdom_visualisation(self):
        self.vis_data = visualise.VisdomVisualiser()
        self.loss_windows = {}
        self.timeboxes = {}
        
        if self.train_loader:
            self.loss_windows['train'] = self.vis_data.vis_plot_loss('train')
            self.timeboxes['train'] = self.vis_data.vis_create_timetracker(self.start_time)

        if self.val_loader:
            self.loss_windows['val'] = self.vis_data.vis_plot_loss('val')
            self.timeboxes['val'] = self.vis_data.vis_create_timetracker(self.start_time)
            self.cam_window = self.vis_data.vis_img_window('cam_val')
        
        if self.train_loader and self.val_loader:
            self.loss_windows['combined'] = self.vis_data.vis_plot_loss("combined")

    def update_loss_values(self, all_recorded_loss, loss_plot_window):
        self.vis_data.custom_update_loss_plot(loss_plot_window, all_recorded_loss, title="<b>Training loss</b>", color='#0000ff')

    def update_vis_timer(self,header,epoch_start,total_n_samples,timebox):
        epoch_end = dt.datetime.now()
        epoch_time = epoch_end-epoch_start
        epoch_spd = epoch_time/(total_n_samples)
        self.vis_data.update_timer(header,timebox,epoch_start,epoch_time,epoch_spd)

    def save_if_best(self, avg_epoch_val_loss, model, out_name):
        if len(self.loss_tracker.all_loss['val']) > 0 and self.loss_tracker.store_models is True:
            if avg_epoch_val_loss > max(self.loss_tracker.all_loss['val']):
                self.loss_tracker.save_model(model, out_name)

    def instantiate_visualizer(self, visualiser):
        if visualiser == 'visdom':
            self.initialize_visdom_visualisation()
        elif visualiser == 'tensorboard':
            self.writer = SummaryWriter('/tmp/log')

class AnnotatedImageAnalysis(ImageAnalysis):
    """Performs semantic segmentation
    TODO: refactor repeated code (e.g. timekeeping)"""
    def __init__(self, model, classes, means, sdevs, train_loader=None, val_loader=None):    
        super().__init__(model, classes, train_loader, val_loader, means, sdevs)
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.means = means
        self.sdevs = sdevs

    def get_batch_loss_and_preds(self, images, labels, model, criterion):
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        loss = criterion(outputs, labels)
        return loss, preds

    def run_singletask_model(self, model, settings, split, optimize=False):
        loss = 0
        accuracies = []                
        for i, batch in enumerate(self.train_loader):
            if optimize:
                settings['optimizer'].zero_grad()
            images, labels = analysis_utils.imgs_labels_to_variables(batch[0], batch[1])
            batch_loss, preds = self.get_batch_loss_and_preds(images, labels, model, settings['criterion'])
            if optimize:
                batch_loss.backward()
                settings['optimizer'].step()
            loss += batch_loss.data[0]
            analysis_utils.add_accuracy(accuracies, preds, labels)
            if (i+1) % settings['report_interval']['train'] == 0:
                print(f"{split}: [{i} : {loss/(i+1):.4f}")
        loss = loss/(i+1)
        accuracies = np.mean(accuracies)
        return loss, accuracies

    def run_multitask_model(self, model, settings, optimize=False):
        """Esoteric method to train multitask model with 2 FCLs and n classes in second FCL
        # Possibly relevant for FCL2: https://discuss.pytorch.org/t/indexing-multi-dimensional-tensors-based-on-1d-tensor-of-indices/1391
        """
        loss_1 = 0
        loss_2 = 0
        accuracies_1 = []
        accuracies_2 = {}

        fc1_negatives = settings['negatives_class']
        for i, batch in enumerate(self.train_loader):
            model.fc1.requires_grad = True
            model.fc2.requires_grad = False            
            if optimize:
                settings['optimizer'][0].zero_grad()
                settings['optimizer'][1].zero_grad()

            images, labels = analysis_utils.imgs_labels_to_variables(batch[0], batch[1])
            outputs = model(images)
            # outputs[0] = var_to_cpu(outputs[0])
            # outputs[1] = var_to_cpu(outputs[1])

            ## Output 1 - 1 binary classifier
            _, preds = torch.max(outputs[0], 1)
            fc1_labels_binary = (labels != fc1_negatives).long()
            settings['criterion'][fc1_negatives] = settings['criterion'][fc1_negatives]
            loss_1 += settings['criterion'][fc1_negatives](outputs[0], fc1_labels_binary)
            analysis_utils.add_accuracy(accuracies_1, preds, labels)

            if optimize:
                loss_1.backward()
                settings['optimizer'][0].step()
                model.fc1.requires_grad = False
                model.fc2.requires_grad = True
            
            # Second classifier runs
            if torch.max(labels.data) > 0:
                matching_labels = torch.masked_select(labels, fc1_labels_binary.byte())
                positive_indices = ((fc1_labels_binary == 1).nonzero())
                fcl2_outputs = outputs[1][positive_indices, :].squeeze(1)

                for i, classifier in enumerate(settings['criterion']):
                    if i != fc1_negatives:
                        class_labels_binary = var_to_cuda((matching_labels == i).long())
                        loss_2 += ((classifier(fcl2_outputs, class_labels_binary)) * settings['class_weights'][i])
            # analysis_utils.add_accuracy(accuracies_2, preds, class_labels_binary)

            if optimize:
                loss_2 = var_to_cuda(Variable(loss_2, requires_grad=True))
                loss_2.backward()
                settings['optimizer'][1].step()
        
        return [loss_1, loss_2]

    def visualise_loss(self, settings, epoch, epoch_start, epoch_accuracy, total_n_batches, split):
        if settings['visualiser'] == 'visdom':
            self.vis_data.custom_update_loss_plot(self.loss_windows[split], self.loss_tracker.all_loss[split], title=f"<b>{split} loss</b>")
            self.update_vis_timer(f"<b>{split}</b>", epoch_start, total_n_batches, self.timeboxes[split])                
            if self.val_loader is not None:
                self.vis_data.custom_combined_loss_plot(self.loss_windows['combined'], self.loss_tracker.all_loss['train'], self.loss_tracker.all_loss['val'])
        elif settings['visualiser'] == 'tensorboard':
            self.writer.add_scalar(f'{split}/Loss', self.loss_tracker.all_loss[split][-1], epoch)
            self.writer.add_scalar(f'{split}/Accuracy', epoch_accuracy, epoch)

    def decay_lr_if_enabled(self, optimizer, epoch, settings):
        if settings['l_rate_decay_epoch']:
            analysis_utils.exp_lr_scheduler(optimizer, epoch, settings['l_rate_decay'], settings['l_rate_decay_epoch'])
        elif settings['l_rate_decay_patience']:
            if self.loss_tracker.is_loss_at_plateau(epochs_until_decay=settings['l_rate_decay_patience']) is True:
                analysis_utils.decay_learning_rate(optimizer, settings['lr_decay'])

    def print_results(self, epoch, loss, accuracy):
        print(f"Train accuracy {epoch}: {accuracy:.4f}")
        print(f"Train {epoch} final loss: {loss:.4f}")        

    def add_cam_img(self, target_img, img_class, cam_layer, epoch):
        gradcam = visualise.GradCam(self.model, cam_layer)
        cam_img = gradcam.create_gradcam_img(img_class, target_img, self.means, self.sdevs)
        to_tensor = ToTensor()
        cam_tensor = to_tensor(cam_img)
        self.writer.add_image(f'{cam_layer}_{img_class}', cam_tensor, epoch)      

    def train(self, settings):
        """Performs model training"""

        if self.loss_tracker.store_loss is True:
            self.loss_tracker.set_loss_file('train')
        if settings['visualiser'] is not None:
            self.instantiate_visualizer(settings['visualiser'])

        for epoch in range(settings['n_epochs']):
            train_epoch_start = dt.datetime.now()
            epoch_now = epoch+1
            model = self.model.train()
            self.decay_lr_if_enabled(settings['optimizer'], epoch, settings)

            epoch_train_loss, epoch_train_accuracy = self.run_singletask_model(model, settings, 'train', optimize=True)

            total_train_batches = len(self.train_loader)
            self.loss_tracker.store_epoch_loss('train', epoch_now, epoch_train_loss, epoch_train_accuracy)
        
            if self.val_loader is not None:
                self.validate(settings)

            if epoch_now % settings['save_interval'] == 0 and self.loss_tracker.store_models is True:
                self.loss_tracker.save_model(model, epoch)

            self.visualise_loss(settings, epoch_now, train_epoch_start, epoch_train_accuracy, total_train_batches, 'train')
            self.print_results(epoch_now, epoch_train_loss, epoch_train_accuracy)

        if settings['shutdown'] is True:
            os.system("shutdown")


    def train_multitask(self, settings):
        """Esoteric method for multitask training"""

        if self.loss_tracker.store_loss is True:
            self.loss_tracker.set_loss_file('train')
        if settings['visualiser'] is not None:
            self.instantiate_visualizer(settings['visualiser'])

        for epoch in range(settings['n_epochs']):
            train_epoch_start = dt.datetime.now()
            epoch_now = epoch+1
            model = self.model.train()
            self.decay_lr_if_enabled(settings['optimizer'], epoch, settings)

            epoch_train_loss = self.run_multitask_model(model, settings, optimize=True)

            print(epoch_train_loss[0])
            print(epoch_train_loss[1])

        if settings['shutdown'] is True:
            os.system("shutdown")            

    def validate(self, settings):
        """For a given model, evaluation criterion,
        and validation loader, performs a single evaluation
        pass."""
        eval_model = self.model.eval()
        val_epoch_start = dt.datetime.now()

        epoch_val_loss = 0
        val_batch_accuracies = []

        if self.loss_tracker.store_loss is True:
            self.loss_tracker.set_loss_file('val')

        epoch_val_loss, epoch_val_accuracy = self.run_singletask_model(eval_model, settings, 'val', optimize=False)

        epoch_now = len(self.loss_tracker.all_loss['val'])+1
        total_val_batches = len(self.val_loader)
        self.loss_tracker.store_epoch_loss('val', epoch_now, epoch_val_loss, epoch_val_accuracy)
        self.save_if_best(epoch_val_loss, eval_model, settings['run_name']+'_best')

        self.visualise_loss(settings, epoch_now, val_epoch_start, epoch_val_accuracy, total_val_batches, 'val')
        if settings['cam_layer'] != None and settings['visualiser'] == 'tensorboard':
            images, labels = next(iter(self.train_loader))
            target_class = labels[0]
            target_img = Variable(images[0].unsqueeze(0))
            self.add_cam_img(target_img, target_class, settings['cam_layer'], epoch_now)
        
        self.print_results(epoch_now, epoch_val_loss, epoch_val_accuracy)

    