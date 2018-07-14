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
from torch.optim.lr_scheduler import ReduceLROnPlateau
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
            if avg_epoch_val_loss == min(self.loss_tracker.all_loss['val']):
                self.loss_tracker.save_model(model, out_name)

    def instantiate_visualizer(self, visualiser):
        if visualiser == 'visdom':
            self.initialize_visdom_visualisation()
        elif visualiser == 'tensorboard':
            self.writer = SummaryWriter('/tmp/log')
    
    def instantiate_loss_tracker(self, output_dir='outputs/'):
        self.loss_tracker = analysis_utils.LossRecorder(output_dir)

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
        self.classes = classes

    def get_batch_loss_and_preds(self, images, labels, criterion):
        outputs = self.model(images)
        _, preds = torch.max(outputs, 1)
        loss = criterion(outputs, labels)
        return loss, preds

    def run_singletask_model(self, settings, split, loader, optimize=False):
        loss = 0
        accuracies = []
        conf_matrix = analysis_utils.ConfusionMatrix(len(self.classes))
        for i, batch in enumerate(loader):
            if optimize:
                settings['optimizer'].zero_grad()
            images, labels = analysis_utils.imgs_labels_to_variables(batch[0], batch[1])
            batch_loss, preds = self.get_batch_loss_and_preds(images, labels, settings['criterion'])
            [conf_matrix.update(int(var_to_cpu(labels[i])), int(var_to_cpu(preds[i]))) for i in range(len(labels))]
            if optimize:
                batch_loss.backward()
                settings['optimizer'].step()
            loss += batch_loss.data[0]
            analysis_utils.add_accuracy(accuracies, preds, labels)
            if (i+1) % settings['report_interval'][split] == 0:
                print(f"{split}: [{i} out of {len(loader)}] : {loss/(i+1):.4f}")
        loss = loss/(i+1)
        accuracies = np.mean(accuracies)
        return loss, accuracies, conf_matrix.matrix

    def visualise_loss(self, settings, epoch, epoch_start, epoch_accuracy, total_n_batches, split):
        if settings['visualiser'] == 'visdom':
            self.vis_data.custom_update_loss_plot(self.loss_windows[split], self.loss_tracker.all_loss[split], title=f"<b>{split} loss</b>")
            self.update_vis_timer(f"<b>{split}</b>", epoch_start, total_n_batches, self.timeboxes[split])                
            if self.val_loader is not None:
                self.vis_data.custom_combined_loss_plot(self.loss_windows['combined'], self.loss_tracker.all_loss['train'], self.loss_tracker.all_loss['val'])
        elif settings['visualiser'] == 'tensorboard':
            self.writer.add_scalar(f'{split}/Loss', self.loss_tracker.all_loss[split][-1], epoch)
            self.writer.add_scalar(f'{split}/Accuracy', epoch_accuracy, epoch)

    def print_results(self, epoch, loss, accuracy, split):
        print(f"{split} {epoch} accuracy: {accuracy:.4f}")
        print(f"{split} {epoch} final loss: {loss:.4f}")        

    def add_cam_img(self, target_img, img_class, cam_layer, epoch, input_size):
        gradcam = visualise.GradCam(self.model, cam_layer)
        cam_img = gradcam.create_gradcam_img(img_class, target_img, self.means, self.sdevs, input_size)
        to_tensor = ToTensor()
        cam_tensor = to_tensor(cam_img)
        self.writer.add_image(f'{cam_layer}_trueclass_{img_class}', cam_tensor, epoch)   

    def train(self, settings):
        """Performs model training"""
        if self.loss_tracker.store_loss is True:
            self.loss_tracker.set_loss_file('train')
        if settings['visualiser'] is not None:
            self.instantiate_visualizer(settings['visualiser'])
        if settings['lr_decay_patience'] in settings.keys():
            lr_scheduler = ReduceLROnPlateau(settings['optimizer'],
                                             'min',
                                             factor=settings['lr_decay'],
                                             patience=settings['lr_decay_patience'])

        for epoch in range(settings['n_epochs']):
            train_epoch_start = dt.datetime.now()
            epoch_now = epoch+1
            self.model = self.model.train()

            epoch_train_loss, epoch_train_accuracy, train_conf_matrix = self.run_singletask_model(settings, 'train', self.train_loader, optimize=True)
            self.loss_tracker.store_epoch_loss('train', epoch_now, epoch_train_loss, epoch_train_accuracy)
            self.loss_tracker.conf_matrix['train'] = train_conf_matrix
        
            if self.val_loader is not None:
                self.validate(settings)

            if epoch_now % settings['save_interval'] == 0 and self.loss_tracker.store_models is True:
                print("Checkpoint-saving model")
                self.loss_tracker.save_model(self.model, epoch)

            total_train_batches = len(self.train_loader)
            self.visualise_loss(settings, epoch_now, train_epoch_start, epoch_train_accuracy, total_train_batches, 'train')
            self.print_results(epoch_now, epoch_train_loss, epoch_train_accuracy, 'train')
            print('Training confusion matrix:\n', train_conf_matrix)

            if settings['lr_decay_patience'] in settings.keys():
                lr_scheduler.step(epoch_train_loss)

        if settings['shutdown'] is True:
            os.system("shutdown")        

    def validate(self, settings):
        """For a given model, evaluation criterion,
        and validation loader, performs a single evaluation
        pass."""
        self.model = self.model.eval()
        val_epoch_start = dt.datetime.now()

        if self.loss_tracker.store_loss is True:
            self.loss_tracker.set_loss_file('val')

        epoch_val_loss, epoch_val_accuracy, val_conf_matrix = self.run_singletask_model(settings, 'val', self.val_loader, optimize=False)

        epoch_now = len(self.loss_tracker.all_loss['val'])+1
        self.loss_tracker.store_epoch_loss('val', epoch_now, epoch_val_loss, epoch_val_accuracy)
        self.save_if_best(epoch_val_loss, self.model, settings['run_name']+'_best')
        self.loss_tracker.conf_matrix['val'] = val_conf_matrix

        total_val_batches = len(self.val_loader)
        self.visualise_loss(settings, epoch_now, val_epoch_start, epoch_val_accuracy, total_val_batches, 'val')
        
        if settings['cam_layer'] != None and settings['visualiser'] == 'tensorboard':
            images, labels = next(iter(self.train_loader))
            target_class = labels[0]
            target_img = Variable(images[0].unsqueeze(0))
            self.add_cam_img(target_img, target_class, settings['cam_layer'], epoch_now, target_img.shape[-1])
        
        self.print_results(epoch_now, epoch_val_loss, epoch_val_accuracy, 'val')
        print('Validation confusion matrix:\n', val_conf_matrix)

    