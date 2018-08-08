import os
import datetime as dt

import numpy as np
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision.transforms import ToTensor
from torch.autograd import Variable
from torch.nn.functional import softmax
from tensorboardX import SummaryWriter

from model_utils import visualise
from model_utils import analysis_utils
from model_utils.analysis_utils import var_to_cpu



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

    def run_singletask_model(self, settings, split, loader, optimizer=False):
        loss = 0
        accuracies = []
        conf_matrix = analysis_utils.ConfusionMatrix(len(self.classes))
        for i, batch in enumerate(loader):
            if optimizer:
                optimizer.zero_grad()
            images, labels = analysis_utils.imgs_labels_to_variables(batch[0], batch[1])
            batch_loss, preds = self.get_batch_loss_and_preds(images, labels, settings['criterion'])
            [conf_matrix.update(int(var_to_cpu(labels[i])), int(var_to_cpu(preds[i]))) for i in range(len(labels))]
            if optimizer:
                batch_loss.backward()
                optimizer.step()
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

    def add_cam_img(self, target_img, cam_layer, input_size, target_class=None, epoch=0):
        gradcam = visualise.GradCam(self.model, cam_layer)
        cam_img = gradcam.create_gradcam_img(target_img, target_class, self.means, self.sdevs, input_size)
        to_tensor = ToTensor()
        cam_tensor = to_tensor(cam_img)
        if target_class:
            caption = f'pred_{gradcam.predicted_class}_true_{target_class}'
        else:
            caption = f'pred_{gradcam.predicted_class}'
        self.writer.add_image(caption, cam_tensor, epoch)   

    def train(self, settings):
        """Performs model training"""
        optimizer = settings['optimizer']
        if self.loss_tracker.store_loss is True:
            self.loss_tracker.set_loss_file('train')
        if settings['visualiser'] is not None:
            self.instantiate_visualizer(settings['visualiser'])
        if 'lr_decay_patience' in settings:
            lr_scheduler = ReduceLROnPlateau(optimizer,
                                             'min',
                                             factor=settings['lr_decay'],
                                             patience=settings['lr_decay_patience'],
                                             verbose=True)

        for epoch in range(settings['n_epochs']):
            train_epoch_start = dt.datetime.now()
            self.model = self.model.train()

            epoch_train_loss, epoch_train_accuracy, train_conf_matrix = self.run_singletask_model(settings, 'train', self.train_loader, optimizer=optimizer)

            epoch_now = len(self.loss_tracker.all_loss['val'])+1
            self.loss_tracker.store_epoch_loss('train', epoch_now, epoch_train_loss, epoch_train_accuracy)
            self.loss_tracker.conf_matrix['train'].append(train_conf_matrix)
        
            if self.val_loader is not None:
                self.validate(settings)

            if epoch_now % settings['save_interval'] == 0 and self.loss_tracker.store_models is True:
                print("Checkpoint-saving model")
                self.loss_tracker.save_model(self.model, epoch)

            total_train_batches = len(self.train_loader)
            self.visualise_loss(settings, epoch_now, train_epoch_start, epoch_train_accuracy, total_train_batches, 'train')
            self.print_results(epoch_now, epoch_train_loss, epoch_train_accuracy, 'train')
            print('Training confusion matrix:\n', train_conf_matrix)

            if 'lr_decay_patience' in settings:
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

        epoch_val_loss, epoch_val_accuracy, val_conf_matrix = self.run_singletask_model(settings, 'val', self.val_loader)

        epoch_now = len(self.loss_tracker.all_loss['val'])+1
        self.loss_tracker.store_epoch_loss('val', epoch_now, epoch_val_loss, epoch_val_accuracy)
        self.save_if_best(epoch_val_loss, self.model, settings['run_name']+'_best')
        self.loss_tracker.conf_matrix['val'].append(val_conf_matrix)

        total_val_batches = len(self.val_loader)
        self.visualise_loss(settings, epoch_now, val_epoch_start, epoch_val_accuracy, total_val_batches, 'val')
        
        if settings['cam_layer'] is not None and settings['visualiser'] == 'tensorboard':
            images, labels = next(iter(self.train_loader))
            target_class = labels[0]
            target_img = Variable(images[0].unsqueeze(0))
            out_size = target_img.shape[-1]
            self.add_cam_img(target_img, settings['cam_layer'], out_size, target_class, epoch_now)
        
        self.print_results(epoch_now, epoch_val_loss, epoch_val_accuracy, 'val')
        print('Validation confusion matrix:\n', val_conf_matrix)

    def infer(self, image, transforms, cam_layer=None, target_class=None, colorramp='inferno'):
        """Takes a single image and computes the most likely class
        """
        self.model = self.model.eval()
        image = transforms(image).unsqueeze(0)  
        if torch.cuda.is_available():
            image = Variable(image.cuda())
        else:
            image = Variable(image)
        output = self.model(image)
        confidence = float(torch.max(softmax(output, dim=1)))
        _, predicted_class_index = torch.max(output, 1)
        predicted_class = self.classes[int(predicted_class_index)]
        
        if cam_layer is not None:
            target_img = image.cpu()
            gradcam = visualise.GradCam(self.model, cam_layer, colorramp)
            cam_img = gradcam.create_gradcam_img(target_img, target_class, self.means, self.sdevs, 224)
        print(f"Predicted class: {predicted_class}")
        print(f"Prediction confidence: {confidence}")
        return [predicted_class, confidence, cam_img]
        