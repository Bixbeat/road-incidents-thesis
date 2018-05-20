from model_utils import visualise
from model_utils import analysis_utils

import os
from os import path
import numpy as np
import datetime as dt

import scipy.misc as misc
import torch
import torch.nn as nn
from torch.optim import SGD, Adam, RMSprop
from torchvision.transforms import Compose, Normalize
from torchvision.transforms import ToTensor

from data_management import data_utils

class ImageAnalysis(object):
    def __init__(self, model, train_loader=None, val_loader=None, means=None, sdevs=None):
        ## Model components
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader

        ## Norm-parameters
        self.means = means
        self.sdevs = sdevs
        
        ## Timing
        self.start_time = dt.datetime.now()

    def initialize_visualisation(self):
        self.vis_data = visualise.Visualiser()
        
        if self.train_loader != None:
            self.train_loss_window = self.vis_data.vis_plot_loss('train')
            self.train_timebox = self.vis_data.vis_create_timetracker(self.start_time)

        if self.val_loader != None:
            self.val_loss_window = self.vis_data.vis_plot_loss('val')
            self.val_timebox = self.vis_data.vis_create_timetracker(self.start_time)
            self.combined_loss_window = self.vis_data.vis_plot_loss("combined")
    
    def update_loss_values(self, all_recorded_loss, loss_plot_window):
        self.vis_data.custom_update_loss_plot(loss_plot_window, all_recorded_loss, title="<b>Training loss</b>", color='#0000ff')

    def update_vis_timer(self,header,epoch_start,total_n_samples,timebox):
        epoch_end = dt.datetime.now()
        epoch_time = epoch_end-epoch_start
        epoch_spd = epoch_time/(total_n_samples)
        self.vis_data.update_timer(header,timebox,epoch_start,epoch_time,epoch_spd)

class AnnotatedImageAnalysis(ImageAnalysis):
    """Performs semantic segmentation
    TODO: refactor repeated code (e.g. timekeeping)"""
    def __init__(self, model, means, sdevs, train_loader=None, val_loader=None):    
        super().__init__(model, train_loader, val_loader, means, sdevs)
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.means = means
        self.sdevs = sdevs
        
        # Timekeeping
        self.start_time = dt.datetime.now()
            
        # Tracking loss
        self.all_train_loss = np.array([])
        self.all_val_loss = np.array([])          
        
        self.loss_tracker = analysis_utils.LossRecorder()

    def get_batch_loss(self, batch, model, criterion, optimizer=None):
        if optimizer:
            optimizer.zero_grad()
        images, labels = analysis_utils.imgs_labels_to_variables(batch[0], batch[1])
        outputs = model(images)
        loss = criterion(outputs, labels)
        if optimizer:
            loss.backward()
            optimizer.step()
        return(loss.data[0])

    def train(self, args):
        """Performs model training
        Args:
        --run_name,           type=str,     'Name used for storage & metadata purposes'
        --model,              type=str,     'Architecture to use'
        --n_epoch,            type=int,     '# of the epochs'
        --batch_size,         type=int      'Batch Size'
        --l_rate,             type=float    'Learning Rate'
        --l_rate_decay,       type=float    'Amount of learning rate decay'
        --l_rate_decay_epoch, type=int      'Epoch at which decay should occur'
        --w_decay,            type=float    'Weight decay'
        """
        self.initialize_visualisation()
        
        # Setup loss
        if self.loss_tracker.store_loss is True:
            self.loss_tracker.set_loss_file('train')

        if torch.cuda.is_available():
            self.model = torch.nn.DataParallel(self.model, device_ids=range(torch.cuda.device_count()))
            criterion = args.criterion.cuda()
        else:
            criterion = args.criterion

        optimizer = args.optimizer(self.model.parameters(), lr=args.l_rate, weight_decay=args.w_decay)

        for epoch in range(args.n_epoch):
            epoch_train_loss = 0

            model = self.model.train()
            train_epoch_start = dt.datetime.now()

            # Use either scheduled decay or decay at plateau
            if args.l_rate_decay_epoch:
                analysis_utils.exp_lr_scheduler(optimizer, epoch, args.l_rate_decay, args.l_rate_decay_epoch)
            elif self.loss_tracker.is_loss_at_plateau(epochs_until_decay=8) is True:
                analysis_utils.decay_learning_rate(optimizer, args.lr_decay)

            for i, batch in enumerate(self.train_loader):
                images, labels = analysis_utils.imgs_labels_to_variables(batch[0], batch[1])

                optimizer.zero_grad()

                outputs = model(images)
                train_loss = criterion(outputs, labels)
                train_loss.backward()
                optimizer.step()

                epoch_train_loss += train_loss.data[0]

                if (i+1) % args.report_interval['train'] == 0:
                    print(f'Train {epoch+1}: [{i} of {len(self.train_loader)}] : {epoch_train_loss/(i+1):.4f}')
                    break
                    # TODO
                    # img, pred = visualise.encoded_img_and_lbl_to_data(image, pred, self.means, self.sdevs)
                    # visualise.plot_pairs(img, pred)

            epoch_now = epoch+1

            # Store loss
            avg_epoch_train_loss = epoch_train_loss/(i+1)

            self.loss_tracker.all_loss['train'] = np.append(self.loss_tracker.all_loss['train'], avg_epoch_train_loss)
            self.vis_data.custom_update_loss_plot(self.train_loss_window, self.all_train_loss, title="<b>Train loss</b>", color="#0000ff")

            self.loss_tracker.save_loss_if_enabled(self.loss_tracker.loss_files['train'], avg_epoch_train_loss, epoch_now)

            # Timekeeping
            total_train_batches = len(self.train_loader)
            self.update_vis_timer("<b>Training</b>", train_epoch_start, total_train_batches, self.train_timebox)

            # Validate model at every epoch if val loader is present           
            if self.val_loader is not None:
                self.validate(criterion, args)
                self.vis_data.custom_combined_loss_plot(self.combined_loss_window, self.loss_tracker.all_loss['train'], self.loss_tracker.all_loss['val'])

            if epoch_now % args.save_interval == 0 and self.loss_tracker.store_models is True:
                self.loss_tracker.save_model(model, epoch)

        # Shutdown after the final epoch
        if args.shutdown is True:
            os.system("shutdown")

    def validate(self, criterion, args):
        """For a given model, evaluation criterion,
        and validation loader, performs a single evaluation
        pass."""
        eval_model = self.model.eval()
        epoch_val_loss = 0
        val_epoch_start = dt.datetime.now()

        if self.loss_tracker.store_loss is True:
            self.loss_tracker.set_loss_file('val')

        for i, (images, labels) in enumerate(self.val_loader):
            images, labels = analysis_utils.imgs_labels_to_variables(images, labels)

            outputs = eval_model(images)
            val_loss = criterion(outputs, labels)

            epoch_val_loss += val_loss.data[0]

            if (i+1) % args.report_interval['val'] == 0:
                print(f"Val [{i} of {len(self.val_loader)}] : {epoch_val_loss/(i+1):.4f}")
                break
                # Plot predictions
                # preds = analysis_utils.get_predictions(outputs)
                # image = images[0]
                # pred = preds[0]

                # img, pred = visualise.encoded_img_and_lbl_to_data(image, pred, self.means, self.sdevs, self.label_colours)
                # visualise.plot_pairs(img, pred)

        # Record validation loss & determine if model is best on val set
        avg_epoch_val_loss = epoch_val_loss/(i+1) 

        if len(self.loss_tracker.all_loss['val']) > 0 and self.loss_tracker.store_models is True:
            if min(self.loss_tracker.all_loss['val']) > avg_epoch_val_loss:
                self.loss_tracker.save_model(eval_model, 'best')

        # Record validation loss & plot results
        self.loss_tracker.all_loss['val'] = np.append(self.loss_tracker.all_loss['val'], avg_epoch_val_loss)
        self.vis_data.custom_update_loss_plot(self.val_loss_window, self.loss_tracker.all_loss['val'], title="<b>Validation loss</b>")

        # Store loss
        epoch_now = len(self.loss_tracker.all_loss['val'])
        self.loss_tracker.save_loss_if_enabled(self.loss_tracker.loss_files['val'], avg_epoch_val_loss, epoch_now)

        # Timekeeping
        total_val_batches = len(self.val_loader)
        self.update_vis_timer("<b>Validation</b>", val_epoch_start, total_val_batches,self.val_timebox)

    def analyse(self, img_directory, transforms, output_dir):
        """With a deployed model and input directory, performs model evaluation
        on the image contents of that folder, then writes them to the output folder.
        """
        pass
        '''
        all_imgs = i_manips.get_images(img_directory)
        model = self.model
                
        for i, img_path in enumerate(all_imgs):
            img = misc.imread(img_path)
            img = np.array(img, dtype=np.int32)
            img = transforms(img).unsqueeze(0)
            
            if torch.cuda.is_available():
                img = Variable(img.cuda())
            else:
                img = Variable(img)
                
            outputs = model(img)
            
            lbl_name = os.path.basename(img_path)
            original_lbl = os.path.join(img_directory, lbl_name)
            out_file = os.path.join(output_dir, lbl_name)

            if self.store_probs == True:
                pred_probs = np.exp(outputs.data.cpu().numpy())
                r_manips.create_output_tile(pred_probs, original_lbl, out_file)                 
            else:
                # Stores argmax labels
                preds = analysis_utils.get_predictions(outputs)[0]            
                bw_predictions = preds.numpy()
                r_manips.create_output_tile(bw_predictions, original_lbl, out_file)
        '''