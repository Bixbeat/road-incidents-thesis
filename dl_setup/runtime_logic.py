from processing_utils import visualise
from processing_utils import analysis_utils

import os
import numpy as np
import datetime as dt

import scipy.misc as misc

import torch
import torch.nn as nn
from torch.optim import SGD, Adam, RMSprop
from torch.autograd import Variable
from torchvision.transforms import Compose, Normalize
from torchvision.transforms import ToTensor

class ImageAnalysis(object):
    def __init__(self, model, train_loader=None, val_loader=None, means=[], sdevs=[]):
        ## Model components
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader

        ## Norm-parameters
        self.means = means
        self.sdevs = sdevs
        
        ## Timing
        self.start_time = dt.datetime.now()

        ## Tracking loss
        self.all_train_loss = np.array([])
        self.all_val_loss = np.array([])     

        ## Switches
        self.shutdown = False  

    def initialize_visualisation(self):
        # Initialise visualisations
        self.vis_data = visualise.Visualiser()
        
        if self.train_loader != None:
            self.train_loss_window = self.vis_data.vis_plot_loss('train')
            self.train_timebox = self.vis_data.vis_create_timetracker(self.start_time)

        if self.val_loader != None:
            self.val_loss_window = self.vis_data.vis_plot_loss('val')
            self.val_timebox = self.vis_data.vis_create_timetracker(self.start_time)
            self.combined_loss_window = self.vis_data.vis_plot_loss("combined")

    def enable_loss_file_saving(self, target_dir, split, run_name):
        self.train_loss_file = f'outputs/models/logs/{run_name}_{split}'

    def update_loss_values(self, avg_epoch_train_loss):
        self.all_train_loss = np.append(self.all_train_loss, avg_epoch_train_loss)
        self.vis_data.custom_update_loss_plot(self.train_loss_window, self.all_train_loss, title="<b>Training loss</b>", color='#0000ff')
        
    def save_loss_if_enabled(self, run_name, avg_epoch_train_loss, epoch):
        if self.train_loss_file is not None:
            analysis_utils.write_loss(file=self.train_loss_file, run_name=run_name, loss=avg_epoch_train_loss, epoch=epoch)

    def update_vis_timer(self, epoch_start, total_n_samples):
        epoch_end = dt.datetime.now()
        epoch_time = epoch_end-epoch_start
        epoch_spd = epoch_time/(total_n_samples)
        self.vis_data.update_timer("<b>Validation</b>",self.val_timebox,epoch_start,epoch_time,epoch_spd)                        

class AnnotatedImageAnalysis(ImageAnalysis):
    """Performs semantic segmentation
    TODO: refactor repeated code (e.g. timekeeping)"""
    def __init__(self, model, means, sdevs, train_loader=None, val_loader=None):
        self.model = model.cuda()
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.means = means
        self.sdevs = sdevs
        
        # Timekeeping
        self.start_time = dt.datetime.now()
            
        # Tracking loss
        self.all_train_loss = np.array([])
        self.all_val_loss = np.array([])          
        
    def train(self,args):
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
        --shutdown,           type=bool,    'Shutdown after training'
        """
        
        if torch.cuda.is_available():
            self.model = torch.nn.DataParallel(self.model, device_ids=range(torch.cuda.device_count()))
    
        optimizer = RMSprop(self.model.parameters(), lr=args.l_rate, weight_decay=args.w_decay)
        criterion = nn.NLLLoss2d().cuda()
    
        for epoch in range(args.n_epoch):
            epoch_train_loss = 0   
            
            model = self.model.train()
            train_epoch_start = dt.datetime.now()
            
            # Decay learning rate
            optimizer = analysis_utils.exp_lr_scheduler(optimizer, epoch, args.l_rate_decay, args.l_rate_decay_epoch)
            
            for i, (images) in enumerate(self.train_loader):
                if torch.cuda.is_available():
                    images = Variable(images.cuda())
                else:
                    images = Variable(images)
    
                optimizer.zero_grad()
    
                outputs = model(images)
                train_loss = criterion(outputs, labels)
                train_loss.backward()
                optimizer.step()
                
                epoch_train_loss += train_loss.data[0]
                    
                if (i+1) % args.visualize_interval == 0:
                    print("Train {}: [{} of {}] : {:.4f}".format(epoch, i, len(self.train_loader), epoch_train_loss/(i+1)))
                    
                    # Plot predictions
                    preds = analysis_utils.get_predictions(outputs)
                    image = images[0]
                    pred = preds[0]
                    
                    img, pred = visualise.encoded_img_and_lbl_to_data(image, pred, self.means, self.sdevs)
                    visualise.plot_pairs(img, pred)
                    
            # Store loss
            avg_epoch_train_loss = epoch_train_loss/(i+1)
            self.update_loss_values(avg_epoch_train_loss)
            # train_loss_file='outputs/models/logs/{}_train'.format(args.run_name)       
            self.save_loss_if_enabled(args.run_name, avg_epoch_train_loss, epoch)
            
            # Timekeeping
            total_num_samples = i+1
            self.update_vis_timer(train_epoch_start, total_num_samples)

            # Validate model at every epoch if val loader is present                                                  
            if self.val_loader != None:
                val_loss_file='outputs/models/logs/{}_val'.format(args.run_name)
                self.validate(criterion, self.val_loader, args.run_name, val_loss_file)
                self.vis_data.custom_combined_loss_plot(self.combined_loss_window, self.all_train_loss, self.all_val_loss)
            
            number_of_runs = epoch+1
            if number_of_runs % args.save_interval == 0:
                torch.save(self.model.state_dict(), "outputs/models/{}_{}.pkl".format(number_of_runs, args.run_name))
                
        # Shutdown after the final epoch
        if args.shutdown == True:
            os.system("shutdown")
                

    def validate(self, criterion, val_loader, run_name, loss_file):
        """For a given model, evaluation criterion,
        and validation loader, performs a single evaluation
        pass."""
        vis_window = self.val_loss_window
        eval_model = self.model.eval()
        epoch_val_loss = 0
        val_epoch_start = dt.datetime.now()
        
        for i, (images, labels, img_path) in enumerate(val_loader):
            if torch.cuda.is_available():
                images = Variable(images.cuda())
                labels = Variable(labels.cuda())
            else:
                images = Variable(images)
                labels = Variable(labels)

            outputs = eval_model(images)
            val_loss = criterion(outputs, labels)

            epoch_val_loss += val_loss.data[0]
                
            if (i+1) %5 == 0:
                print("Val [{} of {}] : {:.4f}".format(i, len(val_loader), epoch_val_loss/(i+1)))
                
                # Plot predictions
                preds = analysis_utils.get_predictions(outputs)
                image = images[0]
                pred = preds[0]
                
                img, pred = visualise.encoded_img_and_lbl_to_data(image, pred, self.means, self.sdevs, self.label_colours)
                visualise.plot_pairs(img, pred)
            
        # Record validation loss & determine if model is best on val set
        avg_epoch_val_loss = epoch_val_loss/(i+1)
        
        if len(self.all_val_loss) > 0:
            if avg_epoch_val_loss < min(self.all_val_loss):
                torch.save(self.model.state_dict(), "outputs/models/{}_best_model.pkl".format(run_name))
        
        # Record validation loss & plot results
        self.all_val_loss = np.append(self.all_val_loss, avg_epoch_val_loss)
        self.vis_data.custom_update_loss_plot(vis_window, self.all_val_loss, title="<b>Validation loss</b>")
        
        epoch_now = len(self.all_val_loss)-1
        analysis_utils.write_loss(file=loss_file, run_name=run_name, loss=avg_epoch_val_loss, epoch=epoch_now)            

        # Timekeeping
        val_epoch_end = dt.datetime.now()
        val_epoch_time = val_epoch_end-val_epoch_start
        val_epoch_spd = val_epoch_time/(i+1)
        self.vis_data.update_timer("<b>Validation</b>",self.val_timebox,val_epoch_start,val_epoch_time,val_epoch_spd)
        
    def analyse(self, img_directory, transforms, output_dir):
        """With a deployed model and input directory, performs model evaluation
        on the image contents of that folder, then writes them to the output folder.
        """
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