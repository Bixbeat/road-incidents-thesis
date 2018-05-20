import argparse
import os
import pickle
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, datasets, transforms
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Normalize, ToTensor

from data_management import image_manipulations as i_manips
from model_utils import analysis_utils
from model_utils import runtime_logic

if __name__ == "__main__":
    # Fix conda slow loading https://github.com/pytorch/pytorch/issues/537    

# =============================================================================
#   HYPERPARAMETERS
# =============================================================================
    seed = 0
    torch.cuda.manual_seed(seed)

    run_name = "ResNet18"
    data_dir = 'hymenoptera_data'

    n_epochs = 250
    workers = 4

    init_l_rate = 1e-5
    l_rate_decay = 0.1
    l_rate_decay_epoch = False # [25, 80, 200]
    w_decay = 1e-4
    
    batch_size = 1
    num_channels = 3
    num_classes = 3
    
    optimizer = optim.SGD
    criterion = nn.CrossEntropyLoss()
    
    shutdown_after = False
    report_results_per_n_batches = {'train':20, 'val':5}

# =============================================================================
#   DATA LOADING
# =============================================================================

    all_image_filepaths = i_manips.get_images(os.path.join(data_dir, 'train'))
    norm_params = i_manips.get_normalize_params(all_image_filepaths, 3)
    means = norm_params['means']
    sdevs = norm_params['sdevs']

    train_transforms = transforms.Compose([
                        transforms.RandomResizedCrop(224),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        transforms.Normalize(means, sdevs)
                    ])

    val_transforms =  transforms.Compose([
                        transforms.Resize(256),
                        transforms.CenterCrop(224),
                        transforms.ToTensor(),
                        transforms.Normalize([means[0], means[1], means[2]], [sdevs[0], sdevs[1], sdevs[2]])
                    ])

    train_data = datasets.ImageFolder(os.path.join(data_dir, 'train'), transform=train_transforms)
    val_data = datasets.ImageFolder(os.path.join(data_dir, 'val'), transform=train_transforms)

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size,
                                                shuffle=True, num_workers=workers)

    val_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size,
                                                shuffle=True, num_workers=workers)                                                
    
# =============================================================================
#   LOAD MODEL
# =============================================================================
    model = models.resnet18(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)    
    
    analysis = runtime_logic.AnnotatedImageAnalysis(model, means, sdevs, train_loader, val_loader)
    analysis.loss_tracker.setup_output_storage(run_name, 'outputs/')
    
# =============================================================================
#   INITIALIZE TRAINER    
# =============================================================================
    parser = argparse.ArgumentParser(description='Hyperparams')
    
    ## Model components
    parser.add_argument('--run_name', nargs='?', type=str, default=run_name,
                        help='Name used for storage & metadata purposes')            
    parser.add_argument('--optimizer', nargs='?', type=str, default=optimizer,
                        help='Name used for storage & metadata purposes')
    parser.add_argument('--criterion', nargs='?', type=str, default=criterion,
                        help='Criterion used')
    
    ## Hyperparameters
    parser.add_argument('--n_epoch', nargs='?', type=int, default=n_epochs, 
                        help='# of the epochs')
    parser.add_argument('--batch_size', nargs='?', type=int, default=batch_size, 
                        help='Batch Size')
    parser.add_argument('--l_rate', nargs='?', type=float, default=init_l_rate, 
                        help='Learning Rate')
    parser.add_argument('--l_rate_decay', nargs='?', type=float, default=l_rate_decay, 
                        help='Amount of learning rate decay')
    parser.add_argument('--w_decay', nargs='?', type=float, default=w_decay,
                        help='Weight decay')
    parser.add_argument('--l_rate_decay_epoch', nargs='?', type=int, default=l_rate_decay_epoch,
                        help='Epoch at which decay should occur')
    
    ##Saving & information
    parser.add_argument('--report_interval', nargs='?', type=float, default=report_results_per_n_batches,
                        help='Epochs between loss function printouts')
    parser.add_argument('--save_interval', nargs='?', type=float, default=25,
                        help='Epochs between model checkpoint saves')    
    parser.add_argument('--shutdown', nargs='?', type=bool, default=shutdown_after,
                        help='Shutdown after training')
    args = parser.parse_args()
    analysis.train(args)