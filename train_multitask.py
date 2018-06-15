import os

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, datasets, transforms
from torchvision.transforms import Compose, Normalize, ToTensor
from copy import deepcopy

from data_management import image_manipulations as i_manips
from model_utils import runtime_logic
from model_utils import model_components
from model_utils.analysis_utils import get_relative_class_weights

if __name__ == "__main__":
    # Fix conda slow loading https://github.com/pytorch/pytorch/issues/537

# =============================================================================
#   HYPERPARAMETERS
# =============================================================================
    seed = 0
    torch.cuda.manual_seed(seed)

    run_name = "ResNet-test"
    data_dir = 'hymenoptera_data'

    # Placeholder to easily extract classes
    train_data = datasets.ImageFolder(os.path.join(data_dir, 'train'))
    classes = deepcopy(train_data.classes)

    n_epochs = 100
    workers = 4

    init_l_rate = 1e-3
    l_rate_decay = 0.1
    l_rate_decay_epoch = False # [25, 80, 200]
    l_rate_decay_patience = 4
    w_decay = 1e-4
    batch_size = 5
    num_channels = 3
    
    criterion = [nn.CrossEntropyLoss() for i in range(3)]
    
    shutdown_after = False
    report_results_per_n_batches = {'train':1, 'val':5}
    save_interval = 9999
    visualiser = 'tensorboard' # Visdom for local, Tensorboard for Colaboratory.
    cam_layer = 'conv1'
    negatives_class = 0

# =============================================================================
#   LOAD MODEL
# =============================================================================
    model = models.resnet18(pretrained=True)
    num_ftrs = model.fc.in_features
    fc1 = nn.Linear(num_ftrs, 2)
    fc2 = nn.Linear(num_ftrs, 2)
    model = model_components.ResNet2FCL(model, [fc1, fc2])
    class_weights = get_relative_class_weights([5,3,2])

    if torch.cuda.is_available():
        model = model.cuda()
        criterion = criterion.cuda()
    
    optimizers = [optim.Adam(model.parameters(), lr=init_l_rate, weight_decay=w_decay) for i in range(2)]
# =============================================================================
#   DATA LOADING
# =============================================================================

    all_image_filepaths = i_manips.get_images(os.path.join(data_dir, 'train'))
    norm_params = i_manips.get_normalize_params(all_image_filepaths, num_channels)
    means = norm_params['means']
    sdevs = norm_params['sdevs']

    train_transforms = transforms.Compose([
                        transforms.Resize(224),
                        transforms.RandomCrop(224),
                        transforms.RandomHorizontalFlip(),
                        transforms.RandomGrayscale(),
                        transforms.RandomRotation(15),
                        transforms.ColorJitter(0.05, 0.05, 0.05, 0.05),
                        transforms.ToTensor(),
                        transforms.Normalize(means, sdevs)
                    ])

    val_transforms =  transforms.Compose([
                        transforms.Resize(224),
                        transforms.RandomCrop(224),
                        transforms.ToTensor(),
                        transforms.Normalize([means[0], means[1], means[2]], [sdevs[0], sdevs[1], sdevs[2]])
                    ])

    train_data = datasets.ImageFolder(os.path.join(data_dir, 'train'), transform=train_transforms)
    val_data = datasets.ImageFolder(os.path.join(data_dir, 'val'), transform=train_transforms)

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size,
                                                shuffle=True, num_workers=workers)

    val_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size,
                                                shuffle=True, num_workers=workers)                                                

    classes = deepcopy(train_data.classes)

# =============================================================================
#   INITIALIZE RUNTIME CLASSES
# =============================================================================
    analysis = runtime_logic.AnnotatedImageAnalysis(model, classes, means, sdevs, train_loader, val_loader)
    analysis.loss_tracker.setup_output_storage(run_name, 'outputs/')

# =============================================================================
#   INITIALIZE TRAINER    
# =============================================================================
    arguments = {# model components
                 'run_name':run_name,
                 'optimizer':optimizers,
                 'criterion':criterion,
                 
                 # Hyperparameters
                 'n_epochs':n_epochs,
                 'batch_size':batch_size,
                 'l_rate_decay':l_rate_decay,
                 'l_rate_decay_epoch':l_rate_decay_epoch,
                 'l_rate_decay_patience':l_rate_decay_patience,
                 
                 # Saving & Information retrieval
                 'report_interval':report_results_per_n_batches,
                 'save_interval':save_interval,
                 'shutdown':shutdown_after,
                 'visualiser':visualiser,
                 'cam_layer':cam_layer,
                 'negatives_class':negatives_class,
                 'class_weights':class_weights
                }

    analysis.train_multitask(arguments)