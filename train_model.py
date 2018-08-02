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
    data_dir = 'hymenoptera'
    # data_dir = '/media/alex/A4A034E0A034BB1E/incidents-thesis/test-run/incidents_cleaned'

    n_epochs = 50
    workers = 4

    init_l_rate = 1e-4
    lr_decay = 0.1
    lr_decay_patience = 2
    w_decay = 1e-4
    
    batch_size = 5
    num_channels = 3
    num_classes = 2
    
    class_weights = get_relative_class_weights(data_dir)
    if torch.cuda.is_available():
        class_weights = class_weights.cuda()
    criterion = nn.CrossEntropyLoss()#class_weights)
    
    shutdown_after = False
    report_results_per_n_batches = {'train':2, 'val':5}
    save_interval = 9999
    visualiser = 'tensorboard' # Visdom for local, Tensorboard for Colaboratory.
    cam_layer = 'conv1'

# =============================================================================
#   LOAD MODEL
# =============================================================================
    model = models.resnet18(pretrained=True)
    num_ftrs = model.fc.in_features
    for param in model.parameters():
        param.requires_grad = False
    model.fc = nn.Linear(num_ftrs, num_classes)

    if torch.cuda.is_available():
        criterion = criterion.cuda()
        model = model.cuda()
    
    optimizer = optim.RMSprop(model.fc.parameters(), lr=init_l_rate, weight_decay=w_decay)
# =============================================================================
#   DATA LOADING
# =============================================================================

    all_image_filepaths = i_manips.get_images(os.path.join(data_dir, 'train'))
    norm_params = i_manips.get_normalize_params(all_image_filepaths, num_channels)
    means = norm_params['means']
    sdevs = norm_params['sdevs']

    train_transforms = transforms.Compose([
                        transforms.Resize((224, 224)),
                        transforms.ToTensor(),
                        transforms.Normalize(means, sdevs)
                    ])

    val_transforms =  transforms.Compose([
                        transforms.Resize((224, 224)),
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
    analysis.instantiate_loss_tracker(output_dir='outputs/')
    analysis.loss_tracker.setup_output_storage(run_name)

# =============================================================================
#   INITIALIZE TRAINER    
# =============================================================================
    arguments = {# model components
                 'run_name':run_name,
                 'optimizer':optimizer,
                 'criterion':criterion,
                 
                 # Hyperparameters
                 'n_epochs':n_epochs,
                 'batch_size':batch_size,
                 'lr_decay':lr_decay,
                 'lr_decay_patience':lr_decay_patience,
                 
                 # Saving & Information retrieval
                 'report_interval':report_results_per_n_batches,
                 'save_interval':save_interval,
                 'shutdown':shutdown_after,
                 'visualiser':visualiser,
                 'cam_layer':cam_layer
                }

    analysis.train(arguments)