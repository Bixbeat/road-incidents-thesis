import os

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, datasets, transforms
from torchvision.transforms import Compose, Normalize, ToTensor
from copy import deepcopy
from PIL import Image

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

    num_channels = 3
    num_classes = 2
    
    cam_layer = 'conv1'

# =============================================================================
#   LOAD MODEL
# =============================================================================
    model = models.resnet18(pretrained=True)
    num_ftrs = model.fc.in_features
    for param in model.parameters():
        param.requires_grad = False
    model.fc = nn.Linear(num_ftrs, num_classes)
# =============================================================================
#   DATA LOADING
# =============================================================================
    all_image_filepaths = i_manips.get_images(os.path.join(data_dir, 'train'))
    norm_params = i_manips.get_normalize_params(all_image_filepaths, num_channels)
    means = norm_params['means']
    sdevs = norm_params['sdevs']

    transforms = transforms.Compose([
                        transforms.Resize((224, 224)),
                        transforms.ToTensor(),
                        transforms.Normalize(means, sdevs)
                    ])

    classes = ['bee', 'ant']

# =============================================================================
#   INITIALIZE RUNTIME CLASSES
# =============================================================================
    analysis = runtime_logic.AnnotatedImageAnalysis(model, classes, means, sdevs)
    img = Image.open(all_image_filepaths[0])
    results = analysis.infer(img, transforms, cam_layer, colorramp='hot')