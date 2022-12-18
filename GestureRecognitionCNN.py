import gzip
import pandas
import time
import numpy as np

from IPython import display

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.optim import SGD
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageFolder
from torchvision.models import resnet18
import torchvision.transforms as T

import matplotlib
matplotlib.rcParams['figure.figsize'] = (9.0, 7.0)
from matplotlib import pyplot

from DataProcessing import DataProcessing
from Classifications import Classifications

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def create_balanced_sampler(dataset):
    def make_weights_for_balanced_classes(images, n_classes):                        
        count = [0] * n_classes                                                      
        for item in images:                                                         
            count[item[1]] += 1                                                     
        weight_per_class = [0.] * n_classes                                      
        N = float(sum(count))                                                   
        for i in range(n_classes):                                                   
            weight_per_class[i] = N/float(count[i])                                 
        weight = [0] * len(images)                                              
        for idx, val in enumerate(images):                                          
            weight[idx] = weight_per_class[val[1]]                                  
        return weight

    n_classes = np.unique(dataset.targets)
    weights = make_weights_for_balanced_classes(dataset.data, len(n_classes))                                                         
    weights = torch.DoubleTensor(weights)                 
    sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights)) 
    return sampler

def compute_accuracy(model, dataloader, device='cpu'):
    training_before = model.training
    model.eval()
    all_predictions = []
    all_targets = []
    
    for i_batch, batch in enumerate(dataloader):
        images, targets = batch
        images = images.to(device)
        targets = targets.to(device)
        with torch.no_grad():
            predictions = model(images)
        all_predictions.append(predictions.cpu().np())
        all_targets.append(targets.cpu().np())

    if all_predictions[0].shape[-1] > 1:
        predictions_np = np.concatenate(all_predictions, axis=0)
        predictions_np = predictions_np.argmax(axis=1)
        targets_np = np.concatenate(all_targets, axis=0)
    else:
        predictions_np = np.concatenate(all_predictions).squeeze(-1)
        targets_np = np.concatenate(all_targets)
        predictions_np[predictions_np >= 0.5] = 1.0
        predictions_np[predictions_np < 0.5] = 0.0

    if training_before:
        model.train()

    return (predictions_np == targets_np).mean()

class GestureRecognitionCNN(nn.Module):
    """
    Cette classe définit un réseau convolutionnel permettant de classifier
    des mouvements de mains
    https://pubmed.ncbi.nlm.nih.gov/31765319/
    """

    def __init__(self):
        super().__init__()
        # Initialiser ici les modules contenant des paramètres à optimiser.
        self.C1 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=2, bias=False)
        self.BN2 = nn.BatchNorm2d(num_features=32)
        
        self.C3 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=2, bias=False)
        self.BN4 = nn.BatchNorm2d(num_features=32)
        
        self.C5 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5, stride=2, bias=False)
        self.BN6 = nn.BatchNorm2d(num_features=32)
        
        self.FC7 = nn.Linear(in_features=32, out_features=6, bias=False)  # out_features = nombre de classes
    
    def forward(self, x):
        ## Effectuer l'inférence du réseau.
        x = F.relu(self.BN2(self.C1(x)))
        x = F.relu(self.BN4(self.C3(x)))
        x = F.relu(self.BN6(self.C5(x)))
        x = F.relu(self.FC7(x))
        return x

# Initialisation des paramètres d'entraînement
# Paramètres recommandés:
# - Nombre d'epochs (nb_epoch = 12)
# - Taux d'apprentissage (learning_rate = 0.01)
# - Momentum (momentum = 0.9)
# - Taille du lot (batch_size = 32)
nb_epoch = 12
learning_rate = 0.01
momentum = 0.9
batch_size = 32

# Chargement des données de test et d'entraînement
accessPath = 'all_data/data_2_electrode_GEL_4072'
fileType = 'csv'
window_length = 25
emgCSV = DataProcessing(accessPath, fileType)
emgCSV.formatCSVFiles(window_length=window_length)
classifications = Classifications(emgCSV.emg_data, subject='0', statistique='mav', window_length=window_length)
classifications.data_segmentation(method='train_test_split', proportions=[0.8, 0.2, 0])
train_set = classifications.trainData
test_set = classifications.testData
print(test_set)


# Création du sampler avec les classes balancées
balanced_train_sampler = create_balanced_sampler(train_set)
balanced_test_sampler = create_balanced_sampler(test_set)

# Création du dataloader d'entraînement
train_loader = Dataloader(train_set, batch_size=batch_size, sampler=balanced_train_sampler)
test_loader = DataLoader(test_set, batch_size=batch_size, sample=balanced_test_sampler)