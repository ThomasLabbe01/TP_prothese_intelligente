import gzip
import pandas
import time
import numpy as np

from IPython import display

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import sigmoid
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
            predictions = predictions.argmax(axis=1)
        all_predictions.append(predictions.cpu().numpy())
        all_targets.append(targets.cpu().numpy())

    if all_predictions[0].shape[-1] > 1:
        predictions_np = np.concatenate(all_predictions, axis=0)
        targets_np = np.concatenate(all_targets, axis=0)
    else:
        predictions_np = np.concatenate(all_predictions).squeeze(-1)
        targets_np = np.concatenate(all_targets)
        predictions_np[predictions_np >= 0.5] = 1.0
        predictions_np[predictions_np < 0.5] = 0.0

    if training_before:
        model.train()

    return (predictions_np == targets_np).mean()

class formatGestureRecognitionDataset(Dataset):

    def __init__(self, dataset):
        super().__init__()
        self.targets = np.array(dataset[1])
        data = []
        for i, value in enumerate(dataset[0]):
            value = value.astype(np.float32)
            image_reshape = (torch.reshape(torch.from_numpy(value), (1, 8, 4)))
            transform = T.Resize(size = (32,32))
            image_resize = transform(image_reshape)
            data.append((image_resize, torch.tensor(self.targets[i])))
        self.data = data
    

    def __getitem__(self, index):
        data = self.data[index][0]
        target = self.targets[index]
        return data, target
    

    def __len__(self):
        return len(self.data)

class GestureRecognitionCNN(nn.Module):
    """
    Cette classe d??finit un r??seau convolutionnel permettant de classifier
    des mouvements de mains
    https://pubmed.ncbi.nlm.nih.gov/31765319/
    """
    def __init__(self, num_classes):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm2d(6),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2))
        self.fc = nn.Linear(400, 120)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(120, 84)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(84, num_classes)
    
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        out = self.relu(out)
        out = self.fc1(out)
        out = self.relu1(out)
        out = self.fc2(out)
        return out


    
# Initialisation des param??tres d'entra??nement
# Param??tres recommand??s:
# - Nombre d'epochs (nb_epoch = 12)
# - Taux d'apprentissage (learning_rate = 0.01)
# - Momentum (momentum = 0.9)
# - Taille du lot (batch_size = 32)
nb_epoch = 250
learning_rate = 0.01
momentum = 0.9
batch_size = 32

# Chargement des donn??es de test et d'entra??nement
accessPath = 'all_data/data_2_electrode_GEL_4072'
fileType = 'csv'
window_length = 150
emgCSV = DataProcessing(accessPath, fileType)
emgCSV.formatCSVFiles(window_length=window_length)
classifications = Classifications(emgCSV.emg_data, subject='0', statistique='mav', window_length=window_length)
classifications.data_segmentation(method='train_test_split', proportions=[0.75, 0.25, 0])

# Format data
trainData = formatGestureRecognitionDataset(classifications.trainData)
testData = formatGestureRecognitionDataset(classifications.testData)
# Cr??ation du sampler avec les classes balanc??es
balanced_train_sampler = create_balanced_sampler(trainData)
balanced_test_sampler = create_balanced_sampler(testData)

# Cr??ation du dataloader d'entra??nement
train_loader = DataLoader(trainData, batch_size=batch_size, sampler=balanced_train_sampler)
test_loader = DataLoader(testData, batch_size=batch_size, sampler=balanced_test_sampler)

def compute_confusion_matrix(model, dataloader, device, num_classes=6):
    
    # Mettre le model en mode ??valuation
    # Calculer toutes les pr??dictions sur le dataloader
    all_predictions = []
    all_targets = []
    for i_batch, batch in enumerate(dataloader):
        images, targets = batch
        images = images.to(device)
        targets = targets.to(device)
        with torch.no_grad():
            predictions = model(images)
            predictions = torch.argmax(predictions, dim=1)
        all_predictions.append(predictions.cpu().numpy())
        all_targets.append(targets.cpu().numpy())

    predictions_np = np.concatenate(all_predictions)
    targets_np = np.concatenate(all_targets)

    matrix = np.zeros((num_classes, num_classes))
    results = np.rint(predictions_np).reshape(-1).astype(int)
    for i in range(len(targets_np)):
        if targets_np[i] == results[i]:
            matrix[results[i]][targets_np[i]] += 1
        else:
            matrix[results[i]][targets_np[i]] += 1
            

    return matrix  # Retourner matrice de confusion


# Instancier votre r??seau GestureRecognitionCNN dans une variable nomm??e "model"
model = GestureRecognitionCNN(num_classes=6)
# ******

# Transf??rer le r??seau sur GPU ou CPU en fonction de la variable "DEVICE"
# Transfer the network to GPU or CPU depending on the "DEVICE" variable
model.to(DEVICE)

# Instancier une fonction d'erreur BinaryCrossEntropy
# et la mettre dans une variable nomm??e criterion
criterion = nn.BCEWithLogitsLoss()

# Instancier l'algorithme d'optimisation SGD
# Ne pas oublier de lui donner les hyperparam??tres
# d'entra??nement : learning rate et momentum!
optimisation = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)


# Mettre le r??seau en mode entra??nement
# Set the network in training mode
# ******
model.train()
# Boucle d'entra??nement / Training loop
for i_epoch in range(nb_epoch):

    start_time, train_losses = time.time(), []
    for i_batch, batch in enumerate(train_loader):
        images, targets = batch
        targets = targets.type(torch.FloatTensor).unsqueeze(-1)

        images = images.to(DEVICE)
        targets = targets.to(DEVICE)
        
        # *** TODO ***
        # Mettre les gradients ?? z??ro
        optimisation.zero_grad()

        # Calculer:
        # 1. l'inf??rence dans une variable "predictions"
        # 2. l'erreur dans une variable "loss"
        predictions = model(images)
        predictions = torch.argmax(predictions, dim=1)
        predictions = predictions.unsqueeze(-1)
        predictions = predictions.type(torch.FloatTensor)
        loss = criterion(predictions, targets)
        loss.requires_grad = True
        # R??tropropager l'erreur et effectuer
        # une ??tape d'optimisation
        loss.backward()
        optimisation.step()
        
        # Accumulation du loss de la batch
        # Accumulating batch loss
        train_losses.append(loss.item())

    print(' [-] epoch {:4}/{:}, train loss {:.6f} in {:.2f}s'.format(
        i_epoch+1, nb_epoch, np.mean(train_losses), time.time()-start_time))

# Affichage du score en test / Display test score
test_acc = compute_accuracy(model, test_loader, DEVICE)
print(' [-] test acc. {:.6f}%'.format(test_acc * 100))

# Affichage de la matrice de confusion / Display confusion matrix
matrix = compute_confusion_matrix(model, test_loader, DEVICE)
print(matrix)

# Lib??re la cache sur le GPU *important sur un cluster de GPU*
# Free GPU cache *important on a GPU cluster*
torch.cuda.empty_cache()