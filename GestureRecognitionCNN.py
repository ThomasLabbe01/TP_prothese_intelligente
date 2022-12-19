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

class formatGestureRecognitionDataset(Dataset):

    def __init__(self, dataset):
        super().__init__()
        self.targets = np.array(dataset[1])
        data = []
        for i, value in enumerate(dataset[0]):
            value = value.astype(np.float32)
            data.append((torch.reshape(torch.from_numpy(value), (1, 8, 4)), torch.tensor(self.targets[i])))
        self.data = data
    

    def __getitem__(self, index):
        data = self.data[index][0]
        target = self.targets[index]
        return data, target
    

    def __len__(self):
        return len(self.data)

class GestureRecognitionCNN(nn.Module):
    """
    Cette classe définit un réseau convolutionnel permettant de classifier
    des mouvements de mains
    https://pubmed.ncbi.nlm.nih.gov/31765319/
    """
    
    def __init__(self):
        super().__init__()
        # Initialiser ici les modules contenant des paramètres à optimiser.
        self.BN1 = nn.BatchNorm2d(num_features=1)
        self.C2 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1 ,bias=False, padding='same')
        self.BN3 = nn.BatchNorm2d(num_features=32)
        
        self.C4 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, bias=False, padding='same')
        self.BN5 = nn.BatchNorm2d(num_features=32)
        
        self.C6 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5, stride=1, bias=False, padding='same')
        self.BN7 = nn.BatchNorm2d(num_features=32)
        
        self.FC8 = nn.Linear(in_features=32, out_features=256, bias=False)
        self.D9 = nn.Dropout(p=0.5, inplace=False) 

        self.output = nn.Linear(in_features=256, out_features=6, bias=False)  # out_features = nombre de classes
    
    
    def forward(self, x):
        ## Effectuer l'inférence du réseau.
        #y = F.relu(self.BN1(x))
        y = F.relu(self.BN3(self.C2(x)))
        y = F.relu(self.BN5(self.C4(y)))
        y = F.relu(self.BN7(self.C6(y)))
        y = y.view(-1, 32)
        y = F.relu(self.FC8(y))
        y = self.D9(y)
        return self.output(y)


    


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

# Format data
trainData = formatGestureRecognitionDataset(classifications.trainData)
testData = formatGestureRecognitionDataset(classifications.testData)
# Création du sampler avec les classes balancées
balanced_train_sampler = create_balanced_sampler(trainData)
balanced_test_sampler = create_balanced_sampler(testData)

# Création du dataloader d'entraînement
train_loader = DataLoader(trainData, batch_size=batch_size, sampler=balanced_train_sampler)
test_loader = DataLoader(testData, batch_size=batch_size, sampler=balanced_test_sampler)

def compute_confusion_matrix(model, dataloader, device):
    
    # Mettre le model en mode évaluation
    # Calculer toutes les prédictions sur le dataloader
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

    predictions_np = np.concatenate(all_predictions)
    targets_np = np.concatenate(all_targets)

    # ******
    # Assigner la classe 0 ou 1 aux prédictions
    # Calculer la matrice de confusion. Attention de bien avoir
    # une matrice 2 par 2 en sortie
    matrix = np.zeros((2, 2))
    results = np.rint(predictions_np).reshape(-1).astype(int)
    for i in range(len(targets_np)):
        if targets_np[i] == results[i]:
            matrix[results[i]][targets_np[i]] += 1
        else:
            matrix[results[i]][targets_np[i]] += 1
            

    return matrix  # Retourner matrice de confusion


# Instancier votre réseau GestureRecognitionCNN dans une variable nommée "model"
model = GestureRecognitionCNN()
# ******

# Transférer le réseau sur GPU ou CPU en fonction de la variable "DEVICE"
# Transfer the network to GPU or CPU depending on the "DEVICE" variable
model.to(DEVICE)

# Instancier une fonction d'erreur BinaryCrossEntropy
# et la mettre dans une variable nommée criterion
criterion = nn.BCELoss()

# Instancier l'algorithme d'optimisation SGD
# Ne pas oublier de lui donner les hyperparamètres
# d'entraînement : learning rate et momentum!
optimisation = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)


# Mettre le réseau en mode entraînement
# Set the network in training mode
# ******
model.train()
# Boucle d'entraînement / Training loop
for i_epoch in range(nb_epoch):

    start_time, train_losses = time.time(), []
    for i_batch, batch in enumerate(train_loader):
        images, targets = batch
        targets = targets.type(torch.FloatTensor).unsqueeze(-1)
        print(images.size())
        print(targets)

        images = images.to(DEVICE)
        targets = targets.to(DEVICE)
        
        # *** TODO ***
        # Mettre les gradients à zéro
        optimisation.zero_grad()

        # Calculer:
        # 1. l'inférence dans une variable "predictions"
        # 2. l'erreur dans une variable "loss"
        predictions = model(images)
        predictions = torch.argmax(predictions, dim=1)
        print(predictions)
        loss = criterion(predictions, targets)

        # Rétropropager l'erreur et effectuer
        # une étape d'optimisation
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

# Libère la cache sur le GPU *important sur un cluster de GPU*
# Free GPU cache *important on a GPU cluster*
torch.cuda.empty_cache()

# *** TODO ***
# Entrez vos commentaires de la discussion ici.
# Enter your discussion comments here
discussion = f"Résultats prédit par la matrice : {100*np.round((matrix[0][0] + matrix[1][1])/(np.sum(matrix.reshape(-1))), 2)}%. Ainsi, le score prédit par la matrice est meilleur que celui indiqué par compute_accuracy"
# ******

frame = {"Comments":[discussion]}
df = pandas.DataFrame(frame)
display.display(df)