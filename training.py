from network import * 
import math
from typing import Any

import torch
from torch import Tensor
from torch.nn.parameter import Parameter, UninitializedParameter
from torch.nn import functional as F
from torch.nn import Module, init
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.optim as optim
import numpy as np
import torchvision
import torchvision.transforms as transforms
import pickle
import pandas as pd
from multiprocessing import Pool

device = torch.device("cpu")
def train_models(seed, 
                M = 3,  
                H = 20, 
                in_dim = 28, 
                out_dim = 10, 
                num_epochs = 20,
                verbose = True):
    '''
    INPUTs:
        seed: int, 
            manual seeds for initialization
        M: int, 
            specifying number of parallel synapses
        save: bool = False, 
            whether to save the model
    OUTPUTs:
        
    '''
    torch.manual_seed(seed)
    models = {
        # '1-NN':  SingleNeuron(in_dim * in_dim, out_dim), \
        # '1-NN with parallel synapse':  ParallelSynapseNeuron(in_dim * in_dim, M, out_dim), \
        
        # '2-NN (H={:d})'.format(H):  TwoLayerNN(in_dim * in_dim, H, out_dim), \
        
        '2-NN with parallel synapse (M={:d}) at hidden layer (H={:d})'.format(M, H):  ParallelSynapseNN1(in_dim * in_dim, M, H, out_dim),\
        # '2-NN with parallel synapase (M={:d}) at input layer (H={:d})'.format(M, H):  ParallelSynapseNN2(in_dim * in_dim, M, H, out_dim)
        }
    accuracies = {}
    losses = {}
    for m in models:
        accuracies[m] = []
        losses[m] = []
            
    # Load the MNIST dataset
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True)

    testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=False)
    
    for model_name in models:
        model = models[model_name]
        # Initialize model, criterion, and optimizer 
        model.to(device)
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        for epoch in range(num_epochs):
            # Training loop
            if 'parallel synapse' in model_name:
                with torch.no_grad():
                    model.parallel_synapse.slope.data = torch.clamp(model.parallel_synapse.slope.data, min = 0)
                    model.parallel_synapse.ampli.data = torch.clamp(model.parallel_synapse.ampli.data, min = 0)
                    
                    mask = ((model.parallel_synapse.thres.data < 0) + (model.parallel_synapse.thres.data > 1)).bool()
                    model.parallel_synapse.thres.data[mask] = torch.rand(mask.sum())
                    # model.parallel_synapse.thres.data = torch.clamp(model.parallel_synapse.thres.data, min = 0, max = 1)
                    
                    mask = model.parallel_synapse.ampli.data < 1e-3
                    model.parallel_synapse.ampli.data[mask] = torch.rand(mask.sum()) 
                
            model.train() 
            running_loss = 0.0
            for i, (inputs, labels) in enumerate(trainloader):
                inputs = inputs.view(-1, 28*28).to(device)
                labels = labels.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = F.nll_loss(outputs, labels) 
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            losses[model_name].append(running_loss / len(trainloader))
            # Validation loop for accuracy
            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for inputs, labels in testloader:
                    inputs = inputs.view(-1, 28*28).to(device)
                    labels = labels.to(device)
                    
                    outputs = model(inputs)
                    _, predicted = outputs.max(1)
                    total += labels.size(0)
                    correct += predicted.eq(labels).sum().item()

            accuracy = 100 * correct / total 
            accuracies[model_name].append(accuracy)
            if verbose and epoch % 10 == 0:
                print(model_name + f" Epoch {epoch+1}/{num_epochs}, Loss: {running_loss / len(trainloader):.4f}, Accuracy: {accuracy:.2f}%")
    
    return {'seed': seed, 
            'M': M,
            'models': models,
            'accuracies': accuracies,
            'losses': losses
            }


import sys

if __name__ == '__main__':
    
    H = int(sys.argv[1])
    M = int(sys.argv[2])
            
    experiment = []
    for i in range(5):
        result_dict = train_models(i, H=H,M=M, num_epochs = 51)
        experiment.append(result_dict)
        # plot_result(result_dict)

        df = pd.DataFrame(experiment)
        with open('./data/MNIST_result_H={:d}_M={:d}_.pkl'.format(H,M), 'wb') as f:
            pickle.dump(df, f)