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

def hingeLoss(actv: Tensor, 
            label: Tensor, 
            margin = 0
            ) -> Tensor:
    '''
    Inputs:
        actv: (n_data, 1) 
        label: (n_data, 1)
        margin: scalar, default = 0
    Outputs:
        loss: scalar
    '''
    return (torch.maximum(torch.zeros_like(actv), margin - actv * label)).sum()

def binary_labels(target: Tensor) -> Tensor:
    '''
    convert target to binary labels: 
    (0, 1, 2, 3, 4) -> (0, 0, 0, 0, 0)
    (5, 6, 7, 8, 9) -> (1, 1, 1, 1, 1)
    '''
    return (target > 4).long() * 2 - 1
def plot_input_histogram_to_parallel_synapse_layer(model) -> None:
    '''
    Inputs:
        model: nn.Module
    
    '''
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    
    testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=False)
    hidden = []
    for inputs, labels in testloader:
        inputs = inputs.view(-1, 28*28)#.to(device)
        
        labels = binary_labels(labels)
        hidden.append(torch.sigmoid(model.fc1(inputs)).detach().cpu())
    hidden = torch.cat(hidden, dim=0)
    plt.figure(figsize=(4, 4))
    plt.hist(hidden.data.cpu().numpy().flatten(), bins=100)
    plt.show(block = False)
    
def train_models_binary(seed,
                        M = 3,  
                        H = 20, 
                        in_dim = 28, 
                        out_dim = 1, 
                        num_epochs = 20,
                        verbose = True):
    '''
    train models for binary classification with MNIST dataset
    
    INPUTs:
        seed: int,
            manual seeds for initialization
        M: int,
            specifying number of parallel synapses
        H: int,
            specifying number of hidden units
        num_epochs: int,
            number of epochs for training
            
    OUTPUTs:
    '''
    assert out_dim == 1
    torch.manual_seed(seed)
    models = {
        '2-NN (H={:d})'.format(H):  TwoLayerNNBinary(in_dim * in_dim, H, out_dim), \
        '2-NN with parallel synapse (M={:d}) at hidden layer (H={:d})'.format(M, H):  ParallelSynapseNN1Binary(in_dim * in_dim, M, H, out_dim)
    }
    accuracies = {}
    losses = {}
    for m in models:
        accuracies[m] = []
        losses[m] = []
        
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True)

    testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=False)
    for model_name in models:
        model = models[model_name]
        # Initialize model, criterion, and optimizer 
        model.to(device)
        if 'parallel synapse' in model_name: 
            lr = 0.00001
            params = [
                {"params": model.fc1.parameters(),"weight_decay": 0.0, "learning_rate": lr},
                {"params": model.parallel_synapse.ampli, "learning_rate": lr},
                {"params": model.parallel_synapse.thres, "learning_rate": lr },
                {"params": model.parallel_synapse.slope, "learning_rate": lr},
                {"params": model.theta, "learning_rate": lr},
            ]
            
        else:
            params = model.parameters()
            lr = 0.00005
        optimizer = optim.Adam(params, lr=lr) #  0.00005 for parallel synapse
        for epoch in range(num_epochs):
            # Training loop
            if 'parallel synapse' in model_name:
                with torch.no_grad():
                    slope_thres = 0.01
                    mask = (model.parallel_synapse.slope.data < slope_thres) 
                    model.parallel_synapse.slope.data = torch.clamp(model.parallel_synapse.slope.data, min = slope_thres)
                    model.parallel_synapse.thres.data[mask] = torch.rand(mask.sum()) * 8
                    
                    ampli_thres = 0.01
                    mask = (model.parallel_synapse.ampli.data**2 < ampli_thres) 
                    model.parallel_synapse.thres.data[mask] = torch.rand(mask.sum()) * 8
                    model.parallel_synapse.ampli.data[mask] = np.sqrt(ampli_thres)
                    
                    model.parallel_synapse.thres.data = torch.clamp(model.parallel_synapse.thres.data, min = -0.5)
                    # 
                    
            model.train() 
            running_loss = 0.0
            for i, (inputs, labels) in enumerate(trainloader):
                inputs = inputs.view(-1, 28*28).to(device)
                labels = labels.to(device)
                optimizer.zero_grad()
                outputs = model(inputs, hidden = 'relu')
                # import pdb; pdb.set_trace()
                loss = hingeLoss(outputs, binary_labels(labels), margin=1) 
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            losses[model_name].append(running_loss / len(trainloader))
            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for inputs, labels in testloader:
                    inputs = inputs.view(-1, 28*28).to(device)
                    labels = labels.to(device)
                    
                    outputs = model(inputs, hidden = 'relu')
                    # import pdb; pdb.set_trace()
                    predicted = torch.sign(outputs).squeeze().long()
                    total += labels.size(0)
                    correct += predicted.eq(binary_labels(labels)).sum().item()
                # import pdb; pdb.set_trace()
            accuracy = 100 * correct / total 
            accuracies[model_name].append(accuracy)
            
            # plt.close(1)
            if verbose and epoch % 10 == 0:
                # plot_input_histogram_to_parallel_synapse_layer(model)
                print(model_name + f" Epoch {epoch+1}/{num_epochs}, Loss: {running_loss / len(trainloader):.4f}, Accuracy: {accuracy:.2f}%")
    
    return {'seed': seed, 
            'M': M,
            'models': models,
            'accuracies': accuracies,
            'losses': losses
            }
            
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
        
        '2-NN (H={:d})'.format(H):  TwoLayerNN(in_dim * in_dim, H, out_dim), \
        
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
                    
                    mask = (model.parallel_synapse.ampli.data < 0.1) 
                    model.parallel_synapse.ampli.data = torch.clamp(model.parallel_synapse.ampli.data, min = 0.1)
                    model.parallel_synapse.thres.data[mask] = torch.rand(mask.sum())  * 10
                    
                    model.parallel_synapse.thres.data = torch.clamp(model.parallel_synapse.thres.data, min = 0, max = 1)
                
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
        result_dict = train_models_binary(i, H=H,M=M, num_epochs = 201)
        experiment.append(result_dict) 

        df = pd.DataFrame(experiment)
        with open('./data/MNIST_result_H={:d}_M={:d}_.pkl'.format(H,M), 'wb') as f:
            pickle.dump(df, f)