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
import datetime
device = torch.device("cpu")

def get_loader(dataset: str) -> Tuple:
    # MNIST datasets
    if dataset == 'MNIST':
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
        trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True)

        testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
        testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=False)
    # CIFAR10 datasets
    elif dataset == 'CIFAR10':
        transform = transforms.Compose(
            [transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                                download=True, transform=transform)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=64,
                                                shuffle=True, num_workers=4)
        testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                            download=True, transform=transform)
        testloader = torch.utils.data.DataLoader(testset, batch_size=64,
                                            shuffle=False, num_workers=4)
    return trainloader, testloader

def multiHingeLoss(output, target, margin = 1):
    '''
    Inputs:
        output: (batch_size, num_classes), direct output from final layer output
        target: (batch_size, num_classes), must be +1/-1
        margin: scalar, default = 1
    Outputs:
        loss: scalar
    '''
    loss = torch.clamp(margin - output * target, min=0)
    return loss.sum(dim=1).mean()

def oneHotLabel(label, num_classes):
    '''
    Inputs:
        label: (batch_size, 1)
        num_classes: scalar
    Outputs:
        one_hot_label: (batch_size, num_classes)
    '''
    batch_size = label.size(0)
    one_hot_label =(-torch.ones(batch_size, num_classes)).scatter_(1, label.view(-1, 1), 1)
    return one_hot_label

def train_models(model, 
                input_dim,
                trainloader: torch.utils.data.DataLoader, 
                testloader: torch.utils.data.DataLoader,
                scaler_reg: float = 0.0, 
                out_dim: int = 10, 
                num_epochs: int = 201,
                verbose: bool = True, 
                device = device,
                model_type: str = 'parallel',
                loss_type: str = 'nll',
                lr: float = 0.001,
                lr_thres: float = 0.05,
                lr_slope: float = 0.05,
                lr_ampli: float = 0.05,
                use_scheduler: bool = False,
                decrease_epoch: int = 50,
                decrease_factor: float = 0.1): 
    
    assert out_dim == 10
    
    params = {  
        'num_epochs': num_epochs,
        'loss_type': loss_type,
        'lr': lr,
        'scaler_reg': scaler_reg
    }
    
    special_params = {'parallel_synapse.thres', 'parallel_synapse.slope', 'parallel_synapse.ampli'}
    other_params = [param for param_name, param in model.named_parameters() if param_name not in special_params]
    if model_type == 'parallel':
        param_groups = [
            {'params': model.parallel_synapse.thres, 'lr': lr_thres},  
            {'params': model.parallel_synapse.slope, 'lr': lr_slope},
            {'params': model.parallel_synapse.ampli, 'lr': lr_ampli},
            {'params': other_params}  # Default learning rate for the rest
        ]
    else:
        param_groups = [
            {'params': other_params}  # Default learning rate for the rest
        ]
    model.to(device)
    optimizer = optim.Adam(param_groups, lr=lr)
    if use_scheduler:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer,  step_size=decrease_epoch, gamma=decrease_factor)

    losses, acc = [], []
    start = datetime.datetime.now()
    for epoch in range(num_epochs):
        model.train() 
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(trainloader):
            if model_type == 'parallel':
                with torch.no_grad():
                    slope_thres = 0.01
                    mask = (model.parallel_synapse.slope.data < slope_thres) 
                    model.parallel_synapse.slope.data = torch.clamp(model.parallel_synapse.slope.data, min = slope_thres)
                    model.parallel_synapse.thres.data[mask] = (torch.rand(mask.sum()) * model.hidden_range[1] + model.hidden_range[0]).to(device)
                    
                    ampli_thres = 0.1
                    mask = (model.parallel_synapse.ampli.data**2 < ampli_thres) 
                    model.parallel_synapse.thres.data[mask] = (torch.rand(mask.sum()) * model.hidden_range[1] + model.hidden_range[0]).to(device)
                    model.parallel_synapse.ampli.data[mask] = np.sqrt(ampli_thres)
                    
                    # model.parallel_synapse.thres.data = torch.clamp(model.parallel_synapse.thres.data, min = model.hidden_range[0] )
        
            inputs = inputs.view(-1, input_dim).to(device)
            
            if loss_type == 'hinge':
                labels = oneHotLabel(labels, out_dim)
                
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs) # default output: direct output from final layer
            
            if loss_type == 'nll':
                outputs = F.log_softmax(outputs, dim = 1)
                loss = F.nll_loss(outputs, labels) 
                
            elif loss_type == 'hinge':
                loss = multiHingeLoss(outputs, labels)
            
            if model_type == 'parallel':
                loss += scaler_reg * (torch.abs(model.parallel_synapse.scaler) - 1).abs().sum()
                
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        
        losses.append(running_loss / len(trainloader))
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for inputs, labels in testloader:
                inputs = inputs.view(-1, input_dim).to(device)
                labels = labels.to(device)
                outputs = model(inputs)
                total += labels.size(0)
                correct += torch.argmax(outputs, dim = 1).eq(labels).sum().item()
                
        acc.append(100 * correct / total)
        if use_scheduler:
            scheduler.step()
        if verbose and epoch % 10 == 0:
            print( f"       Epoch {epoch+1}/{num_epochs}, Loss: {running_loss / len(trainloader):.4f}, Accuracy: {acc[-1]:.2f}%") 

            print(f'        Elapsed time: {datetime.datetime.now() - start}')
    
    return {
            'model': model,
            'accuracy': acc,
            'loss': losses,
            'params': params
    }

import sys

if __name__ == '__main__':
    
    hidden_dim = int(sys.argv[1])
    n_synapse = int(sys.argv[2])
    
    model_type = sys.argv[3]
    loss_type = sys.argv[4]
    
    bias = int(sys.argv[5])
    range_upper = int(sys.argv[6])
    
    dataset = sys.argv[7]
    
    lr_thres = float(sys.argv[8])
    lr_slope = float(sys.argv[9])
    lr_ampli = float(sys.argv[10])
    
    use_scheduler = (sys.argv[11] == 'True')
    decrease_epoch = int(sys.argv[12])
    decrease_factor = float(sys.argv[13])
    assert model_type in ['parallel', '2nn']
    assert loss_type in ['nll', 'hinge']
    # dataset = 'MNIST'
    # dataset = 'CIFAR10'
    
    if dataset == 'MNIST':
        input_dim = 28*28
    elif dataset == 'CIFAR10':
        input_dim = 32*32*3
    
    hidden_range = (0, range_upper) #if hidden_dim < 40 else (0, 20)
    # bias = 10 if hidden_dim < 40 else 20
    # bias = 0 if loss_type == 'hinge' else bias
    
    n_Seed = 20
    
    '''
    multi-seed experiment
    '''
    
    results = []
    if model_type == 'parallel':
        file_name = './results_'+dataset+f'/{model_type}_{loss_type}_H{hidden_dim}_M{n_synapse}_bias{bias}_range{hidden_range[1]}_lr_thres{lr_thres}_lr_ampli{lr_ampli}_lr_slope{lr_slope}_scheduler{use_scheduler}_decrease_{decrease_epoch}_epoch_factor_{decrease_factor}_initialization_v2.pkl'  
    else:
        file_name = './results_'+dataset+f'/{model_type}_{loss_type}_H{hidden_dim}_scheduler{use_scheduler}_decrease_{decrease_epoch}_epoch_factor_{decrease_factor}.pkl'
    try:
        with open(file_name, 'rb') as f:
            results = pickle.load(f)
        i_start = len(results)
    except:
        i_start = 0
    # calculate time for training
    start = datetime.datetime.now()
    
    for i in range(i_start, n_Seed):
        torch.manual_seed(i)
        np.random.seed(i)
        print(f'Running {i+1}/{n_Seed} seed')
        
        end = datetime.datetime.now()
        print(f'    Elapsed time: {end - start}')
        
        # define model
        if model_type == 'parallel':
            model = ParallelSynapse2NN(input_dim=input_dim,
                hidden_dim = hidden_dim, 
                                    n_synapse = n_synapse,
                                    hidden_range = hidden_range, 
                                    additive_bias = bias)
            
        elif model_type == '2nn':
            model = TwoLayerNN(input_dim=input_dim, hidden_dim = hidden_dim)
        
        with open(file_name, 'wb') as f:
            pickle.dump(results, f)
        
        train_loader, test_loader = get_loader(dataset)
        results.append(train_models(model, 
                                    input_dim,
                            train_loader,
                            test_loader,
                            model_type=model_type,
                            loss_type=loss_type,
                            use_scheduler=use_scheduler,
                            lr_ampli=lr_ampli,
                            lr_slope=lr_slope,
                            lr_thres=lr_thres,
                            decrease_epoch = decrease_epoch, 
                            decrease_factor = decrease_factor))
        
        
