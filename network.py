import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from typing import List, Tuple, Dict, Union, Any, cast
from torch import Tensor

'''
naming convention:
    n_data: number of data
    n_synapse: number of synapse per input
    input_dim: dimension of input
'''


class ParallelSynapse(nn.Module):
    '''
    Parallel Synapse Module create parallel synapses per input with a given number of synapse
    '''

    def __init__(self, input_dim: int, n_synapse: int, input_range: Tuple = (-1, 1)) -> None:
        super().__init__()

        self.input_dim = input_dim
        self.n_synapse = n_synapse

        self.thres = nn.Parameter(torch.rand(
            n_synapse, input_dim) * (input_range[1] - input_range[0]) + input_range[0])
        self.slope = nn.Parameter(10 * torch.rand(n_synapse, input_dim))
        self.ampli = nn.Parameter(torch.rand(n_synapse, input_dim))

    def forward(self, input: Tensor):
        '''
        Inputs:
            input: (n_data, input_dim)

        Returns:
            output: (n_data, input_dim)
        '''

        n_data = input.shape[0]
        x = self.slope[None, :, :].expand(n_data, self.n_synapse, self.input_dim) * \
            input[:, None, :].expand(n_data, self.n_synapse, self.input_dim)
        x = x - self.thres[None, :,:].expand(n_data, self.n_synapse, self.input_dim)
        x = self.ampli[None, :, :].expand(n_data, self.n_synapse, self.input_dim) * \
            torch.tanh(x)
        x = x.sum(dim=1).squeeze(1)
        return x

class ParallelSynapseLayer(nn.Module):
    def __init__(self, input_dim: int, n_synapse: int, output_dim: int, input_range: Tuple = (-1, 1)) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.n_synapse = n_synapse
        self.output_dim = output_dim
        
        self.thres = nn.Parameter(torch.rand(
            self.n_synapse, self.input_dim, self.output_dim) * (input_range[1] - input_range[0]) + input_range[0])
        self.slope = nn.Parameter(10 * torch.rand(self.n_synapse, self.input_dim, self.output_dim))
        self.ampli = nn.Parameter(torch.rand(self.n_synapse, self.input_dim, self.output_dim))
    def forward(self, input: Tensor):
        '''
        Inputs:
            input: (n_data, input_dim)

        Returns:
            output: (n_data, output_dim)
        '''
        n_data = input.shape[0]
        x = self.slope[None, :, :,:].expand(n_data, self.n_synapse, self.input_dim, self.output_dim) * \
            input[:, None, :, None].expand(n_data, self.n_synapse, self.input_dim, self.output_dim)
        x = x - self.thres[None, :,:,:].expand(n_data, self.n_synapse, self.input_dim, self.output_dim)
        x = (self.ampli[None, :, :,:]**2).expand(n_data, self.n_synapse, self.input_dim, self.output_dim) * \
            torch.tanh(x)
        x = x.sum(dim=(1,2)).squeeze() 
        return x
        
        
class ParallelSynapseNeuron(nn.Module):
    '''
    single neuron with parallel synapse
    '''

    def __init__(self, input_dim: int, n_synapse: int, output_dim: int, input_range: Tuple = (-1.5, 1.5)) -> None:
        super().__init__()

        self.parallel_synapse = ParallelSynapse(
            input_dim, n_synapse, input_range)
        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, input: Tensor):
        '''
        Inputs:
            input: (n_data, input_dim)

        Returns:
            output: (n_data, output_dim)
        '''
        x = self.parallel_synapse(input)
        x = self.fc(x)
        return F.log_softmax(x, dim = 1)


        
class ParallelSynapseNN1(nn.Module):
    '''
    two layered neural network with parallel synapse at hidden layer
    be careful the input range for parallel synapse
    '''

    def __init__(self, input_dim: int, n_synapse: int, hidden_dim: int, output_dim: int, input_range: Tuple = (-0.5, 7.5)) -> None:
        super().__init__()

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.parallel_synapse = ParallelSynapseLayer(
            hidden_dim, n_synapse, output_dim, input_range)
    
    def forward(self, input: Tensor):
        '''
        Inputs:
            input: (n_data, input_dim)

        Returns:
            output: (n_data, output_dim)
        '''
        x = self.fc1(input) 
        x = torch.sigmoid(x)
        
        x = self.parallel_synapse(x) 
        
        return F.log_softmax(x, dim = 1)
    
class ParallelSynapseNN1Binary(ParallelSynapseNN1):
    '''
    trained to perform binary classification, using hinge loss
    parallel synapses are in the hidden layer
    '''
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        assert self.parallel_synapse.output_dim == 1
    
        self.theta = nn.Parameter(torch.tensor(0.0))    
        
    def forward(self, 
                input: Tensor, 
                hidden: str = 'sigmoid'):
        x = self.fc1(input)
        if hidden == 'sigmoid':
            x = torch.sigmoid(x)
        elif hidden == 'relu':
            x = F.relu(x)
        x = self.parallel_synapse(x)
        return (x - self.theta).squeeze()
    
class ParallelSynapseNN2(nn.Module):
    '''
    two layered neural network with parallel synapse at input layer
    be careful the input range for parallel synapse
    '''

    def __init__(self, input_dim: int, n_synapse: int, hidden_dim: int, output_dim: int, input_range: Tuple = (-1.5, 1.5)) -> None:
        super().__init__()

        self.parallel_synapse = ParallelSynapseLayer(
            input_dim, n_synapse, input_range = input_range)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, input: Tensor):
        '''
        Inputs:
            input: (n_data, input_dim)

        Returns:
            output: (n_data, output_dim)
        '''
        x = self.parallel_synapse(input)
        x = self.fc1(x)
        x = torch.sigmoid(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim = 1)

class TwoLayerNN(nn.Module):
    '''
    two layered neural network
    '''
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int) -> None:
        super().__init__()

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, input: Tensor):
        '''
        Inputs:
            input: (n_data, input_dim)

        Returns:
            output: (n_data, output_dim)
        '''
        x = self.fc1(input)
        x = torch.sigmoid(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim = 1)
class TwoLayerNNBinary(TwoLayerNN):
    '''
    two layered neural network
    '''
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs) 
        assert self.fc2.out_features == 1
        self.theta = nn.Parameter(torch.tensor(0.0))
        
    def forward(self, input: Tensor, hidden = 'sigmoid'):
        x = self.fc1(input)
        if hidden == 'sigmoid':
            x = torch.sigmoid(x)
        elif hidden == 'relu':
            x = F.relu(x) 
        x = self.fc2(x)
        
        return (x - self.theta).squeeze()
class SingleNeuron(nn.Module):
    '''
    single neuron
    '''

    def __init__(self, input_dim: int, output_dim: int, input_range: Tuple = (-1, 1)) -> None:
        super().__init__()

        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, input: Tensor):
        '''
        Inputs:
            input: (n_data, input_dim)

        Returns:
            output: (n_data, output_dim)
        '''
        x = self.fc(input)
        return F.log_softmax(x, dim = 1)
    