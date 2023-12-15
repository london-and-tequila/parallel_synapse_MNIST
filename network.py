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

    
class TwoLayerNN(nn.Module):
    '''
    two layered neural network
    '''
    def __init__(self, 
                input_dim: int = 28*28, 
                hidden_dim: int = 10, 
                output_dim: int = 10,
                drop_out: float = 0.1) -> None:
        super().__init__()

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        # self.dropout = nn.Dropout(drop_out) 
    def forward(self, x: Tensor):
        '''
        Inputs:
            input: (n_data, input_dim)

        Returns:
            output: (n_data, output_dim)
        '''
        # x = self.dropout(x)
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        return x

class ParallelSynapseLayer(nn.Module):
    def __init__(self, 
                input_dim: int, 
                n_synapse: int, 
                output_dim: int, 
                input_range: Tuple = (-1, 1)) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.n_synapse = n_synapse
        self.output_dim = output_dim
        
        self.thres = nn.Parameter(torch.rand(
            self.n_synapse, self.input_dim, self.output_dim) * (input_range[1] - input_range[0]) + input_range[0])
        # normal distribution for threshold:
        # self.thres = torch.abs(torch.normal(0, input_range[1] , size=(self.n_synapse, self.input_dim, self.output_dim)) )
        self.slope = nn.Parameter(5*torch.rand(self.n_synapse, self.input_dim, self.output_dim))
        self.ampli = nn.Parameter(torch.rand(self.n_synapse, self.input_dim, self.output_dim)) 
        # self.ampli = nn.Parameter(torch.ones(self.n_synapse, self.input_dim, self.output_dim) + 0.1*(torch.randn(self.n_synapse, self.input_dim, self.output_dim)))
        self.scaler = nn.Parameter(torch.randint(0,2,(self.input_dim, self.output_dim))*2-torch.ones(self.input_dim, self.output_dim))
        self.bias = nn.Parameter(torch.rand(self.output_dim, ))
    def forward(self, input: Tensor):
        '''
        Inputs:
            input: (n_data, input_dim)

        Returns:
            output: (n_data, output_dim)
        '''
        n_data = input.shape[0]
        x = self.slope[None, :, :,:].expand(n_data, self.n_synapse, self.input_dim, self.output_dim) * \
            (input[:, None, :, None].expand(n_data, self.n_synapse, self.input_dim, self.output_dim) -
            self.thres[None, :,:,:].expand(n_data, self.n_synapse, self.input_dim, self.output_dim))
        x = (self.ampli[None, :, :,:]**2).expand(n_data, self.n_synapse, self.input_dim, self.output_dim) * \
            torch.tanh(x) 
        x = x.sum(dim=(1)).squeeze()
        x = x * self.scaler[None, :, :].expand(n_data, self.input_dim, self.output_dim)
        x = x.sum(dim=(1)).squeeze() + self.bias# shape: (n_data, output_dim)
        return x

class ParallelSynapse2NN(nn.Module):
    def __init__(self, 
                input_dim: int = 28*28, 
                hidden_dim: int = 10, 
                n_synapse: int = 3, 
                output_dim: int = 10, 
                hidden_range: Tuple = (0, 40),
                additive_bias: int = 0) -> None:
        super().__init__()
        self.hidden_range = hidden_range
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.parallel_synapse = ParallelSynapseLayer(hidden_dim, 
                                                n_synapse, 
                                                output_dim, 
                                                input_range = self.hidden_range)
        self.additive_bias = additive_bias
        # self.dropout = nn.Dropout(0.1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x = self.dropout(x)
        x = self.fc1(x)
        
        x = torch.relu(x + self.additive_bias) 
        
        x = self.parallel_synapse(x)
        
        return x

ParallelSynapseLayer_FlipSign = ParallelSynapseLayer
