import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from typing import List, Tuple, Dict, Union, Any, cast
from torch import Tensor

'''
naming convention:
    n_data: number of data, P in the paper
    n_synapse: number of synapse per input, M in the paper
    input_dim: dimension of input
    hidden_dim: dimension of hidden layer, D_{hidden} in the paper
    output_dim: dimension of output, D_{output} in the paper
    
'''

class TwoLayerNN(nn.Module):
    '''
    two layered neural network, benchmark model
    
    Attributes:
        fc1: nn.Linear, hidden layer
        fc2: nn.Linear, output layer
    '''
    def __init__(self, 
                input_dim: int = 28*28, 
                hidden_dim: int = 10, 
                output_dim: int = 10) -> None:
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim) 
        self.fc2 = nn.Linear(hidden_dim, output_dim) 
        
    def forward(self, x: Tensor):
        '''
        Inputs:
            input: (n_data, input_dim)

        Returns:
            output: (n_data, output_dim)
        ''' 
        x = self.fc1(x)
        x = self.bn1(x) 
        hidden = nn.functional.softplus(x)
        x = self.fc2(hidden)
        return x, hidden

class ParallelSynapseLayer(nn.Module):
    '''
    parallel synapse layer, for each input-output connection, create a set of parallel synapses
    
    Attributes:
        input_dim: int, dimension of input, also as N  
        n_synapse: int, number of synapse per input, also as M  
        output_dim: int, dimension of output, also as K 
        
    
        thres: nn.Parameter, threshold for parallel synapses, shape: (n_synapse, input_dim, output_dim), or (M, N, K)
        slope: nn.Parameter, slope for parallel synapses, shape: (n_synapse, input_dim, output_dim) or (M, N, K)
        ampli: nn.Parameter, amplitude for parallel synapses, shape: (n_synapse, input_dim, output_dim) or (M, N, K)
        scaler: nn.Parameter, monotonicity indicator for each input-output connection, also as c_{i,k} in the paper, shape: (input_dim, output_dim) or (N, K)
        bias: nn.Parameter, bias for each output, shape: (output_dim, ) or (K, )
    '''
    def __init__(self, 
                input_dim: int, 
                output_dim: int, 
                n_synapse: int = 3,
                input_range: Tuple = (0, 3),
                slope_init: int = 1,
                ampli_init: int = 2) -> None:
        # input_range is the range of input, used to initialize thresholds,
        # in the case of two layered neural network with parallel synapse layer, input range is the range of estimated hidden layer input
        super().__init__()
        self.input_dim = input_dim
        self.n_synapse = n_synapse
        self.output_dim = output_dim 
        # initialize parameters
        # thresholds are uniformly distributed in the range of input_range
        self.thres = nn.Parameter(torch.rand(
            self.n_synapse, self.input_dim, self.output_dim) * (input_range[1] - input_range[0]) + input_range[0])
        # slopes are uniformly distributed in the range of (0, 5)
        self.slope = nn.Parameter(torch.rand(self.n_synapse, self.input_dim, self.output_dim)+slope_init)
        # amplitudes are uniformly distributed in the range of (0, 1)
        self.ampli = nn.Parameter(torch.rand(self.n_synapse, self.input_dim, self.output_dim)+ampli_init) 
        # bias are randomly initialized to be 0 or 1
        self.bias = nn.Parameter(torch.randn(self.output_dim, ))

    def forward(self, input: Tensor):
        '''
        compute output of parallel synapse layer
        
        Inputs:
            input: (n_data, input_dim) 

        Returns:
            output: (n_data, output_dim), or (P, K)
        '''
        n_data = input.shape[0]
        x = self.slope[None, :, :,:].expand(n_data, self.n_synapse, self.input_dim, self.output_dim) * \
            (input[:, None, :, None].expand(n_data, self.n_synapse, self.input_dim, self.output_dim) -
            self.thres[None, :,:,:].expand(n_data, self.n_synapse, self.input_dim, self.output_dim))
        x = (self.ampli[None, :, :,:]**2).expand(n_data, self.n_synapse, self.input_dim, self.output_dim) * \
            torch.tanh(x)  # also using tanh as activation function 
        x = x.sum(dim=(1)).squeeze() 
        x = x.sum(dim=(1)).squeeze() + self.bias# shape: (n_data, output_dim)
        
        return x

class ParallelSynapse2NN(nn.Module):
    '''
    two layered neural network with parallel synapse layer
    
    Attributes:
        hidden_range: Tuple, range of estimated hidden layer input, used to initialize thresholds
        additive_bias: int, additive bias for hidden layer, used to initialize bias of input to hidden layer, default 0
                if additive_bias larger 0, the hidden layer activation function is relu(x + additive_bias),
                this is used to make the hidden layer input to be more positive
        
        fc1: nn.Linear, hidden layer
        parallel_synapse: ParallelSynapseLayer, parallel synapse layer
        
    '''
    def __init__(self, 
                input_dim: int = 28*28, 
                hidden_dim: int = 20,         
                n_synapse: int = 3, 
                output_dim: int = 10, 
                hidden_range: Tuple = (0, 2), 
                slope_init: int = 5,
                ampli_init: int = 2) -> None:
        super().__init__()
        self.n_synapse = n_synapse
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim 
        
        self.hidden_range = hidden_range
        self.fc1 = nn.Linear(input_dim, hidden_dim) 
        self.bn1 = nn.BatchNorm1d(hidden_dim) 
        
        self.parallel_synapse = ParallelSynapseLayer(hidden_dim, 
                                                output_dim, 
                                                n_synapse = self.n_synapse,
                                                input_range = self.hidden_range,
                                                slope_init = slope_init,
                                                ampli_init = ampli_init)

    def forward(self, x: torch.Tensor) -> torch.Tensor: 
        x = self.fc1(x)
        x = self.bn1(x)  
        # soft relu
        hidden = nn.functional.softplus(x)
        
        x = self.parallel_synapse(hidden)
        
        return x, hidden

