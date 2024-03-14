
import pickle
import matplotlib.pyplot as plt
import numpy as np
from training import *

def get_acc(result) -> np.ndarray:
    '''
    return accuracy from training result
    
    Inputs:
        result: dict, output from train_models_v3
    Outputs:
        acc: np.ndarray, accuracy
    '''
    acc = []
    for i in range(len(result)):
        acc.append(result[i]['accuracy'][:101])
    return np.array(acc)

def get_loss(results):
    '''
    get loss from results dictionary
    '''
    loss = []
    for result in results:
        loss.append(result['loss'][:101])
    return np.array(loss)

def plot_synaptic_distribution(model):
    slope = model.parallel_synapse.slope.detach().numpy().ravel()
    ampli = model.parallel_synapse.ampli.detach().numpy().ravel()
    thres = model.parallel_synapse.thres.detach().numpy().ravel() 
    plt.figure( figsize = (9,3), dpi= 300)
    plt.subplot(1,3,1)
    plt.hist(slope, bins=30, label = f'slope, M = {model.parallel_synapse.slope.shape[0]}')
    plt.xlabel('Slope')
    plt.ylabel('Frequency')
    
    plt.subplot(1,3,2)
    plt.hist(ampli**2, bins=30, label = f'amplitude, M = {model.parallel_synapse.slope.shape[0]}')
    plt.xlabel('Amplitude')
    plt.ylabel('Frequency')
    
    plt.subplot(1,3,3)
    plt.hist(thres, bins=30, label = f'threshold, M = {model.parallel_synapse.slope.shape[0]}') 
    plt.xlabel('Threshold')
    plt.ylabel('Frequency')
    
    plt.tight_layout()
    plt.savefig('fig7.pdf') 
    plt.show()
        
def plot_hidden_actv_linear(model, 
                    testloader,   
                    file_name = 'fig6.pdf') -> None:
    '''
    plot histograms of hidden layer input and final layer input 
    with histogram normalization
    
    also plot learned aggregate synaptic function for each hidden-output connection
    grouped by output unit
    
    Inputs:
        model: nn.Module
        testloader: torch.utils.data.DataLoader 
        file_name: str, file name to save figure 
    '''
    
    hidden = []
    final = []
    for inputs, labels in testloader:
        
        inputs = inputs.view(-1, 28*28)#.to(device)
        f, h = model(inputs)
        hidden.append(h.detach().cpu())
        final.append(f.detach().cpu())
        
    hidden_thres = 0 # if hidden unit input is too small and close to zero, we ignore it
    hidden = torch.cat(hidden, dim=0)
    hidden = hidden.data.cpu().numpy() # n_data x n_hidden
    final = torch.cat(final, dim=0)
    f = model.parallel_synapse
    slope = f.slope
    ampli = f.ampli
    thres = f.thres
    input_dim = slope.shape[1]
    n_data = 100
    output_dim = slope.shape[-1]
    
    hidden_list = [] 
    input_list = []
    for i in np.arange(hidden.shape[1]):
        tmp_hidden = hidden[:, i].ravel()
        
        tmp_hidden = tmp_hidden[tmp_hidden > hidden_thres]
        
        min_val = -1# min(tmp_hidden.min(), model.parallel_synapse.thres[:, i,:].detach().numpy().min()-1)
        max_val = max(tmp_hidden.max(), model.parallel_synapse.thres[:, i,:].detach().numpy().max()+1)
        input_list.append(np.linspace(min_val, max_val, n_data))
        hidden_list.append(tmp_hidden)
    
    input = torch.cat([torch.tensor(input_list[i]).reshape(-1, 1) for i in np.arange(hidden.shape[1])], dim = 1)
    n_data = input.shape[0]
    n_synapse = slope.shape[0]
    
    # aggregate synaptic function
    x = slope[None, :, :, :].expand(n_data, n_synapse, input_dim, output_dim) \
        * (input[:, None, :, None].expand(n_data, n_synapse, input_dim, output_dim)
        - thres[None, :, :, :].expand(n_data, n_synapse, input_dim, output_dim))
    x = torch.tanh(x)
    x = x * (ampli[None, :, :, :]**2 ).expand(n_data, n_synapse, input_dim, output_dim)
    x = x.sum(axis=1).squeeze().detach().numpy() # shape: n_data x input_dim x output_dim
    
    fig, axs = plt.subplots(2,5, figsize = (17, 6), dpi = 300)
    for i in range(min(hidden.shape[1],10)):
        # hidden unit activation histogram 
        ax1 = axs.flatten()[i]
        ax1.hist(hidden_list[i], bins = 50,  alpha=0.75)
        # twin axis to plot aggregate synaptic function
        ax2 = ax1.twinx()
        ax2.plot(input[:,i].numpy(), x[:, i, :], linewidth = 1.5)
        ax2.set_ylabel('Synaptic function')
        ax1.set_xlim([hidden_list[i].min(), hidden_list[i].max()])
        ax1.set_title(f'Hidden unit {i+1}')
        ax1.set_xlabel('Activation')
        ax1.set_ylabel('Frequency') 
    plt.tight_layout() 
    plt.savefig(file_name)
    plt.show()
    
    
def plot_synaptic_func_by_output(model, 
                    testloader,
                    file_name = 'fig4d.pdf') -> None:
    '''
    plot histograms of hidden layer input and final layer input 
    with histogram normalization
    
    also plot learned aggregate synaptic function for each hidden-output connection
    grouped by output unit
    
    Inputs:
        model: nn.Module
        testloader: torch.utils.data.DataLoader
    '''
    
    # collection hidden layer input from test dataset
    hidden = []
    final = []
    for inputs, labels in testloader:
        
        inputs = inputs.view(-1, 28*28)#.to(device)
        f, h = model(inputs)
        hidden.append(h.detach().cpu())
        final.append(f.detach().cpu())
        
    hidden_thres = 0 # if hidden unit input is too small and close to zero, we ignore it
    hidden = torch.cat(hidden, dim=0)
    hidden = hidden.data.cpu().numpy() # n_data x n_hidden
    final = torch.cat(final, dim=0)
    f = model.parallel_synapse
    slope = f.slope
    ampli = f.ampli
    thres = f.thres
    input_dim = slope.shape[1]
    n_data = 100
    output_dim = slope.shape[-1]
    
    hidden_list = [] 
    input_list = []
    for i in np.arange(hidden.shape[1]):
        tmp_hidden = hidden[:, i].ravel()
        
        tmp_hidden = tmp_hidden[tmp_hidden > hidden_thres]
        
        min_val = -1# min(tmp_hidden.min(), model.parallel_synapse.thres[:, i,:].detach().numpy().min()-1)
        max_val = max(tmp_hidden.max(), model.parallel_synapse.thres[:, i,:].detach().numpy().max()+1)
        input_list.append(np.linspace(min_val, max_val, n_data))
        hidden_list.append(tmp_hidden)
    
    input = torch.cat([torch.tensor(input_list[i]).reshape(-1, 1) for i in np.arange(hidden.shape[1])], dim = 1)
    
    n_data = input.shape[0]
    n_synapse = slope.shape[0]
    
    # aggregate synaptic function
    x = slope[None, :, :, :].expand(n_data, n_synapse, input_dim, output_dim) \
        * (input[:, None, :, None].expand(n_data, n_synapse, input_dim, output_dim)
        - thres[None, :, :, :].expand(n_data, n_synapse, input_dim, output_dim))
    x = torch.tanh(x)
    x = x * (ampli[None, :, :, :]**2 ).expand(n_data, n_synapse, input_dim, output_dim)
    x = x.sum(axis=1).squeeze().detach().numpy() # shape: n_data x input_dim x output_dim
    # print(x.shape)
    
    plt.figure(figsize = (12, 5), dpi = 300)
    for i in range(10):
        plt.subplot(2,5,i+1)
        plt.plot(x[:, :, i], linewidth = 1.)  
        plt.title(f'Output unit {i+1}')
        plt.xlabel('Input (arbitrary unit)')
        
        
    plt.tight_layout() 
    plt.savefig(file_name)
    plt.show()
    