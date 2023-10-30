
import pickle
import matplotlib.pyplot as plt
import numpy as np
from training import *

def plot_parallel_synapse_params(model) -> None: 
    '''
    plot histograms of parallel synapses, such as amplitude, slope, threshold, scaler   
    
    Inputs:
        model: nn.Module
    '''
    plt.figure(figsize=(12, 3))
    plt.subplot(1,4,1)
    plt.hist((model.parallel_synapse.ampli.data.cpu().numpy()**2).flatten(), bins=20)
    plt.title('Amplitude histogram')

    plt.subplot(1,4,2)
    plt.hist(model.parallel_synapse.slope.data.cpu().numpy().flatten(), bins=20)
    plt.title('Slope histogram')

    plt.subplot(1,4,3)
    plt.hist(model.parallel_synapse.thres.data.cpu().numpy().flatten(), bins=20)
    plt.title('Threshold histogram')
    
    plt.subplot(1,4,4)
    plt.hist(model.parallel_synapse.scaler.data.cpu().numpy().flatten(), bins=20)
    plt.title('Scaler histogram')
    
    plt.tight_layout()
    plt.show()

def plot_input_histogram_to_parallel_synapse_layer(model, 
                                                testloader, 
                                                hidden_act:str = 'sigmoid', 
                                                is_normalized = False) -> None:
    '''
    plot histograms of hidden layer input and final layer input
    
    Inputs:
        model: nn.Module
    '''
    hidden = []
    final = []
    for inputs, labels in testloader:
        inputs = inputs.view(-1, 28*28)#.to(device)
        
        if hidden_act == 'sigmoid':
            hidden.append(torch.sigmoid(model.fc1(inputs)).detach().cpu())
        else:
            hidden.append(torch.relu(model.fc1(inputs)).detach().cpu())
        final.append(model.parallel_synapse(hidden[-1]).detach().cpu())
        
    hidden_thres = 1e-4
    hidden = torch.cat(hidden, dim=0)
    hidden = hidden.data.cpu().numpy()
    final = torch.cat(final, dim=0)
    plt.figure(figsize=(9, 3 ))
    plt.subplot(1,3,1)
    plt.hist(hidden[hidden>hidden_thres].flatten(), bins=100)
    plt.title('Hidden activation, {:.1f}% > {:.2E}'.format((hidden>hidden_thres).mean() * 100, hidden_thres))
    # plt.legend()
    plt.xlabel('hidden layer input')
    plt.ylabel('count')
    
    f = model.parallel_synapse
    slope = f.slope
    ampli = f.ampli
    thres = f.thres
    input_dim = slope.shape[1]
    n_data = 100
    output_dim = slope.shape[-1]
    input = torch.cat([torch.linspace(0, hidden[:, i].std()*3 + hidden[:, i].mean(), steps = n_data).reshape(-1, 1) for i in range(hidden.shape[1])], dim = 1)

    n_data = input.shape[0]
    n_synapse = slope.shape[0]
    

    x = slope[None, :, :, :].expand(n_data, n_synapse, input_dim, output_dim) \
        * (input[:, None, :, None].expand(n_data, n_synapse, input_dim, output_dim)
        - thres[None, :, :, :].expand(n_data, n_synapse, input_dim, output_dim))
    x = torch.tanh(x)
    x = x * (ampli[None, :, :, :]**2 ).expand(n_data, n_synapse, input_dim, output_dim)
    # x = x * f.scaler[None, None, :, :].expand(n_data, n_synapse, input_dim, output_dim)
    x = x.sum(dim=1).squeeze() 
    # if is_normalized:

    #     input = input * hidden.max(dim = 0)
    
    plt.subplot(1,3,2)
    plt.hist(final.data.cpu().numpy().flatten(), bins=100, label='MNIST')
    ylim = plt.ylim()
    plt.legend()
    plt.xlabel('output')
    plt.title('Histogram, final layer input')
    
    plt.tight_layout()
    plt.show()
    
    plt.figure(figsize = (10, 8))
    
    for i in range(min([20, input_dim])):
        plt.subplot(4,5,i+1)
        plt.plot( torch.linspace(0, 1, steps=n_data), x.detach().numpy()[:,i,:], alpha = 0.8)
        plt.title(str(i+1) + '-th hidden -> output')
        plt.xlabel('Input')
    plt.tight_layout()
    plt.show()
    
def plot_result(result) -> None:
    '''
    plot training result
    
    Inputs:
        result: dict, output from train_models_v3
    '''
    plt.figure(figsize=(6, 3))
    plt.subplot(1,2,1)
    plt.plot(result['loss'])
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss')
    plt.subplot(1,2,2)
    plt.plot(result['accuracy'])
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy')
    plt.tight_layout()
    plt.show()
    
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
        acc.append(result[i]['accuracy'])
    return np.array(acc)

def get_hidden_final_activation_2NN(model, testloader) -> Tuple[np.ndarray, np.ndarray]:
    '''
    return hidden layer activation and final layer activation
    
    Inputs:
        model: nn.Module
        testloader: torch.utils.data.DataLoader
    Outputs:
        hidden: np.ndarray, hidden layer activation
        final: np.ndarray, final layer activation
    '''
    model.eval()
    hidden, final = [], []
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs = inputs.view(-1, 28*28).to(device)
            hidden.append(torch.relu(model.fc1(inputs)).detach().cpu().numpy())
            final.append(model.fc2(torch.relu(model.fc1(inputs))).detach().cpu().numpy())
    return np.concatenate(hidden, axis=0), np.concatenate(final, axis=0)

def plot_hidden_final_loss_2NN(model, hidden, final, result) -> None:
    '''
    plot hidden layer activation, final layer activation, loss, and accuracy, for 2NN
    
    Inputs:
        model: nn.Module
        hidden: np.ndarray, hidden layer activation
        result: dict, output from train_models_v3
    
    '''
    plt.figure(figsize=(12, 3)) 
    plt.subplot(1,4,3)
    plt.hist(hidden[hidden>0.00001].flatten(), bins=100)
    plt.title('Hidden activation, {:.1f}% > 1e-4'.format((hidden>1e-4).mean() * 100))
    plt.xlabel('Activation')
    plt.ylabel('Count')
    plt.subplot(1,4,4)
    plt.hist(final.flatten(), bins=100)
    plt.title('Final activation')
    plt.xlabel('Activation')
    plt.ylabel('Count')
    plt.subplot(1, 4, 2)
    plt.plot(result['loss'])
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss')
    plt.subplot(1, 4, 1)
    plt.plot(result['accuracy'])
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy')
    plt.tight_layout()
    plt.show()