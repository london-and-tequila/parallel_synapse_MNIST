
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


 
    
def plot_input_histogram_to_parallel_synapse_layer(model, 
                                                testloader, 
                                                hidden_act:str = 'relu', 
                                                is_normalized = False,
                                                file_name = '',
                                                is_scaler = True,
                                                is_sign = False) -> None:
    '''
    plot histograms of hidden layer input and final layer input 
    with histogram normalization
    
    also plot learned aggregate synaptic function for each hidden-output connection
    grouped by output unit
    
    Inputs:
        model: nn.Module
        testloader: torch.utils.data.DataLoader
        hidden_act: str, 'sigmoid' or 'relu', default: 'relu'
        is_normalized: bool, whether to normalize input histogram
        file_name: str, file name to save figure
        is_scaler: bool, whether to multiply aggregate synaptic function with the magnitude of scaler, |c_{i,k}|
        is_sign: bool, whether to multiply aggregate synaptic function with the sign of scaler, sgn(c_{i,k})
    '''
    
    # collection hidden layer input from test dataset
    hidden = []
    final = []
    for inputs, labels in testloader:
        inputs = inputs.view(-1, 28*28)#.to(device)
        if hidden_act == 'sigmoid':
            hidden.append(torch.sigmoid(model.fc1(inputs)).detach().cpu())
        elif hidden_act == 'relu':
            hidden.append(torch.relu(model.fc1(inputs)).detach().cpu())
        final.append(model.parallel_synapse(hidden[-1]).detach().cpu())
        
    hidden_thres = 1e-4 # if hidden unit input is too small and close to zero, we ignore it
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
    for i in np.arange(hidden.shape[1]):
        tmp_hidden = hidden[:, i].ravel()
        hidden_list.append(tmp_hidden[tmp_hidden > hidden_thres])
    
    # for each hidden unit, we get its input range, and create input value with n_data bins
    input = torch.cat([torch.linspace(0, hidden_list[i].std()*4 + hidden_list[i].mean(), steps = n_data).reshape(-1, 1) for i in np.arange(hidden.shape[1])], dim = 1)

    all_freq = []
    # also for get hidden unit input histogram, so that we can stretch the aggregate synaptic function 
    # such that the x-axis is input percentile
    for i in np.arange(hidden.shape[1]):
        tmp_hidden = hidden[:, i].ravel()
        tmp_hidden = tmp_hidden[tmp_hidden > hidden_thres]
        freq, _ = np.histogram(tmp_hidden, bins=input[:,i].numpy())
            
        all_freq.append((freq/np.sum(freq)).cumsum())
    freq = np.vstack(all_freq)
    
    n_data = input.shape[0]
    n_synapse = slope.shape[0]
    
    # aggregate synaptic function
    x = slope[None, :, :, :].expand(n_data, n_synapse, input_dim, output_dim) \
        * (input[:, None, :, None].expand(n_data, n_synapse, input_dim, output_dim)
        - thres[None, :, :, :].expand(n_data, n_synapse, input_dim, output_dim))
    x = torch.tanh(x)
    x = x * (ampli[None, :, :, :]**2 ).expand(n_data, n_synapse, input_dim, output_dim)
    if is_scaler and is_sign:
        x = x.detach().numpy() * f.scaler[None, None, :, :].expand(n_data, n_synapse, input_dim, output_dim).detach().numpy()
    elif is_scaler and not is_sign:
        x = x.detach().numpy() * np.abs(f.scaler[None, None, :, :].expand(n_data, n_synapse, input_dim, output_dim).detach().numpy())
    elif not is_scaler and not is_sign:
        x = x.detach().numpy()
    x = x.sum(axis=1).squeeze() # shape: n_data x input_dim x output_dim
    
    
    if is_sign and is_scaler:
        plt.figure(figsize = (8, 7), dpi=300)
        for i in range(20):
            # hidden unit activation histogram
            plt.subplot(4,5,i+1)
            plt.hist(hidden_list[i], bins=input[:,i], density=True)
            plt.title( 'Hidden unit ' +str(i+1))
            plt.xlabel('Activation')
        plt.tight_layout()
        plt.savefig(file_name + '_hidden_activation.pdf')
    plt.show()
    
    # aggregate synaptic function
    plt.figure(figsize = (7.5, 6), dpi=300)
    plt.rcParams.update({'font.size': 8})
    print('transmission function, scaler = ' + str(is_scaler) + ', sign = ' + str(is_sign))
    for i in np.arange(min([20, output_dim])):
        plt.subplot(4,5,i+1)
        plt.plot(freq.T * 100, x[:-1,:,i], linewidth = 1.,alpha = 0.8)# shape: n_data x input_dim x output_dim
        plt.title(   'Output unit ' + str(i+1), fontsize=8)
        plt.xlabel('Input (percentile)', fontsize=8)
    plt.tight_layout()
    plt.savefig(file_name, bbox_inches='tight')
    plt.show()


def plot_parallel_synapse_params(model) -> None: 
    '''
    plot histograms of parallel synapses, such as amplitude, slope, threshold, scaler   
    
    Inputs:
        model: nn.Module
    '''
    plt.figure(figsize=(10, 2.5), dpi=300)
    plt.subplot(1,4,1)
    plt.hist((model.parallel_synapse.ampli.data.cpu().numpy()**2).flatten(), bins=20)
    plt.xlabel('Amplitude')
    plt.ylabel('Frequency')
    

    plt.subplot(1,4,2)
    plt.hist(model.parallel_synapse.slope.data.cpu().numpy().flatten(), bins=20)
    plt.xlabel('Slope')
    plt.ylabel('Frequency')

    plt.subplot(1,4,3)
    plt.hist(model.parallel_synapse.thres.data.cpu().numpy().flatten(), bins=20)
    plt.xlabel('Threshold')
    plt.ylabel('Frequency')
    
    plt.subplot(1,4,4)
    plt.hist(model.parallel_synapse.scaler.data.cpu().numpy().flatten(), bins=20)
    plt.xlabel('Scalar')
    plt.ylabel('Frequency')
    
    plt.tight_layout()
    plt.savefig('./results/parallel_synapse_params.pdf', bbox_inches='tight')
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
    
    
def get_hidden_final_activation_2NN(model, testloader):
    model.eval()
    hidden, final = [], []
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs = inputs.view(-1, 28*28).to(device)
            hidden.append(torch.relu(model.fc1(inputs)).detach().cpu().numpy())
            final.append(model.fc2(torch.relu(model.fc1(inputs))).detach().cpu().numpy())
    return np.concatenate(hidden, axis=0), np.concatenate(final, axis=0)

def plot_hidden_final_loss_2NN(model, hidden, final, result):
    # plot histogram of hidden activation
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