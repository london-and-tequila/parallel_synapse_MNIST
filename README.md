# Neural Networks with Parallel Synapses

This repository implements and analyzes neural networks with parallel synapses - where each connection between neurons consists of nonlinear parallel synapses.

## Code Structure

### network.py
Contains the core neural network implementation with parallel synapses:
- Neural network architecture with configurable number of parallel synapses per connection
- Forward and backward propagation algorithms adapted for parallel weights
- Weight initialization and update methods

### training.py 
Handles the training process:
- Training loop implementation
- Loss calculation and optimization
- Learning rate scheduling
- Training metrics tracking

### analysis.py
Provides analysis tools:
- Evaluation metrics computation
- Weight distribution analysis
- Performance comparison utilities between standard and parallel synapse models

### plot_results.ipynb
Jupyter notebook containing:
- Visualization of training results
- Performance comparisons
- Analysis of learned weight distributions
- Ablation studies on number of parallel synapses
- Key findings and insights

## Key Features

- Implementation of parallel synapses in neural networks
- Comparative analysis with standard neural networks
- Visualization of weight distributions and learning dynamics
- Performance evaluation across different architectures

## Usage

The main components can be used as follows:

1. Define network architecture in `network.py`
2. Configure and run training using `training.py`
3. Analyze results using functions in `analysis.py`
4. View comprehensive results and plots in `plot_results.ipynb`

## Contacts

For questions or additional information, please contact: yus027@ucsd.edu
