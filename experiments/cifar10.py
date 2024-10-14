from typing import List, Optional  # Import necessary typing utilities for type hints
from torch import nn  # Import PyTorch neural network module
from labml import experiment  # Import LabML for experiment tracking
from labml.configs import option  # Import the option decorator for configuration options
from labml_nn.experiments.cifar10 import CIFAR10Configs  # Import CIFAR10 configuration class
from labml_nn.resnet import ResNetBase  # Import the base ResNet class

# Define a configuration class for the ResNet model on CIFAR-10
class Configs(CIFAR10Configs):
    n_blocks: List[int] = [3, 3, 3]  # Define the number of blocks in each layer of the ResNet
    n_channels: List[int] = [16, 32, 64]  # Define the number of output channels for each layer
    bottlenecks: Optional[List[int]] = None  # Optionally define bottleneck layers (set to None for standard ResNet)
    first_kernel_size: int = 3  # Define the kernel size for the first convolutional layer

# Define a function to create the ResNet model
@option(Configs.model)
def _resnet(c: Configs):
    # Initialize the base ResNet model with configuration parameters
    base = ResNetBase(c.n_blocks, c.n_channels, c.bottlenecks, img_channels=3, first_kernel_size=c.first_kernel_size)
    # Create a linear layer for classification with 10 output classes (CIFAR-10)
    classification = nn.Linear(c.n_channels[-1], 10)
    # Combine the ResNet base and the classification layer into a sequential model
    model = nn.Sequential(base, classification)
    # Move the model to the specified device (CPU/GPU)
    return model.to(c.device)

# Main function to set up and run the experiment
def main():
    # Create a new experiment with a specified name and comment
    experiment.create(name='resnet', comment='cifar10')
    # Initialize the configuration object
    conf = Configs()
    # Update the experiment configuration with specific parameters
    experiment.configs(conf, {
        'bottlenecks': [8, 16, 16],  # Define the number of bottleneck channels for each layer
        'n_blocks': [6, 6, 6],  # Update the number of blocks for each layer to 6
        'optimizer.optimizer': 'Adam',  # Set the optimizer to Adam
        'optimizer.learning_rate': 2.5e-4,  # Set the learning rate for the optimizer
        'epochs': 500,  # Set the number of training epochs
        'train_batch_size': 256,  # Set the training batch size
        'train_dataset': 'cifar10_train_augmented',  # Specify the augmented training dataset
        'valid_dataset': 'cifar10_valid_no_augment',  # Specify the validation dataset without augmentation
    })
    
    # Add the PyTorch model to the experiment tracking
    experiment.add_pytorch_models({'model': conf.model})
    # Start the experiment context
    with experiment.start():
        # Run the configuration, training the model
        conf.run()
        
# Entry point of the script
if __name__ == '__main__':
    main()  # Call the main function
