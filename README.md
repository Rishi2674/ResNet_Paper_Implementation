# ResNet_Paper_Implementation

## Overview

This repository contains an implementation of the ResNet (Residual Network) architecture designed for image classification on the CIFAR-10 dataset. ResNet introduces a novel architecture that allows for training very deep networks by utilizing skip connections, which help mitigate the vanishing gradient problem.

## Table of Contents
- [Introduction to ResNet](#introduction-to-resnet)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [References](#references)

## Introduction to ResNet

ResNet is a deep convolutional neural network architecture that was introduced by Kaiming He et al. in the paper ["Deep Residual Learning for Image Recognition"](https://arxiv.org/abs/1512.03385). The main idea behind ResNet is to allow gradients to flow through the network via skip connections, effectively creating shortcut paths for the gradient to propagate backward.

### Key Concepts
- **Residual Block**: The fundamental building block of ResNet. It consists of two or more convolutional layers with a shortcut connection that bypasses one or more layers. The output of the residual block is the sum of the input and the output of the convolutional layers.

  ![Residual Block Diagram](images/residual_building_blocks.png)

- **Skip Connection**: This allows the input to be added directly to the output of the convolutional layers, enabling the network to learn an identity function. If the learned residual is close to zero, the network can easily fit the identity function, which helps in training deeper architectures.

- **Bottleneck Architecture**: A variation of the residual block that reduces the number of parameters by first projecting the input to a lower-dimensional space, performing the main operations, and then projecting back to a higher-dimensional space.

### Advantages of ResNet
- **Improved Training of Deep Networks**: The skip connections prevent the gradient from becoming too small, allowing deeper networks to be trained effectively.
- **Better Generalization**: ResNets tend to generalize better on unseen data due to their increased depth and the nature of residual learning.

## Repository Structure
```bash
ResNet_Paper_Implementation/
├── README.md               # This README file
├── environment.yml         # Conda environment configuration file                
├── models/                  # Contains model definitions
│   ├── __init__.py         # Makes the model directory a package
│   ├── resnet.py           # ResNet model definition
├── experiments/                   # Data handling scripts
│   ├── cifar10.py         # CIFAR-10 dataset loading and preprocessing
├── images/
├── requirements.txt        # Required Python packages

```
## Installation

To set up the project, follow these steps:

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Rishi2674/ResNet_Paper_Implementation.git
   cd ResNet_Paper_Implementation
   ```

2. **Activating the Environment:**
   ```bash
   gconda env create -f environment.yml
   conda activate resnet-env
   ```

3. **Installing necessary requirements**
   ```bash
   pip install -r requirements.txt
   ```

## References
- [Residual Networks](https://arxiv.org/abs/1512.03385)
- [Implementation of ResNets](https://nn.labml.ai/resnet/index.html)
- [Implementation of ResNets on CIFAR10 dataset](https://nn.labml.ai/resnet/experiment.html)



