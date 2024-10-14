from typing import List, Optional
import torch
from torch import nn
from labml_helpers.module import Module

# References: https://arxiv.org/abs/1512.03385v1
# References: https://nn.labml.ai/resnet/index.html
# References: https://nn.labml.ai/resnet/experiment.html



# Class for the shortcut connection with projection, used when the input and output dimensions differ
class ShortcutProjection(Module):
    
    def __init__(self, in_channels: int, out_channels: int, stride: int):
        super().__init__()
        # 1x1 convolution to match the dimensions of the input and output
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride)
        # Batch normalization for the projected shortcut
        self.bn = nn.BatchNorm2d(out_channels)
        
    def forward(self, x: torch.Tensor):
        # Forward pass through the convolution and batch normalization
        return self.bn(self.conv(x))

# Class for the standard residual block in ResNet
class ResidualBlock(Module):
    
    def __init__(self, in_channels: int, out_channels: int, stride: int):
        super().__init__()
        # First convolution layer with ReLU activation and batch normalization
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.act1 = nn.ReLU()
        
        # Second convolution layer with batch normalization
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Shortcut connection: use projection if input and output dimensions differ
        if stride != 1 or in_channels != out_channels:
            self.shortcut = ShortcutProjection(in_channels, out_channels, stride)
        else:
            self.shortcut = nn.Identity()  # No change to the dimensions if they match
        
        # Activation function after adding the shortcut connection
        self.act2 = nn.ReLU()
        
    def forward(self, x: torch.Tensor):
        # Store the shortcut connection output
        shortcut = self.shortcut(x)
        # Forward pass through the first conv layer, batch norm, and activation
        x = self.act1(self.bn1(self.conv1(x)))
        # Forward pass through the second conv layer and batch norm
        x = self.bn2(self.conv2(x))
        # Add the shortcut connection output to the block output
        return self.act2(x + shortcut)

# Class for the bottleneck residual block, used in deeper ResNets
class BottleneckResidualBlock(Module):
    
    def __init__(self, in_channels: int, bottleneck_channels: int, out_channels: int, stride: int):
        super().__init__()
        # First 1x1 convolution to reduce dimensions
        self.conv1 = nn.Conv2d(in_channels, bottleneck_channels, kernel_size=1, stride=1)
        self.bn1 = nn.BatchNorm2d(bottleneck_channels)
        self.act1 = nn.ReLU()
        
        # Second 3x3 convolution to process the features
        self.conv2 = nn.Conv2d(bottleneck_channels, bottleneck_channels, kernel_size=3, stride=stride, padding=1)
        self.bn2 = nn.BatchNorm2d(bottleneck_channels)
        self.act2 = nn.ReLU()
        
        # Third 1x1 convolution to restore the output dimensions
        self.conv3 = nn.Conv2d(bottleneck_channels, out_channels, kernel_size=1, stride=1)
        self.bn3 = nn.BatchNorm2d(out_channels)
        
        # Shortcut connection: use projection if input and output dimensions differ
        if stride != 1 or in_channels != out_channels:
            self.shortcut = ShortcutProjection(in_channels, out_channels, stride)
        else:
            self.shortcut = nn.Identity()  # No change to the dimensions if they match
        
        # Activation function after adding the shortcut connection
        self.act3 = nn.ReLU()
        
    def forward(self, x: torch.Tensor):
        # Store the shortcut connection output
        shortcut = self.shortcut(x)
        # Forward pass through the first conv layer, batch norm, and activation
        x = self.act1(self.bn1(self.conv1(x)))
        # Forward pass through the second conv layer and batch norm
        x = self.act2(self.bn2(self.conv2(x)))
        # Forward pass through the third conv layer and batch norm
        x = self.bn3(self.conv3(x))
        # Add the shortcut connection output to the block output
        return self.act3(x + shortcut)

# Base class for the ResNet architecture
class ResNetBase(Module):
    
    def __init__(self, n_blocks: List[int], n_channels: List[int], bottlenecks: Optional[List[int]] = None,
                 img_channels: int = 3, first_kernel_size: int = 7):
        super().__init__()
        # Ensure the number of blocks matches the number of channels
        assert len(n_blocks) == len(n_channels)
        assert bottlenecks is None or len(bottlenecks) == len(n_channels)
        
        # Initial convolution layer
        self.conv = nn.Conv2d(img_channels, n_channels[0], kernel_size=first_kernel_size, stride=2,
                               padding=first_kernel_size // 2)
        self.bn = nn.BatchNorm2d(n_channels[0])
        
        # Initialize a list to hold all residual blocks
        blocks = []
        prev_channels = n_channels[0]
        
        # Construct the ResNet architecture based on the number of blocks and channels
        for i, channels in enumerate(n_channels):
            # Set stride for the first block to 2 to downsample the feature map
            stride = 2 if len(blocks) == 0 else 1
            
            # Use residual or bottleneck block based on the presence of bottlenecks
            if bottlenecks is None:
                blocks.append(ResidualBlock(prev_channels, channels, stride=stride))
            else:
                blocks.append(BottleneckResidualBlock(prev_channels, bottlenecks[i], channels, stride=stride))
            
            # Update the previous channels for the next block
            prev_channels = channels
            
            # Add additional blocks (all with stride 1)
            for _ in range(n_blocks[i] - 1):
                if bottlenecks is None:
                    blocks.append(ResidualBlock(channels, channels, stride=1))
                else:
                    blocks.append(BottleneckResidualBlock(channels, bottlenecks[i], channels, stride=1))
        
        # Create a sequential container for the blocks
        self.blocks = nn.Sequential(*blocks)
        
    def forward(self, x: torch.Tensor):
        # Forward pass through the initial convolution and batch normalization
        x = self.bn(self.conv(x))
        # Forward pass through all residual blocks
        x = self.blocks(x)
        # Flatten the output for global average pooling
        x = x.view(x.shape[0], x.shape[1], -1)
        # Return the mean across the spatial dimensions
        return x.mean(dim=-1)
