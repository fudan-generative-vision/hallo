"""
This module implements the FaceLocator class, which is a neural network model designed to
locate and extract facial features from input images or tensors. It uses a series of
convolutional layers to progressively downsample and refine the facial feature map.

The FaceLocator class is part of a larger system that may involve facial recognition or
similar tasks where precise location and extraction of facial features are required.

Attributes:
    conditioning_embedding_channels (int): The number of channels in the output embedding.
    conditioning_channels (int): The number of input channels for the conditioning tensor.
    block_out_channels (Tuple[int]): A tuple of integers representing the output channels
        for each block in the model.

The model uses the following components:
- InflatedConv3d: A convolutional layer that inflates the input to increase the depth.
- zero_module: A utility function that may set certain parameters to zero for regularization
    or other purposes.

The forward method of the FaceLocator class takes a conditioning tensor as input and
produces an embedding tensor as output, which can be used for further processing or analysis.
"""

from typing import Tuple

import torch.nn.functional as F
from diffusers.models.modeling_utils import ModelMixin
from torch import nn

from .motion_module import zero_module
from .resnet import InflatedConv3d


class FaceLocator(ModelMixin):
    """
    The FaceLocator class is a neural network model designed to process and extract facial
    features from an input tensor. It consists of a series of convolutional layers that
    progressively downsample the input while increasing the depth of the feature map.

    The model is built using InflatedConv3d layers, which are designed to inflate the
    feature channels, allowing for more complex feature extraction. The final output is a
    conditioning embedding that can be used for various tasks such as facial recognition or
    feature-based image manipulation.

    Parameters:
        conditioning_embedding_channels (int): The number of channels in the output embedding.
        conditioning_channels (int, optional): The number of input channels for the conditioning tensor. Default is 3.
        block_out_channels (Tuple[int], optional): A tuple of integers representing the output channels
            for each block in the model. The default is (16, 32, 64, 128), which defines the
            progression of the network's depth.

    Attributes:
        conv_in (InflatedConv3d): The initial convolutional layer that starts the feature extraction process.
        blocks (ModuleList[InflatedConv3d]): A list of convolutional layers that form the core of the model.
        conv_out (InflatedConv3d): The final convolutional layer that produces the output embedding.

    The forward method applies the convolutional layers to the input conditioning tensor and
    returns the resulting embedding tensor.
    """
    def __init__(
        self,
        conditioning_embedding_channels: int,
        conditioning_channels: int = 3,
        block_out_channels: Tuple[int] = (16, 32, 64, 128),
    ):
        super().__init__()
        self.conv_in = InflatedConv3d(
            conditioning_channels, block_out_channels[0], kernel_size=3, padding=1
        )

        self.blocks = nn.ModuleList([])

        for i in range(len(block_out_channels) - 1):
            channel_in = block_out_channels[i]
            channel_out = block_out_channels[i + 1]
            self.blocks.append(
                InflatedConv3d(channel_in, channel_in, kernel_size=3, padding=1)
            )
            self.blocks.append(
                InflatedConv3d(
                    channel_in, channel_out, kernel_size=3, padding=1, stride=2
                )
            )

        self.conv_out = zero_module(
            InflatedConv3d(
                block_out_channels[-1],
                conditioning_embedding_channels,
                kernel_size=3,
                padding=1,
            )
        )

    def forward(self, conditioning):
        """
        Forward pass of the FaceLocator model.

        Args:
            conditioning (Tensor): The input conditioning tensor.

        Returns:
            Tensor: The output embedding tensor.
        """
        embedding = self.conv_in(conditioning)
        embedding = F.silu(embedding)

        for block in self.blocks:
            embedding = block(embedding)
            embedding = F.silu(embedding)

        embedding = self.conv_out(embedding)

        return embedding
