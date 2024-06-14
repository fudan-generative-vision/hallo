# pylint: disable=E1120
# pylint: disable=E1102
# pylint: disable=W0237

# src/models/resnet.py

"""
This module defines various components used in the ResNet model, such as InflatedConv3D, InflatedGroupNorm,
Upsample3D, Downsample3D, ResnetBlock3D, and Mish activation function. These components are used to construct
a deep neural network model for image classification or other computer vision tasks.

Classes:
- InflatedConv3d: An inflated 3D convolutional layer, inheriting from nn.Conv2d.
- InflatedGroupNorm: An inflated group normalization layer, inheriting from nn.GroupNorm.
- Upsample3D: A 3D upsampling module, used to increase the resolution of the input tensor.
- Downsample3D: A 3D downsampling module, used to decrease the resolution of the input tensor.
- ResnetBlock3D: A 3D residual block, commonly used in ResNet architectures.
- Mish: A Mish activation function, which is a smooth, non-monotonic activation function.

To use this module, simply import the classes and functions you need and follow the instructions provided in
the respective class and function docstrings.
"""

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn


class InflatedConv3d(nn.Conv2d):
    """
    InflatedConv3d is a class that inherits from torch.nn.Conv2d and overrides the forward method.
    
    This class is used to perform 3D convolution on input tensor x. It is a specialized type of convolutional layer
    commonly used in deep learning models for computer vision tasks. The main difference between a regular Conv2d and
    InflatedConv3d is that InflatedConv3d is designed to handle 3D input tensors, which are typically the result of
    inflating 2D convolutional layers to 3D for use in 3D deep learning tasks.
    
    Attributes:
        Same as torch.nn.Conv2d.
        
    Methods:
        forward(self, x):
            Performs 3D convolution on the input tensor x using the InflatedConv3d layer.
            
    Example:
        conv_layer = InflatedConv3d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1)
        output = conv_layer(input_tensor)
    """
    def forward(self, x):
        """
        Forward pass of the InflatedConv3d layer.

        Args:
            x (torch.Tensor): Input tensor to the layer.

        Returns:
            torch.Tensor: Output tensor after applying the InflatedConv3d layer.
        """
        video_length = x.shape[2]

        x = rearrange(x, "b c f h w -> (b f) c h w")
        x = super().forward(x)
        x = rearrange(x, "(b f) c h w -> b c f h w", f=video_length)

        return x


class InflatedGroupNorm(nn.GroupNorm):
    """
    InflatedGroupNorm is a custom class that inherits from torch.nn.GroupNorm.
    It is used to apply group normalization to 3D tensors.

    Args:
        num_groups (int): The number of groups to divide the channels into.
        num_channels (int): The number of channels in the input tensor.
        eps (float, optional): A small constant to add to the variance to avoid division by zero. Defaults to 1e-5.
        affine (bool, optional): If True, the module has learnable affine parameters. Defaults to True.

    Attributes:
        weight (torch.Tensor): The learnable weight tensor for scale.
        bias (torch.Tensor): The learnable bias tensor for shift.

    Forward method:
        x (torch.Tensor): Input tensor to be normalized.
        return (torch.Tensor): Normalized tensor.
    """
    def forward(self, x):
        """
        Performs a forward pass through the CustomClassName.
        
        :param x: Input tensor of shape (batch_size, channels, video_length, height, width).
        :return: Output tensor of shape (batch_size, channels, video_length, height, width).
        """
        video_length = x.shape[2]

        x = rearrange(x, "b c f h w -> (b f) c h w")
        x = super().forward(x)
        x = rearrange(x, "(b f) c h w -> b c f h w", f=video_length)

        return x


class Upsample3D(nn.Module):
    """
    Upsample3D is a PyTorch module that upsamples a 3D tensor.

    Args:
        channels (int): The number of channels in the input tensor.
        use_conv (bool): Whether to use a convolutional layer for upsampling.
        use_conv_transpose (bool): Whether to use a transposed convolutional layer for upsampling.
        out_channels (int): The number of channels in the output tensor.
        name (str): The name of the convolutional layer.
    """
    def __init__(
        self,
        channels,
        use_conv=False,
        use_conv_transpose=False,
        out_channels=None,
        name="conv",
    ):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_conv_transpose = use_conv_transpose
        self.name = name

        if use_conv_transpose:
            raise NotImplementedError
        if use_conv:
            self.conv = InflatedConv3d(self.channels, self.out_channels, 3, padding=1)

    def forward(self, hidden_states, output_size=None):
        """
        Forward pass of the Upsample3D class.

        Args:
            hidden_states (torch.Tensor): Input tensor to be upsampled.
            output_size (tuple, optional): Desired output size of the upsampled tensor.

        Returns:
            torch.Tensor: Upsampled tensor.

        Raises:
            AssertionError: If the number of channels in the input tensor does not match the expected channels.
        """
        assert hidden_states.shape[1] == self.channels

        if self.use_conv_transpose:
            raise NotImplementedError

        # Cast to float32 to as 'upsample_nearest2d_out_frame' op does not support bfloat16
        dtype = hidden_states.dtype
        if dtype == torch.bfloat16:
            hidden_states = hidden_states.to(torch.float32)

        # upsample_nearest_nhwc fails with large batch sizes. see https://github.com/huggingface/diffusers/issues/984
        if hidden_states.shape[0] >= 64:
            hidden_states = hidden_states.contiguous()

        # if `output_size` is passed we force the interpolation output
        # size and do not make use of `scale_factor=2`
        if output_size is None:
            hidden_states = F.interpolate(
                hidden_states, scale_factor=[1.0, 2.0, 2.0], mode="nearest"
            )
        else:
            hidden_states = F.interpolate(
                hidden_states, size=output_size, mode="nearest"
            )

        # If the input is bfloat16, we cast back to bfloat16
        if dtype == torch.bfloat16:
            hidden_states = hidden_states.to(dtype)

        # if self.use_conv:
        #     if self.name == "conv":
        #         hidden_states = self.conv(hidden_states)
        #     else:
        #         hidden_states = self.Conv2d_0(hidden_states)
        hidden_states = self.conv(hidden_states)

        return hidden_states


class Downsample3D(nn.Module):
    """
    The Downsample3D class is a PyTorch module for downsampling a 3D tensor, which is used to 
    reduce the spatial resolution of feature maps, commonly in the encoder part of a neural network.

    Attributes:
        channels (int): Number of input channels.
        use_conv (bool): Flag to use a convolutional layer for downsampling.
        out_channels (int, optional): Number of output channels. Defaults to input channels if None.
        padding (int): Padding added to the input.
        name (str): Name of the convolutional layer used for downsampling.

    Methods:
        forward(self, hidden_states):
            Downsamples the input tensor hidden_states and returns the downsampled tensor.
    """
    def __init__(
        self, channels, use_conv=False, out_channels=None, padding=1, name="conv"
    ):
        """
        Downsamples the given input in the 3D space.

        Args:
            channels: The number of input channels.
            use_conv: Whether to use a convolutional layer for downsampling.
            out_channels: The number of output channels. If None, the input channels are used.
            padding: The amount of padding to be added to the input.
            name: The name of the convolutional layer.
        """
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.padding = padding
        stride = 2
        self.name = name

        if use_conv:
            self.conv = InflatedConv3d(
                self.channels, self.out_channels, 3, stride=stride, padding=padding
            )
        else:
            raise NotImplementedError

    def forward(self, hidden_states):
        """
        Forward pass for the Downsample3D class.

        Args:
            hidden_states (torch.Tensor): Input tensor to be downsampled.

        Returns:
            torch.Tensor: Downsampled tensor.

        Raises:
            AssertionError: If the number of channels in the input tensor does not match the expected channels.
        """
        assert hidden_states.shape[1] == self.channels
        if self.use_conv and self.padding == 0:
            raise NotImplementedError

        assert hidden_states.shape[1] == self.channels
        hidden_states = self.conv(hidden_states)

        return hidden_states


class ResnetBlock3D(nn.Module):
    """
    The ResnetBlock3D class defines a 3D residual block, a common building block in ResNet 
    architectures for both image and video modeling tasks.

    Attributes:
        in_channels (int): Number of input channels.
        out_channels (int, optional): Number of output channels, defaults to in_channels if None.
        conv_shortcut (bool): Flag to use a convolutional shortcut.
        dropout (float): Dropout rate.
        temb_channels (int): Number of channels in the time embedding tensor.
        groups (int): Number of groups for the group normalization layers.
        eps (float): Epsilon value for group normalization.
        non_linearity (str): Type of nonlinearity to apply after convolutions.
        time_embedding_norm (str): Type of normalization for the time embedding.
        output_scale_factor (float): Scaling factor for the output tensor.
        use_in_shortcut (bool): Flag to include the input tensor in the shortcut connection.
        use_inflated_groupnorm (bool): Flag to use inflated group normalization layers.

    Methods:
        forward(self, input_tensor, temb):
            Passes the input tensor and time embedding through the residual block and 
            returns the output tensor.
    """
    def __init__(
        self,
        *,
        in_channels,
        out_channels=None,
        conv_shortcut=False,
        dropout=0.0,
        temb_channels=512,
        groups=32,
        groups_out=None,
        pre_norm=True,
        eps=1e-6,
        non_linearity="swish",
        time_embedding_norm="default",
        output_scale_factor=1.0,
        use_in_shortcut=None,
        use_inflated_groupnorm=None,
    ):
        super().__init__()
        self.pre_norm = pre_norm
        self.pre_norm = True
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut
        self.time_embedding_norm = time_embedding_norm
        self.output_scale_factor = output_scale_factor

        if groups_out is None:
            groups_out = groups

        assert use_inflated_groupnorm is not None
        if use_inflated_groupnorm:
            self.norm1 = InflatedGroupNorm(
                num_groups=groups, num_channels=in_channels, eps=eps, affine=True
            )
        else:
            self.norm1 = torch.nn.GroupNorm(
                num_groups=groups, num_channels=in_channels, eps=eps, affine=True
            )

        self.conv1 = InflatedConv3d(
            in_channels, out_channels, kernel_size=3, stride=1, padding=1
        )

        if temb_channels is not None:
            if self.time_embedding_norm == "default":
                time_emb_proj_out_channels = out_channels
            elif self.time_embedding_norm == "scale_shift":
                time_emb_proj_out_channels = out_channels * 2
            else:
                raise ValueError(
                    f"unknown time_embedding_norm : {self.time_embedding_norm} "
                )

            self.time_emb_proj = torch.nn.Linear(
                temb_channels, time_emb_proj_out_channels
            )
        else:
            self.time_emb_proj = None

        if use_inflated_groupnorm:
            self.norm2 = InflatedGroupNorm(
                num_groups=groups_out, num_channels=out_channels, eps=eps, affine=True
            )
        else:
            self.norm2 = torch.nn.GroupNorm(
                num_groups=groups_out, num_channels=out_channels, eps=eps, affine=True
            )
        self.dropout = torch.nn.Dropout(dropout)
        self.conv2 = InflatedConv3d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1
        )

        if non_linearity == "swish":
            self.nonlinearity = F.silu()
        elif non_linearity == "mish":
            self.nonlinearity = Mish()
        elif non_linearity == "silu":
            self.nonlinearity = nn.SiLU()

        self.use_in_shortcut = (
            self.in_channels != self.out_channels
            if use_in_shortcut is None
            else use_in_shortcut
        )

        self.conv_shortcut = None
        if self.use_in_shortcut:
            self.conv_shortcut = InflatedConv3d(
                in_channels, out_channels, kernel_size=1, stride=1, padding=0
            )

    def forward(self, input_tensor, temb):
        """
        Forward pass for the ResnetBlock3D class.

        Args:
            input_tensor (torch.Tensor): Input tensor to the ResnetBlock3D layer.
            temb (torch.Tensor): Token embedding tensor.

        Returns:
            torch.Tensor: Output tensor after passing through the ResnetBlock3D layer.
        """
        hidden_states = input_tensor

        hidden_states = self.norm1(hidden_states)
        hidden_states = self.nonlinearity(hidden_states)

        hidden_states = self.conv1(hidden_states)

        if temb is not None:
            temb = self.time_emb_proj(self.nonlinearity(temb))[:, :, None, None, None]

        if temb is not None and self.time_embedding_norm == "default":
            hidden_states = hidden_states + temb

        hidden_states = self.norm2(hidden_states)

        if temb is not None and self.time_embedding_norm == "scale_shift":
            scale, shift = torch.chunk(temb, 2, dim=1)
            hidden_states = hidden_states * (1 + scale) + shift

        hidden_states = self.nonlinearity(hidden_states)

        hidden_states = self.dropout(hidden_states)
        hidden_states = self.conv2(hidden_states)

        if self.conv_shortcut is not None:
            input_tensor = self.conv_shortcut(input_tensor)

        output_tensor = (input_tensor + hidden_states) / self.output_scale_factor

        return output_tensor


class Mish(torch.nn.Module):
    """
    The Mish class implements the Mish activation function, a smooth, non-monotonic function 
    that can be used in neural networks as an alternative to traditional activation functions like ReLU.

    Methods:
        forward(self, hidden_states):
            Applies the Mish activation function to the input tensor hidden_states and 
            returns the resulting tensor.
    """
    def forward(self, hidden_states):
        """
        Mish activation function.

        Args:
            hidden_states (torch.Tensor): The input tensor to apply the Mish activation function to.

        Returns:
            hidden_states (torch.Tensor): The output tensor after applying the Mish activation function.
        """
        return hidden_states * torch.tanh(torch.nn.functional.softplus(hidden_states))
