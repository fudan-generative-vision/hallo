# pylint: disable=R0801
# pylint: disable=W0613
# pylint: disable=W0221

"""
temporal_transformers.py

This module provides classes and functions for implementing Temporal Transformers
in PyTorch, designed for handling video data and temporal sequences within transformer-based models.

Functions:
    zero_module(module)
        Zero out the parameters of a module and return it.

Classes:
    TemporalTransformer3DModelOutput(BaseOutput)
        Dataclass for storing the output of TemporalTransformer3DModel.

    VanillaTemporalModule(nn.Module)
        A Vanilla Temporal Module class for handling temporal data.

    TemporalTransformer3DModel(nn.Module)
        A Temporal Transformer 3D Model class for transforming temporal data.

    TemporalTransformerBlock(nn.Module)
        A Temporal Transformer Block class for building the transformer architecture.

    PositionalEncoding(nn.Module)
        A Positional Encoding module for transformers to encode positional information.

Dependencies:
    math
    dataclasses.dataclass
    typing (Callable, Optional)
    torch
    diffusers (FeedForward, Attention, AttnProcessor)
    diffusers.utils (BaseOutput)
    diffusers.utils.import_utils (is_xformers_available)
    einops (rearrange, repeat)
    torch.nn
    xformers
    xformers.ops

Example Usage:
    >>> motion_module = get_motion_module(in_channels=512, motion_module_type="Vanilla", motion_module_kwargs={})
    >>> output = motion_module(input_tensor, temb, encoder_hidden_states)

This module is designed to facilitate the creation, training, and inference of transformer models
that operate on temporal data, such as videos or time-series. It includes mechanisms for applying temporal attention,
managing positional encoding, and integrating with external libraries for efficient attention operations.
"""

# This code is copied from https://github.com/guoyww/AnimateDiff.

import math

import torch
import xformers
import xformers.ops
from diffusers.models.attention import FeedForward
from diffusers.models.attention_processor import Attention, AttnProcessor
from diffusers.utils import BaseOutput
from diffusers.utils.import_utils import is_xformers_available
from einops import rearrange, repeat
from torch import nn


def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    
    Args:
    - module: A PyTorch module to zero out its parameters.

    Returns:
    A zeroed out PyTorch module.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


class TemporalTransformer3DModelOutput(BaseOutput):
    """
    Output class for the TemporalTransformer3DModel.
    
    Attributes:
        sample (torch.FloatTensor): The output sample tensor from the model.
    """
    sample: torch.FloatTensor

    def get_sample_shape(self):
        """
        Returns the shape of the sample tensor.
        
        Returns:
        Tuple: The shape of the sample tensor.
        """
        return self.sample.shape


def get_motion_module(in_channels, motion_module_type: str, motion_module_kwargs: dict):
    """
    This function returns a motion module based on the given type and parameters.
    
    Args:
    - in_channels (int): The number of input channels for the motion module.
    - motion_module_type (str): The type of motion module to create. Currently, only "Vanilla" is supported.
    - motion_module_kwargs (dict): Additional keyword arguments to pass to the motion module constructor.
    
    Returns:
    VanillaTemporalModule: The created motion module.
    
    Raises:
    ValueError: If an unsupported motion_module_type is provided.
    """
    if motion_module_type == "Vanilla":
        return VanillaTemporalModule(
            in_channels=in_channels,
            **motion_module_kwargs,
        )

    raise ValueError


class VanillaTemporalModule(nn.Module):
    """
    A Vanilla Temporal Module class.

    Args:
    - in_channels (int): The number of input channels for the motion module.
    - num_attention_heads (int): Number of attention heads.
    - num_transformer_block (int): Number of transformer blocks.
    - attention_block_types (tuple): Types of attention blocks.
    - cross_frame_attention_mode: Mode for cross-frame attention.
    - temporal_position_encoding (bool): Flag for temporal position encoding.
    - temporal_position_encoding_max_len (int): Maximum length for temporal position encoding.
    - temporal_attention_dim_div (int): Divisor for temporal attention dimension.
    - zero_initialize (bool): Flag for zero initialization.
    """

    def __init__(
        self,
        in_channels,
        num_attention_heads=8,
        num_transformer_block=2,
        attention_block_types=("Temporal_Self", "Temporal_Self"),
        cross_frame_attention_mode=None,
        temporal_position_encoding=False,
        temporal_position_encoding_max_len=24,
        temporal_attention_dim_div=1,
        zero_initialize=True,
    ):
        super().__init__()

        self.temporal_transformer = TemporalTransformer3DModel(
            in_channels=in_channels,
            num_attention_heads=num_attention_heads,
            attention_head_dim=in_channels
            // num_attention_heads
            // temporal_attention_dim_div,
            num_layers=num_transformer_block,
            attention_block_types=attention_block_types,
            cross_frame_attention_mode=cross_frame_attention_mode,
            temporal_position_encoding=temporal_position_encoding,
            temporal_position_encoding_max_len=temporal_position_encoding_max_len,
        )

        if zero_initialize:
            self.temporal_transformer.proj_out = zero_module(
                self.temporal_transformer.proj_out
            )

    def forward(
        self,
        input_tensor,
        encoder_hidden_states,
        attention_mask=None,
    ):
        """
        Forward pass of the TemporalTransformer3DModel.

        Args:
            hidden_states (torch.Tensor): The hidden states of the model.
            encoder_hidden_states (torch.Tensor, optional): The hidden states of the encoder.
            attention_mask (torch.Tensor, optional): The attention mask.

        Returns:
            torch.Tensor: The output tensor after the forward pass.
        """
        hidden_states = input_tensor
        hidden_states = self.temporal_transformer(
            hidden_states, encoder_hidden_states
        )

        output = hidden_states
        return output


class TemporalTransformer3DModel(nn.Module):
    """
    A Temporal Transformer 3D Model class.

    Args:
    - in_channels (int): The number of input channels.
    - num_attention_heads (int): Number of attention heads.
    - attention_head_dim (int): Dimension of attention heads.
    - num_layers (int): Number of transformer layers.
    - attention_block_types (tuple): Types of attention blocks.
    - dropout (float): Dropout rate.
    - norm_num_groups (int): Number of groups for normalization.
    - cross_attention_dim (int): Dimension for cross-attention.
    - activation_fn (str): Activation function.
    - attention_bias (bool): Flag for attention bias.
    - upcast_attention (bool): Flag for upcast attention.
    - cross_frame_attention_mode: Mode for cross-frame attention.
    - temporal_position_encoding (bool): Flag for temporal position encoding.
    - temporal_position_encoding_max_len (int): Maximum length for temporal position encoding.
    """
    def __init__(
        self,
        in_channels,
        num_attention_heads,
        attention_head_dim,
        num_layers,
        attention_block_types=(
            "Temporal_Self",
            "Temporal_Self",
        ),
        dropout=0.0,
        norm_num_groups=32,
        cross_attention_dim=768,
        activation_fn="geglu",
        attention_bias=False,
        upcast_attention=False,
        cross_frame_attention_mode=None,
        temporal_position_encoding=False,
        temporal_position_encoding_max_len=24,
    ):
        super().__init__()

        inner_dim = num_attention_heads * attention_head_dim

        self.norm = torch.nn.GroupNorm(
            num_groups=norm_num_groups, num_channels=in_channels, eps=1e-6, affine=True
        )
        self.proj_in = nn.Linear(in_channels, inner_dim)

        self.transformer_blocks = nn.ModuleList(
            [
                TemporalTransformerBlock(
                    dim=inner_dim,
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=attention_head_dim,
                    attention_block_types=attention_block_types,
                    dropout=dropout,
                    cross_attention_dim=cross_attention_dim,
                    activation_fn=activation_fn,
                    attention_bias=attention_bias,
                    upcast_attention=upcast_attention,
                    cross_frame_attention_mode=cross_frame_attention_mode,
                    temporal_position_encoding=temporal_position_encoding,
                    temporal_position_encoding_max_len=temporal_position_encoding_max_len,
                )
                for d in range(num_layers)
            ]
        )
        self.proj_out = nn.Linear(inner_dim, in_channels)

    def forward(self, hidden_states, encoder_hidden_states=None):
        """
        Forward pass for the TemporalTransformer3DModel.

        Args:
            hidden_states (torch.Tensor): The input hidden states with shape (batch_size, sequence_length, in_channels).
            encoder_hidden_states (torch.Tensor, optional): The encoder hidden states with shape (batch_size, encoder_sequence_length, in_channels).

        Returns:
            torch.Tensor: The output hidden states with shape (batch_size, sequence_length, in_channels).
        """
        assert (
            hidden_states.dim() == 5
        ), f"Expected hidden_states to have ndim=5, but got ndim={hidden_states.dim()}."
        video_length = hidden_states.shape[2]
        hidden_states = rearrange(hidden_states, "b c f h w -> (b f) c h w")

        batch, _, height, weight = hidden_states.shape
        residual = hidden_states

        hidden_states = self.norm(hidden_states)
        inner_dim = hidden_states.shape[1]
        hidden_states = hidden_states.permute(0, 2, 3, 1).reshape(
            batch, height * weight, inner_dim
        )
        hidden_states = self.proj_in(hidden_states)

        # Transformer Blocks
        for block in self.transformer_blocks:
            hidden_states = block(
                hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                video_length=video_length,
            )

        # output
        hidden_states = self.proj_out(hidden_states)
        hidden_states = (
            hidden_states.reshape(batch, height, weight, inner_dim)
            .permute(0, 3, 1, 2)
            .contiguous()
        )

        output = hidden_states + residual
        output = rearrange(output, "(b f) c h w -> b c f h w", f=video_length)

        return output


class TemporalTransformerBlock(nn.Module):
    """
    A Temporal Transformer Block class.

    Args:
    - dim (int): Dimension of the block.
    - num_attention_heads (int): Number of attention heads.
    - attention_head_dim (int): Dimension of attention heads.
    - attention_block_types (tuple): Types of attention blocks.
    - dropout (float): Dropout rate.
    - cross_attention_dim (int): Dimension for cross-attention.
    - activation_fn (str): Activation function.
    - attention_bias (bool): Flag for attention bias.
    - upcast_attention (bool): Flag for upcast attention.
    - cross_frame_attention_mode: Mode for cross-frame attention.
    - temporal_position_encoding (bool): Flag for temporal position encoding.
    - temporal_position_encoding_max_len (int): Maximum length for temporal position encoding.
    """
    def __init__(
        self,
        dim,
        num_attention_heads,
        attention_head_dim,
        attention_block_types=(
            "Temporal_Self",
            "Temporal_Self",
        ),
        dropout=0.0,
        cross_attention_dim=768,
        activation_fn="geglu",
        attention_bias=False,
        upcast_attention=False,
        cross_frame_attention_mode=None,
        temporal_position_encoding=False,
        temporal_position_encoding_max_len=24,
    ):
        super().__init__()

        attention_blocks = []
        norms = []

        for block_name in attention_block_types:
            attention_blocks.append(
                VersatileAttention(
                    attention_mode=block_name.split("_", maxsplit=1)[0],
                    cross_attention_dim=cross_attention_dim
                    if block_name.endswith("_Cross")
                    else None,
                    query_dim=dim,
                    heads=num_attention_heads,
                    dim_head=attention_head_dim,
                    dropout=dropout,
                    bias=attention_bias,
                    upcast_attention=upcast_attention,
                    cross_frame_attention_mode=cross_frame_attention_mode,
                    temporal_position_encoding=temporal_position_encoding,
                    temporal_position_encoding_max_len=temporal_position_encoding_max_len,
                )
            )
            norms.append(nn.LayerNorm(dim))

        self.attention_blocks = nn.ModuleList(attention_blocks)
        self.norms = nn.ModuleList(norms)

        self.ff = FeedForward(dim, dropout=dropout,
                              activation_fn=activation_fn)
        self.ff_norm = nn.LayerNorm(dim)

    def forward(
        self,
        hidden_states,
        encoder_hidden_states=None,
        video_length=None,
    ):
        """
        Forward pass for the TemporalTransformerBlock.

        Args:
            hidden_states (torch.Tensor): The input hidden states with shape
                (batch_size, video_length, in_channels).
            encoder_hidden_states (torch.Tensor, optional): The encoder hidden states
                with shape (batch_size, encoder_length, in_channels).
            video_length (int, optional): The length of the video.

        Returns:
            torch.Tensor: The output hidden states with shape
                (batch_size, video_length, in_channels).
        """
        for attention_block, norm in zip(self.attention_blocks, self.norms):
            norm_hidden_states = norm(hidden_states)
            hidden_states = (
                attention_block(
                    norm_hidden_states,
                    encoder_hidden_states=encoder_hidden_states
                    if attention_block.is_cross_attention
                    else None,
                    video_length=video_length,
                )
                + hidden_states
            )

        hidden_states = self.ff(self.ff_norm(hidden_states)) + hidden_states

        output = hidden_states
        return output


class PositionalEncoding(nn.Module):
    """
    Positional Encoding module for transformers.

    Args:
    - d_model (int): Model dimension.
    - dropout (float): Dropout rate.
    - max_len (int): Maximum length for positional encoding.
    """
    def __init__(self, d_model, dropout=0.0, max_len=24):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x):
        """
        Forward pass of the PositionalEncoding module.

        This method takes an input tensor `x` and adds the positional encoding to it. The positional encoding is
        generated based on the input tensor's shape and is added to the input tensor element-wise.

        Args:
            x (torch.Tensor): The input tensor to be positionally encoded.

        Returns:
            torch.Tensor: The positionally encoded tensor.
        """
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class VersatileAttention(Attention):
    """
    Versatile Attention class.

    Args:
    - attention_mode: Attention mode.
    - temporal_position_encoding (bool): Flag for temporal position encoding.
    - temporal_position_encoding_max_len (int): Maximum length for temporal position encoding.
    """
    def __init__(
        self,
        *args,
        attention_mode=None,
        cross_frame_attention_mode=None,
        temporal_position_encoding=False,
        temporal_position_encoding_max_len=24,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        assert attention_mode == "Temporal"

        self.attention_mode = attention_mode
        self.is_cross_attention = kwargs.get("cross_attention_dim") is not None

        self.pos_encoder = (
            PositionalEncoding(
                kwargs["query_dim"],
                dropout=0.0,
                max_len=temporal_position_encoding_max_len,
            )
            if (temporal_position_encoding and attention_mode == "Temporal")
            else None
        )

    def extra_repr(self):
        """
        Returns a string representation of the module with information about the attention mode and whether it is cross-attention.
        
        Returns:
            str: A string representation of the module.
        """
        return f"(Module Info) Attention_Mode: {self.attention_mode}, Is_Cross_Attention: {self.is_cross_attention}"

    def set_use_memory_efficient_attention_xformers(
        self,
        use_memory_efficient_attention_xformers: bool,
        attention_op = None,
    ):
        """
        Sets the use of memory-efficient attention xformers for the VersatileAttention class.

        Args:
            use_memory_efficient_attention_xformers (bool): A boolean flag indicating whether to use memory-efficient attention xformers or not.

        Returns:
            None

        """
        if use_memory_efficient_attention_xformers:
            if not is_xformers_available():
                raise ModuleNotFoundError(
                    (
                        "Refer to https://github.com/facebookresearch/xformers for more information on how to install"
                        " xformers"
                    ),
                    name="xformers",
                )

            if not torch.cuda.is_available():
                raise ValueError(
                    "torch.cuda.is_available() should be True but is False. xformers' memory efficient attention is"
                    " only available for GPU "
                )

            try:
                # Make sure we can run the memory efficient attention
                _ = xformers.ops.memory_efficient_attention(
                    torch.randn((1, 2, 40), device="cuda"),
                    torch.randn((1, 2, 40), device="cuda"),
                    torch.randn((1, 2, 40), device="cuda"),
                )
            except Exception as e:
                raise e
            processor = AttnProcessor()
        else:
            processor = AttnProcessor()

        self.set_processor(processor)

    def forward(
        self,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        video_length=None,
        **cross_attention_kwargs,
    ):
        """
        Args:
            hidden_states (`torch.Tensor`):
                The hidden states to be passed through the model.
            encoder_hidden_states (`torch.Tensor`, optional):
                The encoder hidden states to be passed through the model.
            attention_mask (`torch.Tensor`, optional):
                The attention mask to be used in the model.
            video_length (`int`, optional):
                The length of the video.
            cross_attention_kwargs (`dict`, optional):
                Additional keyword arguments to be used for cross-attention.

        Returns:
            `torch.Tensor`:
                The output tensor after passing through the model.

        """
        if self.attention_mode == "Temporal":
            d = hidden_states.shape[1]  # d means HxW
            hidden_states = rearrange(
                hidden_states, "(b f) d c -> (b d) f c", f=video_length
            )

            if self.pos_encoder is not None:
                hidden_states = self.pos_encoder(hidden_states)

            encoder_hidden_states = (
                repeat(encoder_hidden_states, "b n c -> (b d) n c", d=d)
                if encoder_hidden_states is not None
                else encoder_hidden_states
            )

        else:
            raise NotImplementedError

        hidden_states = self.processor(
            self,
            hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=attention_mask,
            **cross_attention_kwargs,
        )

        if self.attention_mode == "Temporal":
            hidden_states = rearrange(
                hidden_states, "(b d) f c -> (b f) d c", d=d)

        return hidden_states
