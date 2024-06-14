# pylint: disable=R0801
"""
This module implements the Transformer3DModel, a PyTorch model designed for processing
3D data such as videos. It extends ModelMixin and ConfigMixin to provide a transformer
model with support for gradient checkpointing and various types of attention mechanisms.
The model can be configured with different parameters such as the number of attention heads,
attention head dimension, and the number of layers. It also supports the use of audio modules
for enhanced feature extraction from video data.
"""

from dataclasses import dataclass
from typing import Optional

import torch
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models import ModelMixin
from diffusers.utils import BaseOutput
from einops import rearrange, repeat
from torch import nn

from .attention import (AudioTemporalBasicTransformerBlock,
                        TemporalBasicTransformerBlock)


@dataclass
class Transformer3DModelOutput(BaseOutput):
    """
    The output of the [`Transformer3DModel`].

    Attributes:
        sample (`torch.FloatTensor`):
            The output tensor from the transformer model, which is the result of processing the input
            hidden states through the transformer blocks and any subsequent layers.
    """
    sample: torch.FloatTensor


class Transformer3DModel(ModelMixin, ConfigMixin):
    """
    Transformer3DModel is a PyTorch model that extends `ModelMixin` and `ConfigMixin` to create a 3D transformer model.
    It implements the forward pass for processing input hidden states, encoder hidden states, and various types of attention masks.
    The model supports gradient checkpointing, which can be enabled by calling the `enable_gradient_checkpointing()` method.
    """
    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
        self,
        num_attention_heads: int = 16,
        attention_head_dim: int = 88,
        in_channels: Optional[int] = None,
        num_layers: int = 1,
        dropout: float = 0.0,
        norm_num_groups: int = 32,
        cross_attention_dim: Optional[int] = None,
        attention_bias: bool = False,
        activation_fn: str = "geglu",
        num_embeds_ada_norm: Optional[int] = None,
        use_linear_projection: bool = False,
        only_cross_attention: bool = False,
        upcast_attention: bool = False,
        unet_use_cross_frame_attention=None,
        unet_use_temporal_attention=None,
        use_audio_module=False,
        depth=0,
        unet_block_name=None,
        stack_enable_blocks_name = None,
        stack_enable_blocks_depth = None,
    ):
        super().__init__()
        self.use_linear_projection = use_linear_projection
        self.num_attention_heads = num_attention_heads
        self.attention_head_dim = attention_head_dim
        inner_dim = num_attention_heads * attention_head_dim
        self.use_audio_module = use_audio_module
        # Define input layers
        self.in_channels = in_channels

        self.norm = torch.nn.GroupNorm(
            num_groups=norm_num_groups, num_channels=in_channels, eps=1e-6, affine=True
        )
        if use_linear_projection:
            self.proj_in = nn.Linear(in_channels, inner_dim)
        else:
            self.proj_in = nn.Conv2d(
                in_channels, inner_dim, kernel_size=1, stride=1, padding=0
            )

        if use_audio_module:
            self.transformer_blocks = nn.ModuleList(
                [
                    AudioTemporalBasicTransformerBlock(
                        inner_dim,
                        num_attention_heads,
                        attention_head_dim,
                        dropout=dropout,
                        cross_attention_dim=cross_attention_dim,
                        activation_fn=activation_fn,
                        num_embeds_ada_norm=num_embeds_ada_norm,
                        attention_bias=attention_bias,
                        only_cross_attention=only_cross_attention,
                        upcast_attention=upcast_attention,
                        unet_use_cross_frame_attention=unet_use_cross_frame_attention,
                        unet_use_temporal_attention=unet_use_temporal_attention,
                        depth=depth,
                        unet_block_name=unet_block_name,
                        stack_enable_blocks_name=stack_enable_blocks_name,
                        stack_enable_blocks_depth=stack_enable_blocks_depth,
                    )
                    for d in range(num_layers)
                ]
            )
        else:
            # Define transformers blocks
            self.transformer_blocks = nn.ModuleList(
                [
                    TemporalBasicTransformerBlock(
                        inner_dim,
                        num_attention_heads,
                        attention_head_dim,
                        dropout=dropout,
                        cross_attention_dim=cross_attention_dim,
                        activation_fn=activation_fn,
                        num_embeds_ada_norm=num_embeds_ada_norm,
                        attention_bias=attention_bias,
                        only_cross_attention=only_cross_attention,
                        upcast_attention=upcast_attention,
                    )
                    for d in range(num_layers)
                ]
            )

        # 4. Define output layers
        if use_linear_projection:
            self.proj_out = nn.Linear(in_channels, inner_dim)
        else:
            self.proj_out = nn.Conv2d(
                inner_dim, in_channels, kernel_size=1, stride=1, padding=0
            )

        self.gradient_checkpointing = False

    def _set_gradient_checkpointing(self, module, value=False):
        if hasattr(module, "gradient_checkpointing"):
            module.gradient_checkpointing = value

    def forward(
        self,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        full_mask=None,
        face_mask=None,
        lip_mask=None,
        motion_scale=None,
        timestep=None,
        return_dict: bool = True,
    ):
        """
        Forward pass for the Transformer3DModel.

        Args:
            hidden_states (torch.Tensor): The input hidden states.
            encoder_hidden_states (torch.Tensor, optional): The input encoder hidden states.
            attention_mask (torch.Tensor, optional): The attention mask.
            full_mask (torch.Tensor, optional): The full mask.
            face_mask (torch.Tensor, optional): The face mask.
            lip_mask (torch.Tensor, optional): The lip mask.
            timestep (int, optional): The current timestep.
            return_dict (bool, optional): Whether to return a dictionary or a tuple.

        Returns:
            output (Union[Tuple, BaseOutput]): The output of the Transformer3DModel.
        """
        # Input
        assert (
            hidden_states.dim() == 5
        ), f"Expected hidden_states to have ndim=5, but got ndim={hidden_states.dim()}."
        video_length = hidden_states.shape[2]
        hidden_states = rearrange(hidden_states, "b c f h w -> (b f) c h w")

        # TODO
        if self.use_audio_module:
            encoder_hidden_states = rearrange(
                encoder_hidden_states,
                "bs f margin dim -> (bs f) margin dim",
            )
        else:
            if encoder_hidden_states.shape[0] != hidden_states.shape[0]:
                encoder_hidden_states = repeat(
                    encoder_hidden_states, "b n c -> (b f) n c", f=video_length
                )

        batch, _, height, weight = hidden_states.shape
        residual = hidden_states

        hidden_states = self.norm(hidden_states)
        if not self.use_linear_projection:
            hidden_states = self.proj_in(hidden_states)
            inner_dim = hidden_states.shape[1]
            hidden_states = hidden_states.permute(0, 2, 3, 1).reshape(
                batch, height * weight, inner_dim
            )
        else:
            inner_dim = hidden_states.shape[1]
            hidden_states = hidden_states.permute(0, 2, 3, 1).reshape(
                batch, height * weight, inner_dim
            )
            hidden_states = self.proj_in(hidden_states)

        # Blocks
        motion_frames = []
        for _, block in enumerate(self.transformer_blocks):
            if isinstance(block, TemporalBasicTransformerBlock):
                hidden_states, motion_frame_fea = block(
                    hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    timestep=timestep,
                    video_length=video_length,
                )
                motion_frames.append(motion_frame_fea)
            else:
                hidden_states = block(
                    hidden_states,  # shape [2, 4096, 320]
                    encoder_hidden_states=encoder_hidden_states,  # shape [2, 20, 640]
                    attention_mask=attention_mask,
                    full_mask=full_mask,
                    face_mask=face_mask,
                    lip_mask=lip_mask,
                    timestep=timestep,
                    video_length=video_length,
                    motion_scale=motion_scale,
                )

        # Output
        if not self.use_linear_projection:
            hidden_states = (
                hidden_states.reshape(batch, height, weight, inner_dim)
                .permute(0, 3, 1, 2)
                .contiguous()
            )
            hidden_states = self.proj_out(hidden_states)
        else:
            hidden_states = self.proj_out(hidden_states)
            hidden_states = (
                hidden_states.reshape(batch, height, weight, inner_dim)
                .permute(0, 3, 1, 2)
                .contiguous()
            )

        output = hidden_states + residual

        output = rearrange(output, "(b f) c h w -> b c f h w", f=video_length)
        if not return_dict:
            return (output, motion_frames)

        return Transformer3DModelOutput(sample=output)
