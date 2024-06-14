# pylint: disable=R0801
# pylint: disable=W1203

"""
This file defines the 2D blocks for the UNet model in a PyTorch implementation. 
The UNet model is a popular architecture for image segmentation tasks, 
which consists of an encoder, a decoder, and a skip connection mechanism. 
The 2D blocks in this file include various types of layers, such as ResNet blocks, 
Transformer blocks, and cross-attention blocks, 
which are used to build the encoder and decoder parts of the UNet model. 
The AutoencoderTinyBlock class is a simple autoencoder block for tiny models, 
and the UNetMidBlock2D and CrossAttnDownBlock2D, DownBlock2D, CrossAttnUpBlock2D, 
and UpBlock2D classes are used for the middle and decoder parts of the UNet model. 
The classes and functions in this file provide a flexible and modular way 
to construct the UNet model for different image segmentation tasks.
"""

from typing import Any, Dict, Optional, Tuple, Union

import torch
from diffusers.models.activations import get_activation
from diffusers.models.attention_processor import Attention
from diffusers.models.resnet import Downsample2D, ResnetBlock2D, Upsample2D
from diffusers.models.transformers.dual_transformer_2d import \
    DualTransformer2DModel
from diffusers.utils import is_torch_version, logging
from diffusers.utils.torch_utils import apply_freeu
from torch import nn

from .transformer_2d import Transformer2DModel

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


def get_down_block(
    down_block_type: str,
    num_layers: int,
    in_channels: int,
    out_channels: int,
    temb_channels: int,
    add_downsample: bool,
    resnet_eps: float,
    resnet_act_fn: str,
    transformer_layers_per_block: int = 1,
    num_attention_heads: Optional[int] = None,
    resnet_groups: Optional[int] = None,
    cross_attention_dim: Optional[int] = None,
    downsample_padding: Optional[int] = None,
    dual_cross_attention: bool = False,
    use_linear_projection: bool = False,
    only_cross_attention: bool = False,
    upcast_attention: bool = False,
    resnet_time_scale_shift: str = "default",
    attention_type: str = "default",
    attention_head_dim: Optional[int] = None,
    dropout: float = 0.0,
):
    """ This function creates and returns a UpBlock2D or CrossAttnUpBlock2D object based on the given up_block_type.

    Args:
        up_block_type (str): The type of up block to create. Must be either "UpBlock2D" or "CrossAttnUpBlock2D".
        num_layers (int): The number of layers in the ResNet block.
        in_channels (int): The number of input channels.
        out_channels (int): The number of output channels.
        prev_output_channel (int): The number of channels in the previous output.
        temb_channels (int): The number of channels in the token embedding.
        add_upsample (bool): Whether to add an upsample layer after the ResNet block. Defaults to True.
        resnet_eps (float): The epsilon value for the ResNet block. Defaults to 1e-6.
        resnet_act_fn (str): The activation function to use in the ResNet block. Defaults to "swish".
        resnet_groups (int): The number of groups in the ResNet block. Defaults to 32.
        resnet_pre_norm (bool): Whether to use pre-normalization in the ResNet block. Defaults to True.
        output_scale_factor (float): The scale factor to apply to the output. Defaults to 1.0.

    Returns:
        nn.Module: The created UpBlock2D or CrossAttnUpBlock2D object.
    """
    # If attn head dim is not defined, we default it to the number of heads
    if attention_head_dim is None:
        logger.warning("It is recommended to provide `attention_head_dim` when calling `get_down_block`.")
        logger.warning(f"Defaulting `attention_head_dim` to {num_attention_heads}.")
        attention_head_dim = num_attention_heads

    down_block_type = (
        down_block_type[7:]
        if down_block_type.startswith("UNetRes")
        else down_block_type
    )
    if down_block_type == "DownBlock2D":
        return DownBlock2D(
            num_layers=num_layers,
            in_channels=in_channels,
            out_channels=out_channels,
            temb_channels=temb_channels,
            dropout=dropout,
            add_downsample=add_downsample,
            resnet_eps=resnet_eps,
            resnet_act_fn=resnet_act_fn,
            resnet_groups=resnet_groups,
            downsample_padding=downsample_padding,
            resnet_time_scale_shift=resnet_time_scale_shift,
        )

    if down_block_type == "CrossAttnDownBlock2D":
        if cross_attention_dim is None:
            raise ValueError(
                "cross_attention_dim must be specified for CrossAttnDownBlock2D"
            )
        return CrossAttnDownBlock2D(
            num_layers=num_layers,
            transformer_layers_per_block=transformer_layers_per_block,
            in_channels=in_channels,
            out_channels=out_channels,
            temb_channels=temb_channels,
            dropout=dropout,
            add_downsample=add_downsample,
            resnet_eps=resnet_eps,
            resnet_act_fn=resnet_act_fn,
            resnet_groups=resnet_groups,
            downsample_padding=downsample_padding,
            cross_attention_dim=cross_attention_dim,
            num_attention_heads=num_attention_heads,
            dual_cross_attention=dual_cross_attention,
            use_linear_projection=use_linear_projection,
            only_cross_attention=only_cross_attention,
            upcast_attention=upcast_attention,
            resnet_time_scale_shift=resnet_time_scale_shift,
            attention_type=attention_type,
        )
    raise ValueError(f"{down_block_type} does not exist.")


def get_up_block(
    up_block_type: str,
    num_layers: int,
    in_channels: int,
    out_channels: int,
    prev_output_channel: int,
    temb_channels: int,
    add_upsample: bool,
    resnet_eps: float,
    resnet_act_fn: str,
    resolution_idx: Optional[int] = None,
    transformer_layers_per_block: int = 1,
    num_attention_heads: Optional[int] = None,
    resnet_groups: Optional[int] = None,
    cross_attention_dim: Optional[int] = None,
    dual_cross_attention: bool = False,
    use_linear_projection: bool = False,
    only_cross_attention: bool = False,
    upcast_attention: bool = False,
    resnet_time_scale_shift: str = "default",
    attention_type: str = "default",
    attention_head_dim: Optional[int] = None,
    dropout: float = 0.0,
) -> nn.Module:
    """ This function ...
        Args:
        Returns:
    """
    # If attn head dim is not defined, we default it to the number of heads
    if attention_head_dim is None:
        logger.warning("It is recommended to provide `attention_head_dim` when calling `get_up_block`.")
        logger.warning(f"Defaulting `attention_head_dim` to {num_attention_heads}.")
        attention_head_dim = num_attention_heads

    up_block_type = (
        up_block_type[7:] if up_block_type.startswith("UNetRes") else up_block_type
    )
    if up_block_type == "UpBlock2D":
        return UpBlock2D(
            num_layers=num_layers,
            in_channels=in_channels,
            out_channels=out_channels,
            prev_output_channel=prev_output_channel,
            temb_channels=temb_channels,
            resolution_idx=resolution_idx,
            dropout=dropout,
            add_upsample=add_upsample,
            resnet_eps=resnet_eps,
            resnet_act_fn=resnet_act_fn,
            resnet_groups=resnet_groups,
            resnet_time_scale_shift=resnet_time_scale_shift,
        )
    if up_block_type == "CrossAttnUpBlock2D":
        if cross_attention_dim is None:
            raise ValueError(
                "cross_attention_dim must be specified for CrossAttnUpBlock2D"
            )
        return CrossAttnUpBlock2D(
            num_layers=num_layers,
            transformer_layers_per_block=transformer_layers_per_block,
            in_channels=in_channels,
            out_channels=out_channels,
            prev_output_channel=prev_output_channel,
            temb_channels=temb_channels,
            resolution_idx=resolution_idx,
            dropout=dropout,
            add_upsample=add_upsample,
            resnet_eps=resnet_eps,
            resnet_act_fn=resnet_act_fn,
            resnet_groups=resnet_groups,
            cross_attention_dim=cross_attention_dim,
            num_attention_heads=num_attention_heads,
            dual_cross_attention=dual_cross_attention,
            use_linear_projection=use_linear_projection,
            only_cross_attention=only_cross_attention,
            upcast_attention=upcast_attention,
            resnet_time_scale_shift=resnet_time_scale_shift,
            attention_type=attention_type,
        )

    raise ValueError(f"{up_block_type} does not exist.")


class AutoencoderTinyBlock(nn.Module):
    """
    Tiny Autoencoder block used in [`AutoencoderTiny`]. It is a mini residual module consisting of plain conv + ReLU
    blocks.

    Args:
        in_channels (`int`): The number of input channels.
        out_channels (`int`): The number of output channels.
        act_fn (`str`):
            ` The activation function to use. Supported values are `"swish"`, `"mish"`, `"gelu"`, and `"relu"`.

    Returns:
        `torch.FloatTensor`: A tensor with the same shape as the input tensor, but with the number of channels equal to
        `out_channels`.
    """

    def __init__(self, in_channels: int, out_channels: int, act_fn: str):
        super().__init__()
        act_fn = get_activation(act_fn)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            act_fn,
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            act_fn,
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        )
        self.skip = (
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
            if in_channels != out_channels
            else nn.Identity()
        )
        self.fuse = nn.ReLU()

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        """
        Forward pass of the AutoencoderTinyBlock class.

        Parameters:
        x (torch.FloatTensor): The input tensor to the AutoencoderTinyBlock.

        Returns:
        torch.FloatTensor: The output tensor after passing through the AutoencoderTinyBlock.
        """
        return self.fuse(self.conv(x) + self.skip(x))


class UNetMidBlock2D(nn.Module):
    """
    A 2D UNet mid-block [`UNetMidBlock2D`] with multiple residual blocks and optional attention blocks.

    Args:
        in_channels (`int`): The number of input channels.
        temb_channels (`int`): The number of temporal embedding channels.
        dropout (`float`, *optional*, defaults to 0.0): The dropout rate.
        num_layers (`int`, *optional*, defaults to 1): The number of residual blocks.
        resnet_eps (`float`, *optional*, 1e-6 ): The epsilon value for the resnet blocks.
        resnet_time_scale_shift (`str`, *optional*, defaults to `default`):
            The type of normalization to apply to the time embeddings. This can help to improve the performance of the
            model on tasks with long-range temporal dependencies.
        resnet_act_fn (`str`, *optional*, defaults to `swish`): The activation function for the resnet blocks.
        resnet_groups (`int`, *optional*, defaults to 32):
            The number of groups to use in the group normalization layers of the resnet blocks.
        attn_groups (`Optional[int]`, *optional*, defaults to None): The number of groups for the attention blocks.
        resnet_pre_norm (`bool`, *optional*, defaults to `True`):
            Whether to use pre-normalization for the resnet blocks.
        add_attention (`bool`, *optional*, defaults to `True`): Whether to add attention blocks.
        attention_head_dim (`int`, *optional*, defaults to 1):
            Dimension of a single attention head. The number of attention heads is determined based on this value and
            the number of input channels.
        output_scale_factor (`float`, *optional*, defaults to 1.0): The output scale factor.

    Returns:
        `torch.FloatTensor`: The output of the last residual block, which is a tensor of shape `(batch_size,
        in_channels, height, width)`.

    """

    def __init__(
        self,
        in_channels: int,
        temb_channels: int,
        dropout: float = 0.0,
        num_layers: int = 1,
        resnet_eps: float = 1e-6,
        resnet_time_scale_shift: str = "default",  # default, spatial
        resnet_act_fn: str = "swish",
        resnet_groups: int = 32,
        attn_groups: Optional[int] = None,
        resnet_pre_norm: bool = True,
        add_attention: bool = True,
        attention_head_dim: int = 1,
        output_scale_factor: float = 1.0,
    ):
        super().__init__()
        resnet_groups = (
            resnet_groups if resnet_groups is not None else min(in_channels // 4, 32)
        )
        self.add_attention = add_attention

        if attn_groups is None:
            attn_groups = (
                resnet_groups if resnet_time_scale_shift == "default" else None
            )

        # there is always at least one resnet
        resnets = [
            ResnetBlock2D(
                in_channels=in_channels,
                out_channels=in_channels,
                temb_channels=temb_channels,
                eps=resnet_eps,
                groups=resnet_groups,
                dropout=dropout,
                time_embedding_norm=resnet_time_scale_shift,
                non_linearity=resnet_act_fn,
                output_scale_factor=output_scale_factor,
                pre_norm=resnet_pre_norm,
            )
        ]
        attentions = []

        if attention_head_dim is None:
            logger.warning(
                f"It is not recommend to pass `attention_head_dim=None`. Defaulting `attention_head_dim` to `in_channels`: {in_channels}."
            )
            attention_head_dim = in_channels

        for _ in range(num_layers):
            if self.add_attention:
                attentions.append(
                    Attention(
                        in_channels,
                        heads=in_channels // attention_head_dim,
                        dim_head=attention_head_dim,
                        rescale_output_factor=output_scale_factor,
                        eps=resnet_eps,
                        norm_num_groups=attn_groups,
                        spatial_norm_dim=(
                            temb_channels
                            if resnet_time_scale_shift == "spatial"
                            else None
                        ),
                        residual_connection=True,
                        bias=True,
                        upcast_softmax=True,
                        _from_deprecated_attn_block=True,
                    )
                )
            else:
                attentions.append(None)

            resnets.append(
                ResnetBlock2D(
                    in_channels=in_channels,
                    out_channels=in_channels,
                    temb_channels=temb_channels,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=dropout,
                    time_embedding_norm=resnet_time_scale_shift,
                    non_linearity=resnet_act_fn,
                    output_scale_factor=output_scale_factor,
                    pre_norm=resnet_pre_norm,
                )
            )

        self.attentions = nn.ModuleList(attentions)
        self.resnets = nn.ModuleList(resnets)

    def forward(
        self, hidden_states: torch.FloatTensor, temb: Optional[torch.FloatTensor] = None
    ) -> torch.FloatTensor:
        """
        Forward pass of the UNetMidBlock2D class.

        Args:
            hidden_states (torch.FloatTensor): The input tensor to the UNetMidBlock2D.
            temb (Optional[torch.FloatTensor], optional): The token embedding tensor. Defaults to None.

        Returns:
            torch.FloatTensor: The output tensor after passing through the UNetMidBlock2D.
        """
        # Your implementation here
        hidden_states = self.resnets[0](hidden_states, temb)
        for attn, resnet in zip(self.attentions, self.resnets[1:]):
            if attn is not None:
                hidden_states = attn(hidden_states, temb=temb)
            hidden_states = resnet(hidden_states, temb)

        return hidden_states


class UNetMidBlock2DCrossAttn(nn.Module):
    """
    UNetMidBlock2DCrossAttn is a class that represents a mid-block 2D UNet with cross-attention.
    
    This block is responsible for processing the input tensor with a series of residual blocks,
    and applying cross-attention mechanism to attend to the global information in the encoder.
    
    Args:
        in_channels (int): The number of input channels.
        temb_channels (int): The number of channels for the token embedding.
        dropout (float, optional): The dropout rate. Defaults to 0.0.
        num_layers (int, optional): The number of layers in the residual blocks. Defaults to 1.
        resnet_eps (float, optional): The epsilon value for the residual blocks. Defaults to 1e-6.
        resnet_time_scale_shift (str, optional): The time scale shift type for the residual blocks. Defaults to "default".
        resnet_act_fn (str, optional): The activation function for the residual blocks. Defaults to "swish".
        resnet_groups (int, optional): The number of groups for the residual blocks. Defaults to 32.
        resnet_pre_norm (bool, optional): Whether to apply pre-normalization for the residual blocks. Defaults to True.
        num_attention_heads (int, optional): The number of attention heads for cross-attention. Defaults to 1.
        cross_attention_dim (int, optional): The dimension of the cross-attention. Defaults to 1280.
        output_scale_factor (float, optional): The scale factor for the output tensor. Defaults to 1.0.
    """
    def __init__(
        self,
        in_channels: int,
        temb_channels: int,
        dropout: float = 0.0,
        num_layers: int = 1,
        transformer_layers_per_block: Union[int, Tuple[int]] = 1,
        resnet_eps: float = 1e-6,
        resnet_time_scale_shift: str = "default",
        resnet_act_fn: str = "swish",
        resnet_groups: int = 32,
        resnet_pre_norm: bool = True,
        num_attention_heads: int = 1,
        output_scale_factor: float = 1.0,
        cross_attention_dim: int = 1280,
        dual_cross_attention: bool = False,
        use_linear_projection: bool = False,
        upcast_attention: bool = False,
        attention_type: str = "default",
    ):
        super().__init__()

        self.has_cross_attention = True
        self.num_attention_heads = num_attention_heads
        resnet_groups = (
            resnet_groups if resnet_groups is not None else min(in_channels // 4, 32)
        )

        # support for variable transformer layers per block
        if isinstance(transformer_layers_per_block, int):
            transformer_layers_per_block = [transformer_layers_per_block] * num_layers

        # there is always at least one resnet
        resnets = [
            ResnetBlock2D(
                in_channels=in_channels,
                out_channels=in_channels,
                temb_channels=temb_channels,
                eps=resnet_eps,
                groups=resnet_groups,
                dropout=dropout,
                time_embedding_norm=resnet_time_scale_shift,
                non_linearity=resnet_act_fn,
                output_scale_factor=output_scale_factor,
                pre_norm=resnet_pre_norm,
            )
        ]
        attentions = []

        for i in range(num_layers):
            if not dual_cross_attention:
                attentions.append(
                    Transformer2DModel(
                        num_attention_heads,
                        in_channels // num_attention_heads,
                        in_channels=in_channels,
                        num_layers=transformer_layers_per_block[i],
                        cross_attention_dim=cross_attention_dim,
                        norm_num_groups=resnet_groups,
                        use_linear_projection=use_linear_projection,
                        upcast_attention=upcast_attention,
                        attention_type=attention_type,
                    )
                )
            else:
                attentions.append(
                    DualTransformer2DModel(
                        num_attention_heads,
                        in_channels // num_attention_heads,
                        in_channels=in_channels,
                        num_layers=1,
                        cross_attention_dim=cross_attention_dim,
                        norm_num_groups=resnet_groups,
                    )
                )
            resnets.append(
                ResnetBlock2D(
                    in_channels=in_channels,
                    out_channels=in_channels,
                    temb_channels=temb_channels,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=dropout,
                    time_embedding_norm=resnet_time_scale_shift,
                    non_linearity=resnet_act_fn,
                    output_scale_factor=output_scale_factor,
                    pre_norm=resnet_pre_norm,
                )
            )

        self.attentions = nn.ModuleList(attentions)
        self.resnets = nn.ModuleList(resnets)

        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        temb: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
    ) -> torch.FloatTensor:
        """
        Forward pass for the UNetMidBlock2DCrossAttn class.

        Args:
            hidden_states (torch.FloatTensor): The input hidden states tensor.
            temb (Optional[torch.FloatTensor], optional): The optional tensor for time embeddings.
            encoder_hidden_states (Optional[torch.FloatTensor], optional): The optional encoder hidden states tensor.
            attention_mask (Optional[torch.FloatTensor], optional): The optional attention mask tensor.
            cross_attention_kwargs (Optional[Dict[str, Any]], optional): The optional cross-attention kwargs tensor.
            encoder_attention_mask (Optional[torch.FloatTensor], optional): The optional encoder attention mask tensor.

        Returns:
            torch.FloatTensor: The output tensor after passing through the UNetMidBlock2DCrossAttn layers.
        """
        lora_scale = (
            cross_attention_kwargs.get("scale", 1.0)
            if cross_attention_kwargs is not None
            else 1.0
        )
        hidden_states = self.resnets[0](hidden_states, temb, scale=lora_scale)
        for attn, resnet in zip(self.attentions, self.resnets[1:]):
            if self.training and self.gradient_checkpointing:

                def create_custom_forward(module, return_dict=None):
                    def custom_forward(*inputs):
                        if return_dict is not None:
                            return module(*inputs, return_dict=return_dict)

                        return module(*inputs)

                    return custom_forward

                ckpt_kwargs: Dict[str, Any] = (
                    {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                )
                hidden_states, _ref_feature = attn(
                    hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    cross_attention_kwargs=cross_attention_kwargs,
                    attention_mask=attention_mask,
                    encoder_attention_mask=encoder_attention_mask,
                    return_dict=False,
                )
                hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(resnet),
                    hidden_states,
                    temb,
                    **ckpt_kwargs,
                )
            else:
                hidden_states, _ref_feature = attn(
                    hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    cross_attention_kwargs=cross_attention_kwargs,
                    attention_mask=attention_mask,
                    encoder_attention_mask=encoder_attention_mask,
                    return_dict=False,
                )
                hidden_states = resnet(hidden_states, temb, scale=lora_scale)

        return hidden_states


class CrossAttnDownBlock2D(nn.Module):
    """
    CrossAttnDownBlock2D is a class that represents a 2D cross-attention downsampling block.
    
    This block is used in the UNet model and consists of a series of ResNet blocks and Transformer layers.
    It takes input hidden states, a tensor embedding, and optional encoder hidden states, attention mask,
    and cross-attention kwargs. The block performs a series of operations including downsampling, cross-attention,
    and residual connections.

    Attributes:
        in_channels (int): The number of input channels.
        out_channels (int): The number of output channels.
        temb_channels (int): The number of tensor embedding channels.
        dropout (float): The dropout rate.
        num_layers (int): The number of ResNet layers.
        transformer_layers_per_block (Union[int, Tuple[int]]): The number of Transformer layers per block.
        resnet_eps (float): The ResNet epsilon value.
        resnet_time_scale_shift (str): The ResNet time scale shift type.
        resnet_act_fn (str): The ResNet activation function.
        resnet_groups (int): The ResNet group size.
        resnet_pre_norm (bool): Whether to use ResNet pre-normalization.
        num_attention_heads (int): The number of attention heads.
        cross_attention_dim (int): The cross-attention dimension.
        output_scale_factor (float): The output scale factor.
        downsample_padding (int): The downsampling padding.
        add_downsample (bool): Whether to add downsampling.
        dual_cross_attention (bool): Whether to use dual cross-attention.
        use_linear_projection (bool): Whether to use linear projection.
        only_cross_attention (bool): Whether to use only cross-attention.
        upcast_attention (bool): Whether to upcast attention.
        attention_type (str): The attention type.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        temb_channels: int,
        dropout: float = 0.0,
        num_layers: int = 1,
        transformer_layers_per_block: Union[int, Tuple[int]] = 1,
        resnet_eps: float = 1e-6,
        resnet_time_scale_shift: str = "default",
        resnet_act_fn: str = "swish",
        resnet_groups: int = 32,
        resnet_pre_norm: bool = True,
        num_attention_heads: int = 1,
        cross_attention_dim: int = 1280,
        output_scale_factor: float = 1.0,
        downsample_padding: int = 1,
        add_downsample: bool = True,
        dual_cross_attention: bool = False,
        use_linear_projection: bool = False,
        only_cross_attention: bool = False,
        upcast_attention: bool = False,
        attention_type: str = "default",
    ):
        super().__init__()
        resnets = []
        attentions = []

        self.has_cross_attention = True
        self.num_attention_heads = num_attention_heads
        if isinstance(transformer_layers_per_block, int):
            transformer_layers_per_block = [transformer_layers_per_block] * num_layers

        for i in range(num_layers):
            in_channels = in_channels if i == 0 else out_channels
            resnets.append(
                ResnetBlock2D(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=dropout,
                    time_embedding_norm=resnet_time_scale_shift,
                    non_linearity=resnet_act_fn,
                    output_scale_factor=output_scale_factor,
                    pre_norm=resnet_pre_norm,
                )
            )
            if not dual_cross_attention:
                attentions.append(
                    Transformer2DModel(
                        num_attention_heads,
                        out_channels // num_attention_heads,
                        in_channels=out_channels,
                        num_layers=transformer_layers_per_block[i],
                        cross_attention_dim=cross_attention_dim,
                        norm_num_groups=resnet_groups,
                        use_linear_projection=use_linear_projection,
                        only_cross_attention=only_cross_attention,
                        upcast_attention=upcast_attention,
                        attention_type=attention_type,
                    )
                )
            else:
                attentions.append(
                    DualTransformer2DModel(
                        num_attention_heads,
                        out_channels // num_attention_heads,
                        in_channels=out_channels,
                        num_layers=1,
                        cross_attention_dim=cross_attention_dim,
                        norm_num_groups=resnet_groups,
                    )
                )
        self.attentions = nn.ModuleList(attentions)
        self.resnets = nn.ModuleList(resnets)

        if add_downsample:
            self.downsamplers = nn.ModuleList(
                [
                    Downsample2D(
                        out_channels,
                        use_conv=True,
                        out_channels=out_channels,
                        padding=downsample_padding,
                        name="op",
                    )
                ]
            )
        else:
            self.downsamplers = None

        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        temb: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        additional_residuals: Optional[torch.FloatTensor] = None,
    ) -> Tuple[torch.FloatTensor, Tuple[torch.FloatTensor, ...]]:
        """
        Forward pass for the CrossAttnDownBlock2D class.

        Args:
            hidden_states (torch.FloatTensor): The input hidden states.
            temb (Optional[torch.FloatTensor], optional): The token embeddings. Defaults to None.
            encoder_hidden_states (Optional[torch.FloatTensor], optional): The encoder hidden states. Defaults to None.
            attention_mask (Optional[torch.FloatTensor], optional): The attention mask. Defaults to None.
            cross_attention_kwargs (Optional[Dict[str, Any]], optional): The cross-attention kwargs. Defaults to None.
            encoder_attention_mask (Optional[torch.FloatTensor], optional): The encoder attention mask. Defaults to None.
            additional_residuals (Optional[torch.FloatTensor], optional): The additional residuals. Defaults to None.

        Returns:
            Tuple[torch.FloatTensor, Tuple[torch.FloatTensor, ...]]: The output hidden states and residuals.
        """
        output_states = ()

        lora_scale = (
            cross_attention_kwargs.get("scale", 1.0)
            if cross_attention_kwargs is not None
            else 1.0
        )

        blocks = list(zip(self.resnets, self.attentions))

        for i, (resnet, attn) in enumerate(blocks):
            if self.training and self.gradient_checkpointing:

                def create_custom_forward(module, return_dict=None):
                    def custom_forward(*inputs):
                        if return_dict is not None:
                            return module(*inputs, return_dict=return_dict)

                        return module(*inputs)

                    return custom_forward

                ckpt_kwargs: Dict[str, Any] = (
                    {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                )
                hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(resnet),
                    hidden_states,
                    temb,
                    **ckpt_kwargs,
                )
                hidden_states, _ref_feature = attn(
                    hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    cross_attention_kwargs=cross_attention_kwargs,
                    attention_mask=attention_mask,
                    encoder_attention_mask=encoder_attention_mask,
                    return_dict=False,
                )
            else:
                hidden_states = resnet(hidden_states, temb, scale=lora_scale)
                hidden_states, _ref_feature = attn(
                    hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    cross_attention_kwargs=cross_attention_kwargs,
                    attention_mask=attention_mask,
                    encoder_attention_mask=encoder_attention_mask,
                    return_dict=False,
                )

            # apply additional residuals to the output of the last pair of resnet and attention blocks
            if i == len(blocks) - 1 and additional_residuals is not None:
                hidden_states = hidden_states + additional_residuals

            output_states = output_states + (hidden_states,)

        if self.downsamplers is not None:
            for downsampler in self.downsamplers:
                hidden_states = downsampler(hidden_states, scale=lora_scale)

            output_states = output_states + (hidden_states,)

        return hidden_states, output_states


class DownBlock2D(nn.Module):
    """
    DownBlock2D is a class that represents a 2D downsampling block in a neural network.

    It takes the following parameters:
    - in_channels (int): The number of input channels in the block.
    - out_channels (int): The number of output channels in the block.
    - temb_channels (int): The number of channels in the token embedding.
    - dropout (float): The dropout rate for the block.
    - num_layers (int): The number of layers in the block.
    - resnet_eps (float): The epsilon value for the ResNet layer.
    - resnet_time_scale_shift (str): The type of activation function for the ResNet layer.
    - resnet_act_fn (str): The activation function for the ResNet layer.
    - resnet_groups (int): The number of groups in the ResNet layer.
    - resnet_pre_norm (bool): Whether to apply layer normalization before the ResNet layer.
    - output_scale_factor (float): The scale factor for the output.
    - add_downsample (bool): Whether to add a downsampling layer.
    - downsample_padding (int): The padding value for the downsampling layer.

    The DownBlock2D class inherits from the nn.Module class and defines the following methods:
    - __init__: Initializes the DownBlock2D class with the given parameters.
    - forward: Forward pass of the DownBlock2D class.

    The forward method takes the following parameters:
    - hidden_states (torch.FloatTensor): The input tensor to the block.
    - temb (Optional[torch.FloatTensor]): The token embedding tensor.
    - scale (float): The scale factor for the input tensor.

    The forward method returns a tuple containing the output tensor and a tuple of hidden states.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        temb_channels: int,
        dropout: float = 0.0,
        num_layers: int = 1,
        resnet_eps: float = 1e-6,
        resnet_time_scale_shift: str = "default",
        resnet_act_fn: str = "swish",
        resnet_groups: int = 32,
        resnet_pre_norm: bool = True,
        output_scale_factor: float = 1.0,
        add_downsample: bool = True,
        downsample_padding: int = 1,
    ):
        super().__init__()
        resnets = []

        for i in range(num_layers):
            in_channels = in_channels if i == 0 else out_channels
            resnets.append(
                ResnetBlock2D(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=dropout,
                    time_embedding_norm=resnet_time_scale_shift,
                    non_linearity=resnet_act_fn,
                    output_scale_factor=output_scale_factor,
                    pre_norm=resnet_pre_norm,
                )
            )

        self.resnets = nn.ModuleList(resnets)

        if add_downsample:
            self.downsamplers = nn.ModuleList(
                [
                    Downsample2D(
                        out_channels,
                        use_conv=True,
                        out_channels=out_channels,
                        padding=downsample_padding,
                        name="op",
                    )
                ]
            )
        else:
            self.downsamplers = None

        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        temb: Optional[torch.FloatTensor] = None,
        scale: float = 1.0,
    ) -> Tuple[torch.FloatTensor, Tuple[torch.FloatTensor, ...]]:
        """
        Forward pass of the DownBlock2D class.

        Args:
            hidden_states (torch.FloatTensor): The input tensor to the DownBlock2D layer.
            temb (Optional[torch.FloatTensor], optional): The token embedding tensor. Defaults to None.
            scale (float, optional): The scale factor for the input tensor. Defaults to 1.0.

        Returns:
            Tuple[torch.FloatTensor, Tuple[torch.FloatTensor, ...]]: The output tensor and any additional hidden states.
        """
        output_states = ()

        for resnet in self.resnets:
            if self.training and self.gradient_checkpointing:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs)

                    return custom_forward

                if is_torch_version(">=", "1.11.0"):
                    hidden_states = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(resnet),
                        hidden_states,
                        temb,
                        use_reentrant=False,
                    )
                else:
                    hidden_states = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(resnet), hidden_states, temb
                    )
            else:
                hidden_states = resnet(hidden_states, temb, scale=scale)

            output_states = output_states + (hidden_states,)

        if self.downsamplers is not None:
            for downsampler in self.downsamplers:
                hidden_states = downsampler(hidden_states, scale=scale)

            output_states = output_states + (hidden_states,)

        return hidden_states, output_states


class CrossAttnUpBlock2D(nn.Module):
    """
    CrossAttnUpBlock2D is a class that represents a cross-attention UpBlock in a 2D UNet architecture.
    
    This block is responsible for upsampling the input tensor and performing cross-attention with the encoder's hidden states.
    
    Args:
        in_channels (int): The number of input channels in the tensor.
        out_channels (int): The number of output channels in the tensor.
        prev_output_channel (int): The number of channels in the previous output tensor.
        temb_channels (int): The number of channels in the token embedding tensor.
        resolution_idx (Optional[int]): The index of the resolution in the model.
        dropout (float): The dropout rate for the layer.
        num_layers (int): The number of layers in the ResNet block.
        transformer_layers_per_block (Union[int, Tuple[int]]): The number of transformer layers per block.
        resnet_eps (float): The epsilon value for the ResNet layer.
        resnet_time_scale_shift (str): The type of time scale shift to be applied in the ResNet layer.
        resnet_act_fn (str): The activation function to be used in the ResNet layer.
        resnet_groups (int): The number of groups in the ResNet layer.
        resnet_pre_norm (bool): Whether to use pre-normalization in the ResNet layer.
        num_attention_heads (int): The number of attention heads in the cross-attention layer.
        cross_attention_dim (int): The dimension of the cross-attention layer.
        output_scale_factor (float): The scale factor for the output tensor.
        add_upsample (bool): Whether to add upsampling to the block.
        dual_cross_attention (bool): Whether to use dual cross-attention.
        use_linear_projection (bool): Whether to use linear projection in the cross-attention layer.
        only_cross_attention (bool): Whether to only use cross-attention and no self-attention.
        upcast_attention (bool): Whether to upcast the attention weights.
        attention_type (str): The type of attention to be used in the cross-attention layer.

    Attributes:
        up_block (nn.Module): The UpBlock module responsible for upsampling the input tensor.
        cross_attn (nn.Module): The cross-attention module that performs attention between 
        the decoder's hidden states and the encoder's hidden states.
        resnet_blocks (nn.ModuleList): A list of ResNet blocks that make up the ResNet portion of the block.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        prev_output_channel: int,
        temb_channels: int,
        resolution_idx: Optional[int] = None,
        dropout: float = 0.0,
        num_layers: int = 1,
        transformer_layers_per_block: Union[int, Tuple[int]] = 1,
        resnet_eps: float = 1e-6,
        resnet_time_scale_shift: str = "default",
        resnet_act_fn: str = "swish",
        resnet_groups: int = 32,
        resnet_pre_norm: bool = True,
        num_attention_heads: int = 1,
        cross_attention_dim: int = 1280,
        output_scale_factor: float = 1.0,
        add_upsample: bool = True,
        dual_cross_attention: bool = False,
        use_linear_projection: bool = False,
        only_cross_attention: bool = False,
        upcast_attention: bool = False,
        attention_type: str = "default",
    ):
        super().__init__()
        resnets = []
        attentions = []

        self.has_cross_attention = True
        self.num_attention_heads = num_attention_heads

        if isinstance(transformer_layers_per_block, int):
            transformer_layers_per_block = [transformer_layers_per_block] * num_layers

        for i in range(num_layers):
            res_skip_channels = in_channels if (i == num_layers - 1) else out_channels
            resnet_in_channels = prev_output_channel if i == 0 else out_channels

            resnets.append(
                ResnetBlock2D(
                    in_channels=resnet_in_channels + res_skip_channels,
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=dropout,
                    time_embedding_norm=resnet_time_scale_shift,
                    non_linearity=resnet_act_fn,
                    output_scale_factor=output_scale_factor,
                    pre_norm=resnet_pre_norm,
                )
            )
            if not dual_cross_attention:
                attentions.append(
                    Transformer2DModel(
                        num_attention_heads,
                        out_channels // num_attention_heads,
                        in_channels=out_channels,
                        num_layers=transformer_layers_per_block[i],
                        cross_attention_dim=cross_attention_dim,
                        norm_num_groups=resnet_groups,
                        use_linear_projection=use_linear_projection,
                        only_cross_attention=only_cross_attention,
                        upcast_attention=upcast_attention,
                        attention_type=attention_type,
                    )
                )
            else:
                attentions.append(
                    DualTransformer2DModel(
                        num_attention_heads,
                        out_channels // num_attention_heads,
                        in_channels=out_channels,
                        num_layers=1,
                        cross_attention_dim=cross_attention_dim,
                        norm_num_groups=resnet_groups,
                    )
                )
        self.attentions = nn.ModuleList(attentions)
        self.resnets = nn.ModuleList(resnets)

        if add_upsample:
            self.upsamplers = nn.ModuleList(
                [Upsample2D(out_channels, use_conv=True, out_channels=out_channels)]
            )
        else:
            self.upsamplers = None

        self.gradient_checkpointing = False
        self.resolution_idx = resolution_idx

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        res_hidden_states_tuple: Tuple[torch.FloatTensor, ...],
        temb: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        upsample_size: Optional[int] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
    ) -> torch.FloatTensor:
        """
        Forward pass for the CrossAttnUpBlock2D class.

        Args:
            self (CrossAttnUpBlock2D): An instance of the CrossAttnUpBlock2D class.
            hidden_states (torch.FloatTensor): The input hidden states tensor.
            res_hidden_states_tuple (Tuple[torch.FloatTensor, ...]): A tuple of residual hidden states tensors.
            temb (Optional[torch.FloatTensor], optional): The token embeddings tensor. Defaults to None.
            encoder_hidden_states (Optional[torch.FloatTensor], optional): The encoder hidden states tensor. Defaults to None.
            cross_attention_kwargs (Optional[Dict[str, Any]], optional): Additional keyword arguments for cross attention. Defaults to None.
            upsample_size (Optional[int], optional): The upsample size. Defaults to None.
            attention_mask (Optional[torch.FloatTensor], optional): The attention mask tensor. Defaults to None.
            encoder_attention_mask (Optional[torch.FloatTensor], optional): The encoder attention mask tensor. Defaults to None.

        Returns:
            torch.FloatTensor: The output tensor after passing through the block.
        """
        lora_scale = (
            cross_attention_kwargs.get("scale", 1.0)
            if cross_attention_kwargs is not None
            else 1.0
        )
        is_freeu_enabled = (
            getattr(self, "s1", None)
            and getattr(self, "s2", None)
            and getattr(self, "b1", None)
            and getattr(self, "b2", None)
        )

        for resnet, attn in zip(self.resnets, self.attentions):
            # pop res hidden states
            res_hidden_states = res_hidden_states_tuple[-1]
            res_hidden_states_tuple = res_hidden_states_tuple[:-1]

            # FreeU: Only operate on the first two stages
            if is_freeu_enabled:
                hidden_states, res_hidden_states = apply_freeu(
                    self.resolution_idx,
                    hidden_states,
                    res_hidden_states,
                    s1=self.s1,
                    s2=self.s2,
                    b1=self.b1,
                    b2=self.b2,
                )

            hidden_states = torch.cat([hidden_states, res_hidden_states], dim=1)

            if self.training and self.gradient_checkpointing:

                def create_custom_forward(module, return_dict=None):
                    def custom_forward(*inputs):
                        if return_dict is not None:
                            return module(*inputs, return_dict=return_dict)

                        return module(*inputs)

                    return custom_forward

                ckpt_kwargs: Dict[str, Any] = (
                    {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                )
                hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(resnet),
                    hidden_states,
                    temb,
                    **ckpt_kwargs,
                )
                hidden_states, _ref_feature = attn(
                    hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    cross_attention_kwargs=cross_attention_kwargs,
                    attention_mask=attention_mask,
                    encoder_attention_mask=encoder_attention_mask,
                    return_dict=False,
                )
            else:
                hidden_states = resnet(hidden_states, temb, scale=lora_scale)
                hidden_states, _ref_feature = attn(
                    hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    cross_attention_kwargs=cross_attention_kwargs,
                    attention_mask=attention_mask,
                    encoder_attention_mask=encoder_attention_mask,
                    return_dict=False,
                )

        if self.upsamplers is not None:
            for upsampler in self.upsamplers:
                hidden_states = upsampler(
                    hidden_states, upsample_size, scale=lora_scale
                )

        return hidden_states


class UpBlock2D(nn.Module):
    """
    UpBlock2D is a class that represents a 2D upsampling block in a neural network.
    
    This block is used for upsampling the input tensor by a factor of 2 in both dimensions.
    It takes the previous output channel, input channels, and output channels as input
    and applies a series of convolutional layers, batch normalization, and activation
    functions to produce the upsampled tensor.

    Args:
        in_channels (int): The number of input channels in the tensor.
        prev_output_channel (int): The number of channels in the previous output tensor.
        out_channels (int): The number of output channels in the tensor.
        temb_channels (int): The number of channels in the time embedding tensor.
        resolution_idx (Optional[int], optional): The index of the resolution in the sequence of resolutions. Defaults to None.
        dropout (float, optional): The dropout rate to be applied to the convolutional layers. Defaults to 0.0.
        num_layers (int, optional): The number of convolutional layers in the block. Defaults to 1.
        resnet_eps (float, optional): The epsilon value used in the batch normalization layer. Defaults to 1e-6.
        resnet_time_scale_shift (str, optional): The type of activation function to be applied after the convolutional layers. Defaults to "default".
        resnet_act_fn (str, optional): The activation function to be applied after the batch normalization layer. Defaults to "swish".
        resnet_groups (int, optional): The number of groups in the group normalization layer. Defaults to 32.
        resnet_pre_norm (bool, optional): A flag indicating whether to apply layer normalization before the activation function. Defaults to True.
        output_scale_factor (float, optional): The scale factor to be applied to the output tensor. Defaults to 1.0.
        add_upsample (bool, optional): A flag indicating whether to add an upsampling layer to the block. Defaults to True.

    Attributes:
        layers (nn.ModuleList): A list of nn.Module objects representing the convolutional layers in the block.
        upsample (nn.Module): The upsampling layer in the block, if add_upsample is True.

    """

    def __init__(
        self,
        in_channels: int,
        prev_output_channel: int,
        out_channels: int,
        temb_channels: int,
        resolution_idx: Optional[int] = None,
        dropout: float = 0.0,
        num_layers: int = 1,
        resnet_eps: float = 1e-6,
        resnet_time_scale_shift: str = "default",
        resnet_act_fn: str = "swish",
        resnet_groups: int = 32,
        resnet_pre_norm: bool = True,
        output_scale_factor: float = 1.0,
        add_upsample: bool = True,
    ):
        super().__init__()
        resnets = []

        for i in range(num_layers):
            res_skip_channels = in_channels if (i == num_layers - 1) else out_channels
            resnet_in_channels = prev_output_channel if i == 0 else out_channels

            resnets.append(
                ResnetBlock2D(
                    in_channels=resnet_in_channels + res_skip_channels,
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=dropout,
                    time_embedding_norm=resnet_time_scale_shift,
                    non_linearity=resnet_act_fn,
                    output_scale_factor=output_scale_factor,
                    pre_norm=resnet_pre_norm,
                )
            )

        self.resnets = nn.ModuleList(resnets)

        if add_upsample:
            self.upsamplers = nn.ModuleList(
                [Upsample2D(out_channels, use_conv=True, out_channels=out_channels)]
            )
        else:
            self.upsamplers = None

        self.gradient_checkpointing = False
        self.resolution_idx = resolution_idx

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        res_hidden_states_tuple: Tuple[torch.FloatTensor, ...],
        temb: Optional[torch.FloatTensor] = None,
        upsample_size: Optional[int] = None,
        scale: float = 1.0,
    ) -> torch.FloatTensor:

        """
        Forward pass for the UpBlock2D class.

        Args:
            self (UpBlock2D): An instance of the UpBlock2D class.
            hidden_states (torch.FloatTensor): The input tensor to the block.
            res_hidden_states_tuple (Tuple[torch.FloatTensor, ...]): A tuple of residual hidden states.
            temb (Optional[torch.FloatTensor], optional): The token embeddings. Defaults to None.
            upsample_size (Optional[int], optional): The size to upsample the input tensor to. Defaults to None.
            scale (float, optional): The scale factor to apply to the input tensor. Defaults to 1.0.

        Returns:
            torch.FloatTensor: The output tensor after passing through the block.
        """
        is_freeu_enabled = (
            getattr(self, "s1", None)
            and getattr(self, "s2", None)
            and getattr(self, "b1", None)
            and getattr(self, "b2", None)
        )

        for resnet in self.resnets:
            # pop res hidden states
            res_hidden_states = res_hidden_states_tuple[-1]
            res_hidden_states_tuple = res_hidden_states_tuple[:-1]

            # FreeU: Only operate on the first two stages
            if is_freeu_enabled:
                hidden_states, res_hidden_states = apply_freeu(
                    self.resolution_idx,
                    hidden_states,
                    res_hidden_states,
                    s1=self.s1,
                    s2=self.s2,
                    b1=self.b1,
                    b2=self.b2,
                )

            hidden_states = torch.cat([hidden_states, res_hidden_states], dim=1)

            if self.training and self.gradient_checkpointing:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs)

                    return custom_forward

                if is_torch_version(">=", "1.11.0"):
                    hidden_states = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(resnet),
                        hidden_states,
                        temb,
                        use_reentrant=False,
                    )
                else:
                    hidden_states = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(resnet), hidden_states, temb
                    )
            else:
                hidden_states = resnet(hidden_states, temb, scale=scale)

        if self.upsamplers is not None:
            for upsampler in self.upsamplers:
                hidden_states = upsampler(hidden_states, upsample_size, scale=scale)

        return hidden_states
