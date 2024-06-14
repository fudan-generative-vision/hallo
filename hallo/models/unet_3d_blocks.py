# pylint: disable=R0801
# src/models/unet_3d_blocks.py

"""
This module defines various 3D UNet blocks used in the video model.

The blocks include:
- UNetMidBlock3DCrossAttn: The middle block of the UNet with cross attention.
- CrossAttnDownBlock3D: The downsampling block with cross attention.
- DownBlock3D: The standard downsampling block without cross attention.
- CrossAttnUpBlock3D: The upsampling block with cross attention.
- UpBlock3D: The standard upsampling block without cross attention.

These blocks are used to construct the 3D UNet architecture for video-related tasks.
"""

import torch
from einops import rearrange
from torch import nn

from .motion_module import get_motion_module
from .resnet import Downsample3D, ResnetBlock3D, Upsample3D
from .transformer_3d import Transformer3DModel


def get_down_block(
    down_block_type,
    num_layers,
    in_channels,
    out_channels,
    temb_channels,
    add_downsample,
    resnet_eps,
    resnet_act_fn,
    attn_num_head_channels,
    resnet_groups=None,
    cross_attention_dim=None,
    audio_attention_dim=None,
    downsample_padding=None,
    dual_cross_attention=False,
    use_linear_projection=False,
    only_cross_attention=False,
    upcast_attention=False,
    resnet_time_scale_shift="default",
    unet_use_cross_frame_attention=None,
    unet_use_temporal_attention=None,
    use_inflated_groupnorm=None,
    use_motion_module=None,
    motion_module_type=None,
    motion_module_kwargs=None,
    use_audio_module=None,
    depth=0,
    stack_enable_blocks_name=None,
    stack_enable_blocks_depth=None,
):
    """
    Factory function to instantiate a down-block module for the 3D UNet architecture.
    
    Down blocks are used in the downsampling part of the U-Net to reduce the spatial dimensions
    of the feature maps while increasing the depth. This function can create blocks with or without
    cross attention based on the specified parameters.

    Parameters:
    - down_block_type (str): The type of down block to instantiate.
    - num_layers (int): The number of layers in the block.
    - in_channels (int): The number of input channels.
    - out_channels (int): The number of output channels.
    - temb_channels (int): The number of token embedding channels.
    - add_downsample (bool): Flag to add a downsampling layer.
    - resnet_eps (float): Epsilon for residual block stability.
    - resnet_act_fn (callable): Activation function for the residual block.
    - ... (remaining parameters): Additional parameters for configuring the block.

    Returns:
    - nn.Module: An instance of a down-sampling block module.
    """
    down_block_type = (
        down_block_type[7:]
        if down_block_type.startswith("UNetRes")
        else down_block_type
    )
    if down_block_type == "DownBlock3D":
        return DownBlock3D(
            num_layers=num_layers,
            in_channels=in_channels,
            out_channels=out_channels,
            temb_channels=temb_channels,
            add_downsample=add_downsample,
            resnet_eps=resnet_eps,
            resnet_act_fn=resnet_act_fn,
            resnet_groups=resnet_groups,
            downsample_padding=downsample_padding,
            resnet_time_scale_shift=resnet_time_scale_shift,
            use_inflated_groupnorm=use_inflated_groupnorm,
            use_motion_module=use_motion_module,
            motion_module_type=motion_module_type,
            motion_module_kwargs=motion_module_kwargs,
        )

    if down_block_type == "CrossAttnDownBlock3D":
        if cross_attention_dim is None:
            raise ValueError(
                "cross_attention_dim must be specified for CrossAttnDownBlock3D"
            )
        return CrossAttnDownBlock3D(
            num_layers=num_layers,
            in_channels=in_channels,
            out_channels=out_channels,
            temb_channels=temb_channels,
            add_downsample=add_downsample,
            resnet_eps=resnet_eps,
            resnet_act_fn=resnet_act_fn,
            resnet_groups=resnet_groups,
            downsample_padding=downsample_padding,
            cross_attention_dim=cross_attention_dim,
            audio_attention_dim=audio_attention_dim,
            attn_num_head_channels=attn_num_head_channels,
            dual_cross_attention=dual_cross_attention,
            use_linear_projection=use_linear_projection,
            only_cross_attention=only_cross_attention,
            upcast_attention=upcast_attention,
            resnet_time_scale_shift=resnet_time_scale_shift,
            unet_use_cross_frame_attention=unet_use_cross_frame_attention,
            unet_use_temporal_attention=unet_use_temporal_attention,
            use_inflated_groupnorm=use_inflated_groupnorm,
            use_motion_module=use_motion_module,
            motion_module_type=motion_module_type,
            motion_module_kwargs=motion_module_kwargs,
            use_audio_module=use_audio_module,
            depth=depth,
            stack_enable_blocks_name=stack_enable_blocks_name,
            stack_enable_blocks_depth=stack_enable_blocks_depth,
        )
    raise ValueError(f"{down_block_type} does not exist.")


def get_up_block(
    up_block_type,
    num_layers,
    in_channels,
    out_channels,
    prev_output_channel,
    temb_channels,
    add_upsample,
    resnet_eps,
    resnet_act_fn,
    attn_num_head_channels,
    resnet_groups=None,
    cross_attention_dim=None,
    audio_attention_dim=None,
    dual_cross_attention=False,
    use_linear_projection=False,
    only_cross_attention=False,
    upcast_attention=False,
    resnet_time_scale_shift="default",
    unet_use_cross_frame_attention=None,
    unet_use_temporal_attention=None,
    use_inflated_groupnorm=None,
    use_motion_module=None,
    motion_module_type=None,
    motion_module_kwargs=None,
    use_audio_module=None,
    depth=0,
    stack_enable_blocks_name=None,
    stack_enable_blocks_depth=None,
):
    """
    Factory function to instantiate an up-block module for the 3D UNet architecture.

    Up blocks are used in the upsampling part of the U-Net to increase the spatial dimensions
    of the feature maps while decreasing the depth. This function can create blocks with or without
    cross attention based on the specified parameters.

    Parameters:
    - up_block_type (str): The type of up block to instantiate.
    - num_layers (int): The number of layers in the block.
    - in_channels (int): The number of input channels.
    - out_channels (int): The number of output channels.
    - prev_output_channel (int): The number of channels from the previous layer's output.
    - temb_channels (int): The number of token embedding channels.
    - add_upsample (bool): Flag to add an upsampling layer.
    - resnet_eps (float): Epsilon for residual block stability.
    - resnet_act_fn (callable): Activation function for the residual block.
    - ... (remaining parameters): Additional parameters for configuring the block.

    Returns:
    - nn.Module: An instance of an up-sampling block module.
    """
    up_block_type = (
        up_block_type[7:] if up_block_type.startswith("UNetRes") else up_block_type
    )
    if up_block_type == "UpBlock3D":
        return UpBlock3D(
            num_layers=num_layers,
            in_channels=in_channels,
            out_channels=out_channels,
            prev_output_channel=prev_output_channel,
            temb_channels=temb_channels,
            add_upsample=add_upsample,
            resnet_eps=resnet_eps,
            resnet_act_fn=resnet_act_fn,
            resnet_groups=resnet_groups,
            resnet_time_scale_shift=resnet_time_scale_shift,
            use_inflated_groupnorm=use_inflated_groupnorm,
            use_motion_module=use_motion_module,
            motion_module_type=motion_module_type,
            motion_module_kwargs=motion_module_kwargs,
        )

    if up_block_type == "CrossAttnUpBlock3D":
        if cross_attention_dim is None:
            raise ValueError(
                "cross_attention_dim must be specified for CrossAttnUpBlock3D"
            )
        return CrossAttnUpBlock3D(
            num_layers=num_layers,
            in_channels=in_channels,
            out_channels=out_channels,
            prev_output_channel=prev_output_channel,
            temb_channels=temb_channels,
            add_upsample=add_upsample,
            resnet_eps=resnet_eps,
            resnet_act_fn=resnet_act_fn,
            resnet_groups=resnet_groups,
            cross_attention_dim=cross_attention_dim,
            audio_attention_dim=audio_attention_dim,
            attn_num_head_channels=attn_num_head_channels,
            dual_cross_attention=dual_cross_attention,
            use_linear_projection=use_linear_projection,
            only_cross_attention=only_cross_attention,
            upcast_attention=upcast_attention,
            resnet_time_scale_shift=resnet_time_scale_shift,
            unet_use_cross_frame_attention=unet_use_cross_frame_attention,
            unet_use_temporal_attention=unet_use_temporal_attention,
            use_inflated_groupnorm=use_inflated_groupnorm,
            use_motion_module=use_motion_module,
            motion_module_type=motion_module_type,
            motion_module_kwargs=motion_module_kwargs,
            use_audio_module=use_audio_module,
            depth=depth,
            stack_enable_blocks_name=stack_enable_blocks_name,
            stack_enable_blocks_depth=stack_enable_blocks_depth,
        )
    raise ValueError(f"{up_block_type} does not exist.")


class UNetMidBlock3DCrossAttn(nn.Module):
    """
    A 3D UNet middle block with cross attention mechanism. This block is part of the U-Net architecture
    and is used for feature extraction in the middle of the downsampling path.

    Parameters:
    - in_channels (int): Number of input channels.
    - temb_channels (int): Number of token embedding channels.
    - dropout (float): Dropout rate.
    - num_layers (int): Number of layers in the block.
    - resnet_eps (float): Epsilon for residual block.
    - resnet_time_scale_shift (str): Time scale shift for time embedding normalization.
    - resnet_act_fn (str): Activation function for the residual block.
    - resnet_groups (int): Number of groups for the convolutions in the residual block.
    - resnet_pre_norm (bool): Whether to use pre-normalization in the residual block.
    - attn_num_head_channels (int): Number of attention heads.
    - cross_attention_dim (int): Dimensionality of the cross attention layers.
    - audio_attention_dim (int): Dimensionality of the audio attention layers.
    - dual_cross_attention (bool): Whether to use dual cross attention.
    - use_linear_projection (bool): Whether to use linear projection in attention.
    - upcast_attention (bool): Whether to upcast attention to the original input dimension.
    - unet_use_cross_frame_attention (bool): Whether to use cross frame attention in U-Net.
    - unet_use_temporal_attention (bool): Whether to use temporal attention in U-Net.
    - use_inflated_groupnorm (bool): Whether to use inflated group normalization.
    - use_motion_module (bool): Whether to use motion module.
    - motion_module_type (str): Type of motion module.
    - motion_module_kwargs (dict): Keyword arguments for the motion module.
    - use_audio_module (bool): Whether to use audio module.
    - depth (int): Depth of the block in the network.
    - stack_enable_blocks_name (str): Name of the stack enable blocks.
    - stack_enable_blocks_depth (int): Depth of the stack enable blocks.

    Forward method:
    The forward method applies the residual blocks, cross attention, and optional motion and audio modules
    to the input hidden states. It returns the transformed hidden states.
    """
    def __init__(
        self,
        in_channels: int,
        temb_channels: int,
        dropout: float = 0.0,
        num_layers: int = 1,
        resnet_eps: float = 1e-6,
        resnet_time_scale_shift: str = "default",
        resnet_act_fn: str = "swish",
        resnet_groups: int = 32,
        resnet_pre_norm: bool = True,
        attn_num_head_channels=1,
        output_scale_factor=1.0,
        cross_attention_dim=1280,
        audio_attention_dim=1024,
        dual_cross_attention=False,
        use_linear_projection=False,
        upcast_attention=False,
        unet_use_cross_frame_attention=None,
        unet_use_temporal_attention=None,
        use_inflated_groupnorm=None,
        use_motion_module=None,
        motion_module_type=None,
        motion_module_kwargs=None,
        use_audio_module=None,
        depth=0,
        stack_enable_blocks_name=None,
        stack_enable_blocks_depth=None,
    ):
        super().__init__()

        self.has_cross_attention = True
        self.attn_num_head_channels = attn_num_head_channels
        resnet_groups = (
            resnet_groups if resnet_groups is not None else min(in_channels // 4, 32)
        )

        # there is always at least one resnet
        resnets = [
            ResnetBlock3D(
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
                use_inflated_groupnorm=use_inflated_groupnorm,
            )
        ]
        attentions = []
        motion_modules = []
        audio_modules = []

        for _ in range(num_layers):
            if dual_cross_attention:
                raise NotImplementedError
            attentions.append(
                Transformer3DModel(
                    attn_num_head_channels,
                    in_channels // attn_num_head_channels,
                    in_channels=in_channels,
                    num_layers=1,
                    cross_attention_dim=cross_attention_dim,
                    norm_num_groups=resnet_groups,
                    use_linear_projection=use_linear_projection,
                    upcast_attention=upcast_attention,
                    unet_use_cross_frame_attention=unet_use_cross_frame_attention,
                    unet_use_temporal_attention=unet_use_temporal_attention,
                )
            )
            audio_modules.append(
                Transformer3DModel(
                    attn_num_head_channels,
                    in_channels // attn_num_head_channels,
                    in_channels=in_channels,
                    num_layers=1,
                    cross_attention_dim=audio_attention_dim,
                    norm_num_groups=resnet_groups,
                    use_linear_projection=use_linear_projection,
                    upcast_attention=upcast_attention,
                    use_audio_module=use_audio_module,
                    depth=depth,
                    unet_block_name="mid",
                    stack_enable_blocks_name=stack_enable_blocks_name,
                    stack_enable_blocks_depth=stack_enable_blocks_depth,
                )
                if use_audio_module
                else None
            )

            motion_modules.append(
                get_motion_module(
                    in_channels=in_channels,
                    motion_module_type=motion_module_type,
                    motion_module_kwargs=motion_module_kwargs,
                )
                if use_motion_module
                else None
            )
            resnets.append(
                ResnetBlock3D(
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
                    use_inflated_groupnorm=use_inflated_groupnorm,
                )
            )

        self.attentions = nn.ModuleList(attentions)
        self.resnets = nn.ModuleList(resnets)
        self.audio_modules = nn.ModuleList(audio_modules)
        self.motion_modules = nn.ModuleList(motion_modules)

    def forward(
        self,
        hidden_states,
        temb=None,
        encoder_hidden_states=None,
        attention_mask=None,
        full_mask=None,
        face_mask=None,
        lip_mask=None,
        audio_embedding=None,
        motion_scale=None,
    ):
        """
        Forward pass for the UNetMidBlock3DCrossAttn class.

        Args:
            self (UNetMidBlock3DCrossAttn): An instance of the UNetMidBlock3DCrossAttn class.
            hidden_states (Tensor): The input hidden states tensor.
            temb (Tensor, optional): The input temporal embedding tensor. Defaults to None.
            encoder_hidden_states (Tensor, optional): The encoder hidden states tensor. Defaults to None.
            attention_mask (Tensor, optional): The attention mask tensor. Defaults to None.
            full_mask (Tensor, optional): The full mask tensor. Defaults to None.
            face_mask (Tensor, optional): The face mask tensor. Defaults to None.
            lip_mask (Tensor, optional): The lip mask tensor. Defaults to None.
            audio_embedding (Tensor, optional): The audio embedding tensor. Defaults to None.

        Returns:
            Tensor: The output tensor after passing through the UNetMidBlock3DCrossAttn layers.
        """
        hidden_states = self.resnets[0](hidden_states, temb)
        for attn, resnet, audio_module, motion_module in zip(
            self.attentions, self.resnets[1:], self.audio_modules, self.motion_modules
        ):
            hidden_states, motion_frame = attn(
                hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                return_dict=False,
            )  # .sample
            if len(motion_frame[0]) > 0:
                # if motion_frame[0][0].numel() > 0:
                motion_frames = motion_frame[0][0]
                motion_frames = rearrange(
                    motion_frames,
                    "b f (d1 d2) c -> b c f d1 d2",
                    d1=hidden_states.size(-1),
                )

            else:
                motion_frames = torch.zeros(
                    hidden_states.shape[0],
                    hidden_states.shape[1],
                    4,
                    hidden_states.shape[3],
                    hidden_states.shape[4],
                )

            n_motion_frames = motion_frames.size(2)
            if audio_module is not None:
                hidden_states = (
                    audio_module(
                        hidden_states,
                        encoder_hidden_states=audio_embedding,
                        attention_mask=attention_mask,
                        full_mask=full_mask,
                        face_mask=face_mask,
                        lip_mask=lip_mask,
                        motion_scale=motion_scale,
                        return_dict=False,
                    )
                )[0]  # .sample
            if motion_module is not None:
                motion_frames = motion_frames.to(
                    device=hidden_states.device, dtype=hidden_states.dtype
                )

                _hidden_states = (
                    torch.cat([motion_frames, hidden_states], dim=2)
                    if n_motion_frames > 0
                    else hidden_states
                )
                hidden_states = motion_module(
                    _hidden_states, encoder_hidden_states=encoder_hidden_states
                )
                hidden_states = hidden_states[:, :, n_motion_frames:]

            hidden_states = resnet(hidden_states, temb)

        return hidden_states


class CrossAttnDownBlock3D(nn.Module):
    """
    A 3D downsampling block with cross attention for the U-Net architecture.

    Parameters:
    - (same as above, refer to the constructor for details)

    Forward method:
    The forward method downsamples the input hidden states using residual blocks and cross attention.
    It also applies optional motion and audio modules. The method supports gradient checkpointing
    to save memory during training.
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
        attn_num_head_channels=1,
        cross_attention_dim=1280,
        audio_attention_dim=1024,
        output_scale_factor=1.0,
        downsample_padding=1,
        add_downsample=True,
        dual_cross_attention=False,
        use_linear_projection=False,
        only_cross_attention=False,
        upcast_attention=False,
        unet_use_cross_frame_attention=None,
        unet_use_temporal_attention=None,
        use_inflated_groupnorm=None,
        use_motion_module=None,
        motion_module_type=None,
        motion_module_kwargs=None,
        use_audio_module=None,
        depth=0,
        stack_enable_blocks_name=None,
        stack_enable_blocks_depth=None,
    ):
        super().__init__()
        resnets = []
        attentions = []
        audio_modules = []
        motion_modules = []

        self.has_cross_attention = True
        self.attn_num_head_channels = attn_num_head_channels

        for i in range(num_layers):
            in_channels = in_channels if i == 0 else out_channels
            resnets.append(
                ResnetBlock3D(
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
                    use_inflated_groupnorm=use_inflated_groupnorm,
                )
            )
            if dual_cross_attention:
                raise NotImplementedError
            attentions.append(
                Transformer3DModel(
                    attn_num_head_channels,
                    out_channels // attn_num_head_channels,
                    in_channels=out_channels,
                    num_layers=1,
                    cross_attention_dim=cross_attention_dim,
                    norm_num_groups=resnet_groups,
                    use_linear_projection=use_linear_projection,
                    only_cross_attention=only_cross_attention,
                    upcast_attention=upcast_attention,
                    unet_use_cross_frame_attention=unet_use_cross_frame_attention,
                    unet_use_temporal_attention=unet_use_temporal_attention,
                )
            )
            # TODO:检查维度
            audio_modules.append(
                Transformer3DModel(
                    attn_num_head_channels,
                    in_channels // attn_num_head_channels,
                    in_channels=out_channels,
                    num_layers=1,
                    cross_attention_dim=audio_attention_dim,
                    norm_num_groups=resnet_groups,
                    use_linear_projection=use_linear_projection,
                    only_cross_attention=only_cross_attention,
                    upcast_attention=upcast_attention,
                    use_audio_module=use_audio_module,
                    depth=depth,
                    unet_block_name="down",
                    stack_enable_blocks_name=stack_enable_blocks_name,
                    stack_enable_blocks_depth=stack_enable_blocks_depth,
                )
                if use_audio_module
                else None
            )
            motion_modules.append(
                get_motion_module(
                    in_channels=out_channels,
                    motion_module_type=motion_module_type,
                    motion_module_kwargs=motion_module_kwargs,
                )
                if use_motion_module
                else None
            )

        self.attentions = nn.ModuleList(attentions)
        self.resnets = nn.ModuleList(resnets)
        self.audio_modules = nn.ModuleList(audio_modules)
        self.motion_modules = nn.ModuleList(motion_modules)

        if add_downsample:
            self.downsamplers = nn.ModuleList(
                [
                    Downsample3D(
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
        hidden_states,
        temb=None,
        encoder_hidden_states=None,
        attention_mask=None,
        full_mask=None,
        face_mask=None,
        lip_mask=None,
        audio_embedding=None,
        motion_scale=None,
    ):
        """
        Defines the forward pass for the CrossAttnDownBlock3D class.
        
        Parameters:
        -     hidden_states : torch.Tensor
            The input tensor to the block.
        temb : torch.Tensor, optional
            The token embeddings from the previous block.
        encoder_hidden_states : torch.Tensor, optional
            The hidden states from the encoder.
        attention_mask : torch.Tensor, optional
            The attention mask for the cross-attention mechanism.
        full_mask : torch.Tensor, optional
            The full mask for the cross-attention mechanism.
        face_mask : torch.Tensor, optional
            The face mask for the cross-attention mechanism.
        lip_mask : torch.Tensor, optional
            The lip mask for the cross-attention mechanism.
        audio_embedding : torch.Tensor, optional
            The audio embedding for the cross-attention mechanism.

        Returns:
        --     torch.Tensor
            The output tensor from the block.
        """
        output_states = ()

        for _, (resnet, attn, audio_module, motion_module) in enumerate(
            zip(self.resnets, self.attentions, self.audio_modules, self.motion_modules)
        ):
            # self.gradient_checkpointing = False
            if self.training and self.gradient_checkpointing:

                def create_custom_forward(module, return_dict=None):
                    def custom_forward(*inputs):
                        if return_dict is not None:
                            return module(*inputs, return_dict=return_dict)

                        return module(*inputs)

                    return custom_forward

                hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(resnet), hidden_states, temb
                )

                motion_frames = []
                hidden_states, motion_frame = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(attn, return_dict=False),
                    hidden_states,
                    encoder_hidden_states,
                )
                if len(motion_frame[0]) > 0:
                    motion_frames = motion_frame[0][0]
                    # motion_frames = torch.cat(motion_frames, dim=0)
                    motion_frames = rearrange(
                        motion_frames,
                        "b f (d1 d2) c -> b c f d1 d2",
                        d1=hidden_states.size(-1),
                    )

                else:
                    motion_frames = torch.zeros(
                        hidden_states.shape[0],
                        hidden_states.shape[1],
                        4,
                        hidden_states.shape[3],
                        hidden_states.shape[4],
                    )

                n_motion_frames = motion_frames.size(2)

                if audio_module is not None:
                    # audio_embedding = audio_embedding
                    hidden_states = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(audio_module, return_dict=False),
                        hidden_states,
                        audio_embedding,
                        attention_mask,
                        full_mask,
                        face_mask,
                        lip_mask,
                        motion_scale,
                    )[0]

                # add motion module
                if motion_module is not None:
                    motion_frames = motion_frames.to(
                        device=hidden_states.device, dtype=hidden_states.dtype
                    )
                    _hidden_states = torch.cat(
                        [motion_frames, hidden_states], dim=2
                    )  # if n_motion_frames > 0 else hidden_states
                    hidden_states = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(motion_module),
                        _hidden_states,
                        encoder_hidden_states,
                    )
                    hidden_states = hidden_states[:, :, n_motion_frames:]

            else:
                hidden_states = resnet(hidden_states, temb)
                hidden_states = attn(
                    hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                ).sample
                if audio_module is not None:
                    hidden_states = audio_module(
                        hidden_states,
                        audio_embedding,
                        attention_mask=attention_mask,
                        full_mask=full_mask,
                        face_mask=face_mask,
                        lip_mask=lip_mask,
                        return_dict=False,
                    )[0]
                # add motion module
                if motion_module is not None:
                    hidden_states = motion_module(
                        hidden_states, encoder_hidden_states=encoder_hidden_states
                    )

            output_states += (hidden_states,)

        if self.downsamplers is not None:
            for downsampler in self.downsamplers:
                hidden_states = downsampler(hidden_states)

            output_states += (hidden_states,)

        return hidden_states, output_states


class DownBlock3D(nn.Module):
    """
    A 3D downsampling block for the U-Net architecture. This block performs downsampling operations
    using residual blocks and an optional motion module.

    Parameters:
    - in_channels (int): Number of input channels.
    - out_channels (int): Number of output channels.
    - temb_channels (int): Number of token embedding channels.
    - dropout (float): Dropout rate for the block.
    - num_layers (int): Number of layers in the block.
    - resnet_eps (float): Epsilon for residual block stability.
    - resnet_time_scale_shift (str): Time scale shift for the residual block's time embedding.
    - resnet_act_fn (str): Activation function used in the residual block.
    - resnet_groups (int): Number of groups for the convolutions in the residual block.
    - resnet_pre_norm (bool): Whether to use pre-normalization in the residual block.
    - output_scale_factor (float): Scaling factor for the block's output.
    - add_downsample (bool): Whether to add a downsampling layer.
    - downsample_padding (int): Padding for the downsampling layer.
    - use_inflated_groupnorm (bool): Whether to use inflated group normalization.
    - use_motion_module (bool): Whether to include a motion module.
    - motion_module_type (str): Type of motion module to use.
    - motion_module_kwargs (dict): Keyword arguments for the motion module.

    Forward method:
    The forward method processes the input hidden states through the residual blocks and optional
    motion modules, followed by an optional downsampling step. It supports gradient checkpointing
    during training to reduce memory usage.
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
        output_scale_factor=1.0,
        add_downsample=True,
        downsample_padding=1,
        use_inflated_groupnorm=None,
        use_motion_module=None,
        motion_module_type=None,
        motion_module_kwargs=None,
    ):
        super().__init__()
        resnets = []
        motion_modules = []

        # use_motion_module = False
        for i in range(num_layers):
            in_channels = in_channels if i == 0 else out_channels
            resnets.append(
                ResnetBlock3D(
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
                    use_inflated_groupnorm=use_inflated_groupnorm,
                )
            )
            motion_modules.append(
                get_motion_module(
                    in_channels=out_channels,
                    motion_module_type=motion_module_type,
                    motion_module_kwargs=motion_module_kwargs,
                )
                if use_motion_module
                else None
            )

        self.resnets = nn.ModuleList(resnets)
        self.motion_modules = nn.ModuleList(motion_modules)

        if add_downsample:
            self.downsamplers = nn.ModuleList(
                [
                    Downsample3D(
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
        hidden_states,
        temb=None,
        encoder_hidden_states=None,
    ):
        """
        forward method for the DownBlock3D class.
        
        Args:
            hidden_states (Tensor): The input tensor to the DownBlock3D layer.
            temb (Tensor, optional): The token embeddings, if using transformer.
            encoder_hidden_states (Tensor, optional): The hidden states from the encoder.
        
        Returns:
            Tensor: The output tensor after passing through the DownBlock3D layer.
        """
        output_states = ()

        for resnet, motion_module in zip(self.resnets, self.motion_modules):
            # print(f"DownBlock3D {self.gradient_checkpointing = }")
            if self.training and self.gradient_checkpointing:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs)

                    return custom_forward

                hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(resnet), hidden_states, temb
                )

            else:
                hidden_states = resnet(hidden_states, temb)

                # add motion module
                hidden_states = (
                    motion_module(
                        hidden_states, encoder_hidden_states=encoder_hidden_states
                    )
                    if motion_module is not None
                    else hidden_states
                )

            output_states += (hidden_states,)

        if self.downsamplers is not None:
            for downsampler in self.downsamplers:
                hidden_states = downsampler(hidden_states)

            output_states += (hidden_states,)

        return hidden_states, output_states


class CrossAttnUpBlock3D(nn.Module):
    """
    Standard 3D downsampling block for the U-Net architecture. This block performs downsampling
    operations in the U-Net using residual blocks and an optional motion module.

    Parameters:
    - in_channels (int): Number of input channels.
    - out_channels (int): Number of output channels.
    - temb_channels (int): Number of channels for the temporal embedding.
    - dropout (float): Dropout rate for the block.
    - num_layers (int): Number of layers in the block.
    - resnet_eps (float): Epsilon for residual block stability.
    - resnet_time_scale_shift (str): Time scale shift for the residual block's time embedding.
    - resnet_act_fn (str): Activation function used in the residual block.
    - resnet_groups (int): Number of groups for the convolutions in the residual block.
    - resnet_pre_norm (bool): Whether to use pre-normalization in the residual block.
    - output_scale_factor (float): Scaling factor for the block's output.
    - add_downsample (bool): Whether to add a downsampling layer.
    - downsample_padding (int): Padding for the downsampling layer.
    - use_inflated_groupnorm (bool): Whether to use inflated group normalization.
    - use_motion_module (bool): Whether to include a motion module.
    - motion_module_type (str): Type of motion module to use.
    - motion_module_kwargs (dict): Keyword arguments for the motion module.

    Forward method:
    The forward method processes the input hidden states through the residual blocks and optional
    motion modules, followed by an optional downsampling step. It supports gradient checkpointing
    during training to reduce memory usage.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        prev_output_channel: int,
        temb_channels: int,
        dropout: float = 0.0,
        num_layers: int = 1,
        resnet_eps: float = 1e-6,
        resnet_time_scale_shift: str = "default",
        resnet_act_fn: str = "swish",
        resnet_groups: int = 32,
        resnet_pre_norm: bool = True,
        attn_num_head_channels=1,
        cross_attention_dim=1280,
        audio_attention_dim=1024,
        output_scale_factor=1.0,
        add_upsample=True,
        dual_cross_attention=False,
        use_linear_projection=False,
        only_cross_attention=False,
        upcast_attention=False,
        unet_use_cross_frame_attention=None,
        unet_use_temporal_attention=None,
        use_motion_module=None,
        use_inflated_groupnorm=None,
        motion_module_type=None,
        motion_module_kwargs=None,
        use_audio_module=None,
        depth=0,
        stack_enable_blocks_name=None,
        stack_enable_blocks_depth=None,
    ):
        super().__init__()
        resnets = []
        attentions = []
        audio_modules = []
        motion_modules = []

        self.has_cross_attention = True
        self.attn_num_head_channels = attn_num_head_channels

        for i in range(num_layers):
            res_skip_channels = in_channels if (i == num_layers - 1) else out_channels
            resnet_in_channels = prev_output_channel if i == 0 else out_channels

            resnets.append(
                ResnetBlock3D(
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
                    use_inflated_groupnorm=use_inflated_groupnorm,
                )
            )

            if dual_cross_attention:
                raise NotImplementedError
            attentions.append(
                Transformer3DModel(
                    attn_num_head_channels,
                    out_channels // attn_num_head_channels,
                    in_channels=out_channels,
                    num_layers=1,
                    cross_attention_dim=cross_attention_dim,
                    norm_num_groups=resnet_groups,
                    use_linear_projection=use_linear_projection,
                    only_cross_attention=only_cross_attention,
                    upcast_attention=upcast_attention,
                    unet_use_cross_frame_attention=unet_use_cross_frame_attention,
                    unet_use_temporal_attention=unet_use_temporal_attention,
                )
            )
            audio_modules.append(
                Transformer3DModel(
                    attn_num_head_channels,
                    in_channels // attn_num_head_channels,
                    in_channels=out_channels,
                    num_layers=1,
                    cross_attention_dim=audio_attention_dim,
                    norm_num_groups=resnet_groups,
                    use_linear_projection=use_linear_projection,
                    only_cross_attention=only_cross_attention,
                    upcast_attention=upcast_attention,
                    use_audio_module=use_audio_module,
                    depth=depth,
                    unet_block_name="up",
                    stack_enable_blocks_name=stack_enable_blocks_name,
                    stack_enable_blocks_depth=stack_enable_blocks_depth,
                )
                if use_audio_module
                else None
            )
            motion_modules.append(
                get_motion_module(
                    in_channels=out_channels,
                    motion_module_type=motion_module_type,
                    motion_module_kwargs=motion_module_kwargs,
                )
                if use_motion_module
                else None
            )

        self.attentions = nn.ModuleList(attentions)
        self.resnets = nn.ModuleList(resnets)
        self.audio_modules = nn.ModuleList(audio_modules)
        self.motion_modules = nn.ModuleList(motion_modules)

        if add_upsample:
            self.upsamplers = nn.ModuleList(
                [Upsample3D(out_channels, use_conv=True, out_channels=out_channels)]
            )
        else:
            self.upsamplers = None

        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states,
        res_hidden_states_tuple,
        temb=None,
        encoder_hidden_states=None,
        upsample_size=None,
        attention_mask=None,
        full_mask=None,
        face_mask=None,
        lip_mask=None,
        audio_embedding=None,
        motion_scale=None,
    ):
        """
        Forward pass for the CrossAttnUpBlock3D class.

        Args:
            self (CrossAttnUpBlock3D): An instance of the CrossAttnUpBlock3D class.
            hidden_states (Tensor): The input hidden states tensor.
            res_hidden_states_tuple (Tuple[Tensor]): A tuple of residual hidden states tensors.
            temb (Tensor, optional): The token embeddings tensor. Defaults to None.
            encoder_hidden_states (Tensor, optional): The encoder hidden states tensor. Defaults to None.
            upsample_size (int, optional): The upsample size. Defaults to None.
            attention_mask (Tensor, optional): The attention mask tensor. Defaults to None.
            full_mask (Tensor, optional): The full mask tensor. Defaults to None.
            face_mask (Tensor, optional): The face mask tensor. Defaults to None.
            lip_mask (Tensor, optional): The lip mask tensor. Defaults to None.
            audio_embedding (Tensor, optional): The audio embedding tensor. Defaults to None.

        Returns:
            Tensor: The output tensor after passing through the CrossAttnUpBlock3D.
        """
        for _, (resnet, attn, audio_module, motion_module) in enumerate(
            zip(self.resnets, self.attentions, self.audio_modules, self.motion_modules)
        ):
            # pop res hidden states
            res_hidden_states = res_hidden_states_tuple[-1]
            res_hidden_states_tuple = res_hidden_states_tuple[:-1]
            hidden_states = torch.cat([hidden_states, res_hidden_states], dim=1)

            if self.training and self.gradient_checkpointing:

                def create_custom_forward(module, return_dict=None):
                    def custom_forward(*inputs):
                        if return_dict is not None:
                            return module(*inputs, return_dict=return_dict)

                        return module(*inputs)

                    return custom_forward

                hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(resnet), hidden_states, temb
                )

                motion_frames = []
                hidden_states, motion_frame = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(attn, return_dict=False),
                    hidden_states,
                    encoder_hidden_states,
                )
                if len(motion_frame[0]) > 0:
                    motion_frames = motion_frame[0][0]
                    # motion_frames = torch.cat(motion_frames, dim=0)
                    motion_frames = rearrange(
                        motion_frames,
                        "b f (d1 d2) c -> b c f d1 d2",
                        d1=hidden_states.size(-1),
                    )
                else:
                    motion_frames = torch.zeros(
                        hidden_states.shape[0],
                        hidden_states.shape[1],
                        4,
                        hidden_states.shape[3],
                        hidden_states.shape[4],
                    )

                n_motion_frames = motion_frames.size(2)

                if audio_module is not None:
                    # audio_embedding = audio_embedding
                    hidden_states = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(audio_module, return_dict=False),
                        hidden_states,
                        audio_embedding,
                        attention_mask,
                        full_mask,
                        face_mask,
                        lip_mask,
                        motion_scale,
                    )[0]

                # add motion module
                if motion_module is not None:
                    motion_frames = motion_frames.to(
                        device=hidden_states.device, dtype=hidden_states.dtype
                    )

                    _hidden_states = (
                        torch.cat([motion_frames, hidden_states], dim=2)
                        if n_motion_frames > 0
                        else hidden_states
                    )
                    hidden_states = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(motion_module),
                        _hidden_states,
                        encoder_hidden_states,
                    )
                    hidden_states = hidden_states[:, :, n_motion_frames:]
            else:
                hidden_states = resnet(hidden_states, temb)
                hidden_states = attn(
                    hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                ).sample

                if audio_module is not None:

                    hidden_states = (
                        audio_module(
                            hidden_states,
                            encoder_hidden_states=audio_embedding,
                            attention_mask=attention_mask,
                            full_mask=full_mask,
                            face_mask=face_mask,
                            lip_mask=lip_mask,
                        )
                    ).sample
                # add motion module
                hidden_states = (
                    motion_module(
                        hidden_states, encoder_hidden_states=encoder_hidden_states
                    )
                    if motion_module is not None
                    else hidden_states
                )

        if self.upsamplers is not None:
            for upsampler in self.upsamplers:
                hidden_states = upsampler(hidden_states, upsample_size)

        return hidden_states


class UpBlock3D(nn.Module):
    """
    3D upsampling block with cross attention for the U-Net architecture. This block performs
    upsampling operations and incorporates cross attention mechanisms, which allow the model to
    focus on different parts of the input when upscaling.

    Parameters:
    - in_channels (int): Number of input channels.
    - out_channels (int): Number of output channels.
    - prev_output_channel (int): Number of channels from the previous layer's output.
    - temb_channels (int): Number of channels for the temporal embedding.
    - dropout (float): Dropout rate for the block.
    - num_layers (int): Number of layers in the block.
    - resnet_eps (float): Epsilon for residual block stability.
    - resnet_time_scale_shift (str): Time scale shift for the residual block's time embedding.
    - resnet_act_fn (str): Activation function used in the residual block.
    - resnet_groups (int): Number of groups for the convolutions in the residual block.
    - resnet_pre_norm (bool): Whether to use pre-normalization in the residual block.
    - attn_num_head_channels (int): Number of attention heads for the cross attention mechanism.
    - cross_attention_dim (int): Dimensionality of the cross attention layers.
    - audio_attention_dim (int): Dimensionality of the audio attention layers.
    - output_scale_factor (float): Scaling factor for the block's output.
    - add_upsample (bool): Whether to add an upsampling layer.
    - dual_cross_attention (bool): Whether to use dual cross attention (not implemented).
    - use_linear_projection (bool): Whether to use linear projection in the cross attention.
    - only_cross_attention (bool): Whether to use only cross attention (no self-attention).
    - upcast_attention (bool): Whether to upcast attention to the original input dimension.
    - unet_use_cross_frame_attention (bool): Whether to use cross frame attention in U-Net.
    - unet_use_temporal_attention (bool): Whether to use temporal attention in U-Net.
    - use_motion_module (bool): Whether to include a motion module.
    - use_inflated_groupnorm (bool): Whether to use inflated group normalization.
    - motion_module_type (str): Type of motion module to use.
    - motion_module_kwargs (dict): Keyword arguments for the motion module.
    - use_audio_module (bool): Whether to include an audio module.
    - depth (int): Depth of the block in the network.
    - stack_enable_blocks_name (str): Name of the stack enable blocks.
    - stack_enable_blocks_depth (int): Depth of the stack enable blocks.

    Forward method:
    The forward method upsamples the input hidden states and residual hidden states, processes
    them through the residual and cross attention blocks, and optional motion and audio modules.
    It supports gradient checkpointing during training.
    """
    def __init__(
        self,
        in_channels: int,
        prev_output_channel: int,
        out_channels: int,
        temb_channels: int,
        dropout: float = 0.0,
        num_layers: int = 1,
        resnet_eps: float = 1e-6,
        resnet_time_scale_shift: str = "default",
        resnet_act_fn: str = "swish",
        resnet_groups: int = 32,
        resnet_pre_norm: bool = True,
        output_scale_factor=1.0,
        add_upsample=True,
        use_inflated_groupnorm=None,
        use_motion_module=None,
        motion_module_type=None,
        motion_module_kwargs=None,
    ):
        super().__init__()
        resnets = []
        motion_modules = []

        # use_motion_module = False
        for i in range(num_layers):
            res_skip_channels = in_channels if (i == num_layers - 1) else out_channels
            resnet_in_channels = prev_output_channel if i == 0 else out_channels

            resnets.append(
                ResnetBlock3D(
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
                    use_inflated_groupnorm=use_inflated_groupnorm,
                )
            )
            motion_modules.append(
                get_motion_module(
                    in_channels=out_channels,
                    motion_module_type=motion_module_type,
                    motion_module_kwargs=motion_module_kwargs,
                )
                if use_motion_module
                else None
            )

        self.resnets = nn.ModuleList(resnets)
        self.motion_modules = nn.ModuleList(motion_modules)

        if add_upsample:
            self.upsamplers = nn.ModuleList(
                [Upsample3D(out_channels, use_conv=True, out_channels=out_channels)]
            )
        else:
            self.upsamplers = None

        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states,
        res_hidden_states_tuple,
        temb=None,
        upsample_size=None,
        encoder_hidden_states=None,
    ):
        """
        Forward pass for the UpBlock3D class.

        Args:
            self (UpBlock3D): An instance of the UpBlock3D class.
            hidden_states (Tensor): The input hidden states tensor.
            res_hidden_states_tuple (Tuple[Tensor]): A tuple of residual hidden states tensors.
            temb (Tensor, optional): The token embeddings tensor. Defaults to None.
            upsample_size (int, optional): The upsample size. Defaults to None.
            encoder_hidden_states (Tensor, optional): The encoder hidden states tensor. Defaults to None.

        Returns:
            Tensor: The output tensor after passing through the UpBlock3D layers.
        """
        for resnet, motion_module in zip(self.resnets, self.motion_modules):
            # pop res hidden states
            res_hidden_states = res_hidden_states_tuple[-1]
            res_hidden_states_tuple = res_hidden_states_tuple[:-1]
            hidden_states = torch.cat([hidden_states, res_hidden_states], dim=1)

            # print(f"UpBlock3D {self.gradient_checkpointing = }")
            if self.training and self.gradient_checkpointing:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs)

                    return custom_forward

                hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(resnet), hidden_states, temb
                )
            else:
                hidden_states = resnet(hidden_states, temb)
                hidden_states = (
                    motion_module(
                        hidden_states, encoder_hidden_states=encoder_hidden_states
                    )
                    if motion_module is not None
                    else hidden_states
                )

        if self.upsamplers is not None:
            for upsampler in self.upsamplers:
                hidden_states = upsampler(hidden_states, upsample_size)

        return hidden_states
