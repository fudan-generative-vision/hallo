# pylint: disable=R0801
# pylint: disable=C0303

"""
This module contains various transformer blocks for different applications, such as BasicTransformerBlock,
TemporalBasicTransformerBlock, and AudioTemporalBasicTransformerBlock. These blocks are used in various models,
such as GLIGEN, UNet, and others. The transformer blocks implement self-attention, cross-attention, feed-forward
networks, and other related functions.

Functions and classes included in this module are:
- BasicTransformerBlock: A basic transformer block with self-attention, cross-attention, and feed-forward layers.
- TemporalBasicTransformerBlock: A transformer block with additional temporal attention mechanisms for video data.
- AudioTemporalBasicTransformerBlock: A transformer block with additional audio-specific mechanisms for audio data.
- zero_module: A function to zero out the parameters of a given module.

For more information on each specific class and function, please refer to the respective docstrings.
"""

from typing import Any, Dict, List, Optional

import torch
from diffusers.models.attention import (AdaLayerNorm, AdaLayerNormZero,
                                        Attention, FeedForward)
from diffusers.models.embeddings import SinusoidalPositionalEmbedding
from einops import rearrange
from torch import nn


class GatedSelfAttentionDense(nn.Module):
    """
    A gated self-attention dense layer that combines visual features and object features.

    Parameters:
        query_dim (`int`): The number of channels in the query.
        context_dim (`int`): The number of channels in the context.
        n_heads (`int`): The number of heads to use for attention.
        d_head (`int`): The number of channels in each head.
    """

    def __init__(self, query_dim: int, context_dim: int, n_heads: int, d_head: int):
        super().__init__()

        # we need a linear projection since we need cat visual feature and obj feature
        self.linear = nn.Linear(context_dim, query_dim)

        self.attn = Attention(query_dim=query_dim, heads=n_heads, dim_head=d_head)
        self.ff = FeedForward(query_dim, activation_fn="geglu")

        self.norm1 = nn.LayerNorm(query_dim)
        self.norm2 = nn.LayerNorm(query_dim)

        self.register_parameter("alpha_attn", nn.Parameter(torch.tensor(0.0)))
        self.register_parameter("alpha_dense", nn.Parameter(torch.tensor(0.0)))

        self.enabled = True

    def forward(self, x: torch.Tensor, objs: torch.Tensor) -> torch.Tensor:
        """
        Apply the Gated Self-Attention mechanism to the input tensor `x` and object tensor `objs`.

        Args:
            x (torch.Tensor): The input tensor.
            objs (torch.Tensor): The object tensor.

        Returns:
            torch.Tensor: The output tensor after applying Gated Self-Attention.
        """
        if not self.enabled:
            return x

        n_visual = x.shape[1]
        objs = self.linear(objs)

        x = x + self.alpha_attn.tanh() * self.attn(self.norm1(torch.cat([x, objs], dim=1)))[:, :n_visual, :]
        x = x + self.alpha_dense.tanh() * self.ff(self.norm2(x))

        return x

class BasicTransformerBlock(nn.Module):
    r"""
    A basic Transformer block.

    Parameters:
        dim (`int`): The number of channels in the input and output.
        num_attention_heads (`int`): The number of heads to use for multi-head attention.
        attention_head_dim (`int`): The number of channels in each head.
        dropout (`float`, *optional*, defaults to 0.0): The dropout probability to use.
        cross_attention_dim (`int`, *optional*): The size of the encoder_hidden_states vector for cross attention.
        activation_fn (`str`, *optional*, defaults to `"geglu"`): Activation function to be used in feed-forward.
        num_embeds_ada_norm (:
            obj: `int`, *optional*): The number of diffusion steps used during training. See `Transformer2DModel`.
        attention_bias (:
            obj: `bool`, *optional*, defaults to `False`): Configure if the attentions should contain a bias parameter.
        only_cross_attention (`bool`, *optional*):
            Whether to use only cross-attention layers. In this case two cross attention layers are used.
        double_self_attention (`bool`, *optional*):
            Whether to use two self-attention layers. In this case no cross attention layers are used.
        upcast_attention (`bool`, *optional*):
            Whether to upcast the attention computation to float32. This is useful for mixed precision training.
        norm_elementwise_affine (`bool`, *optional*, defaults to `True`):
            Whether to use learnable elementwise affine parameters for normalization.
        norm_type (`str`, *optional*, defaults to `"layer_norm"`):
            The normalization layer to use. Can be `"layer_norm"`, `"ada_norm"` or `"ada_norm_zero"`.
        final_dropout (`bool` *optional*, defaults to False):
            Whether to apply a final dropout after the last feed-forward layer.
        attention_type (`str`, *optional*, defaults to `"default"`):
            The type of attention to use. Can be `"default"` or `"gated"` or `"gated-text-image"`.
        positional_embeddings (`str`, *optional*, defaults to `None`):
            The type of positional embeddings to apply to.
        num_positional_embeddings (`int`, *optional*, defaults to `None`):
            The maximum number of positional embeddings to apply.
    """

    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        dropout=0.0,
        cross_attention_dim: Optional[int] = None,
        activation_fn: str = "geglu",
        num_embeds_ada_norm: Optional[int] = None,
        attention_bias: bool = False,
        only_cross_attention: bool = False,
        double_self_attention: bool = False,
        upcast_attention: bool = False,
        norm_elementwise_affine: bool = True,
        # 'layer_norm', 'ada_norm', 'ada_norm_zero', 'ada_norm_single'
        norm_type: str = "layer_norm",
        norm_eps: float = 1e-5,
        final_dropout: bool = False,
        attention_type: str = "default",
        positional_embeddings: Optional[str] = None,
        num_positional_embeddings: Optional[int] = None,
    ):
        super().__init__()
        self.only_cross_attention = only_cross_attention

        self.use_ada_layer_norm_zero = (
            num_embeds_ada_norm is not None
        ) and norm_type == "ada_norm_zero"
        self.use_ada_layer_norm = (
            num_embeds_ada_norm is not None
        ) and norm_type == "ada_norm"
        self.use_ada_layer_norm_single = norm_type == "ada_norm_single"
        self.use_layer_norm = norm_type == "layer_norm"

        if norm_type in ("ada_norm", "ada_norm_zero") and num_embeds_ada_norm is None:
            raise ValueError(
                f"`norm_type` is set to {norm_type}, but `num_embeds_ada_norm` is not defined. Please make sure to"
                f" define `num_embeds_ada_norm` if setting `norm_type` to {norm_type}."
            )

        if positional_embeddings and (num_positional_embeddings is None):
            raise ValueError(
                "If `positional_embedding` type is defined, `num_positition_embeddings` must also be defined."
            )

        if positional_embeddings == "sinusoidal":
            self.pos_embed = SinusoidalPositionalEmbedding(
                dim, max_seq_length=num_positional_embeddings
            )
        else:
            self.pos_embed = None

        # Define 3 blocks. Each block has its own normalization layer.
        # 1. Self-Attn
        if self.use_ada_layer_norm:
            self.norm1 = AdaLayerNorm(dim, num_embeds_ada_norm)
        elif self.use_ada_layer_norm_zero:
            self.norm1 = AdaLayerNormZero(dim, num_embeds_ada_norm)
        else:
            self.norm1 = nn.LayerNorm(
                dim, elementwise_affine=norm_elementwise_affine, eps=norm_eps
            )

        self.attn1 = Attention(
            query_dim=dim,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            dropout=dropout,
            bias=attention_bias,
            cross_attention_dim=cross_attention_dim if only_cross_attention else None,
            upcast_attention=upcast_attention,
        )

        # 2. Cross-Attn
        if cross_attention_dim is not None or double_self_attention:
            # We currently only use AdaLayerNormZero for self attention where there will only be one attention block.
            # I.e. the number of returned modulation chunks from AdaLayerZero would not make sense if returned during
            # the second cross attention block.
            self.norm2 = (
                AdaLayerNorm(dim, num_embeds_ada_norm)
                if self.use_ada_layer_norm
                else nn.LayerNorm(
                    dim, elementwise_affine=norm_elementwise_affine, eps=norm_eps
                )
            )
            self.attn2 = Attention(
                query_dim=dim,
                cross_attention_dim=(
                    cross_attention_dim if not double_self_attention else None
                ),
                heads=num_attention_heads,
                dim_head=attention_head_dim,
                dropout=dropout,
                bias=attention_bias,
                upcast_attention=upcast_attention,
            )  # is self-attn if encoder_hidden_states is none
        else:
            self.norm2 = None
            self.attn2 = None

        # 3. Feed-forward
        if not self.use_ada_layer_norm_single:
            self.norm3 = nn.LayerNorm(
                dim, elementwise_affine=norm_elementwise_affine, eps=norm_eps
            )

        self.ff = FeedForward(
            dim,
            dropout=dropout,
            activation_fn=activation_fn,
            final_dropout=final_dropout,
        )

        # 4. Fuser
        if attention_type in {"gated", "gated-text-image"}:  # Updated line
            self.fuser = GatedSelfAttentionDense(
                dim, cross_attention_dim, num_attention_heads, attention_head_dim
            )

        # 5. Scale-shift for PixArt-Alpha.
        if self.use_ada_layer_norm_single:
            self.scale_shift_table = nn.Parameter(
                torch.randn(6, dim) / dim**0.5)

        # let chunk size default to None
        self._chunk_size = None
        self._chunk_dim = 0

    def set_chunk_feed_forward(self, chunk_size: Optional[int], dim: int = 0):
        """
        Sets the chunk size for feed-forward processing in the transformer block.

        Args:
            chunk_size (Optional[int]): The size of the chunks to process in feed-forward layers. 
            If None, the chunk size is set to the maximum possible value.
            dim (int, optional): The dimension along which to split the input tensor into chunks. Defaults to 0.

        Returns:
            None.
        """
        self._chunk_size = chunk_size
        self._chunk_dim = dim

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        timestep: Optional[torch.LongTensor] = None,
        cross_attention_kwargs: Dict[str, Any] = None,
        class_labels: Optional[torch.LongTensor] = None,
    ) -> torch.FloatTensor:
        """
        This function defines the forward pass of the BasicTransformerBlock.

        Args:
            self (BasicTransformerBlock):
                An instance of the BasicTransformerBlock class.
            hidden_states (torch.FloatTensor):
                A tensor containing the hidden states.
            attention_mask (Optional[torch.FloatTensor], optional):
                A tensor containing the attention mask. Defaults to None.
            encoder_hidden_states (Optional[torch.FloatTensor], optional):
                A tensor containing the encoder hidden states. Defaults to None.
            encoder_attention_mask (Optional[torch.FloatTensor], optional):
                A tensor containing the encoder attention mask. Defaults to None.
            timestep (Optional[torch.LongTensor], optional):
                A tensor containing the timesteps. Defaults to None.
            cross_attention_kwargs (Dict[str, Any], optional):
                Additional cross-attention arguments. Defaults to None.
            class_labels (Optional[torch.LongTensor], optional):
                A tensor containing the class labels. Defaults to None.

        Returns:
            torch.FloatTensor:
                A tensor containing the transformed hidden states.
        """
        # Notice that normalization is always applied before the real computation in the following blocks.
        # 0. Self-Attention
        batch_size = hidden_states.shape[0]

        gate_msa = None
        scale_mlp = None
        shift_mlp = None
        gate_mlp = None
        if self.use_ada_layer_norm:
            norm_hidden_states = self.norm1(hidden_states, timestep)
        elif self.use_ada_layer_norm_zero:
            norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(
                hidden_states, timestep, class_labels, hidden_dtype=hidden_states.dtype
            )
        elif self.use_layer_norm:
            norm_hidden_states = self.norm1(hidden_states)
        elif self.use_ada_layer_norm_single:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
                self.scale_shift_table[None] +
                timestep.reshape(batch_size, 6, -1)
            ).chunk(6, dim=1)
            norm_hidden_states = self.norm1(hidden_states)
            norm_hidden_states = norm_hidden_states * \
                (1 + scale_msa) + shift_msa
            norm_hidden_states = norm_hidden_states.squeeze(1)
        else:
            raise ValueError("Incorrect norm used")

        if self.pos_embed is not None:
            norm_hidden_states = self.pos_embed(norm_hidden_states)

        # 1. Retrieve lora scale.
        lora_scale = (
            cross_attention_kwargs.get("scale", 1.0)
            if cross_attention_kwargs is not None
            else 1.0
        )

        # 2. Prepare GLIGEN inputs
        cross_attention_kwargs = (
            cross_attention_kwargs.copy() if cross_attention_kwargs is not None else {}
        )
        gligen_kwargs = cross_attention_kwargs.pop("gligen", None)

        attn_output = self.attn1(
            norm_hidden_states,
            encoder_hidden_states=(
                encoder_hidden_states if self.only_cross_attention else None
            ),
            attention_mask=attention_mask,
            **cross_attention_kwargs,
        )
        if self.use_ada_layer_norm_zero:
            attn_output = gate_msa.unsqueeze(1) * attn_output
        elif self.use_ada_layer_norm_single:
            attn_output = gate_msa * attn_output

        hidden_states = attn_output + hidden_states
        if hidden_states.ndim == 4:
            hidden_states = hidden_states.squeeze(1)

        # 2.5 GLIGEN Control
        if gligen_kwargs is not None:
            hidden_states = self.fuser(hidden_states, gligen_kwargs["objs"])

        # 3. Cross-Attention
        if self.attn2 is not None:
            if self.use_ada_layer_norm:
                norm_hidden_states = self.norm2(hidden_states, timestep)
            elif self.use_ada_layer_norm_zero or self.use_layer_norm:
                norm_hidden_states = self.norm2(hidden_states)
            elif self.use_ada_layer_norm_single:
                # For PixArt norm2 isn't applied here:
                # https://github.com/PixArt-alpha/PixArt-alpha/blob/0f55e922376d8b797edd44d25d0e7464b260dcab/diffusion/model/nets/PixArtMS.py#L70C1-L76C103
                norm_hidden_states = hidden_states
            else:
                raise ValueError("Incorrect norm")

            if self.pos_embed is not None and self.use_ada_layer_norm_single is False:
                norm_hidden_states = self.pos_embed(norm_hidden_states)

            attn_output = self.attn2(
                norm_hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
                **cross_attention_kwargs,
            )
            hidden_states = attn_output + hidden_states

        # 4. Feed-forward
        if not self.use_ada_layer_norm_single:
            norm_hidden_states = self.norm3(hidden_states)

        if self.use_ada_layer_norm_zero:
            norm_hidden_states = (
                norm_hidden_states *
                (1 + scale_mlp[:, None]) + shift_mlp[:, None]
            )

        if self.use_ada_layer_norm_single:
            norm_hidden_states = self.norm2(hidden_states)
            norm_hidden_states = norm_hidden_states * \
                (1 + scale_mlp) + shift_mlp

        ff_output = self.ff(norm_hidden_states, scale=lora_scale)

        if self.use_ada_layer_norm_zero:
            ff_output = gate_mlp.unsqueeze(1) * ff_output
        elif self.use_ada_layer_norm_single:
            ff_output = gate_mlp * ff_output

        hidden_states = ff_output + hidden_states
        if hidden_states.ndim == 4:
            hidden_states = hidden_states.squeeze(1)

        return hidden_states


class TemporalBasicTransformerBlock(nn.Module):
    """
    A PyTorch module that extends the BasicTransformerBlock to include temporal attention mechanisms.
    This class is particularly useful for video-related tasks where capturing temporal information within the sequence of frames is necessary.

    Attributes:
        dim (int): The dimension of the input and output embeddings.
        num_attention_heads (int): The number of attention heads in the multi-head self-attention mechanism.
        attention_head_dim (int): The dimension of each attention head.
        dropout (float): The dropout probability for the attention scores.
        cross_attention_dim (Optional[int]): The dimension of the cross-attention mechanism.
        activation_fn (str): The activation function used in the feed-forward layer.
        num_embeds_ada_norm (Optional[int]): The number of embeddings for adaptive normalization.
        attention_bias (bool): If True, uses bias in the attention mechanism.
        only_cross_attention (bool): If True, only uses cross-attention.
        upcast_attention (bool): If True, upcasts the attention mechanism for better performance.
        unet_use_cross_frame_attention (Optional[bool]): If True, uses cross-frame attention in the UNet model.
        unet_use_temporal_attention (Optional[bool]): If True, uses temporal attention in the UNet model.
    """
    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        dropout=0.0,
        cross_attention_dim: Optional[int] = None,
        activation_fn: str = "geglu",
        num_embeds_ada_norm: Optional[int] = None,
        attention_bias: bool = False,
        only_cross_attention: bool = False,
        upcast_attention: bool = False,
        unet_use_cross_frame_attention=None,
        unet_use_temporal_attention=None,
    ):
        """
        The TemporalBasicTransformerBlock class is a PyTorch module that extends the BasicTransformerBlock to include temporal attention mechanisms. 
        This is particularly useful for video-related tasks, where the model needs to capture the temporal information within the sequence of frames. 
        The block consists of self-attention, cross-attention, feed-forward, and temporal attention mechanisms.

            dim (int): The dimension of the input and output embeddings.
            num_attention_heads (int): The number of attention heads in the multi-head self-attention mechanism.
            attention_head_dim (int): The dimension of each attention head.
            dropout (float, optional): The dropout probability for the attention scores. Defaults to 0.0.
            cross_attention_dim (int, optional): The dimension of the cross-attention mechanism. Defaults to None.
            activation_fn (str, optional): The activation function used in the feed-forward layer. Defaults to "geglu".
            num_embeds_ada_norm (int, optional): The number of embeddings for adaptive normalization. Defaults to None.
            attention_bias (bool, optional): If True, uses bias in the attention mechanism. Defaults to False.
            only_cross_attention (bool, optional): If True, only uses cross-attention. Defaults to False.
            upcast_attention (bool, optional): If True, upcasts the attention mechanism for better performance. Defaults to False.
            unet_use_cross_frame_attention (bool, optional): If True, uses cross-frame attention in the UNet model. Defaults to None.
            unet_use_temporal_attention (bool, optional): If True, uses temporal attention in the UNet model. Defaults to None.

        Forward method:
            hidden_states (torch.FloatTensor): The input hidden states.
            encoder_hidden_states (torch.FloatTensor, optional): The encoder hidden states. Defaults to None.
            timestep (torch.LongTensor, optional): The current timestep for the transformer model. Defaults to None.
            attention_mask (torch.FloatTensor, optional): The attention mask for the self-attention mechanism. Defaults to None.
            video_length (int, optional): The length of the video sequence. Defaults to None.

        Returns:
            torch.FloatTensor: The output hidden states after passing through the TemporalBasicTransformerBlock.
        """
        super().__init__()
        self.only_cross_attention = only_cross_attention
        self.use_ada_layer_norm = num_embeds_ada_norm is not None
        self.unet_use_cross_frame_attention = unet_use_cross_frame_attention
        self.unet_use_temporal_attention = unet_use_temporal_attention

        # SC-Attn
        self.attn1 = Attention(
            query_dim=dim,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            dropout=dropout,
            bias=attention_bias,
            upcast_attention=upcast_attention,
        )
        self.norm1 = (
            AdaLayerNorm(dim, num_embeds_ada_norm)
            if self.use_ada_layer_norm
            else nn.LayerNorm(dim)
        )

        # Cross-Attn
        if cross_attention_dim is not None:
            self.attn2 = Attention(
                query_dim=dim,
                cross_attention_dim=cross_attention_dim,
                heads=num_attention_heads,
                dim_head=attention_head_dim,
                dropout=dropout,
                bias=attention_bias,
                upcast_attention=upcast_attention,
            )
        else:
            self.attn2 = None

        if cross_attention_dim is not None:
            self.norm2 = (
                AdaLayerNorm(dim, num_embeds_ada_norm)
                if self.use_ada_layer_norm
                else nn.LayerNorm(dim)
            )
        else:
            self.norm2 = None

        # Feed-forward
        self.ff = FeedForward(dim, dropout=dropout,
                              activation_fn=activation_fn)
        self.norm3 = nn.LayerNorm(dim)
        self.use_ada_layer_norm_zero = False

        # Temp-Attn
        # assert unet_use_temporal_attention is not None
        if unet_use_temporal_attention is None:
            unet_use_temporal_attention = False
        if unet_use_temporal_attention:
            self.attn_temp = Attention(
                query_dim=dim,
                heads=num_attention_heads,
                dim_head=attention_head_dim,
                dropout=dropout,
                bias=attention_bias,
                upcast_attention=upcast_attention,
            )
            nn.init.zeros_(self.attn_temp.to_out[0].weight.data)
            self.norm_temp = (
                AdaLayerNorm(dim, num_embeds_ada_norm)
                if self.use_ada_layer_norm
                else nn.LayerNorm(dim)
            )

    def forward(
        self,
        hidden_states,
        encoder_hidden_states=None,
        timestep=None,
        attention_mask=None,
        video_length=None,
    ):
        """
        Forward pass for the TemporalBasicTransformerBlock.

        Args:
            hidden_states (torch.FloatTensor): The input hidden states with shape (batch_size, seq_len, dim).
            encoder_hidden_states (torch.FloatTensor, optional): The encoder hidden states with shape (batch_size, src_seq_len, dim).
            timestep (torch.LongTensor, optional): The timestep for the transformer block.
            attention_mask (torch.FloatTensor, optional): The attention mask with shape (batch_size, seq_len, seq_len).
            video_length (int, optional): The length of the video sequence.

        Returns:
            torch.FloatTensor: The output tensor after passing through the transformer block with shape (batch_size, seq_len, dim).
        """
        norm_hidden_states = (
            self.norm1(hidden_states, timestep)
            if self.use_ada_layer_norm
            else self.norm1(hidden_states)
        )

        if self.unet_use_cross_frame_attention:
            hidden_states = (
                self.attn1(
                    norm_hidden_states,
                    attention_mask=attention_mask,
                    video_length=video_length,
                )
                + hidden_states
            )
        else:
            hidden_states = (
                self.attn1(norm_hidden_states, attention_mask=attention_mask)
                + hidden_states
            )

        if self.attn2 is not None:
            # Cross-Attention
            norm_hidden_states = (
                self.norm2(hidden_states, timestep)
                if self.use_ada_layer_norm
                else self.norm2(hidden_states)
            )
            hidden_states = (
                self.attn2(
                    norm_hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    attention_mask=attention_mask,
                )
                + hidden_states
            )

        # Feed-forward
        hidden_states = self.ff(self.norm3(hidden_states)) + hidden_states

        # Temporal-Attention
        if self.unet_use_temporal_attention:
            d = hidden_states.shape[1]
            hidden_states = rearrange(
                hidden_states, "(b f) d c -> (b d) f c", f=video_length
            )
            norm_hidden_states = (
                self.norm_temp(hidden_states, timestep)
                if self.use_ada_layer_norm
                else self.norm_temp(hidden_states)
            )
            hidden_states = self.attn_temp(norm_hidden_states) + hidden_states
            hidden_states = rearrange(
                hidden_states, "(b d) f c -> (b f) d c", d=d)

        return hidden_states


class AudioTemporalBasicTransformerBlock(nn.Module):
    """
    A PyTorch module designed to handle audio data within a transformer framework, including temporal attention mechanisms.

    Attributes:
        dim (int): The dimension of the input and output embeddings.
        num_attention_heads (int): The number of attention heads.
        attention_head_dim (int): The dimension of each attention head.
        dropout (float): The dropout probability.
        cross_attention_dim (Optional[int]): The dimension of the cross-attention mechanism.
        activation_fn (str): The activation function for the feed-forward network.
        num_embeds_ada_norm (Optional[int]): The number of embeddings for adaptive normalization.
        attention_bias (bool): If True, uses bias in the attention mechanism.
        only_cross_attention (bool): If True, only uses cross-attention.
        upcast_attention (bool): If True, upcasts the attention mechanism to float32.
        unet_use_cross_frame_attention (Optional[bool]): If True, uses cross-frame attention in UNet.
        unet_use_temporal_attention (Optional[bool]): If True, uses temporal attention in UNet.
        depth (int): The depth of the transformer block.
        unet_block_name (Optional[str]): The name of the UNet block.
        stack_enable_blocks_name (Optional[List[str]]): The list of enabled blocks in the stack.
        stack_enable_blocks_depth (Optional[List[int]]): The list of depths for the enabled blocks in the stack.
    """
    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        dropout=0.0,
        cross_attention_dim: Optional[int] = None,
        activation_fn: str = "geglu",
        num_embeds_ada_norm: Optional[int] = None,
        attention_bias: bool = False,
        only_cross_attention: bool = False,
        upcast_attention: bool = False,
        unet_use_cross_frame_attention=None,
        unet_use_temporal_attention=None,
        depth=0,
        unet_block_name=None,
        stack_enable_blocks_name: Optional[List[str]] = None,
        stack_enable_blocks_depth: Optional[List[int]] = None,
    ):  
        """
        Initializes the AudioTemporalBasicTransformerBlock module.

        Args:
           dim (int): The dimension of the input and output embeddings.
           num_attention_heads (int): The number of attention heads in the multi-head self-attention mechanism.
           attention_head_dim (int): The dimension of each attention head.
           dropout (float, optional): The dropout probability for the attention mechanism. Defaults to 0.0.
           cross_attention_dim (Optional[int], optional): The dimension of the cross-attention mechanism. Defaults to None.
           activation_fn (str, optional): The activation function to be used in the feed-forward network. Defaults to "geglu".
           num_embeds_ada_norm (Optional[int], optional): The number of embeddings for adaptive normalization. Defaults to None.
           attention_bias (bool, optional): If True, uses bias in the attention mechanism. Defaults to False.
           only_cross_attention (bool, optional): If True, only uses cross-attention. Defaults to False.
           upcast_attention (bool, optional): If True, upcasts the attention mechanism to float32. Defaults to False.
           unet_use_cross_frame_attention (Optional[bool], optional): If True, uses cross-frame attention in UNet. Defaults to None.
           unet_use_temporal_attention (Optional[bool], optional): If True, uses temporal attention in UNet. Defaults to None.
           depth (int, optional): The depth of the transformer block. Defaults to 0.
           unet_block_name (Optional[str], optional): The name of the UNet block. Defaults to None.
           stack_enable_blocks_name (Optional[List[str]], optional): The list of enabled blocks in the stack. Defaults to None.
           stack_enable_blocks_depth (Optional[List[int]], optional): The list of depths for the enabled blocks in the stack. Defaults to None.
        """
        super().__init__()
        self.only_cross_attention = only_cross_attention
        self.use_ada_layer_norm = num_embeds_ada_norm is not None
        self.unet_use_cross_frame_attention = unet_use_cross_frame_attention
        self.unet_use_temporal_attention = unet_use_temporal_attention
        self.unet_block_name = unet_block_name
        self.depth = depth

        zero_conv_full = nn.Conv2d(
            dim, dim, kernel_size=1)
        self.zero_conv_full = zero_module(zero_conv_full)

        zero_conv_face = nn.Conv2d(
            dim, dim, kernel_size=1)
        self.zero_conv_face = zero_module(zero_conv_face)

        zero_conv_lip = nn.Conv2d(
            dim, dim, kernel_size=1)
        self.zero_conv_lip = zero_module(zero_conv_lip)
        # SC-Attn
        self.attn1 = Attention(
            query_dim=dim,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            dropout=dropout,
            bias=attention_bias,
            upcast_attention=upcast_attention,
        )
        self.norm1 = (
            AdaLayerNorm(dim, num_embeds_ada_norm)
            if self.use_ada_layer_norm
            else nn.LayerNorm(dim)
        )

        # Cross-Attn
        if cross_attention_dim is not None:
            if (stack_enable_blocks_name is not None and
                stack_enable_blocks_depth is not None and
                self.unet_block_name in stack_enable_blocks_name and
                self.depth in stack_enable_blocks_depth):
                self.attn2_0 = Attention(
                    query_dim=dim,
                    cross_attention_dim=cross_attention_dim,
                    heads=num_attention_heads,
                    dim_head=attention_head_dim,
                    dropout=dropout,
                    bias=attention_bias,
                    upcast_attention=upcast_attention,
                )
                self.attn2_1 = Attention(
                    query_dim=dim,
                    cross_attention_dim=cross_attention_dim,
                    heads=num_attention_heads,
                    dim_head=attention_head_dim,
                    dropout=dropout,
                    bias=attention_bias,
                    upcast_attention=upcast_attention,
                )
                self.attn2_2 = Attention(
                    query_dim=dim,
                    cross_attention_dim=cross_attention_dim,
                    heads=num_attention_heads,
                    dim_head=attention_head_dim,
                    dropout=dropout,
                    bias=attention_bias,
                    upcast_attention=upcast_attention,
                )
                self.attn2 = None

            else:
                self.attn2 = Attention(
                    query_dim=dim,
                    cross_attention_dim=cross_attention_dim,
                    heads=num_attention_heads,
                    dim_head=attention_head_dim,
                    dropout=dropout,
                    bias=attention_bias,
                    upcast_attention=upcast_attention,
                )
                self.attn2_0=None
        else:
            self.attn2 = None
            self.attn2_0 = None

        if cross_attention_dim is not None:
            self.norm2 = (
                AdaLayerNorm(dim, num_embeds_ada_norm)
                if self.use_ada_layer_norm
                else nn.LayerNorm(dim)
            )
        else:
            self.norm2 = None

        # Feed-forward
        self.ff = FeedForward(dim, dropout=dropout,
                              activation_fn=activation_fn)
        self.norm3 = nn.LayerNorm(dim)
        self.use_ada_layer_norm_zero = False



    def forward(
        self,
        hidden_states,
        encoder_hidden_states=None,
        timestep=None,
        attention_mask=None,
        full_mask=None,
        face_mask=None,
        lip_mask=None,
        motion_scale=None,
        video_length=None,
    ):
        """
        Forward pass for the AudioTemporalBasicTransformerBlock.

        Args:
            hidden_states (torch.FloatTensor): The input hidden states.
            encoder_hidden_states (torch.FloatTensor, optional): The encoder hidden states. Defaults to None.
            timestep (torch.LongTensor, optional): The timestep for the transformer block. Defaults to None.
            attention_mask (torch.FloatTensor, optional): The attention mask. Defaults to None.
            full_mask (torch.FloatTensor, optional): The full mask. Defaults to None.
            face_mask (torch.FloatTensor, optional): The face mask. Defaults to None.
            lip_mask (torch.FloatTensor, optional): The lip mask. Defaults to None.
            video_length (int, optional): The length of the video. Defaults to None.

        Returns:
            torch.FloatTensor: The output tensor after passing through the AudioTemporalBasicTransformerBlock.
        """
        norm_hidden_states = (
            self.norm1(hidden_states, timestep)
            if self.use_ada_layer_norm
            else self.norm1(hidden_states)
        )

        if self.unet_use_cross_frame_attention:
            hidden_states = (
                self.attn1(
                    norm_hidden_states,
                    attention_mask=attention_mask,
                    video_length=video_length,
                )
                + hidden_states
            )
        else:
            hidden_states = (
                self.attn1(norm_hidden_states, attention_mask=attention_mask)
                + hidden_states
            )

        if self.attn2 is not None:
            # Cross-Attention
            norm_hidden_states = (
                self.norm2(hidden_states, timestep)
                if self.use_ada_layer_norm
                else self.norm2(hidden_states)
            )
            hidden_states = self.attn2(
                norm_hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=attention_mask,
            ) + hidden_states

        elif self.attn2_0 is not None:
            norm_hidden_states = (
                self.norm2(hidden_states, timestep)
                if self.use_ada_layer_norm
                else self.norm2(hidden_states)
            )

            level = self.depth
            full_hidden_states = (
                self.attn2_0(
                    norm_hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    attention_mask=attention_mask,
                ) * full_mask[level][:, :, None]
            )
            bz, sz, c = full_hidden_states.shape
            sz_sqrt = int(sz ** 0.5)
            full_hidden_states = full_hidden_states.reshape(
                bz, sz_sqrt, sz_sqrt, c).permute(0, 3, 1, 2)
            full_hidden_states = self.zero_conv_full(full_hidden_states).permute(0, 2, 3, 1).reshape(bz, -1, c)

            face_hidden_state = (
                self.attn2_1(
                    norm_hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    attention_mask=attention_mask,
                ) * face_mask[level][:, :, None]
            )
            face_hidden_state = face_hidden_state.reshape(
                bz, sz_sqrt, sz_sqrt, c).permute(0, 3, 1, 2)
            face_hidden_state = self.zero_conv_face(
                face_hidden_state).permute(0, 2, 3, 1).reshape(bz, -1, c)

            lip_hidden_state = (
                self.attn2_2(
                    norm_hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    attention_mask=attention_mask,
                ) * lip_mask[level][:, :, None]

            ) # [32, 4096, 320]
            lip_hidden_state = lip_hidden_state.reshape(
                bz, sz_sqrt, sz_sqrt, c).permute(0, 3, 1, 2)
            lip_hidden_state = self.zero_conv_lip(
                lip_hidden_state).permute(0, 2, 3, 1).reshape(bz, -1, c)

            if motion_scale is not None:
                hidden_states = (
                    motion_scale[0] * full_hidden_states +
                    motion_scale[1] * face_hidden_state +
                    motion_scale[2] * lip_hidden_state + hidden_states
                )
            else:
                hidden_states = (
                    full_hidden_states +
                    face_hidden_state +
                    lip_hidden_state + hidden_states
                )
        # Feed-forward
        hidden_states = self.ff(self.norm3(hidden_states)) + hidden_states

        return hidden_states

def zero_module(module):
    """
    Zeroes out the parameters of a given module.

    Args:
        module (nn.Module): The module whose parameters need to be zeroed out.

    Returns:
        None.
    """
    for p in module.parameters():
        nn.init.zeros_(p)
    return module
