"""
This module provides the implementation of an Audio Projection Model, which is designed for
audio processing tasks. The model takes audio embeddings as input and outputs context tokens
that can be used for various downstream applications, such as audio analysis or synthesis.

The AudioProjModel class is based on the ModelMixin class from the diffusers library, which
provides a foundation for building custom models. This implementation includes multiple linear
layers with ReLU activation functions and a LayerNorm for normalization.

Key Features:
- Audio embedding input with flexible sequence length and block structure.
- Multiple linear layers for feature transformation.
- ReLU activation for non-linear transformation.
- LayerNorm for stabilizing and speeding up training.
- Rearrangement of input embeddings to match the model's expected input shape.
- Customizable number of blocks, channels, and context tokens for adaptability.

The module is structured to be easily integrated into larger systems or used as a standalone
component for audio feature extraction and processing.

Classes:
- AudioProjModel: A class representing the audio projection model with configurable parameters.

Functions:
- (none)

Dependencies:
- torch: For tensor operations and neural network components.
- diffusers: For the ModelMixin base class.
- einops: For tensor rearrangement operations.

"""

import torch
from diffusers import ModelMixin
from einops import rearrange
from torch import nn


class AudioProjModel(ModelMixin):
    """Audio Projection Model

    This class defines an audio projection model that takes audio embeddings as input
    and produces context tokens as output. The model is based on the ModelMixin class
    and consists of multiple linear layers and activation functions. It can be used
    for various audio processing tasks.

    Attributes:
        seq_len (int): The length of the audio sequence.
        blocks (int): The number of blocks in the audio projection model.
        channels (int): The number of channels in the audio projection model.
        intermediate_dim (int): The intermediate dimension of the model.
        context_tokens (int): The number of context tokens in the output.
        output_dim (int): The output dimension of the context tokens.

    Methods:
        __init__(self, seq_len=5, blocks=12, channels=768, intermediate_dim=512, context_tokens=32, output_dim=768):
            Initializes the AudioProjModel with the given parameters.
        forward(self, audio_embeds):
            Defines the forward pass for the AudioProjModel.
            Parameters:
            audio_embeds (torch.Tensor): The input audio embeddings with shape (batch_size, video_length, blocks, channels).
            Returns:
            context_tokens (torch.Tensor): The output context tokens with shape (batch_size, video_length, context_tokens, output_dim).

    """

    def __init__(
        self,
        seq_len=5,
        blocks=12,  # add a new parameter blocks
        channels=768,  # add a new parameter channels
        intermediate_dim=512,
        output_dim=768,
        context_tokens=32,
    ):
        super().__init__()

        self.seq_len = seq_len
        self.blocks = blocks
        self.channels = channels
        self.input_dim = (
            seq_len * blocks * channels
        )  # update input_dim to be the product of blocks and channels.
        self.intermediate_dim = intermediate_dim
        self.context_tokens = context_tokens
        self.output_dim = output_dim

        # define multiple linear layers
        self.proj1 = nn.Linear(self.input_dim, intermediate_dim)
        self.proj2 = nn.Linear(intermediate_dim, intermediate_dim)
        self.proj3 = nn.Linear(intermediate_dim, context_tokens * output_dim)

        self.norm = nn.LayerNorm(output_dim)

    def forward(self, audio_embeds):
        """
        Defines the forward pass for the AudioProjModel.

        Parameters:
            audio_embeds (torch.Tensor): The input audio embeddings with shape (batch_size, video_length, blocks, channels).

        Returns:
            context_tokens (torch.Tensor): The output context tokens with shape (batch_size, video_length, context_tokens, output_dim).
        """
        # merge
        video_length = audio_embeds.shape[1]
        audio_embeds = rearrange(audio_embeds, "bz f w b c -> (bz f) w b c")
        batch_size, window_size, blocks, channels = audio_embeds.shape
        audio_embeds = audio_embeds.view(batch_size, window_size * blocks * channels)

        audio_embeds = torch.relu(self.proj1(audio_embeds))
        audio_embeds = torch.relu(self.proj2(audio_embeds))

        context_tokens = self.proj3(audio_embeds).reshape(
            batch_size, self.context_tokens, self.output_dim
        )

        context_tokens = self.norm(context_tokens)
        context_tokens = rearrange(
            context_tokens, "(bz f) m c -> bz f m c", f=video_length
        )

        return context_tokens
