import torch
import torch.nn as nn
from torch import Tensor
from typing import List, Tuple
from .utils import exists
from .utils import default
from .module.embedding import TimeEmbedding
from .module.embedding import FourierEmbedding

class UNetFC2(nn.Module):
    '''
    Fully Connected U-Net-like network adapted from the original U-Net.
    This version is for 1D inputs and uses fully connected layers.
    Context, input, and time embedding vectors are merged together.

    x - we can add the Fourier transforamtion
    '''

    def __init__(
        self,
        input_dim: int = 512,
        output_dim: int | None = None,
        time_emb_dim: int = 16,
        hidden_dims: List[int] = [512, 256, 128],
        use_context: bool = False,
        context_dim: int = 32,
        n_fourier : Tuple[int, ...] | None = None,
        #p: int = 0.2
    ):
        super().__init__()


        self.use_context = use_context
        self.time_emb_dim = time_emb_dim
        self.input_dim = input_dim
        self.use_context = use_context
        self.context_dim = context_dim
        #self.p = p
        # Time Embedding

        self.fourier_emb = FourierEmbedding(*n_fourier) if exists(n_fourier) else nn.Identity()

        self.time_emb = nn.Sequential(
            TimeEmbedding(self.time_emb_dim),
            nn.Linear(self.time_emb_dim, self.context_dim*2),
            nn.GELU(),
            nn.Linear(self.context_dim*2, self.time_emb_dim)
        )
        self.cond_emb = nn.Sequential( 
            nn.Linear(self.context_dim, self.context_dim*2),
            nn.GELU(),
            nn.Linear(self.context_dim*2, self.context_dim)
        )

        # Calculate the total input dimension after merging inputs
        total_input_dim = input_dim + self.time_emb_dim
        if self.use_context and self.context_dim is not None:
            total_input_dim += self.context_dim
        if exists(n_fourier):
            total_input_dim += self.input_dim * (2 * self.fourier_emb.n_feat if exists(n_fourier) else 0)


        # Encoder layers
        self.encoder_layers = nn.ModuleList()
        prev_dim = total_input_dim
        for h_dim in hidden_dims:
            self.encoder_layers.append(nn.Sequential(
                nn.Linear(prev_dim, h_dim),
                nn.LayerNorm(h_dim),
                nn.GELU()
            ))
            prev_dim = h_dim #+ self.time_emb_dim


        # Decoder layers
        self.decoder_layers = nn.ModuleList()
        reversed_hidden_dims = list(reversed(hidden_dims))
        for idx, h_dim in enumerate(reversed_hidden_dims):
            self.decoder_layers.append(nn.Sequential(
                nn.Linear(prev_dim, h_dim),
                nn.LayerNorm(h_dim),
                nn.GELU()
))
            prev_dim = h_dim# + self.time_emb_dim

        # Final output layer
        self.output_layer = nn.Linear(prev_dim,self.input_dim)

    def forward(
        self,
        x: Tensor,        # Shape: [batch_size, input_dim]
        time: Tensor,     # Shape: [batch_size, 1]
        context: Tensor = None  # Shape: [batch_size, context_dim]
    ) -> Tensor:
        '''
        Forward pass of the fully connected U-Net.

        Params:
            - x: Input tensor of shape [batch_size, input_dim]
            - time: Time tensor of shape [batch_size, 1]
            - context (optional): Context tensor of shape [batch_size, context_dim]

        Returns:
            - Output tensor of shape [batch_size, output_dim]
        '''
        # Time embedding
        #print(time)
        x = self.fourier_emb(x)
        t_emb = self.time_emb(time)  # [batch_size, time_emb_dim]
        context = self.cond_emb(context) if context is not None else None
        # Merge inputs
        if self.use_context and context is not None:
            x = torch.cat([x, t_emb, context], dim=1)  # [batch_size, total_input_dim]
        else:
            x = torch.cat([x, t_emb], dim=1)  # [batch_size, total_input_dim]

        # Encoder path with skip connections
        skip_connections = []
        for layer in self.encoder_layers:
            x = layer(x) 
            skip_connections.append(x)
            x = torch.cat([x],dim=1)

        # Decoder path with skip connections
        for layer in self.decoder_layers:
            skip = skip_connections.pop()

            #x = layer(x)
            x = torch.cat([layer(x) + skip],dim=1)
            #x = x + skip  # Element-wise addition for skip connection

        # Output layer
        output = self.output_layer(x)

        return output
