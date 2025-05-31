"""Neural network models for time series analysis.

This module contains PyTorch model definitions for time series forecasting
including Time2Vec embeddings and Transformer-based architectures.
"""

import math
from typing import Tuple

import torch
from torch import nn
from torch.utils.data import Dataset
import numpy as np

from config import (DROPOUT_RATE, FC_DROPOUT_RATE, HIDDEN_DIM, 
                   N_HEADS, N_LAYERS, TIME2VEC_DIM_RATIO)


class TimeSeriesDataset(Dataset):
    """Dataset class for time series data with sliding window approach."""
    
    def __init__(self, data: torch.Tensor, seq_len: int = 8):
        """Initialize TimeSeriesDataset.
        
        Args:
            data: Input tensor of shape [T, features]
            seq_len: Length of input sequences
        """
        self.data = data
        self.seq_len = seq_len
    
    def __len__(self) -> int:
        """Return number of available sequences."""
        return len(self.data) - self.seq_len
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get input sequence and target.
        
        Args:
            idx: Index of the sequence
            
        Returns:
            Tuple of (input_sequence, target) where target is next step
            of the second feature (Close price)
        """
        x = self.data[idx:idx + self.seq_len]
        y = self.data[idx + 1:idx + self.seq_len + 1, 1]  # Close price
        return x, y.unsqueeze(-1)


class Time2Vec(nn.Module):
    """Time2Vec embedding layer for temporal feature extraction."""
    
    def __init__(self, in_dim: int, out_dim: int):
        """Initialize Time2Vec layer.
        
        Args:
            in_dim: Input feature dimension
            out_dim: Output dimension for time embeddings
        """
        super().__init__()
        # Linear projection weights and bias
        self.w0 = nn.Parameter(torch.randn(in_dim, 1))
        self.b0 = nn.Parameter(torch.randn(1))
        
        # Sinusoidal projection weights and bias
        self.w = nn.Parameter(torch.randn(in_dim, out_dim - 1))
        self.b = nn.Parameter(torch.randn(out_dim - 1))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of Time2Vec embedding.
        
        Args:
            x: Input tensor of shape [B, T, in_dim]
            
        Returns:
            Time embeddings of shape [B, T, out_dim]
        """
        # Sinusoidal projections
        sin_proj = torch.sin(x @ self.w + self.b)
        
        # Linear projection
        lin_proj = x @ self.w0 + self.b0
        
        return torch.cat([sin_proj, lin_proj], dim=-1)


class StockTransformer(nn.Module):
    """Transformer-based model for stock price prediction."""
    
    def __init__(self, 
                 feat_dim: int, 
                 time2vec_dim: int, 
                 nhead: int = N_HEADS, 
                 nlayers: int = N_LAYERS, 
                 hidden: int = HIDDEN_DIM):
        """Initialize StockTransformer model.
        
        Args:
            feat_dim: Number of input features
            time2vec_dim: Dimension of time embeddings
            nhead: Number of attention heads
            nlayers: Number of transformer layers
            hidden: Hidden dimension in feedforward network
        """
        super().__init__()
        
        # Time embedding layer
        self.time2vec = Time2Vec(feat_dim, time2vec_dim)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=feat_dim + time2vec_dim,
            nhead=nhead,
            dim_feedforward=hidden,
            dropout=DROPOUT_RATE,
            activation='relu'
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, nlayers)
        
        # Global average pooling
        self.pool = nn.AdaptiveAvgPool1d(1)
        
        # Classification head
        self.fc = nn.Sequential(
            nn.Linear(feat_dim + time2vec_dim, hidden),
            nn.ReLU(),
            nn.Dropout(FC_DROPOUT_RATE),
            nn.Linear(hidden, 1)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model.
        
        Args:
            x: Input tensor of shape [B, T, feat_dim]
            
        Returns:
            Predictions of shape [B, 1]
        """
        # Generate time embeddings
        time_vec = self.time2vec(x)
        
        # Concatenate features with time embeddings
        cat = torch.cat([x, time_vec], dim=-1)  # [B, T, D]
        
        # Transformer expects [T, B, D]
        out = self.encoder(cat.permute(1, 0, 2))
        
        # Back to [B, D, T] for pooling
        out = out.permute(1, 2, 0)
        
        # Global average pooling: [B, D, T] -> [B, D]
        pooled = self.pool(out).squeeze(-1)
        
        # Final prediction: [B, D] -> [B, 1]
        return self.fc(pooled)
