"""Time series analysis with Transformer models.

This script implements a complete pipeline for time series forecasting using
Transformer neural networks with Time2Vec embeddings. It includes data
preprocessing, model training, and visualization.
"""

import torch
from torch import nn
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from config import (DEFAULT_CSV_PATH, DEFAULT_FEATURE_COLUMNS, ROLLING_WINDOW,
                   TRAIN_FRACTION, SEQUENCE_LENGTH, BATCH_SIZE, EPOCHS,
                   LEARNING_RATE, TIME2VEC_DIM_RATIO)
from models import TimeSeriesDataset, StockTransformer


def load_and_preprocess_data(csv_path: str,
                            feature_cols=None,
                            rolling_window: int = ROLLING_WINDOW,
                            train_frac: float = TRAIN_FRACTION):
    """Load CSV data and apply preprocessing pipeline.
    
    This function:
    1. Reads CSV file
    2. Selects features and drops NaN values
    3. Applies rolling mean smoothing
    4. Calculates first differences
    5. Applies min-max normalization
    6. Splits into train/test tensors
    
    Args:
        csv_path: Path to CSV file
        feature_cols: List of feature column names
        rolling_window: Window size for rolling mean
        train_frac: Fraction of data for training
        
    Returns:
        Tuple of (train_tensor, test_tensor)
    """
    # Load and select features
    df = pd.read_csv(csv_path)
    feature_cols = feature_cols or DEFAULT_FEATURE_COLUMNS
    df = df[feature_cols].dropna()
    
    # Apply rolling mean smoothing
    df_smoothed = df.rolling(rolling_window).mean().dropna()
    
    # Calculate first differences
    df_diff = df_smoothed.diff().dropna()
    
    # Min-max normalization
    normalized = ((df_diff - df_diff.min()) / 
                  (df_diff.max() - df_diff.min()))
    
    # Convert to tensors and split
    data = normalized.values.astype(np.float32)
    split_idx = int(len(data) * train_frac)
    train_data, test_data = data[:split_idx], data[split_idx:]
    
    return torch.from_numpy(train_data), torch.from_numpy(test_data)


def train_epoch(model, loader, optimizer, loss_fn, device):
    """Train model for one epoch.
    
    Args:
        model: PyTorch model
        loader: DataLoader for training data
        optimizer: Optimizer instance
        loss_fn: Loss function
        device: Device to run training on
        
    Returns:
        Average training loss for the epoch
    """
    model.train()
    total_loss = 0
    
    for batch_x, batch_y in loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        
        optimizer.zero_grad()
        predictions = model(batch_x)
        loss = loss_fn(predictions.squeeze(-1), batch_y.squeeze(-1))
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * batch_x.size(0)
    
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate_model(model, loader, loss_fn, device):
    """Evaluate model on validation/test data.
    
    Args:
        model: PyTorch model
        loader: DataLoader for evaluation data
        loss_fn: Loss function
        device: Device to run evaluation on
        
    Returns:
        Average evaluation loss
    """
    model.eval()
    total_loss = 0
    
    for batch_x, batch_y in loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        predictions = model(batch_x)
        total_loss += loss_fn(
            predictions.squeeze(-1), 
            batch_y.squeeze(-1)
        ).item() * batch_x.size(0)
    
    return total_loss / len(loader.dataset)


def plot_time_series(series: np.ndarray, 
                     title: str = "Time Series", 
                     ylabel: str = "Value"):
    """Plot time series data.
    
    Args:
        series: 1D numpy array of time series values
        title: Plot title
        ylabel: Y-axis label
    """
    plt.figure(figsize=(10, 6))
    plt.plot(series, label=ylabel, linewidth=1.5)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel("Time Step", fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def main():
    """Execute the complete time series analysis pipeline."""
    print("Starting time series analysis with Transformer model...")
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load and preprocess data
    print(f"Loading data from {DEFAULT_CSV_PATH}")
    train_tensor, test_tensor = load_and_preprocess_data(DEFAULT_CSV_PATH)
    
    # Create datasets and data loaders
    train_dataset = TimeSeriesDataset(train_tensor, seq_len=SEQUENCE_LENGTH)
    test_dataset = TimeSeriesDataset(test_tensor, seq_len=SEQUENCE_LENGTH)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=BATCH_SIZE
    )
    
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")
    
    # Initialize model
    feature_dim = train_tensor.shape[1]
    time2vec_dim = int(feature_dim * TIME2VEC_DIM_RATIO)
    
    model = StockTransformer(
        feat_dim=feature_dim,
        time2vec_dim=time2vec_dim
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.MSELoss()
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Training loop
    print(f"Training for {EPOCHS} epochs...")
    for epoch in range(1, EPOCHS + 1):
        train_loss = train_epoch(model, train_loader, optimizer, loss_fn, device)
        val_loss = evaluate_model(model, test_loader, loss_fn, device)
        
        print(f"Epoch {epoch:2d} — train: {train_loss:.4f} │ "
              f"val: {val_loss:.4f}")
    
    # Generate predictions and visualize results
    print("Generating predictions...")
    model.eval()
    test_batch_x, test_batch_y = next(iter(test_loader))
    
    with torch.no_grad():
        predictions = model(test_batch_x.to(device)).cpu().squeeze(-1).numpy()
    
    actual_values = test_batch_y[:, -1].numpy()
    
    # Plot results
    plot_time_series(
        actual_values, 
        title="Actual Close Price Changes", 
        ylabel="Normalized ΔPrice"
    )
    
    plot_time_series(
        predictions, 
        title="Predicted ΔPrice", 
        ylabel="ΔPrice Prediction"
    )
    
    print("Analysis complete!")


if __name__ == "__main__":
    main()
