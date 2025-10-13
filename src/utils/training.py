"""
Training utilities for LSTM and VAE models.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import random


def set_seed(seed: int = 42):
    """
    Set random seed for reproducibility.

    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_lstm(model, train_loader, val_loader, epochs, lr, device, output_weeks=2):
    """
    Train Seq2Seq LSTM model.

    Args:
        model: Seq2SeqLSTM model
        train_loader: Training data loader
        val_loader: Validation data loader
        epochs: Number of training epochs
        lr: Learning rate
        device: Device (cuda or cpu)
        output_weeks: Number of output weeks

    Returns:
        Dictionary containing training history
    """
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    history = {"train_loss": [], "val_loss": []}

    for epoch in range(epochs):
        # Training
        model.train()
        train_losses = []

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs} [Train]")
        for X_batch, y_batch in pbar:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()

            # Forward pass
            output = model(X_batch, output_weeks)

            # Compute loss
            loss = criterion(output, y_batch)

            # Backward pass
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())
            pbar.set_postfix({"loss": loss.item()})

        avg_train_loss = np.mean(train_losses)
        history["train_loss"].append(avg_train_loss)

        # Validation
        model.eval()
        val_losses = []

        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)

                output = model.predict(X_batch, output_weeks)
                loss = criterion(output, y_batch)

                val_losses.append(loss.item())

        avg_val_loss = np.mean(val_losses)
        history["val_loss"].append(avg_val_loss)

        print(
            f"Epoch {epoch + 1}/{epochs} - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}"
        )

    return history


def train_vae(
    model, train_loader, val_loader, epochs, lr, device, output_weeks=2, beta=1.0
):
    """
    Train Seq2Seq VAE model.

    Args:
        model: Seq2SeqVAE model
        train_loader: Training data loader
        val_loader: Validation data loader
        epochs: Number of training epochs
        lr: Learning rate
        device: Device (cuda or cpu)
        output_weeks: Number of output weeks
        beta: Weight for KL divergence term

    Returns:
        Dictionary containing training history
    """
    from ..models.seq2seq_vae import vae_loss

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    history = {
        "train_loss": [],
        "train_recon_loss": [],
        "train_kld_loss": [],
        "val_loss": [],
        "val_recon_loss": [],
        "val_kld_loss": [],
    }

    for epoch in range(epochs):
        # Training
        model.train()
        train_losses = []
        train_recon_losses = []
        train_kld_losses = []

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs} [Train]")
        for X_batch, y_batch in pbar:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()

            # Forward pass
            recon, mu, logvar = model(X_batch, output_weeks)

            # Compute VAE loss
            loss, recon_loss, kld_loss = vae_loss(recon, y_batch, mu, logvar, beta)

            # Backward pass
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())
            train_recon_losses.append(recon_loss.item())
            train_kld_losses.append(kld_loss.item())

            pbar.set_postfix(
                {
                    "loss": loss.item(),
                    "recon": recon_loss.item(),
                    "kld": kld_loss.item(),
                }
            )

        avg_train_loss = np.mean(train_losses)
        avg_train_recon = np.mean(train_recon_losses)
        avg_train_kld = np.mean(train_kld_losses)

        history["train_loss"].append(avg_train_loss)
        history["train_recon_loss"].append(avg_train_recon)
        history["train_kld_loss"].append(avg_train_kld)

        # Validation
        model.eval()
        val_losses = []
        val_recon_losses = []
        val_kld_losses = []

        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)

                recon, mu, logvar = model(X_batch, output_weeks)
                loss, recon_loss, kld_loss = vae_loss(recon, y_batch, mu, logvar, beta)

                val_losses.append(loss.item())
                val_recon_losses.append(recon_loss.item())
                val_kld_losses.append(kld_loss.item())

        avg_val_loss = np.mean(val_losses)
        avg_val_recon = np.mean(val_recon_losses)
        avg_val_kld = np.mean(val_kld_losses)

        history["val_loss"].append(avg_val_loss)
        history["val_recon_loss"].append(avg_val_recon)
        history["val_kld_loss"].append(avg_val_kld)

        print(
            f"Epoch {epoch + 1}/{epochs} - Train Loss: {avg_train_loss:.4f} "
            f"(Recon: {avg_train_recon:.4f}, KLD: {avg_train_kld:.4f}), "
            f"Val Loss: {avg_val_loss:.4f} "
            f"(Recon: {avg_val_recon:.4f}, KLD: {avg_val_kld:.4f})"
        )

    return history
