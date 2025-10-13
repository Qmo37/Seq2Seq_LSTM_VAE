"""
Seq2Seq VAE model for diverse sequence generation.
"""

import torch
import torch.nn as nn


class VAEEncoder(nn.Module):
    """
    VAE Encoder that maps input sequences to latent distribution parameters.

    Args:
        input_size: Number of input features
        hidden_size: Size of hidden state
        latent_dim: Dimension of latent space
        num_layers: Number of LSTM layers
        dropout: Dropout probability
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        latent_dim: int,
        num_layers: int = 1,
        dropout: float = 0.0,
    ):
        super(VAEEncoder, self).__init__()
        self.hidden_size = hidden_size
        self.latent_dim = latent_dim

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # Map hidden state to latent distribution parameters
        self.fc_mu = nn.Linear(hidden_size, latent_dim)
        self.fc_logvar = nn.Linear(hidden_size, latent_dim)

    def forward(self, x):
        """
        Args:
            x: Input sequence (batch_size, seq_len, input_size)

        Returns:
            mu: Mean of latent distribution (batch_size, latent_dim)
            logvar: Log variance of latent distribution (batch_size, latent_dim)
        """
        # Encode sequence
        _, (h_n, _) = self.lstm(x)

        # Use last hidden state
        h = h_n[-1]  # (batch_size, hidden_size)

        # Compute distribution parameters
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)

        return mu, logvar


class VAEDecoder(nn.Module):
    """
    VAE Decoder that generates sequences from latent codes.

    Args:
        latent_dim: Dimension of latent space
        hidden_size: Size of hidden state
        output_size: Number of output features (1 for clicks)
        num_layers: Number of LSTM layers
        dropout: Dropout probability
    """

    def __init__(
        self,
        latent_dim: int,
        hidden_size: int,
        output_size: int,
        num_layers: int = 1,
        dropout: float = 0.0,
    ):
        super(VAEDecoder, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers

        # Map latent code to initial hidden state
        self.fc_z_to_h = nn.Linear(latent_dim, hidden_size)
        self.fc_z_to_c = nn.Linear(latent_dim, hidden_size)

        self.lstm = nn.LSTM(
            input_size=output_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.fc_out = nn.Linear(hidden_size, output_size)

    def forward(self, z, tgt_len):
        """
        Args:
            z: Latent code (batch_size, latent_dim)
            tgt_len: Target sequence length

        Returns:
            outputs: Generated sequence (batch_size, tgt_len, output_size)
        """
        batch_size = z.size(0)

        # Initialize hidden state from latent code
        h_0 = torch.tanh(self.fc_z_to_h(z)).unsqueeze(0)  # (1, batch_size, hidden_size)
        c_0 = torch.tanh(self.fc_z_to_c(z)).unsqueeze(0)  # (1, batch_size, hidden_size)

        # Repeat for multiple layers if needed
        h_0 = h_0.repeat(self.num_layers, 1, 1)
        c_0 = c_0.repeat(self.num_layers, 1, 1)

        hidden = (h_0, c_0)

        # Initialize decoder input
        decoder_input = torch.zeros(batch_size, 1, self.output_size, device=z.device)

        # Generate sequence
        outputs = []
        for t in range(tgt_len):
            output, hidden = self.lstm(decoder_input, hidden)
            output = self.fc_out(output)
            outputs.append(output)
            decoder_input = output

        outputs = torch.cat(outputs, dim=1)  # (batch_size, tgt_len, output_size)

        return outputs


class Seq2SeqVAE(nn.Module):
    """
    Complete Seq2Seq VAE model for diverse sequence generation.

    Architecture:
    - Encoder: Maps input sequences to latent distribution N(μ, σ²)
    - Reparameterization: Samples z from latent distribution
    - Decoder: Generates sequences from z (enables diverse generation)

    Args:
        input_size: Number of input features
        hidden_size: Size of hidden state
        latent_dim: Dimension of latent space
        output_size: Number of output features (1 for clicks)
        num_layers: Number of LSTM layers
        dropout: Dropout probability
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        latent_dim: int = 16,
        output_size: int = 1,
        num_layers: int = 1,
        dropout: float = 0.0,
    ):
        super(Seq2SeqVAE, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.latent_dim = latent_dim
        self.output_size = output_size

        self.encoder = VAEEncoder(
            input_size, hidden_size, latent_dim, num_layers, dropout
        )
        self.decoder = VAEDecoder(
            latent_dim, hidden_size, output_size, num_layers, dropout
        )

    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick: z = μ + σ * ε, where ε ~ N(0, 1)

        Args:
            mu: Mean (batch_size, latent_dim)
            logvar: Log variance (batch_size, latent_dim)

        Returns:
            z: Sampled latent code (batch_size, latent_dim)
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z

    def forward(self, src, tgt_len):
        """
        Args:
            src: Source sequence (batch_size, src_len, input_size)
            tgt_len: Target sequence length

        Returns:
            recon: Reconstructed sequence (batch_size, tgt_len, output_size)
            mu: Mean of latent distribution
            logvar: Log variance of latent distribution
        """
        # Encode to latent distribution
        mu, logvar = self.encoder(src)

        # Sample from latent distribution
        z = self.reparameterize(mu, logvar)

        # Decode to output sequence
        recon = self.decoder(z, tgt_len)

        return recon, mu, logvar

    def generate(self, src, tgt_len, n_samples=1):
        """
        Generate multiple diverse sequences from the same input.

        Args:
            src: Source sequence (batch_size, src_len, input_size)
            tgt_len: Target sequence length
            n_samples: Number of samples to generate per input

        Returns:
            samples: Generated sequences (batch_size, n_samples, tgt_len, output_size)
        """
        self.eval()
        with torch.no_grad():
            batch_size = src.size(0)

            # Encode to latent distribution
            mu, logvar = self.encoder(src)

            # Generate multiple samples
            all_samples = []
            for _ in range(n_samples):
                z = self.reparameterize(mu, logvar)
                sample = self.decoder(z, tgt_len)
                all_samples.append(sample.unsqueeze(1))

            samples = torch.cat(
                all_samples, dim=1
            )  # (batch_size, n_samples, tgt_len, output_size)

        return samples

    def predict(self, src, tgt_len):
        """
        Single prediction using mean of latent distribution (no sampling).

        Args:
            src: Source sequence (batch_size, src_len, input_size)
            tgt_len: Target sequence length

        Returns:
            Predicted sequence (batch_size, tgt_len, output_size)
        """
        self.eval()
        with torch.no_grad():
            mu, _ = self.encoder(src)
            output = self.decoder(mu, tgt_len)
        return output


def vae_loss(recon, target, mu, logvar, beta=1.0):
    """
    VAE loss = Reconstruction loss + β * KL divergence

    Args:
        recon: Reconstructed sequence
        target: Target sequence
        mu: Mean of latent distribution
        logvar: Log variance of latent distribution
        beta: Weight for KL divergence term

    Returns:
        total_loss: Total VAE loss
        recon_loss: Reconstruction loss (MSE)
        kld_loss: KL divergence loss
    """
    # Reconstruction loss (MSE)
    recon_loss = nn.functional.mse_loss(recon, target, reduction="mean")

    # KL divergence: -0.5 * sum(1 + log(σ²) - μ² - σ²)
    kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    kld_loss = kld_loss / mu.size(0)  # Average over batch

    # Total loss
    total_loss = recon_loss + beta * kld_loss

    return total_loss, recon_loss, kld_loss
