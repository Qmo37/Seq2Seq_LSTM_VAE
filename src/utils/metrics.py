"""
Evaluation metrics for comparing LSTM and VAE models.
"""

import torch
import numpy as np
from typing import Tuple


def compute_mse(predictions: np.ndarray, targets: np.ndarray) -> float:
    """
    Compute Mean Squared Error.

    Args:
        predictions: Predicted values (batch_size, seq_len, 1)
        targets: Ground truth values (batch_size, seq_len, 1)

    Returns:
        MSE value
    """
    return np.mean((predictions - targets) ** 2)


def compute_best_of_n_mse(
    samples: np.ndarray, targets: np.ndarray
) -> Tuple[float, np.ndarray]:
    """
    Compute Best-of-N MSE: minimum MSE among N samples.

    For each instance, compute MSE for all N samples and take the minimum.
    This measures how well the model can generate at least one good prediction.

    Args:
        samples: Generated samples (batch_size, n_samples, seq_len, 1)
        targets: Ground truth values (batch_size, seq_len, 1)

    Returns:
        avg_best_mse: Average best MSE across batch
        best_mses: Best MSE for each instance (batch_size,)
    """
    batch_size, n_samples, seq_len, _ = samples.shape

    # Expand targets to match samples shape
    targets_expanded = np.expand_dims(targets, axis=1)  # (batch_size, 1, seq_len, 1)
    targets_expanded = np.repeat(
        targets_expanded, n_samples, axis=1
    )  # (batch_size, n_samples, seq_len, 1)

    # Compute MSE for each sample
    mses = np.mean(
        (samples - targets_expanded) ** 2, axis=(2, 3)
    )  # (batch_size, n_samples)

    # Take minimum MSE for each instance
    best_mses = np.min(mses, axis=1)  # (batch_size,)

    avg_best_mse = np.mean(best_mses)

    return avg_best_mse, best_mses


def compute_diversity(samples: np.ndarray) -> Tuple[float, np.ndarray]:
    """
    Compute diversity as standard deviation across generated samples.

    Higher diversity means the model generates more varied predictions.

    Args:
        samples: Generated samples (batch_size, n_samples, seq_len, 1)

    Returns:
        avg_diversity: Average diversity across batch
        diversities: Diversity for each instance (batch_size,)
    """
    # Compute standard deviation across samples for each instance
    diversities = np.std(samples, axis=1)  # (batch_size, seq_len, 1)

    # Average across sequence and features
    diversities = np.mean(diversities, axis=(1, 2))  # (batch_size,)

    avg_diversity = np.mean(diversities)

    return avg_diversity, diversities


def compute_coverage(
    samples: np.ndarray, targets: np.ndarray, confidence: float = 0.95
) -> Tuple[float, np.ndarray]:
    """
    Compute coverage: proportion of ground truth values within confidence interval.

    Coverage measures whether the generated samples capture the uncertainty range
    that includes the true value.

    Args:
        samples: Generated samples (batch_size, n_samples, seq_len, 1)
        targets: Ground truth values (batch_size, seq_len, 1)
        confidence: Confidence level (default: 0.95 for 95% CI)

    Returns:
        avg_coverage: Average coverage across batch
        coverages: Coverage for each instance (batch_size,)
    """
    batch_size, n_samples, seq_len, _ = samples.shape

    # Compute confidence interval for each instance
    alpha = (1 - confidence) / 2
    lower_percentile = alpha * 100
    upper_percentile = (1 - alpha) * 100

    lower_bounds = np.percentile(
        samples, lower_percentile, axis=1
    )  # (batch_size, seq_len, 1)
    upper_bounds = np.percentile(
        samples, upper_percentile, axis=1
    )  # (batch_size, seq_len, 1)

    # Check if targets are within bounds
    within_bounds = (targets >= lower_bounds) & (
        targets <= upper_bounds
    )  # (batch_size, seq_len, 1)

    # Compute coverage for each instance (proportion of points within bounds)
    coverages = np.mean(within_bounds, axis=(1, 2))  # (batch_size,)

    avg_coverage = np.mean(coverages)

    return avg_coverage, coverages


def evaluate_model(
    model, data_loader, device, output_weeks=2, n_samples=20, is_vae=False
):
    """
    Comprehensive evaluation of a model.

    Args:
        model: Trained model (LSTM or VAE)
        data_loader: Test data loader
        device: Device (cuda or cpu)
        output_weeks: Number of output weeks
        n_samples: Number of samples to generate (for VAE)
        is_vae: Whether model is VAE

    Returns:
        Dictionary containing all metrics and predictions
    """
    model.eval()

    all_predictions = []
    all_targets = []
    all_samples = [] if is_vae else None

    with torch.no_grad():
        for X_batch, y_batch in data_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            if is_vae:
                # Generate multiple samples
                samples = model.generate(X_batch, output_weeks, n_samples)
                all_samples.append(samples.cpu().numpy())

                # Use mean prediction for single-path MSE
                pred = samples.mean(dim=1)
            else:
                # Single prediction
                pred = model.predict(X_batch, output_weeks)

            all_predictions.append(pred.cpu().numpy())
            all_targets.append(y_batch.cpu().numpy())

    # Concatenate all batches
    predictions = np.concatenate(all_predictions, axis=0)
    targets = np.concatenate(all_targets, axis=0)

    # Compute single-path MSE
    mse = compute_mse(predictions, targets)

    results = {"mse": mse, "predictions": predictions, "targets": targets}

    if is_vae:
        samples = np.concatenate(all_samples, axis=0)

        # Compute VAE-specific metrics
        best_of_n_mse, _ = compute_best_of_n_mse(samples, targets)
        diversity, _ = compute_diversity(samples)
        coverage, _ = compute_coverage(samples, targets)

        results.update(
            {
                "best_of_n_mse": best_of_n_mse,
                "diversity": diversity,
                "coverage": coverage,
                "samples": samples,
            }
        )

    return results
