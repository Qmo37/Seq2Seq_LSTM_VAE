"""
Visualization utilities for comparing model predictions.
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def plot_training_curves(lstm_history, vae_history, save_path=None):
    """
    Plot training and validation loss curves for both models.

    Args:
        lstm_history: LSTM training history
        vae_history: VAE training history
        save_path: Optional path to save figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # LSTM curves
    axes[0].plot(lstm_history["train_loss"], label="Train Loss", linewidth=2)
    axes[0].plot(lstm_history["val_loss"], label="Val Loss", linewidth=2)
    axes[0].set_xlabel("Epoch", fontsize=12)
    axes[0].set_ylabel("MSE Loss", fontsize=12)
    axes[0].set_title("Seq2Seq LSTM Training", fontsize=14, fontweight="bold")
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)

    # VAE curves
    axes[1].plot(vae_history["train_loss"], label="Train Total Loss", linewidth=2)
    axes[1].plot(vae_history["val_loss"], label="Val Total Loss", linewidth=2)
    axes[1].plot(
        vae_history["train_recon_loss"],
        label="Train Recon Loss",
        linewidth=2,
        linestyle="--",
        alpha=0.7,
    )
    axes[1].plot(
        vae_history["val_recon_loss"],
        label="Val Recon Loss",
        linewidth=2,
        linestyle="--",
        alpha=0.7,
    )
    axes[1].set_xlabel("Epoch", fontsize=12)
    axes[1].set_ylabel("Loss", fontsize=12)
    axes[1].set_title("Seq2Seq VAE Training", fontsize=14, fontweight="bold")
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()


def plot_predictions(
    ground_truth, lstm_pred, vae_samples, indices=None, n_examples=5, save_path=None
):
    """
    Plot comparison of ground truth, LSTM prediction, and VAE samples.

    Args:
        ground_truth: Ground truth sequences (batch_size, seq_len, 1)
        lstm_pred: LSTM predictions (batch_size, seq_len, 1)
        vae_samples: VAE samples (batch_size, n_samples, seq_len, 1)
        indices: Specific indices to plot (if None, randomly sample)
        n_examples: Number of examples to plot
        save_path: Optional path to save figure
    """
    if indices is None:
        indices = np.random.choice(len(ground_truth), n_examples, replace=False)
    else:
        n_examples = len(indices)

    fig, axes = plt.subplots(1, n_examples, figsize=(4 * n_examples, 4))

    if n_examples == 1:
        axes = [axes]

    for i, idx in enumerate(indices):
        ax = axes[i]

        # Ground truth
        gt = ground_truth[idx].squeeze()
        ax.plot(
            gt,
            "o-",
            color="black",
            linewidth=2.5,
            markersize=8,
            label="Ground Truth",
            zorder=3,
        )

        # LSTM prediction
        lstm = lstm_pred[idx].squeeze()
        ax.plot(
            lstm,
            "s--",
            color="blue",
            linewidth=2,
            markersize=7,
            label="LSTM",
            alpha=0.8,
            zorder=2,
        )

        # VAE samples (semi-transparent)
        vae_samps = vae_samples[idx]  # (n_samples, seq_len, 1)
        for j in range(vae_samps.shape[0]):
            sample = vae_samps[j].squeeze()
            ax.plot(sample, "-", color="red", linewidth=1, alpha=0.15, zorder=1)

        # VAE mean
        vae_mean = vae_samps.mean(axis=0).squeeze()
        ax.plot(
            vae_mean,
            "D-",
            color="red",
            linewidth=2,
            markersize=6,
            label="VAE Mean",
            alpha=0.9,
            zorder=2,
        )

        ax.set_xlabel("Week", fontsize=11)
        ax.set_ylabel("Clicks (Normalized)", fontsize=11)
        ax.set_title(f"Example {idx}", fontsize=12, fontweight="bold")
        ax.grid(True, alpha=0.3)

        if i == 0:
            ax.legend(fontsize=9, loc="best")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()


def plot_comparison(lstm_results, vae_results, save_path=None):
    """
    Create comprehensive comparison plot with metrics and examples.

    Args:
        lstm_results: LSTM evaluation results
        vae_results: VAE evaluation results
        save_path: Optional path to save figure
    """
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    # Metrics comparison
    ax_metrics = fig.add_subplot(gs[0, :])

    metrics_data = {
        "MSE": [lstm_results["mse"], vae_results["mse"]],
        "Best-of-N MSE": [np.nan, vae_results["best_of_n_mse"]],
        "Diversity": [0, vae_results["diversity"]],
        "Coverage": [np.nan, vae_results["coverage"]],
    }

    x = np.arange(len(metrics_data))
    width = 0.35

    lstm_values = [metrics_data[k][0] for k in metrics_data.keys()]
    vae_values = [metrics_data[k][1] for k in metrics_data.keys()]

    bars1 = ax_metrics.bar(
        x - width / 2, lstm_values, width, label="LSTM", color="blue", alpha=0.7
    )
    bars2 = ax_metrics.bar(
        x + width / 2, vae_values, width, label="VAE", color="red", alpha=0.7
    )

    ax_metrics.set_ylabel("Value", fontsize=12)
    ax_metrics.set_title("Model Comparison Metrics", fontsize=14, fontweight="bold")
    ax_metrics.set_xticks(x)
    ax_metrics.set_xticklabels(metrics_data.keys(), fontsize=11)
    ax_metrics.legend(fontsize=11)
    ax_metrics.grid(True, alpha=0.3, axis="y")

    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            if not np.isnan(height):
                ax_metrics.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height,
                    f"{height:.4f}",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                )

    # Example predictions
    indices = np.random.choice(len(lstm_results["targets"]), 6, replace=False)

    for plot_idx in range(6):
        row = (plot_idx // 3) + 1
        col = plot_idx % 3
        ax = fig.add_subplot(gs[row, col])

        idx = indices[plot_idx]

        # Ground truth
        gt = lstm_results["targets"][idx].squeeze()
        ax.plot(
            gt, "o-", color="black", linewidth=2.5, markersize=8, label="GT", zorder=3
        )

        # LSTM
        lstm_pred = lstm_results["predictions"][idx].squeeze()
        ax.plot(
            lstm_pred,
            "s--",
            color="blue",
            linewidth=2,
            markersize=6,
            label="LSTM",
            alpha=0.8,
            zorder=2,
        )

        # VAE samples
        vae_samps = vae_results["samples"][idx]
        for j in range(vae_samps.shape[0]):
            sample = vae_samps[j].squeeze()
            ax.plot(sample, "-", color="red", linewidth=1, alpha=0.12, zorder=1)

        # VAE mean
        vae_mean = vae_samps.mean(axis=0).squeeze()
        ax.plot(
            vae_mean,
            "D-",
            color="red",
            linewidth=2,
            markersize=5,
            label="VAE",
            alpha=0.9,
            zorder=2,
        )

        ax.set_xlabel("Week", fontsize=10)
        ax.set_ylabel("Clicks", fontsize=10)
        ax.set_title(f"Example {idx}", fontsize=11, fontweight="bold")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc="best")

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()


def plot_diversity_analysis(vae_samples, targets, save_path=None):
    """
    Analyze and visualize diversity of VAE samples.

    Args:
        vae_samples: VAE samples (batch_size, n_samples, seq_len, 1)
        targets: Ground truth (batch_size, seq_len, 1)
        save_path: Optional path to save figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Distribution of sample standard deviations
    ax = axes[0, 0]
    stds = np.std(vae_samples, axis=1).flatten()
    ax.hist(stds, bins=50, color="red", alpha=0.7, edgecolor="black")
    ax.axvline(
        stds.mean(),
        color="darkred",
        linestyle="--",
        linewidth=2,
        label=f"Mean: {stds.mean():.3f}",
    )
    ax.set_xlabel("Standard Deviation", fontsize=11)
    ax.set_ylabel("Frequency", fontsize=11)
    ax.set_title("Distribution of Sample Diversity", fontsize=12, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # 2. Sample spread visualization
    ax = axes[0, 1]
    n_examples = min(100, vae_samples.shape[0])
    indices = np.random.choice(vae_samples.shape[0], n_examples, replace=False)

    for idx in indices:
        samples = vae_samples[idx]  # (n_samples, seq_len, 1)
        mean_val = samples.mean()
        std_val = samples.std()
        ax.scatter(mean_val, std_val, color="red", alpha=0.5, s=30)

    ax.set_xlabel("Mean Prediction", fontsize=11)
    ax.set_ylabel("Standard Deviation", fontsize=11)
    ax.set_title("Sample Spread Analysis", fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3)

    # 3. Coverage visualization
    ax = axes[1, 0]
    from .metrics import compute_coverage

    coverage_vals = []
    confidence_levels = np.linspace(0.5, 0.99, 20)

    for conf in confidence_levels:
        cov, _ = compute_coverage(vae_samples, targets, confidence=conf)
        coverage_vals.append(cov)

    ax.plot(
        confidence_levels * 100,
        coverage_vals,
        "o-",
        color="red",
        linewidth=2,
        markersize=6,
    )
    ax.plot([50, 99], [50, 99], "--", color="gray", alpha=0.5, label="Ideal")
    ax.set_xlabel("Confidence Level (%)", fontsize=11)
    ax.set_ylabel("Actual Coverage (%)", fontsize=11)
    ax.set_title("Coverage Calibration", fontsize=12, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # 4. Example with confidence intervals
    ax = axes[1, 1]
    idx = np.random.randint(0, vae_samples.shape[0])
    samples = vae_samples[idx]  # (n_samples, seq_len, 1)
    gt = targets[idx].squeeze()

    mean_pred = samples.mean(axis=0).squeeze()
    std_pred = samples.std(axis=0).squeeze()

    weeks = np.arange(len(mean_pred))
    ax.fill_between(
        weeks,
        mean_pred - 2 * std_pred,
        mean_pred + 2 * std_pred,
        color="red",
        alpha=0.2,
        label="95% CI",
    )
    ax.fill_between(
        weeks,
        mean_pred - std_pred,
        mean_pred + std_pred,
        color="red",
        alpha=0.3,
        label="68% CI",
    )
    ax.plot(
        weeks, mean_pred, "D-", color="red", linewidth=2, markersize=6, label="VAE Mean"
    )
    ax.plot(
        weeks,
        gt,
        "o-",
        color="black",
        linewidth=2.5,
        markersize=8,
        label="Ground Truth",
    )

    ax.set_xlabel("Week", fontsize=11)
    ax.set_ylabel("Clicks (Normalized)", fontsize=11)
    ax.set_title("Uncertainty Quantification", fontsize=12, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()
