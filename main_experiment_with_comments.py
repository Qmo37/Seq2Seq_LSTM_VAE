"""
Seq2Seq LSTM vs VAE for Learning Behavior Prediction - Complete Tutorial Script

This script matches the current Seq2Seq_LSTM_VAE.ipynb notebook structure and provides
a complete end-to-end workflow for comparing two sequence generation models on the
Open University Learning Analytics Dataset (OULAD):

- Seq2Seq LSTM: Deterministic single-path prediction
- Seq2Seq VAE: Probabilistic multi-path generation

Key Features:
1. Comprehensive data preprocessing with OULAD dataset
2. Model training with history tracking
3. Detailed evaluation with win-rate analysis
4. Four publication-quality visualizations

Author: Educational ML Project
Dataset: Open University Learning Analytics Dataset (OULAD)
Last Updated: 2025
"""

# =============================================================================
# SECTION 1: IMPORTS AND ENVIRONMENT SETUP
# =============================================================================

import os
import sys
import json
import warnings
import random

# Add src to path
sys.path.append("src")

# Core libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# Suppress warnings
warnings.filterwarnings("ignore")

# Set matplotlib style
plt.style.use("seaborn-v0_8-darkgrid")

print("=" * 80)
print("Seq2Seq LSTM vs VAE Tutorial - Complete Workflow")
print("=" * 80)
print("\nAll imports successful!")

# =============================================================================
# SECTION 2: CONFIGURATION
# =============================================================================

CONFIG = {
    # Data
    "data_path": "data/raw",
    "input_weeks": 4,
    "output_weeks": 2,
    # Training (Fixed)
    "batch_size": 128,
    "learning_rate": 1e-3,
    "epochs": 20,
    "random_seed": 42,
    # Model Architecture
    "hidden_size": 64,
    "latent_dim": 16,
    "num_layers": 1,
    "dropout": 0.0,
    # VAE
    "beta": 1.0,
    # Evaluation
    "n_samples": 20,
    # Device
    "device": "cuda" if torch.cuda.is_available() else "cpu",
}

print("\n" + "=" * 80)
print("CONFIGURATION")
print("=" * 80)
for key, value in CONFIG.items():
    print(f"  {key:20s}: {value}")
print("=" * 80)

# Set random seeds
random.seed(CONFIG["random_seed"])
np.random.seed(CONFIG["random_seed"])
torch.manual_seed(CONFIG["random_seed"])
if torch.cuda.is_available():
    torch.cuda.manual_seed(CONFIG["random_seed"])

device = torch.device(CONFIG["device"])
print(f"\nUsing device: {device}")

# =============================================================================
# SECTION 3: DATA LOADING AND PREPROCESSING
# =============================================================================

print("\n" + "=" * 80)
print("STEP 1: DATA LOADING AND PREPROCESSING")
print("=" * 80)


def load_and_process_oulad():
    """Load and process OULAD dataset into weekly features"""
    print("\nLoading OULAD data...")

    student_info = pd.read_csv("data/raw/studentInfo.csv")
    student_vle = pd.read_csv("data/raw/studentVle.csv")
    student_assessment = pd.read_csv("data/raw/studentAssessment.csv")

    print(
        f"Loaded: Info={student_info.shape}, VLE={student_vle.shape}, Assessment={student_assessment.shape}"
    )

    # Process VLE data
    vle_date_col = "date" if "date" in student_vle.columns else student_vle.columns[1]
    student_vle["week"] = student_vle[vle_date_col] // 7

    # Process assessment data
    if "date_submitted" in student_assessment.columns:
        student_assessment["week"] = student_assessment["date_submitted"] // 7
    elif "date" in student_assessment.columns:
        student_assessment["week"] = student_assessment["date"] // 7
    else:
        student_assessment["week"] = 0

    # Aggregate clicks
    click_col = "sum_click" if "sum_click" in student_vle.columns else "clicks"
    clicks_df = (
        student_vle.groupby(["id_student", "week"])[click_col].sum().reset_index()
    )
    clicks_df.columns = ["id_student", "week", "clicks"]

    # Process scores
    if "score" in student_assessment.columns:
        if student_assessment["score"].dtype == "object":
            score_map = {"Pass": 70, "Fail": 30, "Distinction": 85, "Withdrawn": 0}
            student_assessment["score"] = (
                student_assessment["score"].map(score_map).fillna(0)
            )

        submit_df = (
            student_assessment.groupby(["id_student", "week"])["score"]
            .agg(["count", "mean"])
            .reset_index()
        )
        submit_df.columns = ["id_student", "week", "submit_cnt", "avg_score"]
    else:
        submit_df = pd.DataFrame(
            columns=["id_student", "week", "submit_cnt", "avg_score"]
        )

    # Create complete grid
    all_students = student_info["id_student"].unique()
    all_weeks = []
    for student in all_students:
        for week in range(30):
            all_weeks.append({"id_student": student, "week": week})

    df = pd.DataFrame(all_weeks)

    # Merge features
    df = df.merge(clicks_df, on=["id_student", "week"], how="left")
    if not submit_df.empty:
        df = df.merge(submit_df, on=["id_student", "week"], how="left")
    else:
        df["submit_cnt"] = 0
        df["avg_score"] = 0

    df.fillna(0, inplace=True)
    df = df.sort_values(["id_student", "week"]).reset_index(drop=True)

    # Derived features
    df["has_submit"] = (df["submit_cnt"] > 0).astype(int)
    df["avg_score_sofar"] = (
        df.groupby("id_student")["avg_score"].expanding().mean().values
    )
    df["clicks_diff1"] = df.groupby("id_student")["clicks"].diff().fillna(0)

    print(f"Created weekly features: {df.shape}")
    print(f"Average clicks per week: {df['clicks'].mean():.1f}")

    return df


weekly_df = load_and_process_oulad()

# =============================================================================
# SECTION 4: CREATE SEQUENCES
# =============================================================================

print("\n" + "=" * 80)
print("STEP 2: CREATING SEQUENCES")
print("=" * 80)


def create_sequences(df, input_weeks=4, output_weeks=2):
    """Create sliding window sequences"""
    feature_cols = ["clicks", "has_submit", "avg_score_sofar", "clicks_diff1"]

    X_list = []
    y_list = []
    student_list = []

    print("\nCreating sequences...")
    for student_id, group in tqdm(df.groupby("id_student")):
        group = group.sort_values("week").reset_index(drop=True)

        if len(group) < input_weeks + output_weeks:
            continue

        for i in range(len(group) - input_weeks - output_weeks + 1):
            X_window = group.iloc[i : i + input_weeks][feature_cols].values
            y_window = group.iloc[i + input_weeks : i + input_weeks + output_weeks][
                ["clicks"]
            ].values

            X_list.append(X_window)
            y_list.append(y_window)
            student_list.append(student_id)

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32)
    student_ids = np.array(student_list)

    print(f"Created {len(X)} sequences")
    print(f"Input shape: {X.shape}")
    print(f"Output shape: {y.shape}")

    return X, y, student_ids


def split_and_normalize(X, y, student_ids):
    """Split by students and normalize"""
    print("\nSplitting data by student ID...")

    unique_students = np.unique(student_ids)
    train_students, test_students = train_test_split(
        unique_students, test_size=0.2, random_state=42
    )
    train_students, val_students = train_test_split(
        train_students, test_size=0.1, random_state=42
    )

    train_mask = np.isin(student_ids, train_students)
    val_mask = np.isin(student_ids, val_students)
    test_mask = np.isin(student_ids, test_students)

    X_train, y_train = X[train_mask], y[train_mask]
    X_val, y_val = X[val_mask], y[val_mask]
    X_test, y_test = X[test_mask], y[test_mask]

    print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

    # Normalize
    X_mean = X_train.mean(axis=(0, 1), keepdims=True)
    X_std = X_train.std(axis=(0, 1), keepdims=True) + 1e-8
    y_mean = y_train.mean()
    y_std = y_train.std() + 1e-8

    X_train = (X_train - X_mean) / X_std
    X_val = (X_val - X_mean) / X_std
    X_test = (X_test - X_mean) / X_std

    y_train = (y_train - y_mean) / y_std
    y_val = (y_val - y_mean) / y_std
    y_test = (y_test - y_mean) / y_std

    print("Data normalized successfully")

    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val,
        "X_test": X_test,
        "y_test": y_test,
        "norm_stats": {"y_mean": y_mean, "y_std": y_std},
    }


X, y, student_ids = create_sequences(weekly_df)
data = split_and_normalize(X, y, student_ids)

# =============================================================================
# SECTION 5: DEFINE MODELS AND DATASET
# =============================================================================

print("\n" + "=" * 80)
print("STEP 3: DEFINING MODELS")
print("=" * 80)


class LearningBehaviorDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class Seq2SeqLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.encoder = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.decoder = nn.LSTM(output_size, hidden_size, batch_first=True)
        self.output_proj = nn.Linear(hidden_size, output_size)

    def forward(self, x, tgt_len):
        batch_size = x.size(0)

        _, (hidden, cell) = self.encoder(x)

        decoder_input = torch.zeros(batch_size, 1, self.output_size, device=x.device)
        outputs = []

        for t in range(tgt_len):
            decoder_output, (hidden, cell) = self.decoder(decoder_input, (hidden, cell))
            output = self.output_proj(decoder_output)
            outputs.append(output)
            decoder_input = output

        return torch.cat(outputs, dim=1)


class Seq2SeqVAE(nn.Module):
    def __init__(self, input_size, hidden_size, latent_dim, output_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.latent_dim = latent_dim
        self.output_size = output_size

        self.encoder = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.mu_proj = nn.Linear(hidden_size, latent_dim)
        self.logvar_proj = nn.Linear(hidden_size, latent_dim)

        self.decoder = nn.LSTM(output_size + latent_dim, hidden_size, batch_first=True)
        self.output_proj = nn.Linear(hidden_size, output_size)

    def encode(self, x):
        _, (hidden, _) = self.encoder(x)
        mu = self.mu_proj(hidden[-1])
        logvar = self.logvar_proj(hidden[-1])
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, tgt_len):
        batch_size = z.size(0)
        decoder_input = torch.zeros(batch_size, 1, self.output_size, device=z.device)
        z_expanded = z.unsqueeze(1)

        outputs = []
        hidden = None

        for t in range(tgt_len):
            decoder_input_with_z = torch.cat([decoder_input, z_expanded], dim=-1)
            decoder_output, hidden = self.decoder(decoder_input_with_z, hidden)
            output = self.output_proj(decoder_output)
            outputs.append(output)
            decoder_input = output

        return torch.cat(outputs, dim=1)

    def forward(self, x, tgt_len):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z, tgt_len)
        return recon, mu, logvar


print("Models defined successfully!")

# Create DataLoaders
print("\nCreating DataLoaders...")
train_dataset = LearningBehaviorDataset(data["X_train"], data["y_train"])
val_dataset = LearningBehaviorDataset(data["X_val"], data["y_val"])
test_dataset = LearningBehaviorDataset(data["X_test"], data["y_test"])

train_loader = DataLoader(train_dataset, batch_size=CONFIG["batch_size"], shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=CONFIG["batch_size"], shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=CONFIG["batch_size"], shuffle=False)

print(
    f"Train batches: {len(train_loader)}, Val: {len(val_loader)}, Test: {len(test_loader)}"
)

# =============================================================================
# SECTION 6: TRAIN LSTM MODEL
# =============================================================================

print("\n" + "=" * 80)
print("STEP 4: TRAINING LSTM MODEL")
print("=" * 80)

lstm_model = Seq2SeqLSTM(
    input_size=4, hidden_size=CONFIG["hidden_size"], output_size=1
).to(device)
optimizer = optim.Adam(lstm_model.parameters(), lr=CONFIG["learning_rate"])

lstm_train_losses = []
lstm_val_losses = []

print("\nTraining LSTM...")
for epoch in range(CONFIG["epochs"]):
    # Training
    lstm_model.train()
    total_train_loss = 0

    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)

        optimizer.zero_grad()
        outputs = lstm_model(X_batch, CONFIG["output_weeks"])
        loss = F.mse_loss(outputs, y_batch)
        loss.backward()
        optimizer.step()

        total_train_loss += loss.item()

    avg_train_loss = total_train_loss / len(train_loader)
    lstm_train_losses.append(avg_train_loss)

    # Validation
    lstm_model.eval()
    total_val_loss = 0
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = lstm_model(X_batch, CONFIG["output_weeks"])
            loss = F.mse_loss(outputs, y_batch)
            total_val_loss += loss.item()

    avg_val_loss = total_val_loss / len(val_loader)
    lstm_val_losses.append(avg_val_loss)

    if (epoch + 1) % 5 == 0:
        print(
            f"LSTM Epoch {epoch + 1}/{CONFIG['epochs']}: Train={avg_train_loss:.6f}, Val={avg_val_loss:.6f}"
        )

print("LSTM training completed!")

# =============================================================================
# SECTION 7: TRAIN VAE MODEL
# =============================================================================

print("\n" + "=" * 80)
print("STEP 5: TRAINING VAE MODEL")
print("=" * 80)

vae_model = Seq2SeqVAE(
    input_size=4,
    hidden_size=CONFIG["hidden_size"],
    latent_dim=CONFIG["latent_dim"],
    output_size=1,
).to(device)
optimizer = optim.Adam(vae_model.parameters(), lr=CONFIG["learning_rate"])

vae_train_losses = []
vae_val_losses = []
vae_train_mse = []
vae_train_kld = []

print("\nTraining VAE...")
for epoch in range(CONFIG["epochs"]):
    # Training
    vae_model.train()
    total_train_loss = 0
    total_mse = 0
    total_kld = 0

    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)

        optimizer.zero_grad()
        recon, mu, logvar = vae_model(X_batch, CONFIG["output_weeks"])

        mse_loss = F.mse_loss(recon, y_batch)
        kl_loss = (
            -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / y_batch.numel()
        )
        loss = mse_loss + CONFIG["beta"] * kl_loss

        loss.backward()
        optimizer.step()

        total_train_loss += loss.item()
        total_mse += mse_loss.item()
        total_kld += kl_loss.item()

    avg_train_loss = total_train_loss / len(train_loader)
    avg_mse = total_mse / len(train_loader)
    avg_kld = total_kld / len(train_loader)

    vae_train_losses.append(avg_train_loss)
    vae_train_mse.append(avg_mse)
    vae_train_kld.append(avg_kld)

    # Validation
    vae_model.eval()
    total_val_loss = 0
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            recon, mu, logvar = vae_model(X_batch, CONFIG["output_weeks"])
            mse_loss = F.mse_loss(recon, y_batch)
            kl_loss = (
                -0.5
                * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
                / y_batch.numel()
            )
            loss = mse_loss + CONFIG["beta"] * kl_loss
            total_val_loss += loss.item()

    avg_val_loss = total_val_loss / len(val_loader)
    vae_val_losses.append(avg_val_loss)

    if (epoch + 1) % 5 == 0:
        print(
            f"VAE Epoch {epoch + 1}/{CONFIG['epochs']}: Train={avg_train_loss:.6f} (MSE={avg_mse:.6f}, KLD={avg_kld:.6f}), Val={avg_val_loss:.6f}"
        )

print("VAE training completed!")

# =============================================================================
# SECTION 8: EVALUATE MODELS
# =============================================================================

print("\n" + "=" * 80)
print("STEP 6: EVALUATING MODELS")
print("=" * 80)


def evaluate_models_detailed():
    """Comprehensive model evaluation with detailed analysis"""
    lstm_model.eval()
    vae_model.eval()

    lstm_predictions = []
    vae_predictions = []
    vae_samples = []
    targets = []

    print("\nEvaluating models...")

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            lstm_pred = lstm_model(X_batch, CONFIG["output_weeks"])
            lstm_predictions.append(lstm_pred.cpu().numpy())

            vae_pred, _, _ = vae_model(X_batch, CONFIG["output_weeks"])
            vae_predictions.append(vae_pred.cpu().numpy())

            mu, logvar = vae_model.encode(X_batch)
            batch_samples = []
            for _ in range(CONFIG["n_samples"]):
                z = vae_model.reparameterize(mu, logvar)
                sample = vae_model.decode(z, CONFIG["output_weeks"])
                batch_samples.append(sample.cpu().numpy())
            vae_samples.append(np.array(batch_samples))

            targets.append(y_batch.cpu().numpy())

    lstm_preds = np.concatenate(lstm_predictions)
    vae_preds = np.concatenate(vae_predictions)
    vae_samples = np.concatenate(vae_samples, axis=1)
    all_targets = np.concatenate(targets)

    # Calculate per-sample MSE
    lstm_mse_per_sample = np.mean((lstm_preds - all_targets) ** 2, axis=(1, 2))
    vae_mse_per_sample = np.mean((vae_preds - all_targets) ** 2, axis=(1, 2))

    # VAE Best-of-N per sample
    vae_best_mse_per_sample = []
    for i in range(len(all_targets)):
        sample_mses = []
        for j in range(CONFIG["n_samples"]):
            mse = np.mean((vae_samples[j, i] - all_targets[i]) ** 2)
            sample_mses.append(mse)
        vae_best_mse_per_sample.append(min(sample_mses))
    vae_best_mse_per_sample = np.array(vae_best_mse_per_sample)

    # Overall metrics
    lstm_mse = lstm_mse_per_sample.mean()
    vae_mse = vae_mse_per_sample.mean()
    best_of_n_mse = vae_best_mse_per_sample.mean()

    diversity = np.std(vae_samples, axis=0).mean()

    lower = np.percentile(vae_samples, 2.5, axis=0)
    upper = np.percentile(vae_samples, 97.5, axis=0)
    coverage = np.mean((all_targets >= lower) & (all_targets <= upper))

    # Top-5 Analysis
    improvement = lstm_mse_per_sample - vae_best_mse_per_sample
    top5_indices = np.argsort(-improvement)[:5]

    print("\n" + "=" * 80)
    print("Top-5 Cases (VAE Best >> LSTM)")
    print("=" * 80)

    top5_df = pd.DataFrame(
        {
            "idx": top5_indices,
            "LSTM_MSE": lstm_mse_per_sample[top5_indices],
            "VAE_best_MSE": vae_best_mse_per_sample[top5_indices],
            "Improvement": improvement[top5_indices],
        }
    )
    print(top5_df.to_string(index=False))

    # Win-rate Analysis
    print("\n" + "=" * 80)
    print("Win-rate by Improvement Bucket")
    print("=" * 80)

    buckets = [
        ("VAE worse >1000", lambda x: x < -1000),
        ("VAE worse 200~1000", lambda x: (x >= -1000) & (x < -200)),
        ("VAE worse 50~200", lambda x: (x >= -200) & (x < -50)),
        ("VAE worse 10~50", lambda x: (x >= -50) & (x < -10)),
        ("VAE slightly worse <10", lambda x: (x >= -10) & (x < 0)),
        ("Tie ±10", lambda x: (x >= 0) & (x < 10)),
        ("VAE better 10~50", lambda x: (x >= 10) & (x < 50)),
        ("VAE better 50~200", lambda x: (x >= 50) & (x < 200)),
        ("VAE better 200~1000", lambda x: (x >= 200) & (x < 1000)),
        ("VAE much better >1000", lambda x: x >= 1000),
    ]

    bucket_stats = []
    for bucket_name, condition in buckets:
        mask = condition(improvement)
        count = np.sum(mask)
        ratio = count / len(improvement) if len(improvement) > 0 else 0
        bucket_stats.append(
            {"Bucket": bucket_name, "Count": count, "Ratio": f"{ratio:.4f}"}
        )

    bucket_df = pd.DataFrame(bucket_stats)
    print(bucket_df.to_string(index=False))

    # Final Results
    print("\n" + "=" * 80)
    print("FINAL EVALUATION RESULTS")
    print("=" * 80)

    y_std = data["norm_stats"]["y_std"]
    lstm_mse_original = lstm_mse * (y_std**2)
    vae_mse_original = vae_mse * (y_std**2)
    best_of_n_original = best_of_n_mse * (y_std**2)

    print(f"LSTM MSE:               {lstm_mse_original:.4f}")
    print(
        f"VAE Best-of-N MSE:      {best_of_n_original:.4f}  (N={CONFIG['n_samples']})"
    )
    print(f"VAE Diversity (std):    {diversity:.4f}")
    print(f"VAE Coverage (95% CI):  {coverage:.4f}")
    print("=" * 80)

    return {
        "lstm_mse": lstm_mse,
        "vae_mse": vae_mse,
        "vae_best_of_n_mse": best_of_n_mse,
        "vae_diversity": diversity,
        "vae_coverage": coverage,
        "top5_df": top5_df,
        "bucket_df": bucket_df,
        "lstm_preds": lstm_preds,
        "vae_all_samples": vae_samples,
        "y_test": all_targets,
        "lstm_mse_per_sample": lstm_mse_per_sample,
    }


eval_output = evaluate_models_detailed()

# Extract metrics
results = {
    "lstm_mse": eval_output["lstm_mse"],
    "vae_mse": eval_output["vae_mse"],
    "vae_best_of_n_mse": eval_output["vae_best_of_n_mse"],
    "vae_diversity": eval_output["vae_diversity"],
    "vae_coverage": eval_output["vae_coverage"],
}

# Expose for visualizations
lstm_preds = eval_output["lstm_preds"]
vae_all_samples = eval_output["vae_all_samples"]
y_test = eval_output["y_test"]
lstm_mse_per_sample = eval_output["lstm_mse_per_sample"]

print("\nEvaluation complete!")

# =============================================================================
# SECTION 9: VISUALIZATION 1 - TRAINING CURVES
# =============================================================================

print("\n" + "=" * 80)
print("STEP 7: GENERATING VISUALIZATIONS")
print("=" * 80)

os.makedirs("results/figures", exist_ok=True)

print("\n1. Training Curves...")

fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# LSTM
axes[0].plot(
    range(1, 21), lstm_train_losses, "b-", linewidth=2, label="Train Loss", marker="o"
)
axes[0].plot(
    range(1, 21), lstm_val_losses, "b--", linewidth=2, label="Val Loss", marker="s"
)
axes[0].set_xlabel("Epoch", fontsize=12)
axes[0].set_ylabel("Loss (MSE)", fontsize=12)
axes[0].set_title("LSTM Training Curves", fontsize=14, fontweight="bold")
axes[0].legend(fontsize=11)
axes[0].grid(True, alpha=0.3)

# VAE
axes[1].plot(
    range(1, 21),
    vae_train_losses,
    "r-",
    linewidth=2,
    label="Train Loss (Total)",
    marker="o",
)
axes[1].plot(
    range(1, 21),
    vae_val_losses,
    "r--",
    linewidth=2,
    label="Val Loss (Total)",
    marker="s",
)
axes[1].plot(
    range(1, 21),
    vae_train_mse,
    "g-",
    linewidth=1.5,
    alpha=0.7,
    label="Train MSE",
    marker="^",
)
axes[1].plot(
    range(1, 21),
    vae_train_kld,
    "orange",
    linewidth=1.5,
    alpha=0.7,
    label="Train KLD",
    marker="v",
)
axes[1].set_xlabel("Epoch", fontsize=12)
axes[1].set_ylabel("Loss", fontsize=12)
axes[1].set_title("VAE Training Curves", fontsize=14, fontweight="bold")
axes[1].legend(fontsize=10)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("results/figures/training_curves.png", dpi=300, bbox_inches="tight")
plt.close()

print("   Saved: training_curves.png")

# =============================================================================
# SECTION 10: VISUALIZATION 2 - SAMPLE PREDICTIONS
# =============================================================================

print("2. Sample Predictions...")

np.random.seed(42)
num_samples = 6
sample_indices = np.random.choice(len(lstm_preds), num_samples, replace=False)

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()

for idx, sample_idx in enumerate(sample_indices):
    ax = axes[idx]

    gt = y_test[sample_idx].flatten()
    lstm_pred = lstm_preds[sample_idx].flatten()
    vae_samples_for_this = vae_all_samples[:, sample_idx]

    if vae_samples_for_this.ndim == 3:
        vae_samples_for_this = vae_samples_for_this.squeeze(-1)

    weeks = np.arange(1, len(gt) + 1)

    # Plot VAE samples
    for i in range(20):
        ax.plot(weeks, vae_samples_for_this[i], color="red", alpha=0.15, linewidth=1)

    # Plot predictions
    ax.plot(
        weeks, lstm_pred, "b-", linewidth=2.5, label="LSTM", marker="s", markersize=8
    )
    vae_mean = vae_samples_for_this.mean(axis=0)
    ax.plot(
        weeks, vae_mean, "r-", linewidth=2.5, label="VAE Mean", marker="^", markersize=8
    )
    ax.plot(
        weeks, gt, "k-", linewidth=3, label="Ground Truth", marker="o", markersize=10
    )

    # Calculate MSE
    lstm_mse = np.mean((gt - lstm_pred) ** 2)
    vae_mse = np.mean((gt - vae_mean) ** 2)
    vae_best_mse = min(
        [np.mean((gt - vae_samples_for_this[i]) ** 2) for i in range(20)]
    )

    ax.set_xlabel("Week", fontsize=11)
    ax.set_ylabel("Clicks (Normalized)", fontsize=11)
    ax.set_title(
        f"Sample {sample_idx}\nLSTM: {lstm_mse:.3f} | VAE: {vae_mse:.3f} | VAE Best: {vae_best_mse:.3f}",
        fontsize=10,
    )
    ax.legend(fontsize=9, loc="best")
    ax.grid(True, alpha=0.3)
    ax.set_xticks([1, 2])

plt.suptitle(
    "Prediction Comparison: GT vs LSTM vs VAE", fontsize=16, fontweight="bold", y=1.00
)
plt.tight_layout()
plt.savefig("results/figures/prediction_comparison.png", dpi=300, bbox_inches="tight")
plt.close()

print("   Saved: prediction_comparison.png")

# =============================================================================
# SECTION 11: VISUALIZATION 3 - DIVERSITY ANALYSIS
# =============================================================================

print("3. Diversity Analysis...")

# Calculate diversity per sample
diversity_per_sample = []
for i in range(vae_all_samples.shape[1]):
    samples = vae_all_samples[:, i]
    if samples.ndim == 3:
        samples = samples.squeeze(-1)
    diversity = np.std(samples)
    diversity_per_sample.append(diversity)

diversity_per_sample = np.array(diversity_per_sample)

fig, axes = plt.subplots(2, 2, figsize=(16, 10))

# Histogram
axes[0, 0].hist(
    diversity_per_sample, bins=50, color="orange", alpha=0.7, edgecolor="black"
)
axes[0, 0].axvline(
    diversity_per_sample.mean(),
    color="red",
    linestyle="--",
    linewidth=2,
    label=f"Mean = {diversity_per_sample.mean():.4f}",
)
axes[0, 0].set_xlabel("Diversity (Std)", fontsize=12)
axes[0, 0].set_ylabel("Frequency", fontsize=12)
axes[0, 0].set_title("Distribution of VAE Diversity", fontsize=13, fontweight="bold")
axes[0, 0].legend(fontsize=11)
axes[0, 0].grid(True, alpha=0.3)

# Box plot
axes[0, 1].boxplot(
    [diversity_per_sample],
    vert=True,
    patch_artist=True,
    boxprops=dict(facecolor="lightblue", alpha=0.7),
)
axes[0, 1].set_ylabel("Diversity (Std)", fontsize=12)
axes[0, 1].set_title("VAE Diversity Box Plot", fontsize=13, fontweight="bold")
axes[0, 1].grid(True, alpha=0.3, axis="y")

# Scatter: Diversity vs LSTM MSE
axes[1, 0].scatter(
    diversity_per_sample, lstm_mse_per_sample, alpha=0.3, c="purple", s=10
)
axes[1, 0].set_xlabel("VAE Diversity (Std)", fontsize=12)
axes[1, 0].set_ylabel("LSTM MSE", fontsize=12)
axes[1, 0].set_title("VAE Diversity vs LSTM MSE", fontsize=13, fontweight="bold")
axes[1, 0].grid(True, alpha=0.3)

# Diversity bins
diversity_bins = [0, 0.1, 0.2, 0.5, 1.0, float("inf")]
diversity_labels = ["0-0.1", "0.1-0.2", "0.2-0.5", "0.5-1.0", ">1.0"]
diversity_binned = pd.cut(
    diversity_per_sample, bins=diversity_bins, labels=diversity_labels
)
diversity_counts = diversity_binned.value_counts().sort_index()

axes[1, 1].bar(
    range(len(diversity_counts)),
    diversity_counts.values,
    color=["red", "orange", "yellow", "lightgreen", "green"],
    alpha=0.7,
    edgecolor="black",
)
axes[1, 1].set_xlabel("Diversity Range", fontsize=12)
axes[1, 1].set_ylabel("Number of Samples", fontsize=12)
axes[1, 1].set_title("VAE Diversity by Range", fontsize=13, fontweight="bold")
axes[1, 1].set_xticks(range(len(diversity_counts)))
axes[1, 1].set_xticklabels(diversity_labels, rotation=45)
axes[1, 1].grid(True, alpha=0.3, axis="y")

plt.suptitle("VAE Diversity Analysis", fontsize=16, fontweight="bold", y=0.995)
plt.tight_layout()
plt.savefig("results/figures/diversity_analysis.png", dpi=300, bbox_inches="tight")
plt.close()

print("   Saved: diversity_analysis.png")

# =============================================================================
# SECTION 12: VISUALIZATION 4 - COMPREHENSIVE DASHBOARD
# =============================================================================

print("4. Comprehensive Dashboard...")

fig = plt.figure(figsize=(20, 12))
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

# 1. MSE Comparison
ax1 = fig.add_subplot(gs[0, 0])
mse_data = [results["lstm_mse"], results["vae_mse"], results["vae_best_of_n_mse"]]
bars = ax1.bar(
    ["LSTM\nMSE", "VAE\nMean", "VAE\nBest-of-N"],
    mse_data,
    color=["blue", "red", "green"],
    alpha=0.7,
    edgecolor="black",
    linewidth=2,
)
ax1.set_ylabel("MSE", fontsize=12, fontweight="bold")
ax1.set_title("MSE Comparison", fontsize=13, fontweight="bold")
ax1.grid(True, alpha=0.3, axis="y")
for bar, value in zip(bars, mse_data):
    ax1.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height(),
        f"{value:.4f}",
        ha="center",
        va="bottom",
        fontsize=10,
        fontweight="bold",
    )

# 2. VAE Metrics
ax2 = fig.add_subplot(gs[0, 1])
vae_metrics = [results["vae_diversity"], results["vae_coverage"]]
bars = ax2.bar(
    ["Diversity\n(Std)", "Coverage\n(95% CI)"],
    vae_metrics,
    color=["orange", "purple"],
    alpha=0.7,
    edgecolor="black",
    linewidth=2,
)
ax2.set_ylabel("Value", fontsize=12, fontweight="bold")
ax2.set_title("VAE Metrics", fontsize=13, fontweight="bold")
ax2.grid(True, alpha=0.3, axis="y")
for bar, value in zip(bars, vae_metrics):
    ax2.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height(),
        f"{value:.3f}",
        ha="center",
        va="bottom",
        fontsize=10,
        fontweight="bold",
    )

# 3. Win Rate
ax3 = fig.add_subplot(gs[0, 2])
vae_best_mse_per_sample = []
for i in range(len(y_test)):
    sample_mses = [np.mean((y_test[i] - vae_all_samples[j, i]) ** 2) for j in range(20)]
    vae_best_mse_per_sample.append(min(sample_mses))
vae_best_mse_per_sample = np.array(vae_best_mse_per_sample)

lstm_win = (lstm_mse_per_sample < vae_best_mse_per_sample).sum()
vae_win = len(y_test) - lstm_win
ax3.pie(
    [lstm_win, vae_win],
    labels=[
        f"LSTM\n({100 * lstm_win / len(y_test):.1f}%)",
        f"VAE\n({100 * vae_win / len(y_test):.1f}%)",
    ],
    colors=["blue", "red"],
    autopct="%1.1f%%",
    startangle=90,
)
ax3.set_title("Win Rate: Best Prediction", fontsize=13, fontweight="bold")

# 4. Training Curves
ax4 = fig.add_subplot(gs[1, :])
epochs = range(1, 21)
ax4.plot(epochs, lstm_train_losses, "b-", linewidth=2, label="LSTM Train", marker="o")
ax4.plot(epochs, lstm_val_losses, "b--", linewidth=2, label="LSTM Val", marker="s")
ax4.plot(epochs, vae_train_losses, "r-", linewidth=2, label="VAE Train", marker="o")
ax4.plot(epochs, vae_val_losses, "r--", linewidth=2, label="VAE Val", marker="s")
ax4.set_xlabel("Epoch", fontsize=12, fontweight="bold")
ax4.set_ylabel("Loss", fontsize=12, fontweight="bold")
ax4.set_title("Training Curves", fontsize=14, fontweight="bold")
ax4.legend(fontsize=11)
ax4.grid(True, alpha=0.3)

# 5-6. Sample Predictions
for i in range(2):
    ax = fig.add_subplot(gs[2, i])
    sample_idx = sample_indices[i]

    gt = y_test[sample_idx].flatten()
    lstm_pred = lstm_preds[sample_idx].flatten()
    vae_samples_for_this = vae_all_samples[:, sample_idx]
    if vae_samples_for_this.ndim == 3:
        vae_samples_for_this = vae_samples_for_this.squeeze(-1)

    weeks = np.arange(1, len(gt) + 1)

    for j in range(20):
        ax.plot(weeks, vae_samples_for_this[j], "r-", alpha=0.15, linewidth=1)

    ax.plot(
        weeks, lstm_pred, "b-", linewidth=2.5, label="LSTM", marker="s", markersize=8
    )
    ax.plot(
        weeks,
        vae_samples_for_this.mean(axis=0),
        "r-",
        linewidth=2.5,
        label="VAE Mean",
        marker="^",
        markersize=8,
    )
    ax.plot(weeks, gt, "k-", linewidth=3, label="GT", marker="o", markersize=10)

    ax.set_xlabel("Week", fontsize=11, fontweight="bold")
    ax.set_ylabel("Clicks", fontsize=11, fontweight="bold")
    ax.set_title(
        f"Example {i + 1} (Sample {sample_idx})", fontsize=12, fontweight="bold"
    )
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xticks([1, 2])

# 7. Summary
ax7 = fig.add_subplot(gs[2, 2])
ax7.axis("off")
summary_text = f"""
MODEL COMPARISON

LSTM:
  MSE: {results["lstm_mse"]:.4f}
  Win Rate: {100 * lstm_win / len(y_test):.1f}%

VAE:
  Mean MSE: {results["vae_mse"]:.4f}
  Best-of-N: {results["vae_best_of_n_mse"]:.4f}
  Diversity: {results["vae_diversity"]:.4f}
  Coverage: {results["vae_coverage"]:.1%}
  Win Rate: {100 * vae_win / len(y_test):.1f}%

CONCLUSION:
{"VAE Best-of-N wins" if results["vae_best_of_n_mse"] < results["lstm_mse"] else "LSTM wins"}
"""
ax7.text(
    0.1,
    0.5,
    summary_text,
    fontsize=11,
    verticalalignment="center",
    family="monospace",
    fontweight="bold",
    bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8),
)

plt.suptitle(
    "Comprehensive Model Comparison Dashboard", fontsize=18, fontweight="bold", y=0.995
)
plt.savefig(
    "results/figures/comprehensive_comparison.png", dpi=300, bbox_inches="tight"
)
plt.close()

print("   Saved: comprehensive_comparison.png")

# =============================================================================
# SECTION 13: SAVE RESULTS
# =============================================================================

print("\n" + "=" * 80)
print("STEP 8: SAVING RESULTS")
print("=" * 80)

os.makedirs("results/checkpoints", exist_ok=True)

# Save models
torch.save(lstm_model.state_dict(), "results/checkpoints/lstm_model.pt")
torch.save(vae_model.state_dict(), "results/checkpoints/vae_model.pt")
print("Models saved to results/checkpoints/")

# Save results
results_summary = {
    "config": CONFIG,
    "lstm": {
        "mse": float(results["lstm_mse"]),
        "n_parameters": sum(p.numel() for p in lstm_model.parameters()),
    },
    "vae": {
        "mse": float(results["vae_mse"]),
        "best_of_n_mse": float(results["vae_best_of_n_mse"]),
        "diversity": float(results["vae_diversity"]),
        "coverage": float(results["vae_coverage"]),
        "n_parameters": sum(p.numel() for p in vae_model.parameters()),
    },
}

with open("results/results_summary.json", "w") as f:
    json.dump(results_summary, f, indent=2)

print("Results saved to results/results_summary.json")

# =============================================================================
# FINAL SUMMARY
# =============================================================================

print("\n" + "=" * 80)
print("EXPERIMENT COMPLETE!")
print("=" * 80)

print("\nGenerated Files:")
print("  1. results/figures/training_curves.png")
print("  2. results/figures/prediction_comparison.png")
print("  3. results/figures/diversity_analysis.png")
print("  4. results/figures/comprehensive_comparison.png")
print("  5. results/checkpoints/lstm_model.pt")
print("  6. results/checkpoints/vae_model.pt")
print("  7. results/results_summary.json")

print("\nKey Findings:")
print(f"  LSTM MSE:            {results['lstm_mse']:.6f}")
print(f"  VAE Best-of-N MSE:   {results['vae_best_of_n_mse']:.6f}")
print(f"  VAE Diversity:       {results['vae_diversity']:.6f}")
print(f"  VAE Coverage:        {results['vae_coverage']:.4f}")

if results["vae_best_of_n_mse"] < results["lstm_mse"]:
    improvement = (
        (results["lstm_mse"] - results["vae_best_of_n_mse"]) / results["lstm_mse"] * 100
    )
    print(f"\n  VAE Best-of-N beats LSTM by {improvement:.1f}%")
else:
    print(f"\n  LSTM outperforms VAE Best-of-N")

print("\n" + "=" * 80)
print("Tutorial Complete!")
print("=" * 80)
