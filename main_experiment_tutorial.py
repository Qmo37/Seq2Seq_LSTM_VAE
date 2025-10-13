"""
Seq2Seq LSTM vs VAE for Learning Behavior Prediction - Tutorial Script

This script compares two sequence generation models on the Open University
Learning Analytics Dataset (OULAD):
- Seq2Seq LSTM: Deterministic single-path prediction
- Seq2Seq VAE: Probabilistic multi-path generation

The tutorial is structured to help you understand:
1. How to load and preprocess educational time-series data
2. How to implement and train sequence-to-sequence models
3. How to evaluate deterministic vs probabilistic models
4. How to interpret and visualize the results

Author: Educational ML Project
Dataset: Open University Learning Analytics Dataset (OULAD)
"""

# =============================================================================
# SECTION 1: IMPORTS AND ENVIRONMENT SETUP
# =============================================================================

# Standard library imports for system operations
import os
import sys
import json

# Add the 'src' directory to Python path so we can import our custom modules
# This allows us to use 'from data import ...' instead of 'from src.data import ...'
sys.path.append('src')

# Core scientific computing libraries
import torch  # PyTorch for deep learning
import numpy as np  # Numerical operations
import pandas as pd  # Data manipulation and analysis
import matplotlib.pyplot as plt  # Visualization

# PyTorch utilities for data handling
from torch.utils.data import DataLoader

# Import our custom modules from the src/ directory
# These modules contain the core functionality for our experiment
from data import load_and_preprocess_data, LearningBehaviorDataset
from models import Seq2SeqLSTM, Seq2SeqVAE
from utils import (
    train_lstm,  # Training loop for LSTM model
    train_vae,   # Training loop for VAE model
    set_seed,    # Set random seeds for reproducibility
    evaluate_model,  # Comprehensive model evaluation
    plot_training_curves,  # Visualize training progress
    plot_comparison,       # Compare model performance
    plot_diversity_analysis  # Analyze VAE diversity
)

# Set matplotlib style for better-looking plots
# 'seaborn-v0_8-darkgrid' provides a professional, publication-ready appearance
plt.style.use('seaborn-v0_8-darkgrid')

print("="*80)
print("Seq2Seq LSTM vs VAE Tutorial")
print("="*80)
print("\nAll imports successful!")

# =============================================================================
# SECTION 2: CONFIGURATION AND HYPERPARAMETERS
# =============================================================================

"""
Configuration Dictionary Explanation:

This dictionary contains all hyperparameters for our experiment. Some are fixed
(as per assignment requirements), while others can be adjusted for experimentation.

FIXED HYPERPARAMETERS (DO NOT CHANGE):
- batch_size: 128 - Number of sequences processed together
- learning_rate: 1e-3 - Step size for gradient descent
- random_seed: 42 - Ensures reproducibility
- input_weeks: 4 - Number of past weeks used for prediction
- output_weeks: 2 - Number of future weeks to predict

ADJUSTABLE HYPERPARAMETERS (CAN EXPERIMENT):
- epochs: Number of training passes through the dataset
- hidden_size: LSTM hidden state dimension (larger = more capacity)
- latent_dim: VAE latent space dimension (controls bottleneck)
- beta: VAE KL divergence weight (controls diversity vs accuracy trade-off)
- n_samples: Number of VAE samples for evaluation
"""

CONFIG = {
    # -------------------------------------------------------------------------
    # Data Configuration
    # -------------------------------------------------------------------------
    'data_path': 'data/raw',  # Directory containing OULAD CSV files
    'input_weeks': 4,   # How many weeks of history to use as input
    'output_weeks': 2,  # How many weeks into the future to predict

    # -------------------------------------------------------------------------
    # Training Configuration (Fixed Requirements)
    # -------------------------------------------------------------------------
    'batch_size': 128,  # Number of sequences per training batch
                        # Larger = faster but more memory; smaller = slower but less memory

    'learning_rate': 1e-3,  # Adam optimizer learning rate (0.001)
                            # Controls how big the parameter update steps are

    'epochs': 20,  # Number of complete passes through training data
                   # ADJUSTABLE: Try 5-30+ depending on convergence

    'random_seed': 42,  # Random seed for reproducibility
                        # Using same seed = same results every run

    # -------------------------------------------------------------------------
    # Model Architecture (Adjustable)
    # -------------------------------------------------------------------------
    'hidden_size': 64,  # Dimension of LSTM hidden state
                        # ADJUSTABLE: Larger (128, 256) = more capacity but slower
                        # Smaller (32) = faster but less expressive

    'latent_dim': 16,  # VAE latent space dimension
                       # ADJUSTABLE: Controls bottleneck size
                       # Larger = more information preserved
                       # Smaller = stronger compression/regularization

    'num_layers': 1,  # Number of stacked LSTM layers
                      # ADJUSTABLE: More layers = deeper model

    'dropout': 0.0,  # Dropout probability (0 = no dropout)
                     # ADJUSTABLE: Use 0.1-0.3 to prevent overfitting

    # -------------------------------------------------------------------------
    # VAE-Specific Configuration
    # -------------------------------------------------------------------------
    'beta': 1.0,  # Weight for KL divergence loss in VAE
                  # ADJUSTABLE: This is crucial for diversity vs accuracy trade-off
                  # beta < 1.0: Prioritize reconstruction (less diversity)
                  # beta = 1.0: Standard VAE (balanced)
                  # beta > 1.0: Prioritize diversity (worse reconstruction)

    # -------------------------------------------------------------------------
    # Evaluation Configuration
    # -------------------------------------------------------------------------
    'n_samples': 20,  # How many samples to generate per input for VAE evaluation
                      # Used for Best-of-N metric and diversity analysis
                      # ADJUSTABLE: More samples = better Best-of-N but slower

    # -------------------------------------------------------------------------
    # Device Configuration
    # -------------------------------------------------------------------------
    'device': 'cuda' if torch.cuda.is_available() else 'cpu'  # Use GPU if available
}

# Print configuration for transparency
print("\n" + "="*80)
print("EXPERIMENT CONFIGURATION")
print("="*80)
for key, value in CONFIG.items():
    print(f"  {key:20s}: {value}")
print("="*80)

# =============================================================================
# SECTION 3: DATA LOADING AND PREPROCESSING
# =============================================================================

"""
Data Pipeline Overview:

The OULAD dataset contains raw student interaction logs. We transform this into
sequences suitable for time-series prediction:

1. Load CSV files: studentInfo, studentVle, studentAssessment
2. Aggregate daily logs into weekly features:
   - clicks: Total weekly clicks from VLE interactions
   - has_submit: Binary indicator of submission (0 or 1)
   - avg_score_sofar: Cumulative average score up to current week
   - clicks_diff1: First-order difference of clicks
3. Create sequences using sliding window:
   - Input X: 4 consecutive weeks × 4 features = (batch, 4, 4)
   - Output y: Next 2 weeks of clicks only = (batch, 2, 1)
4. Split by student ID (not randomly!) to prevent data leakage
5. Normalize using training set statistics only
"""

print("\n" + "="*80)
print("STEP 1: DATA LOADING AND PREPROCESSING")
print("="*80)

# Set random seed BEFORE any random operations to ensure reproducibility
# This affects: data splitting, weight initialization, dropout, etc.
set_seed(CONFIG['random_seed'])
print(f"\nRandom seed set to {CONFIG['random_seed']} for reproducibility")

# Load and preprocess the OULAD dataset
# This function handles all the complex preprocessing steps mentioned above
print("\nLoading and preprocessing OULAD dataset...")
print("  - Reading CSV files from data/raw/")
print("  - Aggregating daily logs into weekly features")
print("  - Creating sliding window sequences")
print("  - Splitting by student ID (train/val/test)")
print("  - Normalizing features using training statistics")

data = load_and_preprocess_data(
    data_path=CONFIG['data_path'],
    input_weeks=CONFIG['input_weeks'],
    output_weeks=CONFIG['output_weeks'],
    random_seed=CONFIG['random_seed']
)

# Display dataset shapes to verify preprocessing
# Expected shapes:
# X: (num_sequences, input_weeks, num_features) = (N, 4, 4)
# y: (num_sequences, output_weeks, 1) = (N, 2, 1)
print("\n" + "-"*80)
print("Dataset Shapes:")
print("-"*80)
print(f"  Training:   X={data['X_train'].shape}, y={data['y_train'].shape}")
print(f"  Validation: X={data['X_val'].shape}, y={data['y_val'].shape}")
print(f"  Test:       X={data['X_test'].shape}, y={data['y_test'].shape}")
print("-"*80)

# Interpretation guide for the shapes
print("\nShape Interpretation:")
print(f"  - Each sequence has {data['X_train'].shape[1]} weeks of input")
print(f"  - Each week has {data['X_train'].shape[2]} features")
print(f"  - We predict {data['y_train'].shape[1]} weeks into the future")
print(f"  - We predict {data['y_train'].shape[2]} feature(s) (clicks only)")

# =============================================================================
# SECTION 4: CREATE PYTORCH DATALOADERS
# =============================================================================

"""
DataLoader Explanation:

PyTorch DataLoaders handle batching and shuffling of data during training.
They automatically:
1. Group sequences into batches for efficient GPU computation
2. Shuffle training data each epoch to prevent overfitting
3. Load data in parallel using multiple workers (optional)

Why shuffle training but not validation/test?
- Training: Shuffling prevents the model from learning order-dependent patterns
- Val/Test: We want consistent evaluation, so no shuffling needed
"""

print("\n" + "="*80)
print("STEP 2: CREATING PYTORCH DATALOADERS")
print("="*80)

# Wrap numpy arrays in PyTorch Dataset objects
# This provides a standard interface for DataLoader
train_dataset = LearningBehaviorDataset(data['X_train'], data['y_train'])
val_dataset = LearningBehaviorDataset(data['X_val'], data['y_val'])
test_dataset = LearningBehaviorDataset(data['X_test'], data['y_test'])

# Create DataLoaders for efficient batching
# shuffle=True for training: Randomize batch order each epoch
# shuffle=False for val/test: Consistent evaluation order
train_loader = DataLoader(
    train_dataset,
    batch_size=CONFIG['batch_size'],
    shuffle=True  # Randomize order for better training
)
val_loader = DataLoader(
    val_dataset,
    batch_size=CONFIG['batch_size'],
    shuffle=False  # Fixed order for consistent validation
)
test_loader = DataLoader(
    test_dataset,
    batch_size=CONFIG['batch_size'],
    shuffle=False  # Fixed order for consistent testing
)

print(f"\nDataLoaders created successfully!")
print(f"  Training batches:   {len(train_loader)} (shuffled)")
print(f"  Validation batches: {len(val_loader)} (not shuffled)")
print(f"  Test batches:       {len(test_loader)} (not shuffled)")
print(f"\nBatch size: {CONFIG['batch_size']} sequences per batch")

# =============================================================================
# SECTION 5: TRAIN SEQ2SEQ LSTM MODEL
# =============================================================================

"""
Seq2Seq LSTM Architecture:

The LSTM model uses an encoder-decoder architecture:

1. ENCODER:
   - Reads input sequence (4 weeks × 4 features)
   - Compresses information into hidden state vector
   - This hidden state is a fixed-size representation of the entire input

2. DECODER:
   - Starts with the encoder's final hidden state
   - Generates output sequence autoregressively (one step at a time)
   - Each step's output becomes the next step's input
   - Produces deterministic predictions (same input → same output)

Key characteristics:
- Deterministic: No randomness, always produces same prediction
- Single-path: Only one possible future sequence
- Fast inference: No need to sample multiple times
- Simple to interpret: Direct mapping from input to output

Loss function: Mean Squared Error (MSE)
- Measures average squared difference between prediction and ground truth
- Good for regression tasks like predicting click counts
"""

print("\n" + "="*80)
print("STEP 3: TRAINING SEQ2SEQ LSTM MODEL")
print("="*80)

# Determine input size from data shape
# This is the number of features per time step (4 features in our case)
input_size = data['X_train'].shape[2]

# Initialize the LSTM model with specified architecture
print("\nInitializing Seq2Seq LSTM model...")
lstm_model = Seq2SeqLSTM(
    input_size=input_size,      # Number of input features (4)
    hidden_size=CONFIG['hidden_size'],  # Hidden state dimension (64)
    output_size=1,              # Number of output features (1 = clicks only)
    num_layers=CONFIG['num_layers'],    # Number of stacked LSTM layers
    dropout=CONFIG['dropout']   # Dropout probability for regularization
)

# Display model architecture
print("\n" + "-"*80)
print("LSTM Model Architecture:")
print("-"*80)
print(lstm_model)

# Count total trainable parameters
# This tells us how complex the model is
total_params = sum(p.numel() for p in lstm_model.parameters())
print(f"\nTotal trainable parameters: {total_params:,}")
print(f"  (More parameters = more capacity but slower training)")
print("-"*80)

# Train the LSTM model
# This function handles the entire training loop:
# - Forward pass (compute predictions)
# - Loss calculation (compare to ground truth)
# - Backward pass (compute gradients)
# - Parameter updates (apply gradients)
# - Validation after each epoch
print("\nStarting LSTM training...")
print(f"  Epochs: {CONFIG['epochs']}")
print(f"  Learning rate: {CONFIG['learning_rate']}")
print(f"  Device: {CONFIG['device']}")

lstm_history = train_lstm(
    model=lstm_model,
    train_loader=train_loader,
    val_loader=val_loader,
    epochs=CONFIG['epochs'],
    lr=CONFIG['learning_rate'],
    device=CONFIG['device'],
    output_weeks=CONFIG['output_weeks']
)

# Save trained model for later use
# This allows us to load the model without retraining
os.makedirs('results/checkpoints', exist_ok=True)
torch.save(lstm_model.state_dict(), 'results/checkpoints/lstm_model.pt')
print("\n✓ LSTM model trained and saved to results/checkpoints/lstm_model.pt")

# =============================================================================
# SECTION 6: TRAIN SEQ2SEQ VAE MODEL
# =============================================================================

"""
Seq2Seq VAE Architecture:

The VAE (Variational Autoencoder) extends the LSTM with probabilistic modeling:

1. ENCODER:
   - Reads input sequence (4 weeks × 4 features)
   - Outputs TWO vectors: μ (mean) and log σ² (log variance)
   - These define a probability distribution over latent space

2. REPARAMETERIZATION TRICK:
   - Sample latent vector: z = μ + σ * ε, where ε ~ N(0,1)
   - This trick allows backpropagation through the sampling operation
   - Key insight: randomness comes from ε, gradients flow through μ and σ

3. DECODER:
   - Takes sampled latent vector z
   - Generates output sequence autoregressively
   - Different samples of z produce different outputs

Key characteristics:
- Probabilistic: Samples from learned distribution
- Multi-path: Can generate diverse future sequences
- Slower inference: Need multiple samples for diversity
- Uncertainty quantification: Spread of samples indicates confidence

Loss function: Reconstruction + β * KL Divergence
- Reconstruction: MSE between prediction and ground truth (accuracy)
- KL Divergence: Similarity to standard normal N(0,1) (regularization)
- β parameter: Controls trade-off between accuracy and diversity
  - β < 1: Prioritize accuracy (less diversity)
  - β = 1: Standard VAE (balanced)
  - β > 1: Prioritize diversity (less accuracy)
"""

print("\n" + "="*80)
print("STEP 4: TRAINING SEQ2SEQ VAE MODEL")
print("="*80)

# Initialize the VAE model
print("\nInitializing Seq2Seq VAE model...")
vae_model = Seq2SeqVAE(
    input_size=input_size,      # Number of input features (4)
    hidden_size=CONFIG['hidden_size'],  # Hidden state dimension (64)
    latent_dim=CONFIG['latent_dim'],    # Latent space dimension (16)
    output_size=1,              # Number of output features (1 = clicks only)
    num_layers=CONFIG['num_layers'],    # Number of stacked LSTM layers
    dropout=CONFIG['dropout']   # Dropout probability
)

# Display model architecture
print("\n" + "-"*80)
print("VAE Model Architecture:")
print("-"*80)
print(vae_model)

# Count total trainable parameters
total_params = sum(p.numel() for p in vae_model.parameters())
print(f"\nTotal trainable parameters: {total_params:,}")
print(f"  (VAE has more parameters than LSTM due to latent space projection)")
print("-"*80)

# Train the VAE model
print("\nStarting VAE training...")
print(f"  Epochs: {CONFIG['epochs']}")
print(f"  Learning rate: {CONFIG['learning_rate']}")
print(f"  Beta (KL weight): {CONFIG['beta']}")
print(f"  Device: {CONFIG['device']}")
print("\nNote: VAE training may take longer than LSTM due to:")
print("  1. More parameters to update")
print("  2. Additional KL divergence computation")
print("  3. Reparameterization trick overhead")

vae_history = train_vae(
    model=vae_model,
    train_loader=train_loader,
    val_loader=val_loader,
    epochs=CONFIG['epochs'],
    lr=CONFIG['learning_rate'],
    device=CONFIG['device'],
    output_weeks=CONFIG['output_weeks'],
    beta=CONFIG['beta']  # Controls diversity vs accuracy trade-off
)

# Save trained VAE model
torch.save(vae_model.state_dict(), 'results/checkpoints/vae_model.pt')
print("\n✓ VAE model trained and saved to results/checkpoints/vae_model.pt")

# =============================================================================
# SECTION 7: VISUALIZE TRAINING CURVES
# =============================================================================

"""
Training Curve Analysis:

Training curves help us understand model learning behavior:

1. TRAINING LOSS:
   - Should decrease over time
   - Measures how well model fits training data
   - If not decreasing: learning rate too high/low, or bad initialization

2. VALIDATION LOSS:
   - Should decrease and stabilize
   - Measures generalization to unseen data
   - If increasing while train loss decreases: OVERFITTING!

3. VAE-SPECIFIC LOSSES:
   - Total loss = Reconstruction loss + β * KL divergence
   - Reconstruction: Similar to LSTM loss (accuracy)
   - KL divergence: Regularization term (prevents collapse)
   - Monitor both to ensure balance

Warning signs:
- Val loss increasing: Model is overfitting to training data
- Loss plateaus early: Learning rate too low or model capacity too small
- Loss oscillates wildly: Learning rate too high
- KL divergence near zero: VAE posterior collapse (bad!)
"""

print("\n" + "="*80)
print("STEP 5: VISUALIZING TRAINING CURVES")
print("="*80)

# Create directory for saving figures
os.makedirs('results/figures', exist_ok=True)

# Generate comprehensive training curve plots
print("\nGenerating training curve comparison...")
plot_training_curves(
    lstm_history=lstm_history,
    vae_history=vae_history,
    save_path='results/figures/training_curves.png'
)

print("✓ Training curves saved to results/figures/training_curves.png")
print("\nWhat to look for in training curves:")
print("  1. Both losses should decrease over time")
print("  2. Validation loss should not diverge from training loss")
print("  3. VAE loss may be higher due to KL divergence penalty")
print("  4. Smooth curves indicate stable training")

# =============================================================================
# SECTION 8: EVALUATE LSTM MODEL
# =============================================================================

"""
LSTM Evaluation:

For deterministic models like LSTM, we measure:

1. MSE (Mean Squared Error):
   - Average squared difference between prediction and ground truth
   - Lower is better
   - Formula: MSE = mean((y_pred - y_true)²)
   - Interpretation: Average prediction error in squared units

Since LSTM is deterministic (same input → same output), we only get:
- One prediction per input
- No diversity metrics
- No uncertainty quantification
- No coverage statistics

This simplicity makes LSTM:
+ Easier to interpret
+ Faster to evaluate
- But lacks uncertainty information
"""

print("\n" + "="*80)
print("STEP 6: EVALUATING LSTM ON TEST SET")
print("="*80)

print("\nRunning LSTM evaluation...")
print("  - Generating predictions for all test sequences")
print("  - Computing MSE against ground truth")
print("  - This may take a moment...")

lstm_results = evaluate_model(
    model=lstm_model,
    data_loader=test_loader,
    device=CONFIG['device'],
    output_weeks=CONFIG['output_weeks'],
    is_vae=False  # LSTM is not a VAE
)

# Display LSTM results
print("\n" + "-"*80)
print("LSTM TEST RESULTS")
print("-"*80)
print(f"  Mean Squared Error (MSE): {lstm_results['mse']:.6f}")
print("-"*80)

print("\nInterpretation:")
print(f"  - On average, LSTM predictions are off by √{lstm_results['mse']:.6f} = {np.sqrt(lstm_results['mse']):.3f} clicks")
print("  - This is a deterministic prediction (no uncertainty)")
print("  - Same input will always produce same output")

# =============================================================================
# SECTION 9: EVALUATE VAE MODEL
# =============================================================================

"""
VAE Evaluation:

For probabilistic models like VAE, we measure multiple aspects:

1. MSE (Mean Squared Error):
   - Use MEAN of samples as prediction
   - Comparable to LSTM's single prediction
   - May be higher than LSTM due to diversity penalty

2. Best-of-N MSE:
   - Take BEST prediction among N samples
   - Shows VAE's potential when allowed multiple attempts
   - Should be LOWER than mean MSE
   - Interpretation: "If we could pick the best future, how good would it be?"

3. Diversity (Standard Deviation):
   - Measures variation across samples
   - Higher = more diverse predictions
   - Low diversity suggests posterior collapse (bad)
   - Interpretation: "How much do predictions vary?"

4. Coverage (95% Confidence Interval):
   - Proportion of ground truth within 95% CI of samples
   - Ideal value: ~0.95 (well-calibrated uncertainty)
   - < 0.95: Under-confident (too diverse)
   - > 0.95: Over-confident (too certain)
   - Interpretation: "Does uncertainty match reality?"

The VAE evaluation requires:
- Multiple samples per input (default: 20)
- More computation time than LSTM
- But provides rich uncertainty information
"""

print("\n" + "="*80)
print("STEP 7: EVALUATING VAE ON TEST SET")
print("="*80)

print("\nRunning VAE evaluation...")
print(f"  - Generating {CONFIG['n_samples']} samples per test sequence")
print("  - Computing multiple metrics (MSE, Best-of-N, Diversity, Coverage)")
print("  - This will take longer than LSTM evaluation...")

vae_results = evaluate_model(
    model=vae_model,
    data_loader=test_loader,
    device=CONFIG['device'],
    output_weeks=CONFIG['output_weeks'],
    n_samples=CONFIG['n_samples'],  # Generate multiple samples
    is_vae=True  # VAE requires special handling
)

# Display VAE results with detailed explanations
print("\n" + "-"*80)
print("VAE TEST RESULTS")
print("-"*80)
print(f"  Mean Squared Error (MSE):     {vae_results['mse']:.6f}")
print(f"    └─ Using mean of {CONFIG['n_samples']} samples as prediction")
print(f"\n  Best-of-N MSE:                {vae_results['best_of_n_mse']:.6f}")
print(f"    └─ Best prediction among {CONFIG['n_samples']} samples")
print(f"\n  Diversity (Std Dev):          {vae_results['diversity']:.6f}")
print(f"    └─ Standard deviation across samples")
print(f"\n  Coverage (95% CI):            {vae_results['coverage']:.4f}")
print(f"    └─ Proportion of ground truth within 95% confidence interval")
print("-"*80)

print("\nInterpretation:")
print(f"  - Mean prediction error: √{vae_results['mse']:.6f} = {np.sqrt(vae_results['mse']):.3f} clicks")
print(f"  - Best possible prediction: √{vae_results['best_of_n_mse']:.6f} = {np.sqrt(vae_results['best_of_n_mse']):.3f} clicks")
print(f"  - Diversity indicates {'high' if vae_results['diversity'] > 10 else 'moderate' if vae_results['diversity'] > 5 else 'low'} variation in predictions")

# Interpret coverage metric
if vae_results['coverage'] < 0.90:
    print(f"  - Coverage ({vae_results['coverage']:.2f}) is LOW → VAE is under-confident (too diverse)")
elif vae_results['coverage'] > 0.98:
    print(f"  - Coverage ({vae_results['coverage']:.2f}) is HIGH → VAE is over-confident (too certain)")
else:
    print(f"  - Coverage ({vae_results['coverage']:.2f}) is well-calibrated → Good uncertainty estimation!")

# =============================================================================
# SECTION 10: COMPARE MODELS
# =============================================================================

"""
Model Comparison Framework:

When comparing LSTM and VAE, we need to consider multiple dimensions:

1. ACCURACY (Single Prediction):
   - LSTM MSE vs VAE MSE (mean)
   - Lower is better
   - LSTM often wins here due to deterministic optimization

2. POTENTIAL (Multiple Attempts):
   - LSTM MSE vs VAE Best-of-N MSE
   - VAE often wins here due to diversity
   - Relevant for applications where you can try multiple scenarios

3. UNCERTAINTY QUANTIFICATION:
   - LSTM: No uncertainty information (deterministic)
   - VAE: Provides confidence intervals via sample spread
   - Critical for risk-sensitive applications

4. COMPUTATIONAL COST:
   - LSTM: Fast (one forward pass per prediction)
   - VAE: Slower (N forward passes for N samples)

5. INTERPRETABILITY:
   - LSTM: Clear input→output mapping
   - VAE: Probabilistic interpretation requires more care

Choose LSTM when:
- Need single best prediction
- Computational efficiency is critical
- Deterministic behavior is preferred
- Simple interpretation is needed

Choose VAE when:
- Need uncertainty estimates
- Want to explore multiple scenarios
- Risk assessment is important
- Diversity in predictions is valuable
"""

print("\n" + "="*80)
print("STEP 8: MODEL COMPARISON AND ANALYSIS")
print("="*80)

# Create comprehensive comparison visualization
print("\nGenerating model comparison plots...")
plot_comparison(
    lstm_results=lstm_results,
    vae_results=vae_results,
    save_path='results/figures/model_comparison.png'
)
print("✓ Comparison plots saved to results/figures/model_comparison.png")

# Create VAE-specific diversity analysis
print("\nGenerating VAE diversity analysis...")
plot_diversity_analysis(
    vae_samples=vae_results['samples'],
    targets=vae_results['targets'],
    save_path='results/figures/diversity_analysis.png'
)
print("✓ Diversity analysis saved to results/figures/diversity_analysis.png")

# =============================================================================
# SECTION 11: CREATE SUMMARY TABLE
# =============================================================================

"""
Summary Table Creation:

This table provides a side-by-side comparison of all metrics for easy
interpretation and reporting.

Key comparisons to analyze:
1. LSTM MSE vs VAE MSE: Which is more accurate for single predictions?
2. VAE Best-of-N vs LSTM MSE: Can VAE beat LSTM with multiple attempts?
3. VAE Diversity: Is the model generating diverse predictions?
4. VAE Coverage: Are uncertainty estimates well-calibrated?

For your report, discuss:
- Why might one model outperform the other?
- What's the practical trade-off?
- When would you choose each model?
"""

print("\n" + "="*80)
print("STEP 9: CREATING SUMMARY TABLE")
print("="*80)

# Create pandas DataFrame for easy formatting
summary = pd.DataFrame({
    'Metric': [
        'MSE',
        'Best-of-N MSE',
        'Diversity',
        'Coverage'
    ],
    'LSTM': [
        f"{lstm_results['mse']:.6f}",
        'N/A',  # LSTM is deterministic, no multi-path generation
        '0.000 (deterministic)',  # No variation
        'N/A'   # No uncertainty quantification
    ],
    'VAE': [
        f"{vae_results['mse']:.6f}",
        f"{vae_results['best_of_n_mse']:.6f}",
        f"{vae_results['diversity']:.6f}",
        f"{vae_results['coverage']:.4f}"
    ]
})

# Display formatted table
print("\n" + "="*80)
print("MODEL COMPARISON SUMMARY")
print("="*80)
print(summary.to_string(index=False))
print("="*80)

# Provide interpretation guidance
print("\n📊 How to interpret this table:")
print("\n1. MSE Comparison:")
if lstm_results['mse'] < vae_results['mse']:
    print(f"   → LSTM wins by {vae_results['mse'] - lstm_results['mse']:.6f}")
    print("   → LSTM is more accurate for single-path prediction")
else:
    print(f"   → VAE wins by {lstm_results['mse'] - vae_results['mse']:.6f}")
    print("   → VAE maintains accuracy despite diversity penalty")

print("\n2. Best-of-N Analysis:")
if vae_results['best_of_n_mse'] < lstm_results['mse']:
    print(f"   → VAE's best prediction beats LSTM by {lstm_results['mse'] - vae_results['best_of_n_mse']:.6f}")
    print("   → VAE's diversity provides better best-case performance")
else:
    print(f"   → LSTM still better than VAE's best by {vae_results['best_of_n_mse'] - lstm_results['mse']:.6f}")
    print("   → Diversity doesn't compensate for accuracy loss")

print("\n3. Diversity & Uncertainty:")
print(f"   → VAE generates diverse predictions (std={vae_results['diversity']:.3f})")
print(f"   → Uncertainty calibration: {vae_results['coverage']:.1%} coverage")

# =============================================================================
# SECTION 12: SAVE RESULTS FOR REPORT
# =============================================================================

"""
Results Export:

We export results in two formats:

1. JSON (results_summary.json):
   - Machine-readable format
   - Contains all configuration and results
   - Useful for programmatic analysis

2. CSV (comparison_table.csv):
   - Human-readable format
   - Easy to copy into Word/Excel
   - Good for report tables

These files can be used to:
- Write the assignment report
- Create additional visualizations
- Perform further analysis
- Document experiment settings
"""

print("\n" + "="*80)
print("STEP 10: SAVING RESULTS FOR REPORT")
print("="*80)

# Create results directory if it doesn't exist
os.makedirs('results', exist_ok=True)

# Prepare comprehensive results dictionary
results_summary = {
    # Include all configuration for reproducibility
    'config': CONFIG,

    # LSTM results
    'lstm': {
        'mse': float(lstm_results['mse']),  # Convert to Python float for JSON
        'n_parameters': sum(p.numel() for p in lstm_model.parameters())
    },

    # VAE results (more comprehensive)
    'vae': {
        'mse': float(vae_results['mse']),
        'best_of_n_mse': float(vae_results['best_of_n_mse']),
        'diversity': float(vae_results['diversity']),
        'coverage': float(vae_results['coverage']),
        'n_parameters': sum(p.numel() for p in vae_model.parameters())
    }
}

# Save as JSON for programmatic access
with open('results/results_summary.json', 'w') as f:
    json.dump(results_summary, f, indent=2)
print("✓ Results saved to results/results_summary.json")

# Save summary table as CSV for easy copying
summary.to_csv('results/comparison_table.csv', index=False)
print("✓ Comparison table saved to results/comparison_table.csv")

# =============================================================================
# SECTION 13: FINAL SUMMARY AND NEXT STEPS
# =============================================================================

print("\n" + "="*80)
print("EXPERIMENT COMPLETE!")
print("="*80)

print("\n📁 Generated Files:")
print("  1. results/checkpoints/lstm_model.pt - Trained LSTM model")
print("  2. results/checkpoints/vae_model.pt - Trained VAE model")
print("  3. results/figures/training_curves.png - Training progress visualization")
print("  4. results/figures/model_comparison.png - Comprehensive comparison")
print("  5. results/figures/diversity_analysis.png - VAE diversity analysis")
print("  6. results/results_summary.json - All metrics in JSON format")
print("  7. results/comparison_table.csv - Summary table for report")

print("\n📝 Next Steps for Your Report:")
print("  1. Copy figures from results/figures/ into your Word document")
print("  2. Use results/comparison_table.csv for metrics table")
print("  3. Discuss advantages/disadvantages of each model:")
print("     - LSTM: Simple, fast, accurate for single prediction")
print("     - VAE: Diverse, uncertainty-aware, better best-case")
print("  4. Analyze when to use each model in practice:")
print("     - LSTM: Production systems, real-time prediction")
print("     - VAE: Risk analysis, scenario planning, uncertainty quantification")

print("\n🔬 Key Findings to Discuss:")
print("  • Single-path accuracy comparison (LSTM MSE vs VAE MSE)")
print("  • Multi-path potential (VAE Best-of-N)")
print("  • Diversity-accuracy trade-off (controlled by β)")
print("  • Uncertainty calibration (coverage metric)")
print("  • Computational efficiency (inference time)")

print("\n🎓 Learning Outcomes:")
print("  ✓ Implemented encoder-decoder architecture")
print("  ✓ Understood deterministic vs probabilistic modeling")
print("  ✓ Applied VAE with reparameterization trick")
print("  ✓ Evaluated beyond simple accuracy metrics")
print("  ✓ Compared model trade-offs in realistic application")

print("\n" + "="*80)
print("Thank you for using this tutorial!")
print("Good luck with your assignment!")
print("="*80)
