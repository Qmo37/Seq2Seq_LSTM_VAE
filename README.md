# Seq2Seq LSTM vs VAE for Learning Behavior Prediction

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Qmo37/Seq2Seq_LSTM_VAE/blob/main/notebooks/Seq2Seq_LSTM_VAE.ipynb)

This project compares Seq2Seq LSTM and Seq2Seq VAE models for generating future learning behavior sequences using the Open University Learning Analytics Dataset (OULAD).

## Project Goal

- Understand the design principles of sequence generation models (LSTM and VAE)
- Implement and train both models on the same educational dataset
- Compare their performance on:
  - Single-path prediction capability
  - Diversity in generation
  - Evaluation metrics (MSE, Best-of-N, Coverage, Diversity)

## Dataset

**Open University Learning Analytics Dataset (OULAD)**
- Source: https://www.kaggle.com/datasets/anlgrbz/student-demographics-online-education-dataoulad

Required files:
- `studentInfo.csv`
- `studentVle.csv`
- `studentAssessment.csv`

### Features

Input features (per week):
- `clicks`: Weekly click count
- `submit_cnt` / `has_submit`: Whether assignment was submitted
- `avg_score_sofar`: Cumulative average assignment score
- `clicks_diff1`: First-order difference of clicks

Sequence structure:
- Input: Past 4 weeks
- Output: Future 2 weeks (clicks prediction)

## Project Structure

```
Seq2Seq_LSTM_VAE/
├── src/
│   ├── data/
│   │   ├── __init__.py
│   │   ├── dataset.py          # Dataset class and data loading
│   │   └── preprocessing.py    # Data preprocessing and feature engineering
│   ├── models/
│   │   ├── __init__.py
│   │   ├── seq2seq_lstm.py     # Seq2Seq LSTM implementation
│   │   └── seq2seq_vae.py      # Seq2Seq VAE implementation
│   └── utils/
│       ├── __init__.py
│       ├── metrics.py          # Evaluation metrics
│       ├── training.py         # Training utilities
│       └── visualization.py    # Visualization functions
├── notebooks/
│   └── main_experiment.ipynb   # Main Colab notebook
├── data/
│   ├── raw/                    # Raw CSV files (not tracked)
│   └── processed/              # Preprocessed data (not tracked)
├── results/
│   ├── figures/                # Generated plots
│   └── checkpoints/            # Model checkpoints (not tracked)
├── requirements.txt
├── README.md
└── CLAUDE.md
```

## Quick Start

### For Google Colab (Recommended)

**Click the "Open in Colab" button above** or click here: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Qmo37/Seq2Seq_LSTM_VAE/blob/main/notebooks/main_experiment.ipynb)

Then:
- The notebook will automatically download the OULAD dataset (Section 0)
- Run all cells to train both models and generate results
- All dependencies will be installed automatically

No setup required! Everything runs in the cloud.

### For Local Development

1. **Clone the repository**:
   ```bash
   git clone https://github.com/YOUR_USERNAME/Seq2Seq_LSTM_VAE.git
   cd Seq2Seq_LSTM_VAE
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Download data** (two options):

   **Option A: Automatic (via notebook)**
   - Open `notebooks/main_experiment.ipynb`
   - Run Section 0 to automatically download and extract data

   **Option B: Manual**
   - Download from [Kaggle OULAD Dataset](https://www.kaggle.com/datasets/anlgrbz/student-demographics-online-education-dataoulad)
   - Extract the following files to `data/raw/`:
     - `studentInfo.csv`
     - `studentVle.csv`
     - `studentAssessment.csv`

4. **Run the notebook**:
   ```bash
   jupyter notebook notebooks/main_experiment.ipynb
   ```

   Or run the standalone Python script:
   ```bash
   python main_experiment_tutorial.py
   ```

### 3. Training Configuration

Fixed hyperparameters:
- Batch size: 128
- Optimizer: Adam (lr=1e-3)
- Loss: MSE for LSTM, MSE + β·KLD for VAE
- Random seed: 42

Adjustable hyperparameters:
- Epochs: 5-30+ (default: 20)
- Hidden size: 64
- Latent dimension: 16 (VAE only)
- Activation function

## Model Architectures

### Seq2Seq LSTM
- Encoder: Reads past 4 weeks of feature sequences
- Decoder: Outputs future 2 weeks of clicks
- Single deterministic path

### Seq2Seq VAE
- Encoder: Maps past sequences to latent space (μ, log σ²)
- Reparameterization: Samples z from latent distribution
- Decoder: Generates future 2 weeks from z
- Capable of diverse multi-path generation

## Evaluation Metrics

- **MSE**: Mean Squared Error (single path)
- **Best-of-N MSE**: Minimum MSE among N samples (VAE)
- **Diversity**: Standard deviation across generated samples (VAE)
- **Coverage**: Proportion of samples within confidence interval (VAE)

## Experimental Results

### Performance Comparison (Second Run - Original Scale)

| Model | Metric | Value | Description |
|-------|--------|-------|-------------|
| **LSTM** | MSE | **3342.40** | Single-path prediction error |
| **VAE** | Best-of-N MSE | **2575.28** | Best prediction among 20 samples |
| **VAE** | Diversity (std) | **0.1853** | Sample variation (diversity metric) |
| **VAE** | Coverage | **0.1940 (19.40%)** | Proportion within prediction range |

**Console Output Evidence:**

![Evaluation Output Screenshot](results/screenshot_evaluation_colab.png)

Complete evaluation output including Top-5 cases, win-rate distribution, and final metrics from the actual Google Colab run.

### Key Findings

**Surprising Result: VAE Outperforms LSTM in Best-of-N**
- VAE Best-of-N MSE (2575.28) is **23% better** than LSTM MSE (3342.40)
- This demonstrates VAE's ability to generate diverse predictions where at least one achieves superior accuracy

**Win Rate Analysis (Δ = LSTM MSE - VAE Best MSE)**

| Improvement Range | Sample Count | Ratio | Interpretation |
|------------------|--------------|-------|----------------|
| VAE差50~200 | 1 | 0.00% | VAE much worse |
| VAE差10~50 | 11 | 0.01% | VAE slightly worse |
| **平手±10** | **76,266** | **52.99%** | Comparable |
| **VAE略勝10~50** | **67,531** | **46.92%** | VAE slightly better |
| VAE勝50~200 | 113 | 0.08% | VAE much better |

**Total test samples: 143,925**

**Analysis:**
- In **99.91% of cases**, LSTM and VAE perform within ±50 MSE
- VAE shows **slight advantage in 47%** of samples (improvement 10~50)
- Only **0.08%** show significant VAE advantage (>50 improvement)
- **Key insight**: When generating 20 diverse samples, VAE can often find at least one better prediction than LSTM's single path

### Top-5 Cases Where VAE Excels

| Sample ID | LSTM MSE | VAE Best MSE | Improvement | Diversity |
|-----------|----------|--------------|-------------|-----------|
| 113967 | 280.15 | 118.58 | **+161.58** | 0.759 |
| 17896 | 128.69 | 33.25 | **+95.43** | 0.846 |
| 17895 | 537.43 | 462.61 | **+74.82** | 0.683 |
| 51917 | 62.89 | 12.97 | **+49.92** | 0.821 |
| 37482 | 67.70 | 19.18 | **+48.52** | 0.717 |

**Observation**: Cases where VAE excels show **higher diversity** (0.68~0.85), suggesting that when VAE successfully generates diverse samples, it can explore better prediction paths.

## Visualization

The project generates four comprehensive comparison plots:

### 1. Training Curves
![Training Curves](results/visualizations/training_curves.png)

**Shows:**
- LSTM and VAE training/validation loss over 20 epochs
- VAE includes both MSE and KLD components
- Both models converge smoothly without overfitting

### 2. Prediction Comparison
![Prediction Comparison](results/visualizations/prediction_comparison.png)

**Shows:**
- Ground truth (black line) vs LSTM (blue line) vs VAE samples (red semi-transparent lines × 20)
- 6 randomly selected test cases
- **Key observation**: VAE's 20 samples show moderate spread (not collapsed), with some samples closer to ground truth than LSTM

### 3. Diversity Analysis
![Diversity Analysis](results/visualizations/diversity_analysis.png)

**Shows:**
- Distribution of VAE diversity across all test samples
- Histogram, box plot, scatter plot (diversity vs LSTM MSE)
- Diversity bins breakdown
- **Average diversity: 0.1853** indicates moderate mode collapse, but sufficient variation for Best-of-N advantage

### 4. Comprehensive Comparison Dashboard
![Comprehensive Comparison](results/visualizations/comprehensive_comparison.png)

**Shows:**
- MSE comparison bar chart (VAE Best-of-N wins)
- VAE-specific metrics (diversity & coverage)
- Win rate pie chart (47% VAE better, 53% comparable)
- Combined training curves
- Sample predictions with detailed MSE values
- Summary statistics table

### Interpretation

**LSTM Advantages:**
- Simpler architecture, faster inference (1× vs 20×)
- More stable single prediction
- Easier to explain to non-technical users

**VAE Advantages (Demonstrated in Second Run):**
- **Best-of-N strategy effective**: 23% improvement over LSTM
- Provides uncertainty quantification (coverage ~19%)
- Can explore multiple future scenarios
- Useful when diversity matters more than single-path accuracy

**When to Use Each Model:**
- **LSTM**: When you need a single, fast, reliable prediction
- **VAE**: When you need to explore multiple possible futures, assess uncertainty, or can afford 20× inference cost for better best-case accuracy
