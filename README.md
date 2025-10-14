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

## Visualization

The project generates comparison plots showing:
- Ground truth (actual future clicks)
- LSTM single-path prediction
- VAE multi-sample predictions (semi-transparent curves)
- Confidence intervals and diversity analysis

## Submission Format

1. **Code**: Colab notebook with both LSTM and VAE models
2. **Analysis**: Word document containing:
   - Model comparison (MSE, Best-of-N, Coverage, Diversity)
   - Visualization results
   - Advantages and disadvantages discussion
3. **Submission**: GitHub repository link

## License

This project is for educational purposes.
