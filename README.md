# financial-time-series-analysis

A comprehensive financial analysis toolkit that combines sentiment analysis from social media and news with advanced time series modeling. This project uses FinBERT to score daily sentiment from tweets and news articles, then fits Vector Autoregression (VAR) models and Transformer neural networks to analyze the relationship between sentiment and stock price movements.

## Table of Contents

- [Installation](#installation)
- [Repository Structure](#repository-structure)
- [Usage](#usage)
- [Configuration](#configuration)
- [Results](#results)
- [Dependencies](#dependencies)

## Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-compatible GPU (optional, for accelerated neural network training)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/ahv15/financial-time-series-analysis.git
cd financial-time-series-analysis
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

3. Set up API keys (required for news collection):
```bash
export CONTEXTUAL_WS_API_KEY="your_rapidapi_key_here"
```

## Repository Structure

```
financial-time-series-analysis/
├── src/                              # Source code modules
│   ├── config.py                     # Configuration parameters and constants
│   ├── utils.py                      # Common utility functions
│   ├── data_collection.py            # Tweet and news collection functions
│   ├── models.py                     # PyTorch neural network models
│   ├── financial_sentiment_var.py    # Main VAR analysis pipeline
│   └── time_series_analysis.py       # Transformer-based forecasting
├── requirements.txt                  # Python package dependencies
└── README.md                        # Project documentation
```

### Module Descriptions

- **`src/config.py`**: Contains all configuration parameters including API limits, model hyperparameters, and default values
- **`src/utils.py`**: Utility functions for sentiment analysis, data preprocessing, and financial data handling
- **`src/data_collection.py`**: Functions for collecting tweets via Twint and news articles via RapidAPI
- **`src/models.py`**: PyTorch implementations of Time2Vec embeddings and Transformer models for time series forecasting
- **`src/financial_sentiment_var.py`**: Complete pipeline that combines sentiment analysis with VAR modeling
- **`src/time_series_analysis.py`**: Advanced neural network approach using Transformer architectures for price prediction

## Usage

### Financial Sentiment VAR Analysis

Run the complete sentiment analysis and VAR modeling pipeline:

```bash
cd src
python financial_sentiment_var.py
```

This script will:
1. Collect tweets and news articles for the configured keyword
2. Score sentiment using FinBERT
3. Download stock price data
4. Fit a VAR model on returns and sentiment
5. Display comprehensive model results

### Transformer-based Time Series Analysis

Run the neural network-based forecasting pipeline:

```bash
cd src
python time_series_analysis.py
```

**Note**: This script expects a CSV file named `TATASTEEL.csv` in the root directory with columns: `Open`, `Close`, `High`, `Low`, `Volume`.

### Example Commands

```bash
# Run sentiment VAR analysis with default parameters
python src/financial_sentiment_var.py

# Run Transformer forecasting
python src/time_series_analysis.py
```

## Configuration

All key parameters can be modified in `src/config.py`:

### Data Collection Parameters
- **`KEYWORD`**: Search term for tweets and news (default: "Jio")
- **`START_DATE`**: Analysis start date (default: "2017-01-01")
- **`END_DATE`**: Analysis end date (default: "2021-12-31")
- **`TICKERS`**: Stock symbols to analyze (default: ["RELIANCE.NS", "BHARTIARTL.NS"])
- **`DAILY_TWEET_LIMIT`**: Maximum tweets per day (default: 10)
- **`NEWS_PAGE_SIZE`**: News articles per API call (default: 100)

### Model Parameters
- **`VAR_LAGS`**: Number of lags for VAR model (default: 5)
- **`SEQUENCE_LENGTH`**: Input sequence length for Transformer (default: 8)
- **`EPOCHS`**: Neural network training epochs (default: 10)
- **`LEARNING_RATE`**: Learning rate for optimization (default: 1e-3)

### API Configuration
Set the `CONTEXTUAL_WS_API_KEY` environment variable with your RapidAPI key for news collection.

## Results

### VAR Model Output
The sentiment VAR analysis generates:
- **Sentiment scores**: Daily aggregated sentiment from tweets and news (-1 to +1 scale)
- **Price returns**: First differences of stock prices
- **VAR model summary**: Statistical relationships between sentiment and returns
- **Model diagnostics**: Lag significance, residual analysis, and fit statistics

### Transformer Model Output
The neural network analysis produces:
- **Training metrics**: Loss curves and convergence statistics
- **Predictions**: Next-step price change forecasts
- **Visualizations**: Actual vs predicted price movements
- **Model architecture**: Time2Vec embeddings + multi-head attention

### Output Files
Results are displayed in the console. For persistent storage, modify the scripts to save:
- Sentiment scores to CSV
- Model predictions to CSV
- Plots to PNG files

## Dependencies

### Core Libraries
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computations
- **yfinance**: Stock price data download
- **transformers**: FinBERT sentiment analysis
- **statsmodels**: VAR model implementation

### Data Collection
- **twint**: Twitter data scraping
- **requests**: API calls for news data
- **nest-asyncio**: Async compatibility

### Machine Learning
- **torch**: PyTorch neural networks
- **matplotlib**: Data visualization

For complete version requirements, see `requirements.txt`.

---

**Note**: This project is for educational and research purposes. Always verify data quality and model assumptions before making financial decisions.
