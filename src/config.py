"""Configuration module for financial time series analysis.

This module contains all configuration parameters and constants used across
the project.
"""

# Data collection parameters
KEYWORD = "Jio"
START_DATE = "2017-01-01"
END_DATE = "2021-12-31"
TICKERS = ["RELIANCE.NS", "BHARTIARTL.NS"]

# Model parameters
VAR_LAGS = 5

# API limits and pagination
DAILY_TWEET_LIMIT = 10
NEWS_PAGE_SIZE = 100

# FinBERT sentiment mapping
SENTIMENT_MAPPING = {
    "positive": 1.0,
    "negative": -1.0,
    "neutral": 0.0
}

# Default API configuration
DEFAULT_NEWS_HOST = "contextualwebsearch-websearch-v1.p.rapidapi.com"
DEFAULT_LANGUAGE = "en"

# Data preprocessing parameters
ROLLING_WINDOW = 10
TRAIN_FRACTION = 0.8

# Model hyperparameters
SEQUENCE_LENGTH = 8
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 1e-3

# Default CSV file for time series analysis
DEFAULT_CSV_PATH = "TATASTEEL.csv"
DEFAULT_FEATURE_COLUMNS = ['Open', 'Close', 'High', 'Low', 'Volume']

# Neural network architecture parameters
TIME2VEC_DIM_RATIO = 1.0  # Ratio of time2vec dimension to feature dimension
N_HEADS = 1
N_LAYERS = 2
HIDDEN_DIM = 64
DROPOUT_RATE = 0.1
FC_DROPOUT_RATE = 0.5
