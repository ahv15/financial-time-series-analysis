"""Utility functions for data processing and analysis.

This module contains common utility functions used across multiple scripts
for data preprocessing, sentiment analysis, and financial data handling.
"""

import datetime as dt
import os
from typing import List, Optional

import pandas as pd
import yfinance as yf
from transformers import BertForSequenceClassification, BertTokenizer, pipeline

from config import SENTIMENT_MAPPING


class FinBERTSentimentAnalyzer:
    """Singleton class for FinBERT sentiment analysis."""
    
    _instance = None
    _tokenizer = None
    _model = None
    _pipeline = None
    
    def __new__(cls):
        """Create singleton instance of FinBERT analyzer."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._initialize_model()
        return cls._instance
    
    @classmethod
    def _initialize_model(cls):
        """Initialize FinBERT model and tokenizer."""
        cls._tokenizer = BertTokenizer.from_pretrained("prosusai/finbert")
        cls._model = BertForSequenceClassification.from_pretrained(
            "prosusai/finbert"
        )
        cls._pipeline = pipeline(
            "sentiment-analysis", 
            model=cls._model, 
            tokenizer=cls._tokenizer
        )
    
    def score_sentiment(self, text: str) -> float:
        """Score sentiment of text using FinBERT.
        
        Args:
            text: Input text to analyze
            
        Returns:
            Sentiment score: 1.0 (positive), -1.0 (negative), 0.0 (neutral)
        """
        label = self._pipeline(
            text, 
            truncation=True, 
            max_length=512
        )[0]["label"].lower()
        return SENTIMENT_MAPPING[label]


def attach_sentiment_scores(df: pd.DataFrame, text_col: str) -> pd.DataFrame:
    """Attach sentiment scores to a DataFrame.
    
    Args:
        df: Input DataFrame containing text data
        text_col: Name of the column containing text to analyze
        
    Returns:
        DataFrame with added 'sentiment' column containing scores
    """
    analyzer = FinBERTSentimentAnalyzer()
    out = df.copy()
    out["sentiment"] = out[text_col].map(analyzer.score_sentiment)
    return out


def calculate_daily_sentiment_mean(df: pd.DataFrame, 
                                   date_col: str = "date") -> pd.Series:
    """Calculate daily mean sentiment from sentiment DataFrame.
    
    Args:
        df: DataFrame with sentiment scores and dates
        date_col: Name of the date column
        
    Returns:
        Series with daily mean sentiment indexed by date
    """
    df[date_col] = pd.to_datetime(df[date_col]).dt.date
    return df.groupby(date_col)["sentiment"].mean()


def download_stock_prices(tickers: List[str], 
                         start: str, 
                         end: str) -> pd.DataFrame:
    """Download adjusted close prices for given tickers.
    
    Args:
        tickers: List of stock ticker symbols
        start: Start date in YYYY-MM-DD format
        end: End date in YYYY-MM-DD format
        
    Returns:
        DataFrame with adjusted close prices, missing values removed
    """
    df = yf.download(tickers, start, end)["Adj Close"]
    if isinstance(df, pd.Series):
        df = df.to_frame(name=tickers[0])
    return df.dropna()


def calculate_price_differences(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate first differences of price data.
    
    Args:
        df: DataFrame with price data
        
    Returns:
        DataFrame with first differences, missing values removed
    """
    return df.diff().dropna()


def validate_date_range(start_date: str, end_date: str) -> None:
    """Validate date range for data collection.
    
    Args:
        start_date: Start date string in YYYY-MM-DD format
        end_date: End date string in YYYY-MM-DD format
        
    Raises:
        ValueError: If date format is invalid or end_date <= start_date
    """
    try:
        start = dt.datetime.fromisoformat(start_date)
        end = dt.datetime.fromisoformat(end_date)
    except ValueError as e:
        raise ValueError(f"Invalid date format: {e}")
    
    if end <= start:
        raise ValueError("End date must be after start date")


def get_api_key(env_var_name: str) -> str:
    """Get API key from environment variable.
    
    Args:
        env_var_name: Name of environment variable containing API key
        
    Returns:
        API key value
        
    Raises:
        RuntimeError: If environment variable is not set
    """
    key = os.getenv(env_var_name)
    if not key:
        raise RuntimeError(f"Set {env_var_name} environment variable.")
    return key
