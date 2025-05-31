"""Financial time series analysis package.

This package provides tools for combining sentiment analysis with time series
modeling for financial data analysis.
"""

__version__ = "1.0.0"
__author__ = "Harshit"

from .config import *
from .utils import FinBERTSentimentAnalyzer
from .data_collection import collect_tweets, collect_news
from .models import TimeSeriesDataset, StockTransformer

__all__ = [
    'FinBERTSentimentAnalyzer',
    'collect_tweets',
    'collect_news', 
    'TimeSeriesDataset',
    'StockTransformer'
]
