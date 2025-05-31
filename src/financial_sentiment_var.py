"""Financial sentiment and VAR analysis pipeline.

This script implements an end-to-end pipeline that:
1. Collects tweets and news for a specified keyword
2. Scores sentiment using FinBERT
3. Aggregates daily sentiment scores
4. Downloads stock prices and calculates returns
5. Fits a VAR model on price returns and sentiment data

All configuration parameters are defined in config.py.
"""

import pandas as pd
from statsmodels.tsa.api import VAR

from config import (KEYWORD, START_DATE, END_DATE, TICKERS, VAR_LAGS,
                   DAILY_TWEET_LIMIT, NEWS_PAGE_SIZE)
from data_collection import collect_tweets, collect_news
from utils import (attach_sentiment_scores, calculate_daily_sentiment_mean,
                   download_stock_prices, calculate_price_differences)


def main():
    """Execute the complete financial sentiment and VAR analysis pipeline."""
    print("Starting financial sentiment VAR analysis...")
    
    # 1. Collect data
    print(f"Collecting tweets for '{KEYWORD}' from {START_DATE} to {END_DATE}")
    tweets_raw = collect_tweets(
        KEYWORD, START_DATE, END_DATE, DAILY_TWEET_LIMIT
    )
    
    print(f"Collecting news for '{KEYWORD}' from {START_DATE} to {END_DATE}")
    news_raw = collect_news(
        KEYWORD, START_DATE, END_DATE, NEWS_PAGE_SIZE
    )
    
    # 2. Score sentiment
    print("Scoring sentiment with FinBERT...")
    tweets_scored = attach_sentiment_scores(tweets_raw, "tweet")
    news_scored = attach_sentiment_scores(news_raw, "text")
    
    # 3. Aggregate daily sentiment
    print("Aggregating daily sentiment...")
    daily_sentiment = pd.concat([
        calculate_daily_sentiment_mean(tweets_scored),
        calculate_daily_sentiment_mean(news_scored)
    ]).groupby(level=0).mean()
    daily_sentiment = daily_sentiment.rename("sentiment")
    
    # 4. Download stock prices and calculate returns
    print(f"Downloading stock prices for {TICKERS}")
    prices = download_stock_prices(TICKERS, START_DATE, END_DATE)
    returns = calculate_price_differences(prices)
    returns.index = returns.index.date
    
    # 5. Merge price returns with sentiment data
    print("Merging price returns with sentiment data...")
    combined_data = returns.join(daily_sentiment, how="inner")
    
    print(f"Combined dataset shape: {combined_data.shape}")
    print(f"Date range: {combined_data.index.min()} to {combined_data.index.max()}")
    
    # 6. Fit VAR model
    print(f"Fitting VAR model with {VAR_LAGS} lags...")
    var_model = VAR(combined_data)
    var_results = var_model.fit(VAR_LAGS)
    
    # Display results
    print("\n" + "="*60)
    print("VAR MODEL RESULTS")
    print("="*60)
    print(var_results.summary())


if __name__ == "__main__":
    main()
