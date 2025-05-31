"""Data collection module for tweets and news articles.

This module handles collection of social media posts and news articles
from various APIs for sentiment analysis.
"""

import datetime as dt
from typing import List

import nest_asyncio
import pandas as pd
import requests
import twint

from config import DEFAULT_NEWS_HOST, DEFAULT_LANGUAGE
from utils import get_api_key, validate_date_range

# Apply nest_asyncio for twint compatibility
nest_asyncio.apply()


def collect_tweets(query: str,
                  start: str,
                  end: str,
                  daily_limit: int,
                  lang: str = DEFAULT_LANGUAGE) -> pd.DataFrame:
    """Collect tweets for a given query and date range.
    
    Args:
        query: Search query string
        start: Start date in YYYY-MM-DD format
        end: End date in YYYY-MM-DD format
        daily_limit: Maximum tweets to collect per day
        lang: Language code for tweets (default: 'en')
        
    Returns:
        DataFrame with columns ['date', 'tweet'] containing collected tweets
    """
    validate_date_range(start, end)
    
    frames: List[pd.DataFrame] = []
    current_date = dt.datetime.fromisoformat(start).date()
    end_date = dt.datetime.fromisoformat(end).date()
    
    while current_date < end_date:
        next_date = current_date + dt.timedelta(days=1)
        
        # Configure twint search
        config = twint.Config()
        config.Search = query
        config.Lang = lang
        config.Since = current_date.strftime("%Y-%m-%d")
        config.Until = next_date.strftime("%Y-%m-%d")
        config.Hide_output = True
        config.Limit = daily_limit
        config.Pandas = True
        
        # Run search and collect results
        twint.run.Search(config)
        df = twint.storage.panda.Tweets_df.copy()
        
        if not df.empty:
            frames.append(df.loc[:, ["date", "tweet"]])
        
        current_date = next_date
    
    return (pd.concat(frames, ignore_index=True) 
            if frames 
            else pd.DataFrame(columns=["date", "tweet"]))


def collect_news(query: str,
                start: str,
                end: str,
                page_size: int,
                host: str = DEFAULT_NEWS_HOST) -> pd.DataFrame:
    """Collect news articles for a given query and date range.
    
    Args:
        query: Search query string
        start: Start date in YYYY-MM-DD format
        end: End date in YYYY-MM-DD format
        page_size: Number of articles per page
        host: API host URL
        
    Returns:
        DataFrame with columns ['date', 'text'] containing news articles
        
    Raises:
        RuntimeError: If API key environment variable is not set
        requests.HTTPError: If API request fails
    """
    validate_date_range(start, end)
    
    api_key = get_api_key("CONTEXTUAL_WS_API_KEY")
    
    headers = {
        "X-RapidAPI-Host": host,
        "X-RapidAPI-Key": api_key
    }
    
    params = {
        "q": query,
        "pageNumber": "1",
        "pageSize": str(page_size),
        "fromPublishedDate": start,
        "toPublishedDate": end,
    }
    
    response = requests.get(
        f"https://{host}/api/search/NewsSearchAPI",
        headers=headers,
        params=params,
        timeout=30
    )
    response.raise_for_status()
    
    json_data = response.json()
    articles = [(article["datePublished"], article["body"]) 
                for article in json_data.get("value", [])]
    
    return pd.DataFrame(articles, columns=["date", "text"])
