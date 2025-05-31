"""financial_sentiment_var.py
==================================================
End-to-end pipeline:
  • Collect tweets & news for KEYWORD
  • Score sentiment with FinBERT (no VADER fallback)
  • Aggregate daily sentiment
  • Download stock prices
  • Fit a VAR model on price returns + sentiment
All logic is in plain functions; edit the PARAMS section to suit.
"""

# ------------------------------ PARAMS ----------------------------------
KEYWORD            = "Jio"
START_DATE         = "2017-01-01"
END_DATE           = "2021-12-31"
TICKERS            = ["RELIANCE.NS", "BHARTIARTL.NS"]
VAR_LAGS           = 5
DAILY_TWEET_LIMIT  = 10
NEWS_PAGE_SIZE     = 100
# -----------------------------------------------------------------------

import datetime as dt
import os
from typing import List

import nest_asyncio, pandas as pd, requests, twint, yfinance as yf
from statsmodels.tsa.api import VAR
from transformers import BertForSequenceClassification, BertTokenizer, pipeline

nest_asyncio.apply()

# ------------------ FinBERT sentiment pipeline (singleton) -------------
_tok   = BertTokenizer.from_pretrained("prosusai/finbert")
_model = BertForSequenceClassification.from_pretrained("prosusai/finbert")
_bert  = pipeline("sentiment-analysis", model=_model, tokenizer=_tok)

def score_finbert(text: str) -> float:
    label = _bert(text, truncation=True, max_length=512)[0]["label"].lower()
    return {"positive": 1.0, "negative": -1.0, "neutral": 0.0}[label]

# ------------------------ Data Collection ------------------------------
def collect_tweets(query: str,
                   start: str,
                   end: str,
                   daily_limit: int,
                   lang: str = "en") -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    cur = dt.datetime.fromisoformat(start).date()
    end_d = dt.datetime.fromisoformat(end).date()
    while cur < end_d:
        nxt = cur + dt.timedelta(days=1)
        cfg = twint.Config()
        cfg.Search, cfg.Lang = query, lang
        cfg.Since, cfg.Until = cur.strftime("%Y-%m-%d"), nxt.strftime("%Y-%m-%d")
        cfg.Hide_output, cfg.Limit, cfg.Pandas = True, daily_limit, True
        twint.run.Search(cfg)
        df = twint.storage.panda.Tweets_df.copy()
        if not df.empty:
            frames.append(df.loc[:, ["date", "tweet"]])
        cur = nxt
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["date", "tweet"])

def collect_news(query: str,
                 start: str,
                 end: str,
                 page_size: int,
                 host: str = "contextualwebsearch-websearch-v1.p.rapidapi.com") -> pd.DataFrame:
    key = os.getenv("CONTEXTUAL_WS_API_KEY")
    if not key:
        raise RuntimeError("Set CONTEXTUAL_WS_API_KEY environment variable.")
    headers = {"X-RapidAPI-Host": host, "X-RapidAPI-Key": key}
    params = {
        "q": query,
        "pageNumber": "1",
        "pageSize": str(page_size),
        "fromPublishedDate": start,
        "toPublishedDate": end,
    }
    r = requests.get(f"https://{host}/api/search/NewsSearchAPI",
                     headers=headers, params=params, timeout=30)
    r.raise_for_status()
    js = r.json()
    return pd.DataFrame([(a["datePublished"], a["body"]) for a in js.get("value", [])],
                        columns=["date", "text"])

# -------------------- Sentiment aggregation ----------------------------
def attach_score(df: pd.DataFrame, text_col: str) -> pd.DataFrame:
    out = df.copy()
    out["sentiment"] = out[text_col].map(score_finbert)
    return out

def daily_mean(df: pd.DataFrame, date_col: str = "date") -> pd.Series:
    df[date_col] = pd.to_datetime(df[date_col]).dt.date
    return df.groupby(date_col)["sentiment"].mean()

# ----------------------- Price utilities --------------------------------
def download_prices(tickers: List[str], start: str, end: str) -> pd.DataFrame:
    df = yf.download(tickers, start, end)["Adj Close"]
    if isinstance(df, pd.Series):
        df = df.to_frame(name=tickers[0])
    return df.dropna()

def difference(df: pd.DataFrame) -> pd.DataFrame:
    return df.diff().dropna()

# --------------------------- Main Logic ---------------------------------
if __name__ == "__main__":
    # 1. Collect data
    tweets_raw = collect_tweets(KEYWORD, START_DATE, END_DATE, DAILY_TWEET_LIMIT)
    news_raw   = collect_news(KEYWORD, START_DATE, END_DATE, NEWS_PAGE_SIZE)

    # 2. Score sentiment
    tweets_scored = attach_score(tweets_raw, "tweet")
    news_scored   = attach_score(news_raw,   "text")

    # 3. Aggregate daily sentiment
    daily_sent = pd.concat(
        [daily_mean(tweets_scored), daily_mean(news_scored)]
    ).groupby(level=0).mean()
    daily_sent = daily_sent.rename("sentiment")

    # 4. Price returns + merge
    prices  = download_prices(TICKERS, START_DATE, END_DATE)
    returns = difference(prices)
    returns.index = returns.index.date
    df = returns.join(daily_sent, how="inner")

    # 5. VAR model
    res = VAR(df).fit(VAR_LAGS)
    print(res.summary())