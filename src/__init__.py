# src/__init__.py
import pandas as pd
import numpy as np
import nltk
from textblob import TextBlob
from textblob.sentiments import PatternAnalyzer
from typing import Union, Optional, Tuple, Any
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import re
from collections import Counter
import talib
import os

# Download NLTK resources
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)

# Data Loading and Preprocessing
def load_news_data(file_path):
    """Load and preprocess news data from raw_analyst_ratings.csv."""
    df = pd.read_csv(file_path)
    df['date'] = pd.to_datetime(df['date'], utc=True)  # Ensure UTC timezone
    df['headline_length'] = df['headline'].apply(len)
    return df

def load_stock_data(stock_symbol, folder_path='data/yfinance_data'):
    """Load stock price data for a given stock symbol."""
    file_path = os.path.join(folder_path, f'{stock_symbol}_historical_data.csv')
    df = pd.read_csv(file_path)
    df['Date'] = pd.to_datetime(df['Date'])  # Ensure date is datetime
    df['daily_return'] = df['Close'].pct_change()  # Compute daily returns
    df['stock'] = stock_symbol
    return df

def merge_news_stock(news_df, stock_df):
    """Merge news and stock data on date."""
    news_df['date_only'] = news_df['date'].dt.date
    stock_df['date_only'] = stock_df['Date'].dt.date
    merged = pd.merge(
        news_df.groupby(['date_only', 'stock'])['sentiment'].mean().reset_index(),
        stock_df,
        left_on=['date_only', 'stock'],
        right_on=['date_only', 'stock'],
        how='inner'
    )
    return merged

# Text Processing
def clean_text(text):
    """Clean headline text for analysis."""
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    return text

# Sentiment Analysis
def get_sentiment(headline: str) -> float:
    """Compute sentiment score using TextBlob.
    
    Args:
        headline (str): The text to analyze
        
    Returns:
        float: Sentiment score between -1 (negative) and 1 (positive)
    """
    try:
        # Clean and convert to string
        text = str(headline)
        # Create TextBlob object with explicit analyzer
        blob = TextBlob(text, analyzer=PatternAnalyzer())
        # Get sentiment and convert to float
        sentiment: Any = blob.sentiment
        # Access polarity directly from the sentiment object
        polarity = getattr(sentiment, 'polarity', 0.0)
        return float(polarity)
    except Exception as e:
        print(f"Error analyzing sentiment for headline: {e}")
        return 0.0  # Return neutral sentiment in case of error

# Time Series Analysis
def get_publication_frequency(df, freq='D'):
    """Compute publication frequency over time."""
    return df.groupby(df['date'].dt.floor(freq)).size()

# Technical Indicators
def compute_technical_indicators(prices, stock_symbol):
    """Compute MA, RSI, and MACD for a given stock's price series."""
    try:
        ma = talib.SMA(prices, timeperiod=20)
        rsi = talib.RSI(prices, timeperiod=14)
        macd, signal, _ = talib.MACD(prices, fastperiod=12, slowperiod=26, signalperiod=9)
        return pd.DataFrame({
            'Date': prices.index,
            'stock': stock_symbol,
            'MA20': ma,
            'RSI': rsi,
            'MACD': macd,
            'MACD_Signal': signal
        })
    except Exception as e:
        print(f"Error computing technical indicators: {e}")
        return None

# Plotting Utility
def plot_distribution(data, title, xlabel, ylabel, filename=None):
    """Plot a distribution with seaborn."""
    plt.figure(figsize=(10, 6))
    sns.histplot(data, bins=30, kde=True, color='skyblue')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if filename:
        plt.savefig(filename)
    plt.show()

def plot_bar(data, title, xlabel, ylabel, filename=None):
    """Plot a bar chart."""
    plt.figure(figsize=(10, 6))
    data.plot(kind='bar', color='lightgreen')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if filename:
        plt.savefig(filename)
    plt.show()