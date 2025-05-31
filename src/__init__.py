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
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# Define which functions to export
__all__ = [
    'load_news_data',
    'load_stock_data',
    'merge_news_stock',
    'clean_text',
    'get_sentiment',
    'get_publication_frequency',
    'compute_technical_indicators',
    'plot_distribution',
    'plot_bar',
    'plot_time_series',
    'extract_keywords',
    'analyze_keywords',
    'plot_sentiment_comparison'
]

# Download NLTK resources
def download_nltk_resources():
    """Download required NLTK resources."""
    resources = {
        'punkt': 'tokenizers/punkt',
        'stopwords': 'corpora/stopwords',
        'averaged_perceptron_tagger': 'taggers/averaged_perceptron_tagger',
        'wordnet': 'corpora/wordnet'
    }
    
    for resource, path in resources.items():
        try:
            nltk.data.find(path)
        except LookupError:
            print(f"Downloading {resource}...")
            nltk.download(resource, quiet=True)
    print("NLTK resources downloaded successfully.")

# Download resources when module is imported
download_nltk_resources()

# Initialize stopwords
stop_words = set(stopwords.words('english'))

def extract_keywords(text):
    """Extract keywords from text."""
    if not isinstance(text, str):
        return []
    try:
        words = word_tokenize(clean_text(text))
        return [word for word in words if word not in stop_words and len(word) > 2]
    except Exception as e:
        print(f"Error processing text: {e}")
        return []

# Data Loading and Preprocessing
def load_news_data(file_path='data/raw_analyst_ratings.csv'):
    """Load and preprocess news data from raw_analyst_ratings.csv."""
    try:
        # Read the CSV file
        df = pd.read_csv(file_path)
        print("Columns in the dataset:", df.columns.tolist())
        print("\nFirst few rows:")
        print(df.head())
        
        # Check for missing values
        print("\nMissing values per column:")
        print(df.isnull().sum())
        
        # Clean the date column
        if 'date' in df.columns:
            # First, try to identify the date format
            sample_date = df['date'].iloc[0]
            print(f"\nSample date format: {sample_date}")
            
            # Convert date column with error handling
            try:
                # Try parsing without timezone first
                df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d %H:%M:%S', errors='coerce')
                # Drop rows where date conversion failed
                df = df.dropna(subset=['date'])
                print(f"\nSuccessfully converted dates. Remaining rows: {len(df)}")
            except Exception as e:
                print(f"Error converting dates: {e}")
                # Try alternative format if the first attempt fails
                try:
                    df['date'] = pd.to_datetime(df['date'], errors='coerce')
                    df = df.dropna(subset=['date'])
                    print(f"\nSuccessfully converted dates using alternative method. Remaining rows: {len(df)}")
                except Exception as e:
                    print(f"Error in alternative date conversion: {e}")
                    raise
        
        # Clean the headline column if it exists
        if 'headline' in df.columns:
            df['headline_length'] = df['headline'].apply(lambda x: len(str(x)) if pd.notnull(x) else 0)
        
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        raise

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
        # Ensure the plots directory exists
        os.makedirs('plots', exist_ok=True)
        # Save to plots directory
        plt.savefig(os.path.join('plots', os.path.basename(filename)))
    plt.show()

def plot_bar(data, title, xlabel, ylabel, filename=None):
    """Plot a bar chart."""
    plt.figure(figsize=(10, 6))
    data.plot(kind='bar', color='lightgreen')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if filename:
        # Ensure the plots directory exists
        os.makedirs('plots', exist_ok=True)
        # Save to plots directory
        plt.savefig(os.path.join('plots', os.path.basename(filename)))
    plt.show()

def plot_time_series(data, title, xlabel, ylabel, filename=None):
    """Plot a time series."""
    plt.figure(figsize=(12, 6))
    data.plot()
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(rotation=45)
    plt.tight_layout()
    if filename:
        # Ensure the plots directory exists
        os.makedirs('plots', exist_ok=True)
        # Save to plots directory
        plt.savefig(os.path.join('plots', os.path.basename(filename)))
    plt.show()

# Add new function for keyword analysis
def analyze_keywords(df, column='headline', top_n=10):
    """Analyze keywords in a DataFrame column.
    
    Args:
        df (pd.DataFrame): DataFrame containing the text column
        column (str): Name of the column containing text
        top_n (int): Number of top keywords to return
        
    Returns:
        list: List of (keyword, count) tuples
    """
    all_keywords = []
    for text in df[column]:
        all_keywords.extend(extract_keywords(text))
    return Counter(all_keywords).most_common(top_n)

def plot_sentiment_comparison(stock_sentiment, filename=None):
    """Plot top and bottom stocks by sentiment.
    
    Args:
        stock_sentiment (pd.Series): Series containing stock sentiment scores
        filename (str, optional): If provided, save the plot to this file
    """
    try:
        # Get top and bottom stocks
        top_5 = stock_sentiment.nlargest(5)
        bottom_5 = stock_sentiment.nsmallest(5)
        
        # Combine them using pd.concat
        combined_stocks = pd.concat([top_5, bottom_5])
        
        # Plot
        plt.figure(figsize=(12, 6))
        ax = combined_stocks.plot(kind='bar', color=['green']*5 + ['red']*5)
        plt.title('Top/Bottom 5 Stocks by Average Sentiment')
        plt.xlabel('Stock Symbol')
        plt.ylabel('Average Sentiment Score')
        plt.xticks(rotation=45, ha='right')
        
        # Add value labels on top of bars
        for i, v in enumerate(combined_stocks.values):
            offset = 0.01 if float(v) >= 0 else -0.01
            ax.text(i, float(v) + offset, 
                   f'{v:.3f}', 
                   ha='center', 
                   va='bottom' if float(v) >= 0 else 'top')
        
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if filename:
            # Ensure the plots directory exists
            os.makedirs('plots', exist_ok=True)
            # Save to plots directory
            plt.savefig(os.path.join('plots', os.path.basename(filename)))
        plt.show()
        
    except Exception as e:
        print(f"Error plotting sentiment comparison: {e}")