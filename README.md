# Financial News Analysis Project

This project analyzes financial news data and its relationship with stock market movements. It combines natural language processing techniques with technical analysis to provide insights into market sentiment and trends.

## Features

- **News Data Processing**
  - Load and clean financial news data
  - Extract keywords and analyze text patterns
  - Perform sentiment analysis on news headlines

- **Stock Data Analysis**
  - Load historical stock price data
  - Calculate technical indicators (MA, RSI, MACD)
  - Merge news sentiment with stock price data

- **Visualization**
  - Distribution plots for headline lengths
  - Time series plots for publication frequency
  - Bar charts for keyword analysis

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd week-1
```

2. Create and activate a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -e .
```

## Usage

```python
from src import load_news_data, analyze_keywords, plot_distribution

# Load news data
news_df = load_news_data('data/raw_analyst_ratings.csv')

# Analyze keywords
keyword_counts = analyze_keywords(news_df, column='headline', top_n=10)
print("Top 10 Keywords:", keyword_counts)

# Plot headline length distribution
plot_distribution(
    news_df['headline_length'],
    title='Distribution of Headline Lengths',
    xlabel='Headline Length (characters)',
    ylabel='Frequency',
    filename='headline_length_distribution.png'
)
```

## Project Structure

```
week-1/
├── data/
│   ├── raw_analyst_ratings.csv
│   └── [stock]_historical_data.csv files
├── plots/
├── src/
│   └── __init__.py
├── requirements.txt
├── setup.py
└── README.md
```

## Dependencies

- pandas >= 1.5.0
- numpy >= 1.21.0
- nltk >= 3.6.0
- textblob >= 0.15.3
- matplotlib >= 3.4.0
- seaborn >= 0.11.0
- TA-Lib >= 0.4.24

