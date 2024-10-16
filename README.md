https://sideprojectpredictionapp-cnkz8apzjk8bk3vgkvykdp.streamlit.app
# 📈 Stock Price Prediction and News Sentiment Analysis App

This app predicts stock prices using a Long Short-Term Memory (LSTM) neural network and analyzes the sentiment of recent news articles related to a specific stock. Built with Streamlit, the app fetches stock data, scrapes relevant news, performs sentiment analysis, and visualizes the results. The prediction model incorporates sentiment analysis to enhance its accuracy.

## Features

- **Dynamic Stock Price Prediction**: Uses LSTM to forecast future stock prices based on historical data.
- **News Sentiment Analysis**: Scrapes news articles for a given stock, filters relevant articles, and calculates sentiment to gauge market sentiment.
- **Interactive Interface**: Allows users to input stock tickers and adjust parameters such as date range, sequence length, and model complexity.
- **Visualizations**: Displays historical stock price data, sentiment trends, and prediction intervals.
- **Handles Restricted Content**: Skips news articles from restricted sources like CNBC and Bloomberg to avoid access issues.

## How It Works

1. **Fetches Stock Data**: Uses `yfinance` to download historical stock prices based on user input.
2. **News Scraping**: Fetches recent news articles from [TickerTick API](https://api.tickertick.com/) related to the stock. Filters relevant articles based on financial-impact keywords.
3. **Sentiment Analysis**: Analyzes the sentiment of each article using TextBlob, filtering out articles that are restricted or inaccessible.
4. **LSTM Model Prediction**: Trains an LSTM model on the stock data and incorporates sentiment analysis to predict future prices.
5. **Visualization**: Plots historical stock prices, displays sentiment over time, and presents the predicted price with confidence intervals.

## Getting Started

### Prerequisites

- Python 3.7 or higher
- An internet connection for fetching data and news articles


