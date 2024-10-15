import subprocess
import sys

# Function to install packages
def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# List of required packages
required_packages = ["yfinance", "numpy", "pandas", "scikit-learn", "tensorflow", 
                     "plotly", "requests", "beautifulsoup4", "textblob"]

# Install each package if it's not already installed
for package in required_packages:
    try:
        __import__(package)
    except ImportError:
        install(package)

import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense
from datetime import datetime
import plotly.graph_objs as go
import requests
from textblob import TextBlob
from bs4 import BeautifulSoup
from sklearn.metrics import mean_absolute_error, mean_squared_error
import tensorflow as tf
import time

# Define keywords related to financial impact
keywords = ["earnings", "profit", "revenue", "investment", "acquisition", "merger",
            "regulation", "forecast", "financial", "market", "stock", "share", "dividend"]

# List of restricted sources
restricted_sources = ["cnbc.com", "bloomberg.com"]

# Function to filter articles based on keywords
def is_relevant_article(title, keywords):
    title = title.lower()
    return any(keyword in title for keyword in keywords)

# Function to filter relevant articles based on title or description
def filter_relevant_articles(articles, keywords):
    return [article for article in articles if is_relevant_article(article['title'], keywords)]

# Fetch filtered news from TickerTick API
def fetch_tickertick_news(ticker, max_results=5):
    try:
        query = f"(and T:curated tt:{ticker} (or T:earning T:market T:sec_fin T:trade))"
        url = f"https://api.tickertick.com/feed?q={query}&n={max_results}"
        response = requests.get(url)
        response.raise_for_status()
        news_data = response.json().get('stories', [])
        
        articles = []
        for story in news_data:
            date = datetime.fromtimestamp(story['time'] / 1000)
            articles.append({
                'title': story['title'],
                'link': story['url'],
                'date': date,
                'site': story['site']
            })
        
        return articles
    except requests.RequestException as e:
        st.error(f"Error fetching news articles for {ticker}: {e}")
        return []

# Function to fetch full article content
def fetch_article_content(url):
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/85.0.4183.102 Safari/537.36",
            "Referer": "https://www.google.com",
        }
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')

        article_text = ""
        for tag in ['article', 'div', 'section']:
            elements = soup.find_all(tag, class_=lambda x: x and ('content' in x or 'body' in x or 'article' in x))
            if elements:
                for elem in elements:
                    article_text += elem.get_text(separator=" ", strip=True)
                break

        if not article_text:
            article_text = soup.get_text(separator=" ", strip=True)

        return article_text[:1000] + "..." if len(article_text) > 1000 else article_text

    except requests.HTTPError as e:
        if response.status_code == 403:
            st.write("⚠️ Access to this article is restricted (403 Forbidden).")
        return None  # Skip content on access restriction
    except requests.RequestException:
        return None  # Skip content on general errors

# Function to analyze sentiment of each article's content
def analyze_article_content(articles):
    sentiments = []
    for article in articles:
        # Skip restricted sources
        if any(restricted in article['site'] for restricted in restricted_sources):
            st.write(f"⚠️ Content restricted for source: '{article['site']}'. Skipping this article.")
            continue
        
        full_text = fetch_article_content(article['link'])
        
        if full_text is None:
            st.write(f"⚠️ Failed to fetch content for: '{article['title']}' (Source: {article['site']}). Skipping sentiment analysis for this article.")
            continue

        st.write(f"Title: {article['title']}")
        st.write(f"Source: {article['site']}")
        st.write(f"Content Snippet: {full_text[:300]}")
        st.write(f"[Read More]({article['link']})")
        
        sentiment = analyze_sentiment(full_text)
        sentiments.append(sentiment)
        st.write(f"Sentiment Score: {sentiment:.2f}")
        
        # Sleep briefly to avoid hitting rate limits
        time.sleep(1)

    return sentiments

# Function to perform sentiment analysis
def analyze_sentiment(text):
    blob = TextBlob(text)
    return blob.sentiment.polarity

@st.cache_resource
def train_model(X_train, y_train, lstm_units):
    model = Sequential()
    model.add(LSTM(units=lstm_units, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(Dropout(0.2))
    model.add(LSTM(units=lstm_units))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train, y_train, epochs=100, batch_size=32)
    return model

@tf.function
def predict_with_uncertainty(model, X, num_samples=100):
    predictions = tf.stack([model(X, training=True) for _ in range(num_samples)])
    mean = tf.reduce_mean(predictions, axis=0)
    std = tf.math.reduce_std(predictions, axis=0)
    return mean, std

# Main function to analyze stock and display results
def analyze_stock(ticker, start_date, end_date, sequence_length, lstm_units):
    news_articles = fetch_tickertick_news(ticker, max_results=5)
    relevant_articles = filter_relevant_articles(news_articles, keywords)
    
    if relevant_articles:
        st.write("### Relevant News Articles Affecting Stock Price")
        sentiments = analyze_article_content(relevant_articles)
        
        if sentiments:
            avg_sentiment = np.mean(sentiments)
            st.write(f"### Average Sentiment for {ticker}: {avg_sentiment:.2f}")
        else:
            avg_sentiment = 0
            st.write(f"No relevant articles could be accessed for sentiment analysis for {ticker}.")
    else:
        st.write("No relevant news articles found for the specified ticker.")
        avg_sentiment = 0

    # Fetch historical data
    data = yf.download(ticker, start=start_date, end=end_date)
    if data.empty:
        st.warning(f"No data available for {ticker} in the selected date range.")
        return

    st.write(f"### Historical Data for {ticker}")
    st.write(data.describe())

    # Plot historical prices
    fig = go.Figure(data=[go.Candlestick(
        x=data.index, open=data['Open'], high=data['High'],
        low=data['Low'], close=data['Close']
    )])
    fig.update_layout(title=f'{ticker} Price History', xaxis_title='Date', yaxis_title='Price (USD)')
    st.plotly_chart(fig)

    # Prepare data for LSTM model
    close_prices = data[['Adj Close']]
    sentiment_data = pd.DataFrame({'Sentiment': [avg_sentiment] * len(close_prices)}, index=close_prices.index)
    combined_data = pd.concat([close_prices, sentiment_data], axis=1)

    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaled_data = scaler.fit_transform(combined_data)

    X, y = [], []
    for i in range(sequence_length, len(scaled_data)):
        X.append(scaled_data[i-sequence_length:i])
        y.append(scaled_data[i, 0])

    X, y = np.array(X), np.array(y)

    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    with st.spinner(f"Training model for {ticker}..."):
        try:
            model = train_model(X_train, y_train, lstm_units)
        except Exception as e:
            st.error(f"Error training model for {ticker}: {str(e)}")
            return

    try:
        y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))

        st.write(f"Model Performance Metrics for {ticker}:")
        st.write(f"Mean Absolute Error: ${mae:.2f}")
        st.write(f"Root Mean Square Error: ${rmse:.2f}")
    except Exception as e:
        st.error(f"Error evaluating model for {ticker}: {str(e)}")
        return

    try:
        last_sequence = scaled_data[-sequence_length:]
        mean, std = predict_with_uncertainty(model, last_sequence.reshape(1, sequence_length, 2))
        lower_bound = scaler.inverse_transform(np.hstack((mean - 1.96 * std, [[avg_sentiment]])))[0, 0]
        upper_bound = scaler.inverse_transform(np.hstack((mean + 1.96 * std, [[avg_sentiment]])))[0, 0]
        next_price_prediction = scaler.inverse_transform(np.hstack((mean, [[avg_sentiment]])))[0, 0]

        st.metric(f"Predicted Price for {ticker}", f"${next_price_prediction:.2f}")
        st.write(f"95% Prediction Interval: ${lower_bound:.2f} to ${upper_bound:.2f}")
    except Exception as e:
        st.error(f"Error making prediction for {ticker}: {str(e)}")
        return

    if sentiments:
        daily_sentiment = pd.DataFrame({'Sentiment': sentiments}, index=data.index[-len(sentiments):])
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=data.index, y=data['Adj Close'], name='Stock Price'))
        fig.add_trace(go.Scatter(x=daily_sentiment.index, y=daily_sentiment['Sentiment'], name='Sentiment', yaxis='y2'))
        fig.update_layout(
            title=f'{ticker} Price and Sentiment Over Time',
            xaxis_title='Date',
            yaxis_title='Price (USD)',
            yaxis2=dict(title='Sentiment', overlaying='y', side='right')
        )
        st.plotly_chart(fig)
    else:
        st.write("No sentiment data available for visualization.")

# Streamlit app setup
st.title("📈 Enhanced Stock Price Prediction with LSTM and News Sentiment")
st.sidebar.header("Input Parameters")
tickers = st.sidebar.text_input("Enter Stock Tickers (comma-separated, e.g., AAPL,GOOGL,MSFT):", "AAPL,GOOGL,MSFT").split(',')
start_date = st.sidebar.date_input("Start Date", datetime(2017, 1, 1))
end_date = st.sidebar.date_input("End Date", datetime.today())
sequence_length = st.sidebar.slider("Sequence Length", 30, 90, 60)
lstm_units = st.sidebar.slider("LSTM Units", 20, 100, 50)

if st.sidebar.button("Analyze and Predict"):
    for ticker in tickers:
        st.write(f"## Analysis for {ticker}")
        analyze_stock(ticker.strip(), start_date, end_date, sequence_length, lstm_units)
