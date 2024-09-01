import os
import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import json  # Importing JSON
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, LSTM
from langchain import LLMChain, PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.utilities import SerpAPIWrapper
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Disable oneDNN optimizations for compatibility
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Define the prompt template
prompt_template = PromptTemplate(
    input_variables=["stock_symbol", "time_step", "last_days", "predicted_price", "prediction_date", "bought_price",
                     "decision"],
    template="""
    Based on the stock data provided for {stock_symbol}, here's an overview:

    - Recent Trend: The stock price over the last {time_step} days shows {last_days}.
    - Predicted Price: The estimated price on {prediction_date} is {predicted_price}.
    - Decision: f{decision}

    For making an informed decision, consider the following aspects:

    1. **Fundamental Analysis**:
       - **Company Performance**: Research financial health, including revenue, profits, debt levels, and future growth prospects.
       - **Valuation**: Analyze valuation metrics like P/E ratio, P/B ratio, and Dividend Yield.
       - **Competitive Landscape**: Evaluate the company's position in its industry and potential impacts from competitors.

    2. **Technical Analysis**:
       - **Moving Averages**: Compare current performance with moving averages.
       - **Trading Volume**: Look at volume patterns to identify trends.
       - **Chart Patterns**: Study price charts for potential patterns.

    3. **Market Sentiment and News Events**:
       - **Economic Conditions**: Consider factors like economic outlook, interest rates, and geopolitical events.
       - **Industry News**: Stay updated on news affecting the industry and the company.
       - **Analyst Ratings**: Review analyst opinions for additional insights.

    **Note**: Past performance is not indicative of future results. Diversification is key to managing risk. Consult with a financial advisor for personalized advice.
    """
)

# Initialize the LLM
llm = ChatGoogleGenerativeAI(model='gemini-1.5-pro', google_api_key=os.getenv('GEMINI_API_KEY'))

# Initialize the SerpAPIWrapper tool
serpapi_tool = SerpAPIWrapper(serpapi_api_key=os.getenv('SERP_API_KEY'))

# Create the LLMChain with the prompt template
llm_chain = LLMChain(llm=llm, prompt=prompt_template)


# Function to create dataset for the LSTM model
def create_dataset(dataset, time_step=1):
    X, y = [], []
    for i in range(len(dataset) - time_step - 1):
        X.append(dataset[i:(i + time_step), 0])
        y.append(dataset[i + time_step, 0])
    return np.array(X), np.array(y)


# Function to load and preprocess data
@st.cache
def load_and_preprocess_data(stock_symbol, start_date, end_date):
    data = yf.download(stock_symbol, start=start_date, end=end_date)
    data = data[['Close']].dropna()
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)
    return data, scaled_data, scaler


# Function to generate a buy/hold/sell decision
def generate_decision(bought_price, predicted_price):
    if predicted_price > bought_price * 1.05:  # Sell if profit is more than 5%
        return "SELL"
    elif predicted_price < bought_price * 0.95:  # Buy if loss is more than 5%
        return "BUY"
    else:
        return "HOLD"


# Function to generate analysis using LLMChain and real-time data
def generate_analysis(stock_symbol, time_step, last_days, bought_price, predicted_price, prediction_date):
    decision = generate_decision(float(bought_price), predicted_price)

    # Create the prompt with LLMChain
    response = llm_chain.run({
        "stock_symbol": stock_symbol,
        "time_step": time_step,
        "last_days": last_days,
        "predicted_price": predicted_price,
        "prediction_date": prediction_date,
        "bought_price": bought_price,
        "decision": decision
    })

    # Fetch real-time news using the SerpAPIWrapper
    query = f"Latest news about {stock_symbol} - summarize top 5 articles."
    news_results = serpapi_tool.run(query)
    news_summarizer = ChatGoogleGenerativeAI(model="gemini-1.5-flash")
    top_news = news_summarizer.invoke(news_results + "organise and return top 5 news")

    return response, top_news


# Streamlit app UI
st.title('Stock Price Prediction and Analysis')

# User inputs
stock_symbol = st.text_input('Enter Stock Symbol', 'RELIANCE.NS')
start_date = st.date_input('Select Start Date', pd.to_datetime('2010-01-01'))
end_date = st.date_input('Select End Date', pd.to_datetime('2024-08-30'))
time_step = st.slider('Select Time Step', min_value=1, max_value=120, value=60)
bought_price = st.text_input('Enter the Price When you Bought')
prediction_date = st.date_input('Select Prediction Date')

if st.button('Predict'):
    if not stock_symbol or prediction_date is pd.NaT or start_date >= end_date:
        st.error(
            "Please provide a valid stock symbol, prediction date, and ensure the start date is before the end date.")
    else:
        st.write(f'Predicting for stock symbol: {stock_symbol}')

        # Load and preprocess data
        data, scaled_data, scaler = load_and_preprocess_data(stock_symbol, start_date, end_date)

        # Split data into training and testing sets
        train_size = int(len(scaled_data) * 0.8)
        train_data, test_data = scaled_data[:train_size], scaled_data[train_size:]
        X_train, y_train = create_dataset(train_data, time_step)
        X_test, y_test = create_dataset(test_data, time_step)
        X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
        X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

        # Load or train the model
        model_file = 'stock_model.h5'
        if os.path.exists(model_file):
            model = load_model(model_file)
        else:
            model = Sequential()
            model.add(LSTM(50, return_sequences=True, input_shape=(time_step, 1)))
            model.add(LSTM(50, return_sequences=False))
            model.add(Dense(25))
            model.add(Dense(1))
            model.compile(optimizer='adam', loss='mean_squared_error')
            model.fit(X_train, y_train, batch_size=64, epochs=100, validation_data=(X_test, y_test))
            model.save(model_file)

        # Make prediction
        if len(data) < time_step:
            st.error("Not enough data for the specified time step.")
        else:
            last_days = data['Close'][-time_step:].values
            last_days_str = ', '.join([f"${x:.2f}" for x in last_days])
            last_days_scaled = scaler.transform(last_days.reshape(-1, 1))
            input_data = last_days_scaled.reshape(1, time_step, 1)
            predicted_scaled_price = model.predict(input_data)
            predicted_price = scaler.inverse_transform(predicted_scaled_price)[0][0]

            # Generate analysis and fetch real-time news
            analysis, news_summary = generate_analysis(stock_symbol, time_step, last_days_str, bought_price,
                                                       predicted_price, prediction_date)

            # Display the results
            st.write("### Analysis")
            st.write(analysis)

            st.write("### Real-Time News")
            st.write(news_summary)
