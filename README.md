<h1>Stock Price Prediction and Analysis</h1>
This project is a Streamlit application for predicting stock prices and generating analysis using machine learning (LSTM) and generative AI models. It integrates real-time stock data from Yahoo Finance, performs predictions using an LSTM neural network, and provides a detailed analysis with recommendations based on the predicted price and the user's buying price. The project also includes real-time news updates for the selected stock symbol.

Features
Stock Price Prediction: Predict future stock prices using historical data with an LSTM neural network.
Customizable Time Step: Users can adjust the time step for predictions to fine-tune the model.
Real-Time Stock Data: Fetch historical stock data from Yahoo Finance for analysis.
Detailed Stock Analysis: Leverages a generative AI model to provide insights and recommendations based on the predicted stock price.
Real-Time News Summary: Get the latest news summaries related to the stock symbol using the SerpAPI and Google Generative AI models.
Tech Stack
Frontend: Streamlit
Data Source: Yahoo Finance
Machine Learning: TensorFlow and Keras
Generative AI: Google Generative AI (Gemini) via LangChain
News Fetching: SerpAPI
Environment Management:dotenv

Usage
Open the Streamlit app: After running the app, Streamlit will provide a local URL (usually http://localhost:8501/). Open it in your web browser.

Input Parameters:

Stock Symbol: Enter the stock symbol you wish to analyze (e.g., RELIANCE.NS).
Start and End Date: Choose the date range for historical data.
Time Step: Select the time step for the LSTM model.
Bought Price: Enter the price at which you purchased the stock.
Prediction Date: Select the date for which you want to predict the stock price.
View Predictions and Analysis: Click the "Predict" button to view the predicted stock price, analysis, and the latest news summary.

Model Training
The model is an LSTM-based neural network that predicts future stock prices based on historical data.
The model is trained or loaded from a pre-trained file (stock_model.h5). If the model file does not exist, the script will automatically train a new model.
Troubleshooting
Not enough data for the specified time step: Ensure that the selected time step does not exceed the available data points.
Invalid Dates: Make sure the start date is earlier than the end date, and both dates are valid.
API Key Issues: Ensure that your .env file contains valid API keys for Google Generative AI and SerpAPI.
Contributing
Contributions are welcome! Please fork the repository and create a pull request with your changes. Make sure to follow the coding standards and include appropriate tests.

Contact
If you have any questions or need further assistance, feel free to contact me at charantej928@gmail.com

