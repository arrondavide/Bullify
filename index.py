
import yfinance as yf
from datetime import datetime, timedelta
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
import numpy as np

# Ensure you have the VADER lexicon downloaded
nltk.download('vader_lexicon')

# Initialize VADER sentiment analyzer
sia = SentimentIntensityAnalyzer()

# Define the list of stock symbols and their names
stock_symbols = {
    "BP": "BP plc",
    "EQNR": "Equinor",
    "SU": "Suncor Energy",
    "E": "Eni",
    "VLO": "Valero Energy",
    "FANG": "Diamondback Energy",
    "CTRA": "Coterra Energy",
    "DINO": "HF Sinclair"
}



# Sidebar navigation
page = st.sidebar.radio("Navigation", ["Stock Prediction", "Live Market", "Market Sentiment"])

# Stock Prediction page
if page == "Stock Prediction":
    # Load stock data based on selected symbol and predict
    selected_stock = st.sidebar.selectbox("Select Stock Symbol", list(stock_symbols.keys()))

    if selected_stock:
        st.subheader(f"Stock Selected: {stock_symbols[selected_stock]} ({selected_stock})")
        
        # Load stock data
        data = yf.download(tickers=selected_stock, period='5y', interval='1d')
        opn = data[['Open']]
        
        # Normalize the data
        normalizer = MinMaxScaler(feature_range=(0, 1))
        ds_scaled = normalizer.fit_transform(np.array(opn).reshape(-1, 1))
        
        # Split the data into training and testing sets
        train_size = int(len(ds_scaled) * 0.70)
        ds_train, ds_test = ds_scaled[0:train_size, :], ds_scaled[train_size:len(ds_scaled), :1]
        
        # Create dataset for LSTM
        def create_ds(dataset, step):
            Xtrain, Ytrain = [], []
            for i in range(len(dataset) - step - 1):
                a = dataset[i:(i + step), 0]
                Xtrain.append(a)
                Ytrain.append(dataset[i + step, 0])
            return np.array(Xtrain), np.array(Ytrain)
        
        # Parameters
        time_stamp = 100
        X_train, y_train = create_ds(ds_train, time_stamp)
        X_test, y_test = create_ds(ds_test, time_stamp)
        
        # Reshape data to fit into LSTM model
        X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
        X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
        
        # Create the LSTM model
        model = Sequential()
        model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
        model.add(LSTM(units=50, return_sequences=True))
        model.add(LSTM(units=50))
        model.add(Dense(units=1, activation='linear'))
        model.compile(loss='mean_squared_error', optimizer='adam')
        
        # Show a spinner while model is training
        with st.spinner("Predicting... Please wait."):
            model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=100, batch_size=64, verbose=0)
        
        # Predict on train and test data
        train_predict = model.predict(X_train)
        test_predict = model.predict(X_test)
        
        # Inverse transform to get actual values
        train_predict = normalizer.inverse_transform(train_predict)
        test_predict = normalizer.inverse_transform(test_predict)
        
        # Combine predictions
        test = np.vstack((train_predict[-len(test_predict):], test_predict))
        
        # Prepare future prediction
        fut_inp = ds_test[-time_stamp:].reshape(1, -1)
        tmp_inp = list(fut_inp[0])
        
        # Predict next 30 days
        lst_output = []
        i = 0
        while(i < 30):
            if(len(tmp_inp) > time_stamp):
                fut_inp = np.array(tmp_inp[1:])
                fut_inp = fut_inp.reshape(1, -1)
                fut_inp = fut_inp.reshape((1, time_stamp, 1))
                yhat = model.predict(fut_inp, verbose=0)
                tmp_inp.extend(yhat[0].tolist())
                tmp_inp = tmp_inp[1:]
                lst_output.extend(yhat.tolist())
                i = i + 1
            else:
                fut_inp = fut_inp.reshape((1, time_stamp, 1))
                yhat = model.predict(fut_inp, verbose=0)
                tmp_inp.extend(yhat[0].tolist())
                lst_output.extend(yhat.tolist())
                i = i + 1
        
        # Transform predictions back to original scale
        lst_output = normalizer.inverse_transform(lst_output)
        
        # Create tabs for prediction and recommendation
        tab1, tab2 = st.tabs(["Stock Prediction", "Buy/Sell Recommendation"])
        
        with tab1:
            # Plot historical and predicted data using Plotly
            historical_data = go.Scatter(x=data.index, y=normalizer.inverse_transform(ds_scaled).flatten(), mode='lines', name='Historical Data')
            predicted_data = go.Scatter(x=pd.date_range(start=data.index[-1], periods=30).tolist(), y=lst_output.flatten(), mode='lines', name='Predicted Data', line=dict(color='red'))
            
            prediction_start_line = go.Scatter(x=[data.index[-1], data.index[-1]], y=[0, np.max(normalizer.inverse_transform(ds_scaled))], mode='lines', name='Prediction Start', line=dict(color='blue', dash='dash'))
            
            layout = dict(title=f'{stock_symbols[selected_stock]} Stock Price Prediction',
                          xaxis=dict(title='Time'),
                          yaxis=dict(title='Stock Price'))
            
            fig = go.Figure(data=[historical_data, predicted_data, prediction_start_line], layout=layout)
            
            st.plotly_chart(fig)
        
        with tab2:
            # Recommendation System
            predicted_changes = np.diff(lst_output, axis=0)
            positive_changes = np.sum(predicted_changes > 0)
            negative_changes = np.sum(predicted_changes <= 0)
            
            # Calculate buy/sell percentages
            buy_percentage = positive_changes / len(predicted_changes) * 100
            sell_percentage = negative_changes / len(predicted_changes) * 100
            
            # Plot the recommendation using Plotly
            labels = ['Buy', 'Sell']
            sizes = [buy_percentage, sell_percentage]
            colors = ['green', 'red']
            explode = (0.1, 0)
            
            fig_recommendation = go.Figure(data=[go.Pie(labels=labels, values=sizes, hole=0.3, pull=[0.1, 0])])
            fig_recommendation.update_layout(title='Buy/Sell Recommendation', showlegend=True)
            
            st.plotly_chart(fig_recommendation)
            
            # Print the recommendation
            if buy_percentage > sell_percentage:
                recommendation = 'Buy'
            else:
                recommendation = 'Sell'
            
            st.write(f"Recommendation: {recommendation}")
            st.write(f"Buy: {buy_percentage:.2f}%, Sell: {sell_percentage:.2f}%")
            
            # Function to predict and recommend for a customizable number of days
            def predict_and_recommend(days):
                next_days_pred = lst_output[:days]
                next_days_changes = np.diff(next_days_pred, axis=0)
            
                buy_days = np.sum(next_days_changes > 0)
                sell_days = np.sum(next_days_changes <= 0)
            
                # Calculate buy/sell percentages for the given number of days
                buy_percentage_days = buy_days / len(next_days_changes) * 100
                sell_percentage_days = sell_days / len(next_days_changes) * 100
            
                # Plot the recommendation for the given number of days using Plotly
                sizes_days = [buy_percentage_days, sell_percentage_days]
                fig_recommendation_days = go.Figure(data=[go.Pie(labels=labels, values=sizes_days, hole=0.3, pull=[0.1, 0])])
                fig_recommendation_days.update_layout(title=f'Buy/Sell Recommendation for Next {days} Days', showlegend=True)
                
                st.plotly_chart(fig_recommendation_days)
            
                # Print the recommendation for the given number of days
                if buy_percentage_days > sell_percentage_days:
                    recommendation_days = 'Buy'
                else:
                    recommendation_days = 'Sell'
            
                st.write(f"Recommendation for next {days} days: {recommendation_days}")
                st.write(f"Buy: {buy_percentage_days:.2f}%, Sell: {sell_percentage_days:.2f}%")
            
            # User input for number of days
            num_days = st.sidebar.number_input("Enter number of days for prediction", value=30)
            predict_and_recommend(num_days)

# Live Market page
elif page == "Live Market":
    st.subheader("Live Stock Market Data")

    # Fetch live market data for the selected stocks
    selected_stock = st.sidebar.selectbox("Select Stock Symbol", list(stock_symbols.keys()))
    
    if selected_stock:
        ticker = yf.Ticker(selected_stock)
        live_price = ticker.history(period="1d")['Close'][0]
        
        st.write(f"Current price of {stock_symbols[selected_stock]} ({selected_stock}): ${live_price:.2f}")

        live_data = ticker.info
        st.write("Live Data:")
        for key, value in live_data.items():
            st.write(f"{key}: {value}")

# Market Sentiment page
elif page == "Market Sentiment":
    st.subheader("Market Sentiment Analysis")

    # Fetch stock news based on selected symbol
    selected_stock = st.sidebar.selectbox("Select Stock Symbol", list(stock_symbols.keys()))
    
    if selected_stock:
        st.write(f"Analyzing sentiment for: {stock_symbols[selected_stock]} ({selected_stock})")
        
        # Define the date range
        end_date = datetime.today()
        start_date = end_date - timedelta(days=30)  # Past month
        
        # Fetch stock news
        ticker = yf.Ticker(selected_stock)
        news = ticker.news
        
        # Filter news for the past month and analyze sentiment
        news_data = []
        for article in news:
            publish_time = datetime.fromtimestamp(article['providerPublishTime'])
            if publish_time >= start_date:
                title = article['title']
                link = article['link']
                # Analyze sentiment
                sentiment = sia.polarity_scores(title)
                news_data.append({
                    'Title': title,
                    'Published at': publish_time.strftime('%Y-%m-%d %H:%M:%S'),
                    'Link': link,
                    'Compound': sentiment['compound'],
                    'Positive': sentiment['pos'],
                    'Neutral': sentiment['neu'],
                    'Negative': sentiment['neg']
                })
        
        # Create a DataFrame
        df = pd.DataFrame(news_data)

        if not df.empty:
            # Display sentiment data
            st.write("Sentiment Data for the past month:")
            st.dataframe(df[['Title', 'Published at', 'Compound', 'Positive', 'Neutral', 'Negative', 'Link']])
            
            # Plot pie chart for sentiment distribution
            sentiment_counts = df[['Positive', 'Neutral', 'Negative']].mean()
            fig_pie = go.Figure(data=[go.Pie(labels=sentiment_counts.index,
                                             values=sentiment_counts.values,
                                             title='Average Sentiment Distribution')])
            st.plotly_chart(fig_pie)
            
            # Plot bar chart for sentiment scores
            fig_bar = go.Figure(data=[
                go.Bar(name='Positive', x=df['Published at'], y=df['Positive']),
                go.Bar(name='Neutral', x=df['Published at'], y=df['Neutral']),
                go.Bar(name='Negative', x=df['Published at'], y=df['Negative'])
            ])
            fig_bar.update_layout(barmode='stack', title='Sentiment Scores Over Time')
            st.plotly_chart(fig_bar)
            
            # Provide a recommendation based on average sentiment
            average_sentiment = df['Compound'].mean()
            recommendation = "Buy" if average_sentiment > 0.2 else "Sell" if average_sentiment < -0.2 else "Hold"
            
            st.write(f"Average Sentiment Score: {average_sentiment:.2f}")
            st.write(f"Recommendation: {recommendation}")
        else:
            st.write("No news articles found for the selected stock in the past month.")
