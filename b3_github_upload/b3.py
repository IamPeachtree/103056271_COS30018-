# File: stock_prediction.py
# Authors: Bao Vo and Cheong Koo
# Date: 14/07/2021(v1); 19/07/2021 (v2); 02/07/2024 (v3)

# Code modified from:
# Title: Predicting Stock Prices with Python
# Youtuble link: https://www.youtube.com/watch?v=PuZY9q-aKLw
# By: NeuralNine

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pandas_datareader as web
import datetime as dt
import tensorflow as tf

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, InputLayer
# b2 inclusions
from sklearn.model_selection import train_test_split
# b3 inclussion
import mplfinance as mpf

# DATA_SOURCE = "yahoo"
COMPANY = 'CBA.AX'
FEATURES = ["Adj Close", "Volume", "Open", "High", "Low"]
TRAIN_START = '2020-01-01'     # Start date to read
TRAIN_END = '2024-07-02'       # End date to read
# Number of days to look back to base the prediction
PREDICTION_DAYS = 60 # Original


import yfinance as yf

# Get the data for the stock AAPL

# fit values between 0 and 1
scaler = MinMaxScaler(feature_range=(0, 1)) 


def scale_features(df, features):
    # copy the dataframe to not mess with the original 
    df2 = df.copy()
    # set to only the columns that are chosen as features
    
    df2 = df2.loc[:, features]
    # I am not sure why this stopped it from crashing but it did
    scaler_features = MinMaxScaler(feature_range=(0, 1))
    # remove the non-feature column
    #df2_features = df2.drop(columns=[features])
    # Scale the features with the new svaler
    df2_scaled = scaler_features.fit_transform(df2)
    # Convert back
    df2_scaled = pd.DataFrame(df2_scaled, columns=df2.columns)
    #save for future access if wanted
    df2_scaled.to_csv("scaled_features.csv", index = False)

def split_date(df, features, split_date_from, to_shuffle):
#date spllit logic
    print("splitting by date")
    df['Date'] = pd.to_datetime(df.index)
    if to_shuffle:
        df = df.sample(frac = 1)
    # Split the data by date with pandas datetime format. this  will be the format to split by date time
    # this will work even if the chosen date ins not in the dataset
    train_df = df[df['Date'] < pd.to_datetime(split_date_from)] # all columns greater than date
    test_df = df[df['Date'] >= pd.to_datetime(split_date_from)] # collumns less than or including date
    # Create training data
    #apply scaler transformation to dataset part
    train_df = scaler.fit_transform(train_df[features].values.reshape(-1, 1)).flatten()
    x_train = np.array([train_df[i - PREDICTION_DAYS:i] for i in range(PREDICTION_DAYS, len(train_df))])
    y_train = train_df[PREDICTION_DAYS:]
    # Create testing data
    #apply scaler transformation to dataset part
    test_df = scaler.transform(test_df[features].values.reshape(-1, 1)).flatten()
    x_test = np.array([test_df[i - PREDICTION_DAYS:i] for i in range(PREDICTION_DAYS, len(test_df))])
    y_test = test_df[PREDICTION_DAYS:]

    return x_train, x_test, y_train, y_test

def standard_split(df, features, test_size, to_shuffle):
    scaled_data = scaler.fit_transform(df[features].values.reshape(-1, 1)).flatten()
    # set up x and y inputs for train_test_split
    x = np.array([scaled_data[i-PREDICTION_DAYS:i] for i in range(PREDICTION_DAYS, len(scaled_data))])
    y = scaled_data[PREDICTION_DAYS:]
    # Split the data. test_size is the ratio split. shuffle is set to false
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=1, shuffle=to_shuffle)
    
    return x_train, x_test, y_train, y_test

#default arguments included
def data_process(company, data_start, data_end, features, split_by_date = False, 
                 split_date_from = "2022-3-7", to_shuffle = False,save= True, test_size = 0.3, scales = False, 
                 load_data = False, load_csv_file_name ="dataset.csv"):
    
    #if load = true
    if load_data:
        data = pd.read_csv(load_csv_file_name)
    else:    
        data = yf.download(company, data_start, data_end)
        df = pd.DataFrame(data)
        if save == True:
            df.to_csv("dataset.csv")
    # deal with NaN
    df.dropna()
    if scales:  # if true scale the features columns aka everything other than the prediction target
        scale_features(df, features)
    if split_by_date:
       x_train, x_test, y_train, y_test = split_date(df, features, split_date_from)    
    else:
        x_train, x_test, y_train, y_test = standard_split(df, features, test_size, to_shuffle)
        # Reshape
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    # Generate model inputs for test data
    actual_prices_scaled = pd.DataFrame(y_test).values
    actual_prices = scaler.inverse_transform(actual_prices_scaled.reshape(-1, 1)).flatten()
    
    total_dataset = df[features]
    
    model_inputs = total_dataset[len(total_dataset) - len(actual_prices) - PREDICTION_DAYS:].values 
    model_inputs = model_inputs.reshape(-1, 1)
    model_inputs = scaler.transform(model_inputs)
    
    # Generate x_test using the same method as in the provided code
    x_test = np.array([model_inputs[i-PREDICTION_DAYS:i, 0] for i in range(PREDICTION_DAYS, len(model_inputs))])
    # Reshape x_test
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    # return all processed values required for training and predictions
    return x_train, x_test, y_train, y_test, actual_prices, model_inputs    

x_train, x_test, y_train, y_test, actual_prices, model_inputs  = data_process(COMPANY, TRAIN_START, TRAIN_END, FEATURES, split_by_date = False, 
                 split_date_from = "2022-3-7", to_shuffle = False,save= True, test_size = 0.3, scales = True, 
                 load_data = False, load_csv_file_name ="dataset.csv")

#------------------------------------------------------------------------------
# Build the Model
## TO DO:
# 1) Check if data has been built before. 
# If so, load the saved data
# If not, save the data into a directory
# 2) Change the model to increase accuracy?
#------------------------------------------------------------------------------
model = Sequential() # Basic neural network
model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=50))
model.add(Dropout(0.2))
model.add(Dense(units=1)) 

model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(x_train, y_train, epochs=25, batch_size=32)


predicted_prices = model.predict(x_test)
predicted_prices = scaler.inverse_transform(predicted_prices)



def candle_stick_chart(number_of_trade_days = 0):
    num = number_of_trade_days
    #loading datasset. define first column as index and parse_dates if true will parse and represent as a datetime value
    df = pd.read_csv("dataset.csv", index_col=0, parse_dates= True)
    #always start with first entry of dataset
    start_date = df.index[0]
    #if a user choses 1 or more days this will subtract 1 to ensure the index lines up with the chosen values
    if number_of_trade_days != 0:
        number_of_trade_days = number_of_trade_days - 1

    end_date = df.index[number_of_trade_days]
    #ranged_df = pd.date_range(start= start_date, end= end_date)
    #date_range did not work so a slice of the dataset was used
    ranged_df = df.loc[start_date: end_date]
    #assign table name. if num = 0 set to 1 as first index is one day
    if num == 0:
        num = 1
    #using 
    mpf.plot(ranged_df ,type='candle', title=f'Shares {num} trade day(s)', ylabel = "Price $")

def box_plot(consecutive_trade_days = 0):
    num = consecutive_trade_days
#loading datasset. define first column as index and parse_dates if true will parse and represent as a datetime value
    df = pd.read_csv("dataset.csv", index_col=0, parse_dates= True)
    #always start with first entry of dataset
    start_date = df.index[0]
    #if a user choses 1 or more days this will subtract 1 to ensure the index lines up with the chosen values
    if consecutive_trade_days != 0:
        consecutive_trade_days = consecutive_trade_days - 1
    end_date = df.index[consecutive_trade_days]
    #date_range did not work so a slice of the dataset was used
    ranged_df = df.loc[start_date: end_date]
    #volume has been excluded from the target columns as it is a large int and will outscale all the other numbers
    target_columns = ["Open","High","Low","Close","Adj Close"]
    #determin figure size
    plt.figure(figsize=(10, 10))
    # plot as box plot. for each column value in taget columns given the index value of the tupple. 
    # add to box plot with name of target column
    plt.boxplot([ranged_df[col] for col in target_columns], labels=target_columns)
    #assign table name. if num = 0 set to 1 as first index is one day
    if num == 0:
        num = 1
    plt.title(f"Box Plot {num} day(s)")
    #assign axis name
    plt.ylabel("Values")
    #load to screen
    plt.show()

candle_stick_chart(10)
box_plot(10)

plt.plot(actual_prices, color="black", label=f"Actual {COMPANY} Price")
plt.plot(predicted_prices, color="green", label=f"Predicted {COMPANY} Price")
plt.title(f"{COMPANY} Share Price")
plt.xlabel("Time")
plt.ylabel(f"{COMPANY} Share Price")
plt.legend()
plt.show()


real_data = [model_inputs[len(model_inputs) - PREDICTION_DAYS:, 0]]
real_data = np.array(real_data)
real_data = np.reshape(real_data, (real_data.shape[0], real_data.shape[1], 1))

prediction = model.predict(real_data)
prediction = scaler.inverse_transform(prediction)
print(f"Prediction: {prediction}")
