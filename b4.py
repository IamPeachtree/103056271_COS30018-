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
import os
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, InputLayer, Bidirectional
# b2 inclusions
from sklearn.model_selection import train_test_split
# b3 inclussion
import mplfinance as mpf
from collections import deque
# b4 includes
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import layers
from tensorflow import keras
# DATA_SOURCE = "yahoo"
COMPANY = 'CBA.AX'
FEATURES = ["Adj Close", "Volume", "Open", "High", "Low"]
TRAIN_START = '2020-01-01'     # Start date to read
TRAIN_END = '2024-07-02'       # End date to read
# Number of days to look back to base the prediction
PREDICTION_DAYS = 50 # Original


import yfinance as yf

# Get the data for the stock AAPL

# fit values between 0 and 1
scaler = MinMaxScaler(feature_range=(0, 1)) 


def load_and_process_data(company, start_date, end_date, feature_columns=['Close', 'Volume', 'Open', 'High', 'Low'], n_steps=50, nan_strategy='drop', split_method='ratio', test_size=0.2, split_by_date=None, save_local=False, local_path=None, scale=True, scaler_type='minmax', shuffle=True):
    """
        Params:
        company(string): the stock we want to load
        start_date(string): the start date
        end_date(string): the end date
        feature_columns (list): list of features to include (e.g. 'Close')
        n_steps(int): number of steps(in this case days) to check in order to make the prediction
        nan_strategy(string): how we will handle missing data
        split_method(string): how we will split the data
        test_size(float): when using ratio split - what percentage of data will be used for testing
        split_by_date(string): when used - the date to split the data
        save_local(bool): do we want to save downloaded data locally?
        local_path(string): path to the local data file
        scale(bool): do we want the data to be scaled?
        scaler_type(string): the type of scaler to use (minmax or standard in this case)
        shuffle(bool): does the training data need to be shuffled?

        returns: 
        x_train,x_test (arrays) : training and testing feature sets
        y_train, y_test (arrays): training and testing target sets
        scaler(obj): scaler object to transform data for LTSM 
        feature_columns(list): list of features
    """
## 1. a. This function will allow you to specify the start date and the end date for the whole dataset as inputs. 
## 1. d. This function will have the option to allow you to store the downloaded data on your local machine for future uses and to load the data locally to save time.
    if local_path and os.path.exists(local_path): ## if there is saved data locally, then import it. 
        data = pd.read_csv(local_path, index_col=0, parse_dates=True) ##https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_csv.html#pandas.read_csv
    else: #otherwise, download the data and if save_local is True then save it locally. 
        data = yf.download(company,start_date,end_date) ##https://pypi.org/project/yfinance/
        if save_local and local_path:
            data.to_csv(local_path) ##https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.to_csv.html

##1. b. This function will allow you to deal with the NaN issue in the data.
    if nan_strategy == 'drop':              ##https://pandas.pydata.org/pandas-docs/stable/user_guide/missing_data.html#dropping-missing-data
        data = data.dropna()  #drop the rows with NaN values
    elif nan_strategy == 'fill_mean':
        data = data.fillna(data.mean()) #fill the NaN values with he mean of each column
    elif nan_strategy == 'fill_median':
        data = data.fillna(data.median())#fill the NaN values with he median of each column
    else: 
        data = data.dropna()
    data = data[feature_columns] ##use the feature_columns
##1. e. This function will also allow you to have an option to scale your feature columns and store the scalers in a data structure to allow future access to these scalers.
##https://machinelearningmastery.com/standardscaler-and-minmaxscaler-transforms-in-python/
    scaler = None ##initialise scaler
    if scale:
        if scaler_type =='minmax':
            scaler = MinMaxScaler() ##https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html Scales between 0 and 1
        elif scaler_type =='standard': 
            scaler = StandardScaler() ##https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html#sklearn.preprocessing.StandardScaler Standardises data
        else:
            scaler = MinMaxScaler() ##default scaler is minmax 
        data[feature_columns] = scaler.fit_transform(data[feature_columns])

    X, y = [], [] ## feature sequences and target prices respectively 
    sequences = deque(maxlen=n_steps) ## stores last 'n' prices

    for i in range(len(data)):
        sequences.append(data.iloc[i].values)
        if len(sequences) == n_steps:
            X.append(np.array(sequences))
            y.append(data['Close'].iloc[i])

    #converts x and y into numpy arrays
    X,y = np.array(X), np.array(y)

##1. c. This function will also allow you to use different methods to split the data into train/test data; e.g. you can split it according to some specified ratio of train/test and you can specify to split it by date or randomly.
    if split_method == 'ratio': ##split by ratio 
        split_index = int(len(X) * (1 - test_size)) ##https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html
        X_train, X_test = X[:split_index], X[split_index:]
        y_train, y_test = y[:split_index], y[split_index:]
    elif split_method == 'date' and split_by_date:
        split_date = pd.to_datetime(split_by_date) ##split based on dates
        train_data = data.loc[:split_date]
        test_data = data.loc[split_date:]
        X_train = X[:len(train_data)]
        X_test = X[len(train_data):]
        y_train = y[:len(train_data)]
        y_test = y[len(train_data):]
    else:
        raise ValueError("Error! Invalid split_method.")
    
    if shuffle: 
        indices = np.arange(len(X_train))
        np.random.shuffle(indices)
        X_train, y_train = X_train[indices], y_train[indices]

    return X_train, X_test, y_train, y_test, scaler, feature_columns 

COMPANY = 'CBA.AX'
TRAIN_START = '2020-01-01'     # Start date to read
TRAIN_END = '2023-08-01'       # End date to read

X_train, X_test, y_train, y_test, scaler , feature_columns = load_and_process_data (
    company=COMPANY,
    start_date=TRAIN_START, 
    end_date=TRAIN_END,
    nan_strategy='fill_mean',
    feature_columns=['Close','Volume', 'Open', 'High', 'Low'],
    n_steps= 50, 
    split_method='ratio',
    test_size=0.2,
    save_local=True,
    local_path='CBA_data.csv',
    scale=True,
    scaler_type='minmax',
    shuffle = True
)
#------------------------------------------------------------------------------
# Build the Model
## TO DO:
# 1) Check if data has been built before. 
# If so, load the saved data
# If not, save the data into a directory
# 2) Change the model to increase accuracy?
#------------------------------------------------------------------------------
# new addition to ensure data does not get overtrained/ over fit via unessisary repiitions and restoring best model
# from tensorflow.keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='loss', patience = 5, restore_best_weights=True)

# argumnets and default arguments to be used
def create_model(sequence_length, n_features, neurons =256, dl_network="CNN", n_layers=2, dropout=0.2,
                loss="mean_absolute_error", optimizer="rmsprop", bidirectional=False, l2 = 0.1):
    # create a model with linear layer stack
    model = Sequential()
    # added in l2 regularisaiton into the model to help against overfitting.
    # one thing this does is increase the error in the training set
    # this will shink the coeficient towards 0 but not to zero thus retaining all features with reduced influence
    # balance features over feature selection
    # lower bias but higher variance
    # https://medium.com/@fernando.dijkinga/explaining-l1-and-l2-regularization-in-machine-learning-2356ee91c8e3
    # https://johnthas.medium.com/regularization-in-tensorflow-using-keras-api-48aba746ae21
    # loop for creating total layers in model based on user input or layers. each layer will consist of of x layers
    for i in range(n_layers):
        # if at the start of the loop
        if i == 0:
            # first layer
            # if bidirectional is true all layers will be of bidirectional type which allows the model to process inputs in forwards and backwards. 
            # return_sequence = true is because further layers require full sequences
            # batch_input_shape specifies the shape o the input. None will allow for variable batch sizes
            if bidirectional:
                model.add(Bidirectional(dl_network(neurons, return_sequences=True), batch_input_shape=(None, sequence_length, n_features)))
            else:
                #add a layer based on user argument with chosen number of neurons
                model.add(dl_network(neurons, return_sequences=True, batch_input_shape=(None, sequence_length, n_features)))
        #if i is equal to the final value in the number of layers. last itteration of the loop
        elif i == n_layers - 1:
            # last layer
            # does not have batch_shape and return sequence is faluse as this is the last layer that is added
            # if the if statement is met and it does not pass to another normal dl_network layer. it goes to the dense layers
            if bidirectional:
                model.add(Bidirectional(dl_network(neurons, return_sequences=False)))
            else:
                #add a layer based on user argument with chosen number of neurons
                model.add(dl_network(neurons, return_sequences=False))
       # in all other instances of the loop that is not the first nor the last
        else:
            # hidden layers
            # intermedate layer that is between first and last layers.
            # return sequence is true due to pasing onto eachother
            if bidirectional:
                model.add(Bidirectional(dl_network(neurons, return_sequences=True)))
            else:
                model.add(dl_network(neurons, return_sequences=True))
        # add dropout after each layer at the end of any itteration fo the loop.
        # this will remove a random percentage of the neurones in the model to create a more resilient model that does not rely on
        model.add(Dropout(dropout))
    #new addition flatten layer
    # flatten data to a one-dimensional vector or the fully conneted layer
    model.add(layers.Flatten())
    # additional dense layer for relu activation which will change any values bellow 0 to 0 and any above are unchanged
    model.add(Dense(4, activation='relu', kernel_regularizer= keras.regularizers.l2(l2)))
    #creates the final fully connected layer of 1 neuron
    # linear activation is used fro regression
    model.add(Dense(1, activation="linear"))
    # compile and return the model with its 
    model.compile(loss=loss, metrics=["mean_absolute_error"], optimizer=optimizer)
    return model
# arguments for model
model = create_model(PREDICTION_DAYS, len(FEATURES), loss='huber_loss', neurons=128, dl_network= LSTM, n_layers= 4,
                    dropout=0.3, optimizer='adam', bidirectional=False, l2 = 0.01)
# adam optimizer is reliable
model.compile(optimizer='adam', loss='mean_squared_error')
# added callbacks to the fitting
# batch size is the number of samples fed to each iteration in the training process. So it is th number of times it passes through a neural network
model.fit(X_train, y_train, epochs=40, batch_size=20, callbacks=[early_stopping])




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


predicted_prices = model.predict(X_test)
#predicted_prices = scaler.inverse_transform(predicted_prices)


plt.plot(y_test, color="black", label=f"Actual {COMPANY} Price")
plt.plot(predicted_prices, color="green", label=f"Predicted {COMPANY} Price")
plt.title(f"{COMPANY} Share Price")
plt.xlabel("Time")
plt.ylabel(f"{COMPANY} Share Price")
plt.legend()
plt.show()


real_data = [X_test[len(X_test) - PREDICTION_DAYS:, 0]]
real_data = np.array(real_data)

# error at reshape cannoty reshape array of size 250 into shape(1,50,1)
# real_data = np.reshape(real_data, (real_data.shape[0], real_data.shape[1], 1))

prediction = model.predict(real_data)
prediction = scaler.inverse_transform(prediction)
print(f"Prediction: {prediction}")
