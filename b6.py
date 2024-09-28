import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pandas_datareader as web
import datetime as dt
import tensorflow as tf
import yfinance as yf

from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, LSTM, InputLayer, Bidirectional, GRU, RNN, Conv1D, MaxPooling1D, Flatten, Dense
from tensorflow.keras.callbacks import EarlyStopping

# only get errors
# import tensorflow_decision_forests as tfdf

#import for b6
# import tensorflow_probability as tfp
import pmdarima as pm
#------------------------------------------------------------------------------
DATA_SOURCE = "yahoo"
# COMPANY = "TSLA"
# TRAIN_START = dt.datetime(2012, 5, 23)     # Start date to read
# TRAIN_END = dt.datetime(2020, 1, 7)       # End date to read
COMPANY = "CBA.AX"
TRAIN_START = '2020-01-01'     # Start date to read
TRAIN_END = '2024-07-02'       # End date to read
PRICE_VALUE = "Close"
PREDICTION_DAYS = 60 # Original
SPLIT_TYPE = ["ratio", "date", "random"]

def load_data(company=COMPANY, data_source=DATA_SOURCE, start_date=TRAIN_START, end_date=TRAIN_END, split_by="ratio", ratio=0.8, column_name="Close", store_data=False, store_scalers=False):
    data = yf.download(COMPANY, TRAIN_START, TRAIN_END) # Read data using yahoo
    #data = web.DataReader(COMPANY, DATA_SOURCE, TRAIN_START, TRAIN_END) # Read data using yahoo
    data.dropna(inplace=True)       # Delete NaN values

    PRICE_VALUE = column_name
    scaler = MinMaxScaler(feature_range=(0, 1)) 
    scaled_data = scaler.fit_transform(data[PRICE_VALUE].values.reshape(-1, 1)) 

    input_data = []
    output_data = []
    scaled_data = scaled_data[:,0] # Turn the 2D array back to a 1D array
    #this loop
    for x in range(PREDICTION_DAYS, len(scaled_data)):
        input_data.append(scaled_data[x-PREDICTION_DAYS:x])
        output_data.append(scaled_data[x])

    # turn into numpy array
    input_data, output_data = np.array(input_data), np.array(output_data)
    # reshape input data to format
    input_data = np.reshape(input_data, (input_data.shape[0], input_data.shape[1], 1))

    if split_by=="ratio":
        train_samples = int(ratio * len(input_data))
        x_train = input_data[:train_samples]
        y_train = output_data[:train_samples]
        x_test = input_data[train_samples:]
        y_test = output_data[train_samples:]

    # Please implement split by date and split by random

    if store_data:
        folder_name = COMPANY + TRAIN_START + TRAIN_END + ".csv"
        data.to_csv(folder_name)
        # To load it: data = pd.load_csv(folder_name)

    if store_scalers:
        scaler_filename = "scaler.save"
        joblib.dump(scaler, scaler_filename)
        # To load it: scaler = joblib.load(scaler_filename)

    
    return data, x_train, y_train, x_test, y_test, scaler

data, x_train, y_train, x_test, y_test, scaler = load_data()

# print(x_train)
# print(y_train)
# Visualisation
import plotly.graph_objects as go
window_size = 1
def plot_graph(window_size=window_size):
    data_mean = data.rolling(window_size).mean()
    data_mean = data_mean.iloc[window_size-1 :: window_size, :]
    trace1 = {
            'x': data_mean.index,
            'open': data_mean["Open"],
            'close': data_mean['Close'],
            'high': data_mean["High"],
            'low': data_mean["Low"],
            'type': 'candlestick',
            'name': COMPANY,
            'showlegend': True
        }
    fig = go.Figure(data=[trace1])
    fig.show()

#autoregressive Integrated moving average
# https://www.tensorflow.org/probability/api_docs/python/tfp/sts/AutoregressiveIntegratedMovingAverage
# https://alkaline-ml.com/pmdarima/modules/generated/pmdarima.arima.auto_arima.html
# input needs to be a 1d array of floats. np or panda works
def create_arima(data_1d):
    arima = pm.arima.auto_arima(data_1d, # user input for model. this should be a 1d dataset of pandas series or np
                                start_p = 3, # starting value of p (order)--> number of time lags of auto regressive model (AR)
                                start_q=3,  # starting value of q (order) --> moving average (MA)
                                max_p=6, #  max value for lag time p
                                max_q=6, # max value of moving average (MA)
                                seasonal = True, # should the model be seasonal?
                                stepwise= True, # to use a stepwise algorithm to identify optimal parameters
                                error_action = 'ignore', # if unalbe to fit set behaviour
                                suppress_warnings = True, # don't give me errors
                                trace = True, # to print the status of the fits.
                                max_D = 6, # the maximum value of D being the order of seasonal differnces. D is automaticly selected based on results
                                max_Q = 6 # maximum value of Q which is the order od moving average portion of seasonl modll
                                )
    return arima

arima = create_arima(y_train)
print(arima.summary())


def arima_prediction(arima, period_pred =PREDICTION_DAYS):
    arima_pred = pd.DataFrame(arima.predict(period_pred))
    arima_pred = scaler.inverse_transform(arima_pred)
    return arima_pred


arima_p = arima_prediction(arima, period_pred=1)
print(arima_p.shape)
print(arima_p)

def create_sarimax(arima, data, period_pred = PREDICTION_DAYS):
    # create a dataframe based on the prediction of arima model
    sarimax_fc = pd.DataFrame(arima.predict(period_pred))
    # name the column
    sarimax_fc.columns = ["Sarimax_predictions"]
    # select last index from list
    last_index = data.index[-1]
    # create range of future dates for forcasting starting at the next day 
    #pdtimedelta sets the starting date as the day following the last day in the dataset
    # preiod + 1 is to ensure we are starting from the next day
    #[1:] slice data range to exclude first element to account for original data for last day
    future_dates = pd.date_range(start=last_index + pd.Timedelta(days=1), periods=period_pred + 1)[1:]  # Business days ('B')
    
    # Assign the new index to the predictions DataFrame
    sarimax_fc.index = future_dates
    
    return sarimax_fc

sarimax_predictions = create_sarimax(arima, data, period_pred = 1)
print(sarimax_predictions.shape)
print(sarimax_predictions)
# scale data back to proper number
sarimax_predictions = scaler.inverse_transform(sarimax_predictions)
print(sarimax_predictions)

# reference code
# cannot run. I only get errors with importing forest
# https://www.tensorflow.org/decision_forests
# def create_forest(train_data, test_data):
#     # apply training and testing data for decision tree
#     train_ds = tfdf.keras.pd_dataframe_to_tf_dataset(train_data, label="tree")
#     test_ds = tfdf.keras.pd_dataframe_to_tf_dataset(test_data, label="tree")
#     # create random forest model
#     model_f = tfdf.keras.RandomForestModel()
#     # fit model data
#     model_f.fit(train_ds)
#     model_f.compile(metrics=["accuracy"])
#     # get result
#     result = model_f.evaluate(test_ds, return_dict=True)
#     print(f"Random forest result: {result}")
    
#     return result

# d_tree = create_forest(y_train, y_test)

early_stopping = EarlyStopping(monitor='loss', patience = 5, restore_best_weights=True)

def create_cnn(sequence_length, n_features, kernel_size=3,dropout_rate=0.3,loss="mean_absolute_error", optimizer="adam"):
    model = Sequential()
    # 1 dementional convolution layer for time series with a 64 filter. relu activation converts all negatives into neurons with a value of 0
    model.add(Conv1D(16,kernel_size, padding='same', activation='relu', input_shape=(sequence_length,n_features)))
    #use default arguments. Is for down sampling and helps reduce overfitting
    model.add(MaxPooling1D())
    model.add(Conv1D(8,kernel_size, padding='same',activation = 'relu'))
    #use default argument
    #https://www.tensorflow.org/api_docs/python/tf/keras/layers/MaxPool1D
    # will select the maximum value from the options it is given.
    # reduce overfitting by reducing total parameters, decrease size of inputs for more computational efficiency
    # will maintain the most significant features
    model.add(MaxPooling1D())
    model.add(Conv1D(4,kernel_size, padding='same',activation = 'relu'))
    # a fully connected layer with 64 neurons. combines features extracted by conv1d
    model.add(Dense(16, activation='relu'))
    # standard dropout layer to address overfitting
    model.add(Dropout(dropout_rate))
    # convert multi-dimensional features into 1 dimentional vector
    model.add(Flatten())
    # model.add(Dropout(dropout_rate))
    #single neuron for final value prediciotns
    model.add(Dense(1, activation='relu'))
    model.compile(loss=loss, metrics=["mean_absolute_error"], optimizer=optimizer)
    return model

cnn_model = create_cnn(sequence_length=x_train.shape[1], n_features=1)
cnn_model.fit(x_train, y_train, epochs=100, batch_size=15, callbacks=[early_stopping])

# cnn_predictions = cnn_model.predict(x_test)
# cnn_predictions = scaler.inverse_transform(cnn_predictions)

def create_model(sequence_length, n_features, units=[50,100,150,200], layer_names=[LSTM, GRU, RNN, Dropout, Dense], n_layers=3, dropout_rate=0.3, loss="mean_absolute_error", optimizer="adam", bidirectional=False):
    # You can try "print(x_train.shape)" to see what is the meaning of it. Basically, x_train.shape[1] is the number of rows in the train data.
    model = Sequential()
    for i in range(n_layers):
        if i == 0:
            # first layer
            if bidirectional:
                model.add(Bidirectional(layer_names[0](units[0], return_sequences=True), batch_input_shape=(None, sequence_length, n_features)))
            else:
                model.add(layer_names[0](units[0], return_sequences=True, batch_input_shape=(None, sequence_length, n_features)))
        elif i == n_layers - 1:
            # last layer
            if bidirectional:
                model.add(Bidirectional(layer_names[0](units[0], return_sequences=False)))
            else:
                model.add(layer_names[0](units[0], return_sequences=False))
        else:
            # hidden layers
            if bidirectional:
                model.add(Bidirectional(layer_names[0](units[0], return_sequences=True)))
            else:
                model.add(layer_names[0](units[0], return_sequences=True))
        # add dropout after each layer
        model.add(layer_names[3](dropout_rate))
    model.add(layer_names[4](1, activation="linear"))
    model.compile(loss=loss, metrics=["mean_absolute_error"], optimizer=optimizer)
    return model

model = create_model(sequence_length=x_train.shape[1], n_features=1, bidirectional=True, dropout_rate = 0.2)

#test 1,2,3
# model.fit(x_train, y_train, epochs=25, batch_size=32)
# test 4
model.fit(x_train, y_train, epochs=100, batch_size=15, callbacks=[early_stopping])



y_test = y_test.reshape(-1, 1)
actual_prices = scaler.inverse_transform(y_test)

predicted_prices = model.predict(x_test)
predicted_prices = scaler.inverse_transform(predicted_prices)

plt.plot(actual_prices, color="black", label=f"Actual {COMPANY} Price")
plt.plot(predicted_prices, color="green", label=f"Predicted {COMPANY} Price")
plt.title(f"{COMPANY} Share Price")
plt.xlabel("Time")
plt.ylabel(f"{COMPANY} Share Price")
plt.legend()
plt.show()


real_data = [x_test[len(x_test) - PREDICTION_DAYS:, 0]]
real_data = np.array(real_data)
real_data = np.reshape(real_data, (real_data.shape[0], real_data.shape[1], 1))

prediction = model.predict(real_data)
print(f"No scaling {prediction}")
print(prediction.shape)
prediction = scaler.inverse_transform(prediction)
print(f"Prediction: {prediction}")

cnn_pred = cnn_model.predict(real_data)
cnn_pred = scaler.inverse_transform(cnn_pred)
print(f"cnn prediction: {cnn_pred}")
# ensembling for fancy people
# https://blog.paperspace.com/ensembling-neural-network-models/
    

# for basic ensemble predictions average the result between all he datapoints
def model_ensemble(result_list):
# average out the predictions to account for variations
    pred =0
    for i in result_list:
        pred += i
    pred = pred/ len(result_list)
    print(f"One day ensemble prediction: {pred}")

model_results = [prediction, sarimax_predictions]
# test 1 
# model_ensemble(model_results)


# # test 2
# model_results = [prediction, arima_p]
# model_ensemble(model_results)

#test 3, 4
# model_results = [prediction, sarimax_predictions, arima_p]

# model_ensemble(model_results)

# # test 5
# model_results = [prediction, sarimax_predictions, arima_p]

# model_ensemble(model_results)

# test 6
# model_results = [prediction, sarimax_predictions, arima_p, cnn_pred]

# model_ensemble(model_results)

# test 7, 8
model_results = [prediction, cnn_pred]

model_ensemble(model_results)