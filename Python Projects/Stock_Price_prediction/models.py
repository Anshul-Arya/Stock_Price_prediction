# -*- coding: utf-8 -*-
"""
Created on Thu Sep  3 04:44:35 2020

@author: Anshul Arya
"""

#------------------------------------------------#
# Import Libraries
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.style as style
import seaborn as sns

from keras.models import Sequential
from keras.layers import LSTM, Dropout, Dense

from sklearn.preprocessing import MinMaxScaler
from Functions import add_fignum

pd.set_option("display.Max_columns",10)
#------------------------------------------------#

# Read the dataset
df = pd.read_csv("file:///C:/Users/Anshul Arya/Desktop/DataScience/" \
                     "Stock-Price-Prediction-Project-Code/" \
                     "NSE-Tata-Global-Beverages-Limited/" \
                     "NSE-Tata-Global-Beverages-Limited.csv")

print(df.head())
# Check the datatypes of each column
df.dtypes
"""
The Column 'Date' is in Object Format, let;s change it to proper Datetime format
"""
df["Date"] = pd.to_datetime(df["Date"], format = "%Y-%m-%d")
# Set Indec to Date Values for Time Series Analysis
df.index = df["Date"]

# Analyzing the Closing prices from dataset
figtext_args, figtext_kwargs = add_fignum("Fig 1. Closing Price")
style.use("fivethirtyeight")
plt.figure(figsize = (16,8))
plt.plot(df["Close"], label = "Close Price History")
plt.title("Closing Price History", loc = "left", weight = "bold", 
          fontdict = dict(fontsize = 18, color = "darkblue"))
plt.xlabel("Year", weight = "bold")
plt.ylabel("Closing Price", weight = "bold")
plt.figtext(*figtext_args, **figtext_kwargs)

# Sort the dataset on date time and filter "Date" and "close Price"
data = df.sort_index(ascending = True, axis = 0)
new_data = pd.DataFrame(index = range(0,len(df)), columns = ["Date", "Close"])

for i in range(0,len(df)):
    new_data["Date"][i] = data["Date"][i]
    new_data["Close"][i] = data["Close"][i]

new_data.index = new_data.Date
new_data.drop("Date", axis = 1, inplace = True)
# Normalize the new filtered dataset
dataset = new_data.values

train = dataset[0:987,:]
test = dataset[987:,:]

scalar = MinMaxScaler(feature_range=(0,1))
scaled_data = scalar.fit_transform(dataset)

x_train, y_train = [], []

for i in range(60, len(train)):
    x_train.append(scaled_data[i-60:i,0])
    y_train.append(scaled_data[i,0])

x_train, y_train = np.array(x_train), np.array(y_train)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1],1))

# Build and Train LSTM Model
lstm_model = Sequential()
lstm_model.add(LSTM(units = 50, return_sequences = True, 
                    input_shape = (x_train.shape[1],1)))
lstm_model.add(LSTM(units=50))
lstm_model.add(Dense(1))

inputs_data = new_data[len(new_data) - len(test) - 60:].values
inputs_data = inputs_data.reshape(-1,1)
inputs_data = scalar.transform(inputs_data)

lstm_model.compile(loss='mean_squared_error', optimizer="adam")
lstm_model.fit(x_train, y_train, epochs=1, batch_size=1, verbose=2)

# Take a sample of dataset to make stock price predictions using the LSTM model
X_test = []
for i in range(60, inputs_data.shape[0]):
    X_test.append(inputs_data[i-60:i,0])

X_test = np.array(X_test)
X_test = np.reshape(X_test,(X_test.shape[0],X_test.shape[1],1))
pred_closing_price = lstm_model.predict(X_test)
pred_closing_price = scalar.inverse_transform(pred_closing_price)

# Save the LSTM Model
lstm_model.save("saved_model.h5")

# Visualize the predicted stocks cost with actual cost
train_data = new_data[:987]
valid_data = new_data[987:]
valid_data["Predicted"] = pred_closing_price

figtext_args, figtext_kwargs = add_fignum("Fig 2. Predicted Closing Price")
style.use("fivethirtyeight")
plt.figure(figsize = (16,8))
plt.plot(train_data["Close"])
plt.plot(valid_data[['Close', 'Predicted']])
plt.title("Closing Price Prediction", loc = "left", weight = "bold", 
          fontdict = dict(fontsize = 18))
plt.xlabel("Year", weight = "bold")
plt.ylabel("Closing Price", weight = "bold")
plt.figtext(*figtext_args, **figtext_kwargs)