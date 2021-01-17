import math
import pandas_datareader as web
import numpy as np
import pandas as import pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')

#Get stock quote
df = web.DataReader('AAPL', data_source='yahoo', start='2012-01-01', end='2019-12-17')

#Get the number of rows and column in dataset
df.shape
#Visualize the closing price history
plt.figure(figsize=(16,8))
plt.title('Close price history')
plt.plot(df['Close'])
plt.xlabel('Data', fontsize=18)
plt.ylabel('Close price usd $', fontsize=18)
plt.show()

