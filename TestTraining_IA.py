import json

import keras.saving.save
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
from ModelTools import ModelTools
import datetime
import math
from keras.callbacks import ModelCheckpoint

currencies_list = [
    ['BTCUSDT', '16 Dec 2020', 'btc_predict_model-1.30.2.h5']

]

"""
x['ETHUSDT', '17 Aug 2017', 'eth_predict_model-1.30.0.h5'],
x['BTCUSDT', '17 Aug 2017', 'btc_predict_model-1.30.1.h5']
 x  ['DOTUSDT', '18 Aug 2020', 'dot_predict_model-1.30.0.h5']
        x['AVAXUSDT', '22 Sep 2020', 'avax_predict_model-1.30.0.h5']
        x['ADAUSDT', '17 Apr 2018', 'ada_predict_model-1.30.0.h5'],
        ['BNBUSDT', '06 Nov 2017', 'bnb_predict_model-1.30.0.h5'],
        ['SOLUSDT', '11 Aug 2020', 'sol_predict_model-1.30.0.h5'],
        """

for i in range(0, len(currencies_list)):
    date = datetime.datetime.now()
    ts = date.timestamp()
    ts = ts * 1000
    date = datetime.datetime.fromtimestamp((ts - 86400000) / 1000)
    # f'{date.strftime("%d")} {date.strftime("%b")}, {date.strftime("%Y")}'

    currency = currencies_list[i]

    bot = ModelTools(currency[0], currency[1], f'{date.strftime("%d")} {date.strftime("%b")}, {date.strftime("%Y")}')
    klines = bot.klines
    rsi = bot.rsi
    dmi = bot.dmi
    rsi_index = len(rsi) - len(klines)
    dmi_index = len(dmi) - len(klines)
    trainingSet_alt = {
        'Close': [],
        'High': [],
        'Low': [],
        '+DI': [],
        '-DI': [],
        'ADX': [],
        'RSI': []
    }

    for i in range(0, len(klines)):
        dt = datetime.datetime.fromtimestamp(klines[i][6] / 1000)
        y = dt.strftime("%Y")

        kl = klines[i]
        r = rsi[rsi_index if rsi_index >= 0 else 0]
        d = dmi[dmi_index if dmi_index >= 0 else 0]

        if kl[6] == r['date'] == d['date']:
            trainingSet_alt['Close'].append(float(kl[4]))
            trainingSet_alt['High'].append(float(kl[2]))
            trainingSet_alt['Low'].append(float(kl[3]))
            trainingSet_alt['+DI'].append(d['DI']['positive'])
            trainingSet_alt['-DI'].append(d['DI']['negative'])
            trainingSet_alt['ADX'].append(d['ADX'])
            trainingSet_alt['RSI'].append(r['rsi'])

        rsi_index += 1
        dmi_index += 1

    trainingSet_alt = pd.DataFrame(data=trainingSet_alt)

    data = trainingSet_alt.filter(['Close', '+DI', '-DI', 'ADX', 'RSI'])
    dataset = data.values
    training_data_len = math.ceil(len(dataset))  # TODO

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(dataset)

    scaler_output = MinMaxScaler(feature_range=(0, 1))
    scaled_data_output = scaler.fit_transform(dataset[:, [0]])
    obj = scaler_output.fit(dataset)

    train_data = scaled_data[0: training_data_len, :]

    x_train = []
    y_train = []
    for i in range(30, len(train_data)):
        x_train.append(scaled_data[i - 30: i, [0, 1, 2, 3, 4]])
        y_train.append(scaled_data[i, [0, 1, 2, 3, 4]])

    x_train, y_train = np.array(x_train), np.array(y_train)

    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], x_train.shape[2]))

    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(x_train.shape[1], x_train.shape[2])))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dense(25))
    model.add(Dense(5))

    model.compile(optimizer='adam', loss='mse')

    model.fit(x_train, y_train, batch_size=1, epochs=1500)