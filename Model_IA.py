import datetime
import math

import keras.models
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from ModelTools import ModelTools


class Model:
    def __init__(self, currency: str, start_day: int, num_days: int):
        self.currency = currency.upper()
        self.start_day = self.get_start_day(start_day)
        self.num_days = num_days
        self.model_tools = ModelTools(self.currency, self.format_start_day(self.start_day), "")
        self.model = self.get_model(self.currency)
        self.model_filepath = ''
        self.dataset = None
        self.scaler = None
        self.obj_scaler = None
        self.scaled_data = None
        self.fragment_data = None

        self.prepare_project_data()

    def get_model(self, currency: str):

        if currency == 'BTCUSDT':
            self.model_filepath = '../models/btc_predict_model-1.30.1.h5'
            return keras.models.load_model('../models/btc_predict_model-1.30.1.h5')

        elif currency == 'ETCHUSDT':
            self.model_filepath = '../models/eth_predict_model-1.30.0.h5'
            return keras.models.load_model('../models/eth_predict_model-1.30.0.h5')

    def get_start_day(self, sd: int):
        if sd == 'default':
            return datetime.datetime.fromisoformat('2020-12-16')
        elif len(str(sd)) == 13:
            return datetime.datetime.fromtimestamp(sd / 1000)
        elif len(str(sd)) == 10:
            return datetime.datetime.fromtimestamp(sd)
        else:
            return datetime.datetime.now()

    def format_start_day(self, sd: datetime.datetime):
        return f'{sd.strftime("%d")} {sd.strftime("%b")}, {sd.strftime("%Y")}'

    def visualize(self, predictions):
        plt.figure(figsize=(16, 8))
        plt.title('Model')
        plt.xlabel('Date', fontsize=18)
        plt.ylabel('Close Price USD ($)', fontsize=18)
        plt.plot(predictions[:, [0]])
        plt.legend(['Predictions'], loc='lower right')
        plt.show()

    def prepare_project_data(self):
        klines = self.model_tools.klines
        rsi = self.model_tools.rsi
        dmi = self.model_tools.dmi
        rsi_index = len(rsi) - len(klines)
        dmi_index = len(dmi) - len(klines)
        data_set_dict = {
            'Close': [],
            'High': [],
            'Low': [],
            '+DI': [],
            '-DI': [],
            'ADX': [],
            'RSI': [],
            'Date': []
        }

        for i in range(0, len(klines) - 1):
            dt = datetime.datetime.fromtimestamp(klines[i][6] / 1000)
            y = dt.strftime("%Y")

            kl = klines[i]
            r = rsi[rsi_index if rsi_index >= 0 else 0]
            d = dmi[dmi_index if dmi_index >= 0 else 0]

            if kl[6] == r['date'] == d['date']:
                data_set_dict['Close'].append(float(kl[4]))
                data_set_dict['High'].append(float(kl[2]))
                data_set_dict['Low'].append(float(kl[3]))
                data_set_dict['+DI'].append(d['DI']['positive'])
                data_set_dict['-DI'].append(d['DI']['negative'])
                data_set_dict['ADX'].append(d['ADX'])
                data_set_dict['RSI'].append(r['rsi'])
                data_set_dict['Date'].append(datetime.datetime.fromtimestamp(kl[6] / 1000))

            rsi_index += 1
            dmi_index += 1

        data_set_dict = pd.DataFrame(data=data_set_dict)

        data = data_set_dict.filter(['Close', '+DI', '-DI', 'ADX', 'RSI'])
        self.dataset = data.values
        start_data_len = math.ceil(len(self.dataset) - 360)  # TODO

        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.scaled_data = self.scaler.fit_transform(self.dataset)
        self.obj_scaler = self.scaler.fit(self.dataset)
        self.fragment_data = self.scaled_data[start_data_len:, :]

    def predict_asset(self):
        x_data = []
        for i in range(30, len(self.fragment_data) + 1):
            if i == len(self.fragment_data):
                x_data.append(self.fragment_data[i - 30:, :])
            else:
                x_data.append(self.fragment_data[i - 30:  i, :])

        x_data = np.array(x_data)
        x_data = np.reshape(x_data, (x_data.shape[0], x_data.shape[1], x_data.shape[2]))

        predictions = self.model.predict(x_data)

        x_data_loop = []

        for i in range(0, self.num_days):
            if i == 0:
                x_data_loop.append(predictions[len(predictions) - 31:, [0, 1, 2, 3, 4]])
            else:
                if i == 1:
                    predictions = predictions.tolist()
                x_data_loop = np.array(x_data_loop)
                pr = self.model.predict(x_data_loop)
                predictions.append(pr.tolist()[0])
                x_data_loop = x_data_loop.tolist()
                x_1 = x_data_loop[0]
                x_1.remove(x_1[0])
                x_1.append(pr[0, [0, 1, 2, 3, 4]].tolist())

        predictions = np.array(predictions)
        predictions = self.obj_scaler.inverse_transform(predictions)

        self.visualize(predictions)

    def retraining_model(self):
        x_data = []
        y_data = []
        for i in range(30, len(self.fragment_data)):
            x_data.append(self.fragment_data[i - 30:  i, :])
            y_data.append(self.fragment_data[i, :])

        x_data = np.array(x_data)
        y_data = np.array(y_data)

        x_data = np.reshape(x_data, (x_data.shape[0], x_data.shape[1], x_data.shape[2]))

        self.model.fit(x_data, y_data, batch_size=1, epochs=100)
        self.model.save(self.model_filepath)


"""
# f'{date.strftime("%d")} {date.strftime("%b")}, {date.strftime("%Y")}'
    bot = ModelTools('BTCUSDT', "22 Sep 2020", f'{date.strftime("%d")} {date.strftime("%b")}, {date.strftime("%Y")}')
    klines = bot.klines
    rsi = bot.rsi
    dmi = bot.dmi
    rsi_index = len(rsi) - len(klines)
    dmi_index = len(dmi) - len(klines)
    test_set = {
        'Close': [],
        'High': [],
        'Low': [],
        '+DI': [],
        '-DI': [],
        'ADX': [],
        'RSI': [],
        'Date': []
    }

    for i in range(0, len(klines) - 1):
        dt = datetime.datetime.fromtimestamp(klines[i][6] / 1000)
        y = dt.strftime("%Y")

        kl = klines[i]
        r = rsi[rsi_index if rsi_index >= 0 else 0]
        d = dmi[dmi_index if dmi_index >= 0 else 0]

        if kl[6] == r['date'] == d['date']:
            test_set['Close'].append(float(kl[4]))
            test_set['High'].append(float(kl[2]))
            test_set['Low'].append(float(kl[3]))
            test_set['+DI'].append(d['DI']['positive'])
            test_set['-DI'].append(d['DI']['negative'])
            test_set['ADX'].append(d['ADX'])
            test_set['RSI'].append(r['rsi'])
            test_set['Date'].append(datetime.datetime.fromtimestamp(kl[6] / 1000))

        rsi_index += 1
        dmi_index += 1

    test_set = pd.DataFrame(data=test_set)

    data = test_set.filter(['Close', '+DI', '-DI', 'ADX', 'RSI'])
    dataset = data.values
    test_data_len = math.ceil(len(dataset) - 360)  # TODO
    date = np.array(test_set.filter(['Date']).values)[test_data_len:]

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(dataset)

    scaler_output = MinMaxScaler(feature_range=(0, 1))
    scaled_data_output = scaler.fit_transform(dataset[:, [0]])
    obj = scaler_output.fit(dataset)
    model = keras.models.load_model('btc_predict_model-1.30.1.h5')
    test_data = scaled_data[test_data_len:, :]
    """
