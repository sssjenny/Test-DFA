import configparser

import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
from keras.layers import *
from keras.models import *
from keras import layers
from keras.optimizers import RMSprop
from keras import optimizers

'''
Function: 读取文件
Variable: 
    df_data_5minute: 用于数据分析
    origin_data: 用于最后生成csv文件时提取日期列
'''
def compute_r2(y_true, y_pred):
    molecular = np.power(y_true - y_pred, 2).sum()
    denominator = np.power(y_true - np.mean(y_true), 2).sum()

    return 1 - molecular / denominator

def dfa_trained_auto():
    import pandas as pd
    df_data_5minute = pd.read_csv('6m.csv')
    origin_data = pd.read_csv('6m.csv')

    #选取一定日期范围的数据，并删除其中
    origin_data = origin_data[:19200]
    df_data_5minute = df_data_5minute[:19200]
    df_data_5minute.drop('Unnamed: 0', axis=1, inplace=True)

    df=df_data_5minute
    close = df['close']
    df.drop(labels=['close'], axis=1,inplace = True)
    df.insert(0, 'close', close)
    origin_data_train = df.iloc[:int(df.shape[0] * 0.8), :]
    origin_data_test = df.iloc[int(df.shape[0] * 0.8):, :]
    print(origin_data_train.shape, origin_data_test.shape)

    # -*- coding: utf-8 -*-
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler(feature_range=(0, 1))

    data_train = scaler.fit_transform(origin_data_train)
    data_test = scaler.fit_transform(origin_data_test)

    output_dim = 1
    batch_size = 256
    seq_len = 5
    hidden_size = 128
    max_features = 10000
    max_len = 500

    TIME_STEPS = 5
    INPUT_DIM = 7

    X_train = np.array([data_train[i : i + seq_len, :] for i in range(data_train.shape[0] - seq_len)])
    Y_train = np.array([data_train[i + seq_len, 0] for i in range(data_train.shape[0]- seq_len)])
    X_test = np.array([data_test[i : i + seq_len, :] for i in range(data_test.shape[0]- seq_len)])
    y_test = np.array([data_test[i + seq_len, 0] for i in range(data_test.shape[0] - seq_len)])

    config = configparser.ConfigParser()
    config.read('config.ini', encoding='GB18030')
    count = config.getint('config', 'count') + 1

    model = load_model('model/' + 'model_dfa_' + count.__str__() + '.h5')

    config.set('config', 'count', count.__str__())
    openConfig = open('config.ini', 'w')
    config.write(openConfig)
    openConfig.close()

    y_pred = model.predict(X_test)
    y_other = np.array([data_test[i + seq_len, 1:] for i in range(data_test.shape[0] - seq_len)])
    y_pred = y_pred.reshape(1, -1)
    y_test = y_test.reshape(1, -1)
    y_pred_other = np.insert(y_other, 0, values=y_pred, axis=1)
    y_test_other = np.insert(y_other, 0, values=y_test, axis=1)
    y_pred = scaler.inverse_transform(y_pred_other)
    y_test = scaler.inverse_transform(y_test_other)
    y_pred = y_pred[:, 0]
    y_test = y_test[:, 0]

    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r_square = compute_r2(y_test, y_pred)

    y_pred = y_pred[48-seq_len:]
    origin_data = origin_data['Unnamed: 0']
    origin_data = origin_data[-y_pred.shape[0]:]
    origin_data = pd.DataFrame({'Unnamed: 0': origin_data})
    origin_data['close'] = y_pred
    origin_data.to_csv("pred_dfa.csv", index=False)

    return r_square