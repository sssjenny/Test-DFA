import configparser
import datetime
import pandas as pd
import datetime
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.layers import *
from keras.models import *
from keras import layers

from sklearn.metrics import explained_variance_score, r2_score, mean_squared_error, mean_absolute_error


def dfa_auto():
    # !/usr/bin/env python
    # coding: utf-8

    # 记录程序启动时间
    starttime = datetime.datetime.now()

    '''
    Function: 读取文件
    Variable: 
        df_data_5minute: 用于数据分析
        origin_data: 用于最后生成csv文件时提取日期列
    '''
    df_data_5minute = pd.read_csv('6m.csv')
    origin_data = pd.read_csv('6m.csv')

    # 选取一定日期范围的数据，并删除其中
    origin_data = origin_data[:19200]
    df_data_5minute = df_data_5minute[:19200]
    df_data_5minute.drop('Unnamed: 0', axis=1, inplace=True)

    df = df_data_5minute
    close = df['close']
    df.drop(labels=['close'], axis=1, inplace=True)
    df.insert(0, 'close', close)
    origin_data_train = df.iloc[:int(df.shape[0] * 0.8), :]
    origin_data_test = df.iloc[int(df.shape[0] * 0.8):, :]
    print(origin_data_train.shape, origin_data_test.shape)

    # -*- coding: utf-8 -*-

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

    X_train = np.array([data_train[i: i + seq_len, :] for i in range(data_train.shape[0] - seq_len)])
    Y_train = np.array([data_train[i + seq_len, 0] for i in range(data_train.shape[0] - seq_len)])
    X_test = np.array([data_test[i: i + seq_len, :] for i in range(data_test.shape[0] - seq_len)])
    y_test = np.array([data_test[i + seq_len, 0] for i in range(data_test.shape[0] - seq_len)])

    print(X_train.shape, Y_train.shape, X_test.shape, y_test.shape)

    model = Sequential()
    model.add(layers.Input(shape=(TIME_STEPS, INPUT_DIM)))
    model.add(Conv1D(filters=64,
                     kernel_size=1))
    model.add(MaxPooling1D(pool_size=5))
    model.add(layers.Dropout(0.2))
    model.add(layers.GRU(units=64, return_sequences=True))
    model.add(layers.Dropout(0.2))
    model.add(layers.GRU(units=64))
    model.add(layers.Dropout(0.2))
    model.add(Dense(1, activation='tanh'))
    model.summary()

    model.compile(optimizer='adam', loss='mse', metrics=['mae'])

    history = model.fit(X_train, Y_train, epochs=20, batch_size=64, validation_data=(X_test, y_test))

    # 验证损失度
    history_dict = history.history
    loss_values = history_dict['loss']
    val_loss_values = history_dict['val_loss']

    epochs = range(1, len(loss_values) + 1)

    config = configparser.ConfigParser()
    config.read('config.ini', encoding='GB18030')

    count = config.getint('config', 'count') + 1
    model.save('model/' + 'model_dfa_' + count.__str__() + '.h5')
    config.set('config', 'count', count.__str__())
    config.set('config', 'filename', 'pred_dfa.csv')
    openConfig = open('config.ini', 'w')
    config.write(openConfig)
    openConfig.close()

    plt.plot(epochs, loss_values, 'bo', label='Training loss')
    plt.plot(epochs, val_loss_values, 'b', label='Validation loss')
    plt.legend()
    plt.show()

    endtime = datetime.datetime.now()
    print("运行时间: " + (endtime - starttime).seconds.__str__() + "s")

    test_mse, test_mae = model.evaluate(X_test, y_test)

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

    y_pred = y_pred[48 - seq_len:]
    origin_data = origin_data['Unnamed: 0']
    origin_data = origin_data[-y_pred.shape[0]:]
    origin_data = pd.DataFrame({'Unnamed: 0': origin_data})
    origin_data['close'] = y_pred
    origin_data.to_csv("pred_dfa.csv", index=False)

    return mse, mae
