import pandas as pd
import math
import configparser

'''
Function: 从文件中读取数据
params: 
    pred_data: 预测数据
    actual_data: 实际数据
    date: 日期数据
'''
pred_data = pd.read_csv("pred_null.csv")
# pred_data = pd.read_csv("pred_hurst.csv")
# pred_data = pd.read_csv("pred_dfa.csv")
actual_data = pd.read_csv("6m.csv")

date = pred_data["Unnamed: 0"].values
actual_data = actual_data.loc[19200*0.8+48:19199, ["close"]]
actual_data = actual_data["close"].values
pred_data = pred_data["close"].values

config = configparser.ConfigParser()
config.read('config.ini', encoding='GB18030')

X = config.getfloat('config', 'x')  #信号范围
origin_money = 10000    #最初投资额
money = 10000   #动态钱数
status = 0  #持仓方向，0是未持仓，1是多头，2是空头
money_History = []  #钱数历史数据
money_History.append(money)
hasTransactionToday = False #当日是否开仓
numOfHand = 0

for i in range(1, 3791):
    if(i % 48 == 0):
        hasTransactionToday = False
    now_close = actual_data[i]      #此5min实际价格
    last_close = actual_data[i-1]   #上5min实际价格
    next_pred = pred_data[i+1]      #上5min预测价格
    signal_1 = (now_close - last_close) / last_close    #趋势信号1
    signal_2 = (next_pred - now_close) / now_close      #趋势信号2

    if(status == 0):
        if(signal_2 > X):
            print(date[i], end=" ")
            numOfHand = math.floor(money / now_close)
            money -= now_close * numOfHand
            money -= now_close * 0.000023 * numOfHand
            status = 1
            hasTransactionToday = True
            print("多头")
        elif(signal_2 < -X):
            print(date[i], end=" ")
            numOfHand = math.floor(money / now_close)
            money += now_close * numOfHand
            money -= now_close * 0.000023 * numOfHand
            status = 2
            hasTransactionToday = True
            print("空头")
    elif(status == 1):
        if(signal_1 < -X and signal_2 > X):
            print(date[i], end=" ")
            money += now_close * numOfHand
            if(hasTransactionToday):
                money -= now_close * 0.000345 * numOfHand
            else:
                money -= now_close * 0.000023 * numOfHand
            print("多头平仓")
            numOfHand = math.floor(money / now_close)
            money -= now_close * numOfHand
            money -= now_close * 0.000023 * numOfHand
            hasTransactionToday = True
            print("多头")
        elif(signal_2 < -X):
            print(date[i], end=" ")
            money += now_close * numOfHand
            if (hasTransactionToday):
                money -= now_close * 0.000345 * numOfHand
            else:
                money -= now_close * 0.000023 * numOfHand
            print("多头平仓")
            numOfHand = math.floor(money / now_close)
            money += now_close * numOfHand
            money -= now_close * 0.000023 * numOfHand
            status = 2
            hasTransactionToday = True
            print("空头")
        elif(signal_1 < -X and signal_2 >= -X and signal_2 <= X):
            print(date[i], end=" ")
            money += now_close * numOfHand
            if (hasTransactionToday):
                money -= now_close * 0.000345 * numOfHand
            else:
                money -= now_close * 0.000023 * numOfHand
            print("多头平仓")
            status = 0
    elif(status == 2):
        if(signal_2 > X):
            print(date[i], end=" ")
            money -= now_close * numOfHand
            if (hasTransactionToday):
                money -= now_close * 0.000345 * numOfHand
            else:
                money -= now_close * 0.000023 * numOfHand
            print("空头平仓")
            numOfHand = math.floor(money / now_close)
            money -= now_close * numOfHand
            money -= now_close * 0.000023 * numOfHand
            status = 1
            hasTransactionToday = True
            print("多头")
        if(signal_1 > X and signal_2 < -X):
            print(date[i], end=" ")
            money -= now_close * numOfHand
            if (hasTransactionToday):
                money -= now_close * 0.000345 * numOfHand
            else:
                money -= now_close * 0.000023 * numOfHand
            print("空头平仓")
            numOfHand = math.floor(money / now_close)
            money += now_close * numOfHand
            money -= now_close * 0.000023 * numOfHand
            hasTransactionToday = True
            print("空头")
        if(signal_1 > X and signal_2 >= -X and signal_2 <= X):
            print(date[i], end=" ")
            money -= now_close * numOfHand
            if (hasTransactionToday):
                money -= now_close * 0.000345 * numOfHand
            else:
                money -= now_close * 0.000023 * numOfHand
            print("空头平仓")
            status = 0
    money_History.append(money)

    #最后一天强制平仓
    if(i == 3790):
        if(status == 1):
            print(date[i], end=" ")
            money += now_close * numOfHand
            if (hasTransactionToday):
                money -= now_close * 0.000345 * numOfHand
            else:
                money -= now_close * 0.000023 * numOfHand
            print("多头平仓")
        elif(status == 2):
            print(date[i], end=" ")
            money -= now_close * numOfHand
            if (hasTransactionToday):
                money -= now_close * 0.000345 * numOfHand
            else:
                money -= now_close * 0.000023 * numOfHand
            print("空头平仓")

print("收益率为：" + ((money-origin_money)/origin_money).__str__())
import matplotlib.pyplot as plt
plt.plot(money_History)
plt.show()

