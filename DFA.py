import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt

train_data = pd.read_csv('5m.csv', encoding='gbk')
close = train_data['close']
close = np.array(close)

class DFA:
    def __init__(self, x):
        self.x = x
        account_x = 0  # x总和
        for i in range(0, len(self.x)):
            account_x += self.x[i]
        self.avg_x = (account_x * 1.0) / len(self.x)  # x均值
        self.N = len(self.x)

    # DFA方法
    def DFA_Main(self, s):
        # 步骤二：分割等长区间
        Y_Interval = []
        for i in range(0, self.N):
            Y_Interval.append(self.Y(i))

        timeIntervalArray = []  # 等长时间区间数组
        timeIntervalIndexArray = [] #等长时间区间数组横坐标
        self.N_s = math.floor((self.N * 1.0) / s)
        #正着分一遍
        for i in range(0, self.N_s):
            timeInterval = []
            for j in range(0, s):
                timeInterval.append(Y_Interval[i * s + j])
            timeIntervalArray.append(timeInterval)
            timeIntervalIndexArray.append(i+1)
        #反着分一遍
        for i in range(0, self.N_s):
            timeInterval = []
            for j in range(0, s):
                timeInterval.append(Y_Interval[len(Y_Interval)-1-(i*s+j)])
            timeIntervalArray.append(timeInterval)
            timeIntervalIndexArray.append(i+self.N_s+1)

        #步骤三：求各区间拟合式Z()
        self.Z_param = []
        for i in range(0, 2 * self.N_s):
            self.Z_param.append(self.LeastSquares(timeIntervalIndexArray, timeIntervalArray[i]))

        #步骤四：计算均方误差F²()

        #步骤五：计算q阶波动函数F_q()
        return self.F_q(s, 2)


    # 离差序列Y求法
    def Y(self, i):
        # 步骤一：计算离差序列Y
        Y = 0  # 轮廓序列
        for j in range(0, i):
            Y += (self.x[j] - self.avg_x)
        return Y

    # 残差序列Z求法
    def Z(self, i, k, b):
        return k * i + b

    #均方误差F
    def F(self, s, v):
        sum = 0
        if(v <= self.N_s):
            for i in range(1, s):
               sum += (self.Y((v - 1) * s + i) - self.Z(i, self.Z_param[v][0], self.Z_param[v][1])) * (self.Y((v - 1) * s + i) - self.Z(i, self.Z_param[v][0], self.Z_param[v][1]))
            return (sum * 1.0) / s
        else:
            for i in range(1, s):
               sum += (self.Y(self.N - (v - self.N_s) * s + i) - self.Z(i, self.Z_param[v][0], self.Z_param[v][1])) * (self.Y(self.N - (v - self.N_s) * s + i) - self.Z(i, self.Z_param[v][0], self.Z_param[v][1]))
            return (sum * 1.0) / s

    #q阶波动函数F_q
    def F_q(self, s, q):
        sum = 0
        for v in range(1, 2 * self.N_s):
            sum += math.pow(self.F(s, v), q / 2)

        if(q == 0):
            return math.pow(math.e, (sum * 1.0) / (2 * self.N_s))
        else:
            return math.pow((sum * 1.0) / (2 * self.N_s), 1.0 / q)

    #最小二乘法
    def LeastSquares(self, x, y):
        sum_x = 0
        sum_y = 0
        sum_xy = 0
        sum_square_x = 0
        #选定拟合范围
        minLen = min(len(x), len(y))

        #求平均值
        for i in range(0, minLen):
            sum_x += x[i]
            sum_y += y[i]
            sum_xy += x[i] * y[i]
            sum_square_x += x[i] * x[i]
        avg_x = sum_x / minLen
        avg_y = sum_y / minLen
        avg_xy = sum_xy / minLen
        avg_square_x = sum_square_x / minLen

        k = ((avg_xy - avg_x * avg_y) * 1.0)/(avg_square_x - avg_x * avg_x)
        b = avg_y - k * avg_x

        return k, b

def calDFA(x):
    dfa = DFA(x)

    # log_10(F(s)) ~ alog_10(s), 用最小二乘法求a
    log_F_s_array = []  # F_q的对数的数组
    log_s_array = []  # s的对数的数组
    for s in range(2, 49):
        log_F_s_array.append(np.log10(dfa.DFA_Main(s)))
        log_s_array.append(np.log10(s))

    # 最小二乘法斜率为广义Hurst指数
    k, b = dfa.LeastSquares(log_s_array, log_F_s_array)

    return k

if __name__ == '__main__':
    closegroup = []
    for i in range(1, 707):
        for j in range(1, 7):
            closegroup.append(close[(i - 1) * 48 + j - 1])

    countgroup = 668
    hrustgroup = []
    hurstline = []
    for i in range(1, countgroup):
        hurst_exponent = calDFA(closegroup[(i - 1) * 6:(i - 1) * 6 + 240])
        hrustgroup.append(hurst_exponent)
        for j in range(1, 49):
            hurstline.append(hurst_exponent)
        print(i, end=" ")
        print(hurst_exponent)

    for i in range(len(hurstline), 33936):
        hurstline.append('')
    allDataFrame = pd.read_csv('5m.csv')
    allDataFrame['hurst'] = hurstline
    allDataFrame.to_csv('6m.csv', index=False)

    plt.plot(range(1, countgroup), hrustgroup, 'b', label='Hrust')
    plt.legend()
    plt.show()