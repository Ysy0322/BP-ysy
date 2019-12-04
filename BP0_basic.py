from symtable import Symbol

import numpy as np
import random

random.seed(0)


def rand(a, b):
    return (b - a) * random.random() + a


def init_weight(m, n):
    w = [0.0] * m
    for i in range(m):
        w[i] = [0.0] * n
        for j in range(n):
            w[i][j] = rand(-1, 1)
    return w


def init_bias(m):
    b = [0.0] * m
    for i in range(m):
        b[i] = rand(-1, 1)
    return b


# tanh激活函数
def tanh(x):
    return np.tanh(x)


# tanh函数的梯度函数
def tanh_derivative(x):
    return 1 - np.tanh(x) * np.tanh(x)


def active_function(x, deriv=False):
    if deriv == True:
        return 1 - np.tanh(x) * np.tanh(x)  # tanh函数的导数
    return np.tanh(x)


class BPNetwork:
    def __init__(self):
        # 输入神经元数
        self.input_n = 0
        # 输入神经元数组
        self.input_cells = []
        # 输出神经元数
        self.output_m = 0
        # 输出神经元数组
        self.output_cells = []
        # 输入层到隐藏层权值数组
        self.input_w = []
        # 隐藏层到输出层权值数组
        self.output_w = []
        # 输出层的Bias
        self.output_b = []
        # 隐藏层以及，每个隐藏层神经元数
        self.hidden_set = []
        # 隐藏层之间的weight数组
        self.hidden_ws = []
        # 隐藏层的Bias
        self.hidden_bs = []
        # 隐藏层的输出数组
        self.hidden_results = []
        # 输出层的delta调整值数组
        self.output_deltas = []
        # 隐藏层的delta调整值数组
        self.hidden_deltases = []

    '''
    初始化输入层，隐藏层，输出层数组
    input_n 输入神经元数
    output_m 输出神经元数
    hidden_set 是一个数组，每一位是隐藏层每一层神经元数
    '''

    def setup(self, input_n, output_m, hidden_set):
        self.input_n = input_n
        self.output_m = output_m
        self.hidden_set = [0.0] * len(hidden_set)
        for i in range(len(hidden_set)):
            self.hidden_set[i] = hidden_set[i]
        self.input_cells = [1.0] * self.input_n
        self.output_cells = [1.0] * self.output_m
        # 初始化weights和bias
        self.input_w = init_weight(self.input_n, self.hidden_set[0])
        self.hidden_ws = [0.0] * (len(self.hidden_set) - 1)
        for i in range(len(self.hidden_set) - 1):
            self.hidden_ws[i] = init_weight(self.hidden_set[i], self.hidden_set[i + 1])
        self.output_w = init_weight(self.hidden_set[len(self.hidden_set) - 1], self.output_m)
        self.output_b = init_bias(self.output_m)
        self.hidden_bs = [0.0] * len(self.hidden_set)
        for i in range(len(self.hidden_set)):
            self.hidden_bs[i] = init_bias(self.hidden_set[i])

        self.hidden_results = [0.0] * (len(self.hidden_set))

    '''
    向前传播
    Sj = tanh((Σi=前一层输入神经元个数weight[i][j] * input[i]) + bias[j])
    '''

    def forward_propagate(self, input):
        for i in range(len(input)):
            self.input_cells[i] = input[i]
        # 输入层到第一层隐藏层
        self.hidden_results[0] = [0.0] * self.hidden_set[0]
        for h in range(self.hidden_set[0]):
            total = 0.0
            for i in range(self.input_n):
                total += self.input_w[i][h] * self.input_cells[i]
            self.hidden_results[0][h] = tanh(total + self.hidden_bs[0][h])

        # 隐藏层之间，result从第二层开始(hidden_results[1])
        for k in range(len(self.hidden_set) - 1):
            self.hidden_results[k + 1] = [0.0] * self.hidden_set[k + 1]
            for h in range(self.hidden_set[k + 1]):
                total = 0.0
                for i in range(self.hidden_set[k]):
                    total += self.hidden_ws[k][i][h] * self.hidden_results[k][i]
                self.hidden_results[k + 1][h] = tanh(total + self.hidden_bs[k + 1][h])

        # 最后一层隐藏层到输出层
        self.output_cells = [0.0] * self.output_m
        for h in range(self.output_m):
            total = 0.0
            for i in range(self.hidden_set[len(self.hidden_set) - 1]):
                total += self.output_w[i][h] * self.hidden_results[len(self.hidden_set) - 1][i]
            self.output_cells[h] = tanh(total + self.output_b[h])
        # 返回前向传播输出结果
        return self.output_cells[:]

    '''
    dj : 实际输出
    yj : 期望输出
    φ(x) : 激活函数
    总误差函数E(w,b) = 0.5*∑(dj-yj)*(dj-yj)
    根据梯度下降法，对E(w,b)分别对w, b 求偏导
    计算得到局部梯度δj
    j是输出层节点的时候，δj=(dj-yj)φ'(Sj)
    j是隐藏层节点的时候，δj=∑(i-∞)(δi · Wij)φ'(Sj)  注：δi 符合上一个公式
    '''

    def calculate_delta(self, expect):
        self.output_deltas = [0.0] * self.output_m
        # 最后一层隐藏层层到输出层
        for o in range(self.output_m):
            error = expect[o] - self.output_cells[o]
            self.output_deltas[o] = tanh_derivative(self.output_cells[o]) * error
        # 隐藏层之间(包含输入层到第一层隐藏层，k=0)，从最后一层向前计算delta
        tmp_deltas = self.output_deltas
        tmp_w = self.output_w
        self.hidden_deltases = [0.0] * (len(self.hidden_set))
        k = len(self.hidden_set) - 1
        while k >= 0:
            self.hidden_deltases[k] = [0.0] * (self.hidden_set[k])
            # δj=∑i( · Wij)φj'(Sj)
            for o in range(self.hidden_set[k]):
                error = 0.0
                for i in range(len(tmp_deltas)):
                    error += tmp_deltas[i] * tmp_w[o][i]
                self.hidden_deltases[k][o] = tanh_derivative(self.hidden_results[k][o]) * error
            k = k - 1
            if k >= 0:
                tmp_w = self.hidden_ws[k]
                tmp_deltas = self.hidden_deltases[k + 1]

    '''
    更新weight
    learn 为学习率，即每次权值调整的rate
    W'ij= Wij + learn* δij*Xi
    '''

    def update_weight(self, learn):
        # 更新隐藏层到输出层的output_w
        k = len(self.hidden_set) - 1
        for i in range(self.hidden_set[k]):
            for o in range(self.output_m):
                change = self.output_deltas[o] * self.hidden_results[k][i]
                self.output_w[i][o] += change * learn

        # 更新隐藏层之间的hidden_w,从最后一层向前更新
        while k > 0:
            for i in range(self.hidden_set[k - 1]):
                for o in range(self.hidden_set[k]):
                    change = self.hidden_deltases[k][o] * self.hidden_results[k - 1][i]
                    self.hidden_ws[k - 1][i][o] += change * learn
            k = k - 1

        # 更新输入层到隐藏层的input_w
        for i in range(self.input_n):
            for o in range(self.hidden_set[0]):
                change = self.hidden_deltases[0][o] * self.input_cells[i]
                self.input_w[i][o] += change * learn

    '''
    更新bias
    learn 为学习率，即每次权值调整的rate
    B'ij= Bij + learn* δij
    '''

    def update_bias(self, learn):
        # 更新隐藏层bias
        k = len(self.hidden_bs) - 1
        while k >= 0:
            for i in range(self.hidden_set[k]):
                self.hidden_bs[k][i] = self.hidden_bs[k][i] + learn * self.hidden_deltases[k][i]
            k = k - 1
        # 更新输出层bias
        for o in range(self.output_m):
            self.output_b[o] += self.output_deltas[o] * learn

    '''
    计算损失值
    dj : 实际输出
    yj : 期望输出
    损失函数为平方差函数 0.5*(dj-yj)**2
    '''

    def get_loss(self, expect, output_cell):
        error = 0.0
        for o in range(len(output_cell)):
            error += 0.5 * (expect[o] - output_cell[o]) ** 2
        return error

    # 计算output的平均损失值
    def get_average_loss(self, datas, expects):
        error = 0
        predicate_res = []
        for i in range(len(datas)):
            predicate_res.append(self.forward_propagate(datas[i]))
            error += self.get_loss(expects[i], self.output_cells)
        error = error / len(datas)
        return error, predicate_res
