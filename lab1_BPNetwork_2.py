import datetime

import numpy as np
import random
import utils

# 这个文件是用python写的分类bp网，只使用了非线性函数，而没有使用softmax，因为太慢，所以不再使用，而是使用java版本的

random.seed(0)


def rand(a, b):
    return (b - a) * random.random() + a


def generate_w(m, n):
    w = [0.0] * m
    for i in range(m):
        w[i] = [0.0] * n
        for j in range(n):
            w[i][j] = rand(-0.69, 1)  # rand(-1,1) #
    return w


def generate_b(m):
    b = [0.0] * m
    for i in range(m):
        b[i] = rand(-2.409, 0.02)  # rand(-1,1)#
    return b


def sigmoid(x, deriv=False):
    if deriv == True:
        return 1 - np.tanh(x) * np.tanh(x)  # tanh函数的导数
    return np.tanh(x)


class BPNetwork:
    def __init__(self):
        self.input_n = 0
        self.input_cells = []
        self.output_n = 0
        self.output_cells = []
        self.input_w = []
        self.output_w = []
        self.hidden_bs = []
        self.output_b = []
        self.hidden_results = []
        self.output_deltas = []
        self.hidden_deltases = []

    def setup(self, input_n, output_n, hidden_set):
        self.input_n = input_n + 1
        self.output_n = output_n
        self.hidden_ns = [0.0] * len(hidden_set)
        self.input_cells = [1.0] * self.input_n
        self.output_cells = [1.0] * self.output_n
        # 初始化weights和bias
        self.input_w = generate_w(self.input_n, self.hidden_ns[0])
        self.hidden_ws = [0.0] * (len(self.hidden_ns) - 1)
        for i in range(len(self.hidden_ns) - 1):
            self.hidden_ws[i] = generate_w(self.hidden_ns[i], self.hidden_ns[i + 1])

        self.output_w = generate_w(self.hidden_ns[len(self.hidden_ns) - 1], self.output_n)
        self.output_b = generate_b(self.output_n)
        self.hidden_bs = [0.0] * len(self.hidden_ns)
        for i in range(len(self.hidden_ns)):
            self.hidden_bs[i] = generate_b(self.hidden_ns[i])

        self.hidden_results = [0.0] * (len(self.hidden_ns))

    def forward_propagate(self, input):
        for i in range(len(input)):
            self.input_cells[i] = input[i]
        # 输入层
        self.hidden_results[0] = [0.0] * self.hidden_ns[0]
        for h in range(self.hidden_ns[0]):
            total = 0.0
            for i in range(self.input_n):
                total += self.input_w[i][h] * self.input_cells[i]
            self.hidden_results[0][h] = sigmoid(total + self.hidden_bs[0][h])
        # 隐藏层
        for k in range(len(self.hidden_ns) - 1):
            self.hidden_results[k + 1] = [0.0] * self.hidden_ns[k + 1]
            for h in range(self.hidden_ns[k + 1]):
                total = 0.0
                for i in range(self.hidden_ns[k]):
                    total += self.hidden_ws[k][i][h] * self.hidden_results[k][i]
                self.hidden_results[k + 1][h] = sigmoid(total + self.hidden_bs[k + 1][h])
        # 输出层
        for h in range(self.output_n):
            total = 0.0
            for i in range(self.hidden_ns[len(self.hidden_ns) - 1]):
                total += self.output_w[i][h] * self.hidden_results[len(self.hidden_ns) - 1][i]
            self.output_cells[h] = sigmoid(total + self.output_b[h])
        return self.output_cells[:]

    def get_deltas(self, label):
        self.output_deltas = [0.0] * self.output_n
        # 输出层 deltas
        for o in range(self.output_n):
            if o == label:
                error = 1 - self.output_cells[o]
            else:
                error = (-1) * self.output_cells[o]
            self.output_deltas[o] = sigmoid(self.output_cells[o], True) * error
        # 隐藏层deltas
        tmp_deltas = self.output_deltas
        tmp_w = self.output_w
        self.hidden_deltases = [0.0] * (len(self.hidden_ns))
        k = len(self.hidden_ns) - 1
        while k >= 0:
            self.hidden_deltases[k] = [0.0] * (self.hidden_ns[k])
            for o in range(self.hidden_ns[k]):
                error = 0.0
                for i in range(len(tmp_deltas)):
                    error += tmp_deltas[i] * tmp_w[o][i]
                self.hidden_deltases[k][o] = sigmoid(self.hidden_results[k][o], True) * error
            k = k - 1
            if k >= 0:
                tmp_w = self.hidden_ws[k]
                tmp_deltas = self.hidden_deltases[k + 1]
            else:
                break

    def renew_w(self, learn):
        # 更新隐藏层→输出层的权重
        k = len(self.hidden_ns) - 1
        for i in range(self.hidden_ns[k]):
            for o in range(self.output_n):
                change = self.output_deltas[o] * self.hidden_results[k][i]
                self.output_w[i][o] += change * learn

        # 更新隐藏层的权重
        while k > 0:
            for i in range(self.hidden_ns[k - 1]):
                for o in range(self.hidden_ns[k]):
                    change = self.hidden_deltases[k][o] * self.hidden_results[k - 1][i]
                    self.hidden_ws[k - 1][i][o] += change * learn
            k = k - 1

        # 更新输入层→隐藏层权重
        for i in range(self.input_n):
            for o in range(self.hidden_ns[0]):
                change = self.hidden_deltases[0][o] * self.input_cells[i]
                self.input_w[i][o] += change * learn

    def renew_b(self, learn):
        k = len(self.hidden_bs) - 1
        while k >= 0:
            for i in range(self.hidden_ns[k]):
                self.hidden_bs[k][i] = self.hidden_bs[k][i] + learn * self.hidden_deltases[k][i]
            k = k - 1
        for o in range(self.output_n):
            self.output_b[o] += self.output_deltas[o] * learn

    def back_propagate(self, label, learn):
        self.get_deltas(np.argmax(label))
        self.renew_w(learn)
        self.renew_b(learn)

    '''
    计算正确率
    '''

    def calculate_correct(self, train_data, train_label):
        count = 0
        for i in range(len(train_data)):
            predicate = self.forward_propagate(train_data[i].astype(int))
            label = np.argmax(predicate)
            if label == train_label[i].index(1):
                count += 1
        return count / len(train_data)

    '''
    训练
    每10次打印一次正确率
    '''

    def train(self, train_data, train_label, learn, times):
        for i in range(times):
            for w in range(len(train_data)):
                self.forward_propagate(train_data[w])
                self.back_propagate(train_label[w], learn=learn)

            j = i
            if j > 0 and j % 10 == 0:
                print("第 " + str(j) + " 次" + "反向传播训练, 训练集正确率为 " + str(self.calculate_correct(train_data, train_label)))

    #
    # def train(self, input_datas, labels, learn=0.05, limit=100000):
    #     print("训练开始时间:" + str(datetime.datetime.now()))
    #
    #     for j in range(limit):
    #         # input_datas, labels = self.upset_train_data(input_datas, labels)
    #         error = 0.0
    #         for i in range(len(input_datas)):
    #             input = input_datas[i]
    #             label = labels[i]
    #             error += self.back_propagate(input, label, learn)
    #         print(error)

    # def test(self):
    #     self.setup(784, 12, [10])
    #
    #     self.train(word_train, res_train, 0.1, 50)
    #     count = 0.0
    #     for i in range(len(word_train)):
    #         output_result = self.forward_propagate(word_train[i])
    #         if np.argmax(output_result) == res_train[i]:
    #             count += 1
    #     correction = count / (len(word_train))
    #     print("训练结束时间:" + str(datetime.datetime.now()))
    #     print("训练集正确率：")
    #     print(correction)
    #
    #     count = 0.0
    #     print("测试开始时间:" + str(datetime.datetime.now()))
    #     for i in range(len(word_test)):
    #         output_result = self.forward_propagate(word_test[i])
    #         if max_index(output_result) == word_test[i]:
    #             count += 1
    #     correction = count / (len(word_test))
    #     print("测试结束时间:" + str(datetime.datetime.now()))
    #     print("测试集正确率为：")
    #     print(correction)

    # def upset_train_data(self, datas, labels):
    #     index = rand(0, len(datas))
    #     index = int(index)
    #     index_mid = int(index / 2)
    #     for i in range(index_mid):
    #         tmp_data = datas[index_mid + i]
    #         tmp_label = labels[index_mid + i]
    #         datas[index_mid + i] = datas[i]
    #         datas[i] = tmp_data
    #         labels[index_mid + i] = labels[i]
    #         labels[i] = tmp_label
    #     remain_mid = int(int(len(datas) - index) / 2)
    #     for i in range(remain_mid):
    #         tmp_data = datas[index + i]
    #         datas[index + i] = datas[index + remain_mid + i]
    #         datas[index + remain_mid + i] = tmp_data
    #         tmp_label = labels[index + i]
    #         labels[index + i] = labels[index + remain_mid + i]
    #         labels[index + remain_mid + i] = tmp_label
    #     return datas, labels


word_train, word_test, res_train, res_test = utils.get_train_data()

bp = BPNetwork()
if __name__ == '__main__':
    '''
    初始化神经网络的结构
    输入层 28 * 28 = 784
    输出层 12
    '''
    bp.setup(784, 12, [20])
    # 初始化学习率，训练次数
    learn = 0.1
    times = 50
    print("训练开始: " + datetime.datetime.now().strftime('%Y.%m.%d-%H:%M:%S'))

    bp.train(word_train, res_train, learn, times)

    print("训练结束: " + datetime.datetime.now().strftime('%Y.%m.%d-%H:%M:%S'))

    print("测试开始: " + datetime.datetime.now().strftime('%Y.%m.%d-%H:%M:%S'))

    count = bp.calculate_correct(word_test)

    print("测试结束: " + datetime.datetime.now().strftime('%Y.%m.%d-%H:%M:%S'))

    print("测试集的正确率为: ")
    print(count)
