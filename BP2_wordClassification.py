import datetime

import numpy
import scipy

import utils
import BP0_basic


def softmax(x):
    """
    Compute the softmax function for each row of the input x.

    Arguments:
    x -- A N dimensional vector or M x N dimensional numpy matrix.

    Return:
    x -- You are allowed to modify x in-place
    """
    orig_shape = x.shape

    if len(x.shape) > 1:
        # Matrix
        exp_minmax = lambda x: numpy.exp(x - numpy.max(x))
        denom = lambda x: 1.0 / numpy.sum(x)
        x = numpy.apply_along_axis(exp_minmax, 1, x)
        denominator = numpy.apply_along_axis(denom, 1, x)

        if len(denominator.shape) == 1:
            denominator = denominator.reshape((denominator.shape[0], 1))

        x = x * denominator
    else:
        # Vector
        x_max = numpy.max(x)
        x = x - x_max
        numerator = numpy.exp(x)
        denominator = 1.0 / numpy.sum(numerator)
        x = numerator.dot(denominator)

    assert x.shape == orig_shape
    return x


def sigmoid(x):
    return 1 / 1 + numpy.exp(-x)


class BPNetwork:
    def __int__(self):
        # 输入神经元数
        self.input_n = 0
        # 输入神经元数组
        self.input_cells = []
        # 输出神经元数
        self.output_m = 0
        # 输出神经元数组
        self.output_cells = []
        # 隐藏层神经元数
        self.hidden_set = 0
        # 隐藏层结果
        self.hidden_result = []
        # 输入层到隐藏层权值数组
        self.input_w = []
        # 隐藏层到输出层权值数组
        self.output_w = []
        # 输入层到隐藏层的Bias
        self.input_b = []
        # 隐藏层到输出层的Bias
        self.output_b = []
        # 隐藏层到输出层的delta
        self.output_delta = []
        # 输入层到隐藏层的delta
        self.input_delta = []

    '''
    初始化神经网络的结构，一层隐藏层
    '''

    def setup(self, input_n, output, hidden_size):
        self.input_n = input_n
        self.output_m = output
        self.output_cells = [0.0] * output
        self.hidden_set = hidden_size
        self.hidden_result = [0.0] * hidden_size
        self.input_w = BP0_basic.init_weight(input_n, hidden_size)
        self.output_w = BP0_basic.init_weight(hidden_size, output)
        self.input_b = BP0_basic.init_bias(hidden_size)
        self.output_b = BP0_basic.init_bias(output)
        self.input_delta = [0.0] * self.input_n
        self.output_delta = [0.0] * self.hidden_set

    '''
    一次向前传播
    '''

    def forward_propagate(self, input_cells):
        # 输入层到隐藏层
        self.input_cells = [0.0] * self.input_n
        for i in range(len(input_cells)):
            self.input_cells[i] = input_cells[i]
        # print(self.input_cells)
        for i in range(self.hidden_set):
            for w in range(self.input_n):
                self.hidden_result[i] = sigmoid(self.input_w[w][i] * self.input_cells[w])
        # 隐藏层到输出层
        for i in range(self.output_m):
            for w in range(self.hidden_set):
                self.output_cells[i] = sigmoid(self.output_w[w][i] * self.hidden_result[i])

        self.output_cells = softmax(numpy.array(self.output_cells))

        return self.output_cells[:]

    '''
    包含向前传播，计算误差，更新weight和bias的一次反向传播
    '''

    def back_propagate(self, input_cells, target, learn):
        # 向前传播
        self.forward_propagate(input_cells)

        # 计算误差,delta
        output_error = numpy.array(self.output_cells) - numpy.array(target)
        hidden_error = numpy.dot(numpy.array(self.output_w), output_error)
        for i in range(self.output_m):
            self.output_delta[i] = output_error[i] * self.output_cells[i] * (1 - numpy.array(self.output_cells[i]))
        for i in range(self.hidden_set):
            self.input_delta[i] = hidden_error[i] * self.hidden_result[i] * (1 - numpy.array(self.hidden_result[i]))

        # 更新input_w,output_w,input_b,output_b
        for i in range(self.hidden_set):
            for j in range(self.output_m):
                self.output_w[i][j] += learn * self.output_delta[j] * self.hidden_result[i]
                self.output_b[j] += learn * self.output_delta[j]
        for i in range(self.input_n):
            for j in range(self.hidden_set):
                self.input_w[i][j] += learn * self.input_delta[j] * self.input_cells[i]
                self.input_b[j] += learn * self.input_delta[j]
        #
        # self.input_w += learn * numpy.multiply(numpy.array(self.input_delta) * numpy.array(self.input_cells))
        # self.output_b += numpy.multiply(learn, self.input_delta)

    '''
    训练
    '''

    def train(self, input_cells, target, learn, times):
        for i in range(times):
            self.back_propagate(input_cells, target, learn)

    def test(self, word_test, res_test):
        error = 0
        for i in range(len(word_test)):
            predicate = bp.forward_propagate(word_test[i])
            label = numpy.argmax(predicate)
            if label == res_test[i]:
                error += 0
            error += 1
        return error/len(word_test)

'''
测试
'''
bp = BPNetwork()
if __name__ == '__main__':
    '''
    初始化神经网络的结构
    输入层 28 * 28 = 784
    输出层 12
    '''
    bp.setup(784, 12, 50)
    # 初始化学习率，训练次数
    learn = 0.1
    times = 2
    word_train, word_test, res_train, res_test = utils.get_train_data()
    print("训练开始: " + str(datetime.datetime.now()))
    for w in range(12):
        for i in range(len(word_train[w])):
            bp.train(word_train[w][i], res_train[w][i], learn, times)
    print("训练结束: " + str(datetime.datetime.now()))

    print("测试开始: " + str(datetime.datetime.now()))
    error = 0
    for w in range(12):
        error += bp.test(word_test[w])
    print("测试集的正确率为: ")
    print(1 - error / 12)
