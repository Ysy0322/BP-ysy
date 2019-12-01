import numpy as np
import random
import math
import matplotlib
import pylab

# 相同种子下每次运行生成的随机数相同
random.seed(0)


# 生成(a,b)之间的随机数
def rand(a, b):
    return a + (b - a) * random.random()


# 初始化权值，生成m*n的Weight矩阵，并初始化每一个Weight的值
def generate_w(m, n):
    # m*n 的-1到1之间的随机数矩阵
    w = [0.0] * m
    for i in range(m):
        w[i] = [0.0] * n
        for j in range(n):
            # w[i][j] = rand(-1, -1)
            w[i][j] = rand(-1, 1)
            # w[i][j] = rand(-0.69, 1)  # rand(-1,1) #
    # w = -0.5 + 2 * np.random.random((m, n))
    return w


# 初始化阈值，生成m的数组，并初始化每一个Bias的值
def generate_b(m):
    b = [0.0] * m
    for i in range(m):
        b[i] = rand(-1, 1)
        # b[i] = rand(-2.409, 0.02)
    # b = -1 + 2 * np.random.random(m)
    return b


# tanh激活函数
def tanh(x):
    return np.tanh(x)


# tanh函数的梯度函数
def tanh_derivative(x):
    return 1 - np.tanh(x) * np.tanh(x)


class BP:
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
        self.hidden_w = []
        # 隐藏层的Bias
        self.hidden_b = []
        # 隐藏层的输出数组
        self.hidden_results = []
        # 输出层的delta调整值数组
        self.output_delta = []
        # 隐藏层的delta调整值数组
        self.hidden_delta = []

    '''
    初始化输入层，隐藏层，输出层数组
    input_n 输入神经元数
    output_m 输出神经元数
    hidden_set 隐藏层每一层神经元数
    input_n是定义输入的参数个数，但并不等于输入层神经元个数，
    self.input_n等于输入的参数input_n+1,
    原因是需要新增一个神经元用来调节bias，
    且这个神经元的输入值记为1，
    这样就可以通过用output_delta来调节input_b了
    '''

    def setup(self, input_n, output_m, hidden_set):
        self.input_n = input_n
        # self.input_n = input_n + 1  # 输入层个数加上一个Bias值
        self.output_m = output_m
        self.hidden_set = [0.0] * len(hidden_set)
        # for i in range(len(hidden_set)):
        #     self.hidden_set[i] = hidden_set[i] + 1  # 隐藏层最后加上一个Bias值
        for i in range(len(hidden_set)):
            self.hidden_set[i] = hidden_set[i]  # 隐藏层最后加上一个Bias值
        self.input_cells = [1.0] * self.input_n
        self.output_cells = [1.0] * self.output_m
        # 初始化weights，input的bais用的是hidden_b[0]
        self.input_w = generate_w(self.input_n, self.hidden_set[0])
        # 输出层weights和bias
        self.output_w = generate_w(self.hidden_set[len(hidden_set) - 1], self.output_m)
        self.output_b = generate_b(self.output_m)
        # 隐藏层weights和bias
        self.hidden_w = [0.0] * (len(self.hidden_set) - 1)
        for i in range(len(self.hidden_set) - 1):
            self.hidden_w[i] = generate_w(self.hidden_set[i], self.hidden_set[i + 1])
        self.hidden_b = [0.0] * len(self.hidden_set)
        for i in range(len(self.hidden_set)):
            self.hidden_b[i] = generate_b(self.hidden_set[i])
        self.hidden_results = [0.0] * (len(self.hidden_set))

    '''
    向前传播
    Sj = tanh((Σi=前一层输入神经元个数weight[i][j] * input[i]) + bias[j])
    '''

    def forward(self, input):
        for i in range(len(input)):
            self.input_cells[i] = input[i]
        # 输入层到第一层隐藏层
        self.hidden_results[0] = [0.0] * self.hidden_set[0]
        for i in range(self.input_n):
            result_i = 0.0
            for j in range(len(self.input_cells)):  # self.input_n
                result_i += self.input_cells[j] * self.input_w[j][i]
            self.hidden_results[0][i] = tanh(result_i + self.hidden_b[0][i])

        # 隐藏层之间，result从第二层开始
        for k in range(len(self.hidden_set) - 1):
            self.hidden_results[k + 1] = [0.0] * self.hidden_set[k + 1]
            for i in range(self.hidden_set[k + 1]):
                result_k1_j = 0.0
                for j in range(self.hidden_set[k]):
                    result_k1_j += self.hidden_results[k][j] * self.hidden_w[k][j][i]
                self.hidden_results[k + 1][i] = tanh(result_k1_j + self.hidden_b[k + 1][i])

        # 最后一层隐藏层到输出层
        self.output_cells = [0.0] * self.output_m
        for i in range(self.output_m):
            result_i = 0.0
            for j in range(self.hidden_set[len(self.hidden_set) - 1]):
                result_i += self.output_w[j][i] * self.hidden_results[len(self.hidden_set) - 1][j]
            self.output_cells[i] = tanh(result_i + self.output_b[i])
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
        # 最后一层隐藏层层到输出层
        self.output_delta = [0.0] * self.output_m
        for i in range(self.output_m):
            self.output_delta[i] = (expect[i] - self.output_cells[i]) * tanh_derivative(self.output_cells[i])
        # 隐藏层之间(包含输入层到第一层隐藏层，k=0)，从最后一层向前计算delta
        tmp_delta = self.output_delta
        tmp_w = self.output_w
        self.hidden_delta = [0.0] * len(self.hidden_set)
        k = len(self.hidden_set) - 1
        while k >= 0:
            self.hidden_delta[k] = [0.0] * (self.hidden_set[k])
            # δj=∑i( · Wij)φj'(Sj)
            for i in range(self.hidden_set[k]):
                error = 0.0
                for j in range(len(tmp_delta)):
                    error += tmp_delta[j] * tmp_w[i][j]
                self.hidden_delta[k][i] = tanh_derivative(self.hidden_results[k][i]) * error
            k -= 1
            if k >= 0:
                tmp_delta = self.hidden_delta[k + 1]
                tmp_w = self.hidden_w[k]
            else:
                break

    '''
    更新weight
    μ 为学习率，即每次权值调整的rate
    W'ij= Wij - μ* δij*Xi
    '''

    def update_weight(self, μ):
        # 更新隐藏层到输出层的output_w
        k = len(self.hidden_set) - 1
        for i in range(self.hidden_set[k]):
            for j in range(self.output_m):
                self.output_w[i][j] += μ * self.output_delta[j] * self.hidden_results[k][i]
        # 更新隐藏层之间的hidden_w,从最后一层向前更新
        while k > 0:
            for i in range(self.hidden_set[k - 1]):
                for j in range(self.hidden_set[k]):
                    self.hidden_w[k - 1][i][j] += μ * self.hidden_delta[k][j] * self.hidden_results[k - 1][i]
            k -= 1
        # 更新输入层到隐藏层的input_w
        for i in range(self.input_n):
            for j in range(self.hidden_set[0]):
                self.input_w[i][j] += μ * self.hidden_delta[0][j] * self.input_cells[i]

    '''
    更新bias
    μ 为学习率，即每次权值调整的rate
    B'ij= Bij - μ* δij
    '''

    def update_bias(self, μ):
        # 更新输出层Bias
        for i in range(self.output_m):
            self.output_b[i] += μ * self.output_delta[i]
        # 更新隐藏层Bias
        k = len(self.hidden_set) - 1
        while k >= 0:
            for i in range(self.hidden_set[k]):
                self.hidden_b[k][i] += μ * self.hidden_delta[k][i]
            k -= 1

    '''
    反向传播算法对训练集进行训练
    repeat：
        向前传播，计算output_cells
        反向传播，计算δ, 更新weight和bias
    '''

    def bp(self, tests, tests_res, learn=0.05, limit=100000):
        for j in range(limit):
            for i in range(len(tests)):
                self.forward(tests[i])
                self.calculate_delta(tests_res[i])
                self.update_weight(learn)
                self.update_bias(learn)

    # 得到训练集
    def train_set(self):
        train = []
        train_res = []
        for i in range(-10, 11, 1):
            train.append([i * math.pi / 10])
            train_res.append(np.sin([i * math.pi / 10]))

        return train, train_res

    # 得到测试集
    def test_set(self):
        test = []
        test_res = []
        for i in range(-100, 101, 1):
            test.append([i * math.pi / 100])
            test_res.append(np.sin([i * math.pi / 100]))
        return test, test_res

    # 计算损失值
    def loss(self, output, expect):
        error = 0.0
        for i in range(len(expect)):
            error += 0.5 * (output[i] - expect[i]) * (output[i] - expect[i])

        return error

    # 计算预测结果的平均误差
    def loss_average(self, tests, expects):
        error = 0.0
        predicate_res = []
        for i in range(len(tests)):
            predicate_res.append(self.forward(tests[i]))
            error += self.loss(self.output_cells, expects[i])
        error_average = error / len(tests)
        return error_average, predicate_res

    # 测试
    def predicate(self):
        learn = 0.03
        times = 3000
        train, train_res = self.train_set()
        self.setup(1, 1, [10, 5, 5,5])
        self.bp(train, train_res, learn, times)

        test, test_res = self.test_set()
        error_average, predicate_res = self.loss_average(test, test_res)
        print(error_average)

        # 画图
        pylab.plt.scatter(train, train_res, marker='x', color='g', label='train set')

        x = np.arange(-1 * np.pi, np.pi, 0.01)
        x = x.reshape((len(x), 1))
        y = np.sin(x)
        pylab.plot(x, y, label='standard sinx')

        pylab.plot(test, predicate_res, label='predicate sinx, learn = '+str(learn), linestyle='--', color='r')

        pylab.plt.legend(loc='best')
        pylab.plt.show()


if __name__ == '__main__':
    bp = BP()
    bp.predicate()
