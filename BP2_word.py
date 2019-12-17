# coding: utf-8
import sys, os
sys.path.append(os.pardir)
import numpy as np
from collections import OrderedDict
from dataset.mnist import load_mnist
import time


# ---------------------------------Affine层--------------------------------------------------
class Affine:

    def __init__(self, W, b):
        self.W = W
        self.b = b
        self.x = None
        self.dW = None
        self.db = None

    def forward(self, x):
        self.x = x
        out = np.dot(x, self.W) + self.b

        return out

    def backward(self, dout):
        dx = np.dot(dout, self.W.T) # T代表转置
        self.dW = np.dot(self.x.T, dout) # x要在前面乘
        self.db = np.sum(dout, axis=0) # 偏置反向传播需要汇总为偏置的元素

        return dx


# ---------------------------------------Relu层----------------------------------------------
class Relu():
    """Relu函数，反向传播时，x>0则会将上游的值原封不动的传递给下游（dx = dout）
                            x<0则会将信号停在这里（dout=0）
        先将输入数据转换为True和False的mask数组"""
    def __init__(self):
        self.mask = None # mask轮廓的含义,mask是由True/Fase组成的numpy数组。

    def forward(self, x):
        self.mask = (x <= 0) # mask会将x元素小于等于0的地方保存为True，其他地方都保存为False
        out = x.copy() # False的地方输出为x
        out[self.mask] = 0 # 将True的地方输出为0

        return out

    def backward(self, dout):
        dout[self.mask] = 0 # 前面保存了mask，True的地方反向传播会停在这个地方，故TRUE的地方设置为0，False的地方是将上游的值原封不动的传递给下游
        dx = dout

        return dx


# coding: utf-8
import numpy as np


# ---------------------------激活函数定义---------------------------------------------------
def softmax(x):
    if x.ndim == 2: # 多维数组
        x = x.T # 转置
        x = x - np.max(x, axis=0)
        y = np.exp(x) / np.sum(np.exp(x), axis=0)
        return y.T

    x = x - np.max(x) # 溢出对策
    return np.exp(x) / np.sum(np.exp(x))


# ---------------------------定义损失函数---------------------------------------------------
def cross_entropy_error(y, t):
    delta = 1e-07 # 设置一个微小值，避免对数的参数为0导致无穷大
    return - np.sum(t * np.log(y + delta)) # 注意这个log对应的是ln


# -----------------------------------------------------------------------
class SoftmaxWitgLoss:
    def __init__(self):
        self.loss = None # 损失
        self.y = None # softmax的输出
        self.t = None # 监督数据（one-hot ）

    def forward(self, x, t):
        self.t = t
        self.y = softmax(x)
        self.loss = cross_entropy_error(self.y, self.t)

        return self.loss

    def backward(self, dout=1):
        batch_size = self.t.shape[0]
        dx = (self.y - self.t) / batch_size # 得到单个数据的误差

        return dx


# ---------------------------定义数值梯度（梯度下降法）函数,偏导计算--------------------------------------
# # 数值微分（这里加上这个是为了后面对比数值微分和误差反向传播两种方法求的梯度之间的误差）
def numerical_gradient_no_batch(f, x):
    h = 1e-04 # 0.0001
    grad = np.zeros_like(x) # 生成和x形状相同的数组

    for index in range(x.size):
        tmp_value = x[index] # 先将数组的值存着

        x[index] = tmp_value + h # f(x+h)
        fxh1 = f(x)

        x[index] = tmp_value - h # f(x-h)
        fxh2 = f(x)

        grad[index] = (fxh1 - fxh2) / (2*  h)
        x[index] = tmp_value # 数值还原

    return grad


def numerical_gradient(f, X):
    if X.ndim == 1:
        return numerical_gradient_no_batch(f, X)
    else:
        grad = np.zeros_like(X)

        for idx, x in enumerate(X):
            grad[idx] = numerical_gradient_no_batch(f, x)

        return grad


# ----------------------------------两层神经网络---------------------------------------------
class TwoLayersNet:

    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        """权重的初始化是高斯分布，乘以一个weight_init_std
        偏置的初始化是为0"""
        # 初始化权重
        self.params = {} # 参数字典

        # 初始化输入层到隐藏层的权重和偏置，高斯随机生成形状为（input_szie, hidden_size）的二维数组
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)

        self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)

        # 生成层
        self.layers = OrderedDict() # 创建有序字典（（可以记住向字典中添加元素的顺序，在反向传播时只需要按相反的顺序调用各层即可））

        # 以正确的顺序链接各层

        # Affine1层
        self.layers['Affine1'] = Affine(self.params['W1'], self.params['b1'])

        # Relu1层
        self.layers['Relu1'] = Relu()

        # Affine2层
        self.layers['Affine2'] = Affine(self.params['W2'], self.params['b2'])

        # softmax_with_loss层
        self.lastlayer = SoftmaxWitgLoss()

    # 进行识别(推理)
    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)
        return x

    # 计算损失函数的值 x输入数据，t监督数据
    def loss(self, x, t):
        y = self.predict(x)

        return self.lastlayer.forward(y, t)

    # 计算识别精度
    def accuracy(self, x, t):
        y = self.predict(x) # 推理
        y = np.argmax(y, axis=1) # 最大值的索引
        if t.ndim != 1: t = np.argmax(t, axis=1) # 如果监督数据不是按照one-hot表示的（监督数据不是1维的）
        # 如果索引相同，即识别正确，计算精度
        accuracy = np.sum(y == t) / float(x.shape[0]) # x.shape[0]是整个数据的数量
        return accuracy

    # 计算权重参数梯度
    # 通过误差反向传播计算关于权重的梯度
    def gradient(self, x, t):

        # forward
        self.loss(x, t)

        # backward
        dout = 1
        dout = self.lastlayer.backward(dout)

        layers = list(self.layers.values())
        layers.reverse() # 逆序
        for layer in layers:
            dout = layer.backward(dout)

        # 设定
        grads = {}
        grads['W1'] = self.layers['Affine1'].dW
        grads['b1'] = self.layers['Affine1'].db
        grads['W2'] = self.layers['Affine2'].dW
        grads['b2'] = self.layers['Affine2'].db

        return grads # 返回各个参数的梯度

        # 计算权重参数梯度 x:输入数据, t:监督数据
        # 通过数值微分计算（用于对比）

    def numerical_gradient(self, x, t):
        loss_W = lambda W: self.loss(x, t)
        grads = {}
        grads['W1'] = numerical_gradient(loss_W, self.params['W1'])
        grads['b1'] = numerical_gradient(loss_W, self.params['b1'])
        grads['W2'] = numerical_gradient(loss_W, self.params['W2'])
        grads['b2'] = numerical_gradient(loss_W, self.params['b2'])

        return grads  # 返回各个参数的梯度


# ----------------------------梯度确认-------------------------------------------------------
# 通过数值微分和反向误差传播进行对比结果
# 读入数据
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

network = TwoLayersNet(input_size=784, hidden_size=50, output_size=10)

x_batch = x_train[:3] # 使用部分数据
t_batch = t_train[:3]

# 计算梯度，数值微分
grad_numerical = network.numerical_gradient(x_batch, t_batch)

# 误差反向传播
grad_backprop = network.gradient(x_batch, t_batch)

# 求两种梯度法的梯度误差
for key in grad_numerical.keys():
    # 求各个权重参数对应元素的差的绝对值，并计算其平均值
    diff = np.average(np.abs(grad_backprop[key] - grad_numerical[key]))
    print(key + ":" + str(diff))

# -------------------------------使用误差反向传播法进行神经网络的学习---------------------------------
start = time.time()

# 读入数据
iters_num = 10000 # 迭代次数
train_size = x_train.shape[0] # 要训练的数据
batch_size = 100
learning_rate = 0.1

train_loss_list = []
train_acc_list = []
test_acc_list = []

iter_per_epoch = max(train_size / batch_size, 1)

for i in range(iters_num):
    batch_mask = np.random.choice(train_size, batch_size) # 随机选择
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    # 使用误差反向传播法求梯度
    grad = network.gradient(x_batch, t_batch)

    # 更新
    for key in ('W1', 'b1', 'W2', 'b2'):
        network.params[key] -= learning_rate * grad[key]

    loss = network.loss(x_batch, t_batch)
    train_loss_list.append(loss)
    if i % iter_per_epoch == 0:
        train_acc = network.accuracy(x_train, t_train)
        test_acc = network.accuracy(x_test, t_test)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        print(train_acc, test_acc)

end = time.time()
print("耗时：", (end - start))


