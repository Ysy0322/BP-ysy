import numpy as np
import math
from matplotlib import pylab
import BP

bp = BP.BPNetwork()


# 拟合sinx函数的训练集
def get_train():
    train_set = []
    train_res = []
    for i in range(-10, 11, 1):
        train_set.append([i * math.pi / 10])
        train_res.append(np.sin([i * math.pi / 10]))
    return train_set, train_res


# 拟合sinx函数的测试集
def get_test():
    test = []
    test_res = []
    for i in range(-100, 101, 1):
        test.append([i * math.pi / 100])
        test_res.append(np.sin([i * math.pi / 100]))
    return test, test_res


'''
反向传播算法对训练集进行训练
repeat：
    向前传播，计算output_cells
    反向传播，计算δ, 更新weight和bias
'''


def back_propagate(input, expects, learn=0.05, limit=10000):
    for j in range(limit):
        for i in range(len(input)):
            bp.forward_propagate(input[i])
            bp.calculate_delta(expects[i])
            bp.update_weight(learn)
            bp.update_bias(learn)


if __name__ == '__main__':
    '''
    初始化神经网络的结构
    神经网络层数的选择：
    根据经验公式：
    h = (n+m)
    '''
    bp.setup(1, 1, [10,10])
    # 初始化学习率，训练次数
    learn = 0.08
    times = 3000
    train, train_res = get_train()

    back_propagate(train, train_res, learn, times)

    test, test_res = get_test()
    average_loss, predicate_res = bp.get_average_loss(test, test_res)
    print(average_loss)

    # 画图
    pylab.plt.scatter(train, train_res, marker='x', color='g', label='train set')

    x = np.arange(-1 * np.pi, np.pi, 0.01)
    x = x.reshape((len(x), 1))
    y = np.sin(x)
    pylab.plot(x, y, label='standard sinx')

    pylab.plot(test, predicate_res, label='predicate sinx, learn = ' + str(learn) + ' times = ' + str(times),
               linestyle='--',
               color='r')

    pylab.plt.legend(loc='best')
    pylab.plt.show()
