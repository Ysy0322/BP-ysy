import datetime
import numpy
import utils
import scipy


class BPNetwork:
    def __int__(self):
        # 输入神经元数
        self.input_n = 0
        # 输入神经元数组 1 * 784 * 1
        self.input_cells = []
        # 输出神经元数
        self.output_m = 0
        # 输出神经元数组 1 * 12
        self.output_cells = []
        # 隐藏层神经元数
        self.hidden_set = 0
        # 隐藏层结果 1 * set
        self.hidden_result = []
        # 输入层到隐藏层权值数组 784 * set
        self.input_w = []
        # 隐藏层到输出层权值数组 set * 12
        self.output_w = []
        # 输入层到隐藏层的Bias 1 * set
        self.input_b = []
        self.output_b = []
        # 隐藏层到输出层的delta 1 * 12
        self.output_delta = []
        # 输入层到隐藏层的delta 1 * set
        self.input_delta = []

    '''
    初始化神经网络的结构，一层隐藏层
    '''

    def setup(self, input_n, output, hidden_size):
        self.input_n = input_n
        self.input_cells = [1.0] * input_n
        self.output_m = output
        self.output_cells = [0.0] * output
        self.hidden_set = hidden_size
        self.hidden_result = [0.0] * hidden_size
        self.input_w = utils.generate_random(input_n, hidden_size)
        self.output_w = utils.generate_random(hidden_size, output)
        self.input_b = numpy.random.random(hidden_size)
        self.output_b = numpy.random.random(output)
        self.input_delta = [0.0] * self.input_n
        self.output_delta = [0.0] * self.hidden_set

    '''
    一次向前传播
    '''

    def softmax(self, x):
        x = x - numpy.max(x)
        result = numpy.exp(x) / numpy.sum(numpy.exp(x))
        return result

    def activation_function(self, x):
        return scipy.special.expit(x)

    def forward_propagate(self, train):
        # 第一层到隐藏层
        self.input_cells = numpy.array(train)
        self.input_cells.shape = (784, 1)
        self.hidden_result = numpy.dot(numpy.transpose(self.input_w), self.input_cells)

        self.input_b.shape = (self.hidden_set, 1)
        self.hidden_result = self.activation_function(self.hidden_result + self.input_b)

        # 隐藏层到输出层
        self.output_cells = numpy.dot(numpy.transpose(self.output_w), self.hidden_result)
        self.output_b.shape = (self.output_m, 1)
        self.output_cells = self.activation_function(self.output_cells + self.output_b)

        return self.output_cells

    def back_propagate(self, target, learn):
        target = numpy.array(target)
        target.shape = (12, 1)
        # 输出层到隐藏层
        output_error = target - self.output_cells
        self.output_delta = numpy.array(output_error * numpy.multiply(self.output_cells, (1 - self.output_cells)))
        self.output_delta.shape = (12, 1)
        self.hidden_result.shape = (self.hidden_set, 1)
        self.output_b.shape = (self.output_m, 1)
        self.output_w += learn * numpy.dot(self.hidden_result, self.output_delta.T)
        self.output_b += learn * self.output_delta
        # 隐藏层到输入层
        hidden_error = numpy.dot(self.output_w, output_error)
        hidden_error.shape = (self.hidden_set, 1)
        self.input_delta = hidden_error * self.hidden_result * (1 - self.hidden_result)
        self.input_delta = numpy.array(self.input_delta)
        self.input_delta.shape = (self.hidden_set, 1)
        self.input_cells.shape = (784, 1)
        self.input_b.shape = (self.hidden_set, 1)
        self.input_w += learn * numpy.dot(self.input_cells, self.input_delta.T)
        self.input_b += learn * self.input_delta

    '''
    计算正确率
    '''

    def calculate_correct(self, train_data, train_label):
        predict_label = []
        count = 0
        for i in range(len(train_data)):
            predicate = self.forward_propagate(train_data[i].astype(int))
            label = numpy.argmax(predicate)
            predict_label.append(label + 1)
            if label == train_label[i].index(1):
                count += 1
        return count / len(train_data), predict_label

    '''
    训练
    每2次打印一次正确率
    '''

    def train(self, times, train_data, train_label, learn):
        i = 1
        train_corrects = []
        test_corrects = []
        while i <= times:
            for w in range(len(train_data)):
                self.forward_propagate(train_data[w])
                self.back_propagate(train_label[w], learn=learn)

            # j = i
            # if j % 2 == 0 or j == times:
            print("第 " + str(i) + " 次" + "反向传播训练")
            train_correct, train_label_predict = self.calculate_correct(train_data, train_label)
            train_corrects.append(train_correct)
            '''
            for test
            '''
            # print("训练集正确率为: " + str(train_correct))
            # test_correct, test_label_predict = self.calculate_correct(word_test, res_test)
            # test_corrects.append(test_correct)
            # print("测试集正确率为: " + str(test_correct))

            i += 1
        return train_corrects, test_corrects


word_train, word_test, res_train, res_test = utils.get_data()
train, train_label = utils.get_all_data()
test = utils.get_test_image_matrix()

bp = BPNetwork()

if __name__ == '__main__':
    '''
    初始化神经网络的结构
    输入层 28 * 28 = 784
    输出层 12
    '''
    hid = 100
    bp.setup(784, 12, hid)
    # 初始化学习率，训练次数
    learn = 0.01
    times = 150
    print("训练开始: " + datetime.datetime.now().strftime('%Y.%m.%d-%H:%M:%S'))
    print("隐藏层节点数: " + str(hid) + '\n' +
          "学习率: " + str(learn) + '\n' +
          "训练次数: " + str(times))
    # train_corrects, test_corrects = bp.train(times, word_train, res_train, learn)
    train_corrects, test_corrects = bp.train(times, train, train_label, learn)
    print("训练结束: " + datetime.datetime.now().strftime('%Y.%m.%d-%H:%M:%S'))
    print("测试开始: " + datetime.datetime.now().strftime('%Y.%m.%d-%H:%M:%S'))
    predict = []
    for i in range(len(test)):
        res = bp.forward_propagate(test[i])
        label = numpy.argmax(res)
        predict.append(label+1)
    utils.save_predict(predict, "out\\predict_2")

    '''
    for test
    '''
    # count, predict_label = bp.calculate_correct(word_test, res_test)
    # utils.save_predict(predict_label, "out\\predict_1")

    print("测试结束: " + datetime.datetime.now().strftime('%Y.%m.%d-%H:%M:%S'))
    # print("测试集的正确率为: " + str(count))
    # utils.draw(train_corrects, test_corrects, hid, learn)
