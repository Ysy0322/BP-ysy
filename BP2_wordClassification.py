import datetime

import numpy
import scipy

import utils
import BP0_basic


def softmax(x):
    return numpy.exp(x) / numpy.sum(numpy.exp(x), axis=0)


def sigmoid(x):
    return 1 / 1 + numpy.exp(-x)


# 测试结果
scores = [3.0, 1.0, 0.2]


# print(softmax(scores))


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
        self.input_b = BP0_basic.init_bias(input_n)
        self.output_b = BP0_basic.init_bias(hidden_size)

    '''
    一次向前传播
    '''

    def forward_propagate(self, input_cells):
        # 输入层到隐藏层
        self.input_cells = input_cells
        print(self.input_cells)
        for i in range(self.hidden_set):
            for w in range(self.input_n):
                self.hidden_result[i] = sigmoid(self.input_w[w][i] * self.input_cells[w])
        # 隐藏层到输出层
        for i in range(self.output_m):
            for w in range(self.hidden_set):
                self.output_cells[i] = sigmoid(self.output_w[w][i] * self.hidden_result[i])

        return self.output_cells[:]

    '''
    包含向前传播，计算误差，更新weight和bias的一次反向传播
    '''

    def back_propagate(self, input_cells, target, learn):
        # 向前传播
        self.forward_propagate(input_cells)

        # 计算误差
        output_error = numpy.array(self.output_cells) - numpy.array(target)
        hidden_error = numpy.dot(numpy.array(self.output_w), output_error)

        # 更新input_w,output_w
        self.output_w += learn * numpy.dot((numpy.array(output_error) * numpy.array(self.output_cells) * (1.0 - numpy.array(self.output_cells))),
                                           numpy.transpose(self.hidden_result))

        self.input_w += learn * numpy.dot((numpy.array(hidden_error) * numpy.array(self.hidden_result) * (1.0 - numpy.array(self.hidden_result))),
                                          numpy.transpose(inputs))

        # 更新input_b,output_b

    '''
    训练
    '''

    def train(self, input_cells, target, learn, times):
        for i in range(times):
            self.back_propagate(input_cells, target, learn)


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
    bp.setup(784, 12, 100)
    # 初始化学习率，训练次数
    learn = 0.1
    times = 100
    word_train, word_test, res_train, res_test = utils.get_train_data()
    print("训练开始: " + str(datetime.datetime.now()))
    for w in range(12):
        for i in range(len(word_train[w])):
            bp.train(word_train[w][i], res_train[w][i], learn, times)
    print("训练结束: " + str(datetime.datetime.now()))
    error = 0
    for w in 12:
        for i in range(len(word_test[w])):
            predicate = bp.forward_propagate(word_test[w][i])
            label = numpy.argmax(predicate)
            if label == res_test[w][i]:
                error += 0
            error += 1

    print("测试集的正确率为: ")
    print(1 - error / (12 * len(word_test)))


class neuralNetwork:
    # initialise the neural network
    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
        # set number of nodes in each input,hidden,output layer
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes
        # learning rate
        self.lr = learningrate

        # link weight matrices ,wih and who
        # weight inside the arrays are w_i_j, where link is from node i to node j in the next layer
        # w11 w21
        # w12 w22 etc
        self.wih = (numpy.random.normal(0.0, pow(self.hnodes, -0.5), (self.hnodes, self.inodes)))
        self.who = (numpy.random.normal(0.0, pow(self.onodes, -0.5), (self.onodes, self.hnodes)))

        # activation function is the sigmoid function
        self.activation_function = lambda x: scipy.special.expit(x)

        pass

    # train the neural network
    def train(self, inputs_list, targets_list):
        # convert inputs list to 2d array
        inputs = numpy.array(inputs_list, ndmin=2).T
        targets = numpy.array(targets_list, ndmin=2).T

        # calculate signals into hidden layer
        hidden_inputs = numpy.dot(self.wih, inputs)
        # calculate the signals emerging from hidden layer
        hidden_outputs = self.activation_function(hidden_inputs)

        # calculate signals into final output layer
        final_inputs = numpy.dot(self.who, hidden_outputs)
        # calculate the signals emerging from final output layer
        final_outputs = self.activation_function(final_inputs)

        # output layer error is the (target-actual)
        output_errors = targets - final_outputs
        # hidden layer error is the output_errors,split by weights,recombined at hidden nodes
        hidden_errors = numpy.dot(self.who.T, output_errors)

        # update the weights for the links between the hidden and output layers
        self.who += self.lr * numpy.dot((output_errors * final_outputs * (1.0 - final_outputs)),
                                        numpy.transpose(hidden_outputs))

        # update the weights for the links between the input and hidden layers
        self.wih += self.lr * numpy.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)),
                                        numpy.transpose(inputs))

        pass

    # query the neural network
    def query(self, inputs_list):
        # convert inputs list to 2d array
        inputs = numpy.array(inputs_list, ndmin=2).T

        # calculate signals into hidden layer
        hidden_inputs = numpy.dot(self.wih, inputs)
        # calculate the signals emerging from hidden layer
        hidden_outputs = self.activation_function(hidden_inputs)

        # calculate signals into final output layer
        final_inputs = numpy.dot(self.who, hidden_outputs)
        # calculate the signals emerging from final output layer
        final_outputs = self.activation_function(final_inputs)

        return final_outputs


input_nodes = 784
hidden_nodes = 200
output_nodes = 10

# learning rate is 0.3
learning_rate = 0.1

# create instance of neural network
n = neuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)

# train the neural network

# load the mnist training data csv file into a list
training_data_file = open("mnist_dataset/mnist_train.csv", 'r')
training_data_list = training_data_file.readlines()
training_data_file.close()

# epochs is the number of times the training data set is used for training
epochs = 5
for e in range(epochs):
    # go through all records in the training data set
    for record in training_data_list:
        all_values = record.split(',')
        # scale and shift the inputs
        inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
        # create the target output values (all 0.01, except the desired label which is 0.99)
        targets = numpy.zeros(output_nodes) + 0.01
        # all_values[0] is the target label for this record
        targets[int(all_values[0])] = 0.99
        n.train(inputs, targets)
        pass
