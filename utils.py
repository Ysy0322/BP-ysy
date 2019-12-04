# -*- coding: utf-8 -*-
import os
import numpy
from PIL import Image
from sklearn.model_selection import train_test_split
from pylab import savetxt

'''
读取特定文件夹下的bmp格式图片文件列表
'''


def get_file_list(path):
    # for i in os.listdir(path):
    #     if i.endswith('.bpm'):
    #         os.path.join(path, i)
    # return os.path
    return [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.bmp')]


'''
image_file_matrix = [0.0] * 12 文字图片文件的矩阵
image_bit_matrix = [0.0] * 12 文字图片bit的矩阵
image_bit_matrix[w][i]
是第w个文字的第i个图片的bit矩阵
'''


def get_image_matrix():
    image_file_matrix = [0.0] * 12
    image_bit_matrix = [0.0] * 12

    for w in range(12):
        image_file_matrix[w] = get_file_list("train\\" + str(w + 1))

    for w in range(12):
        image_bit_matrix[w] = [0.0] * 620
        for i in range(620):
            image = numpy.array(Image.open(image_file_matrix[w][i]))
            image_bit_matrix[w][i] = numpy.ndarray.flatten(image)
            numpy.array(image.reshape(1, 784))
            #
        # savetxt("word image\\word" + str(1 + w) + '.txt', image_bit_matrix[w], fmt="%0i")
    return image_bit_matrix


# get_image_matrix()

x = [[[1, 1, 0], [1, 0, 1], [1, 0, 0], [1, 1, 1]], [[0, 1, 0], [0, 0, 1], [0, 0, 0]], ]
y = [1, 1, 1, 1]


def split(x, y):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=0)
    return x_train, x_test, y_train, y_test


# print(split(x, y))


def get_train_data():
    res_total_matrix = [0.0] * 12
    word_train = [0.0] * 12
    res_train = [0.0] * 12
    word_test = [0.0] * 12
    res_test = [0.0] * 12
    image_matrix = get_image_matrix()

    for w in range(12):
        res_total_matrix[w] = [0.0] * 620
        for i in range(620):
            res_total_matrix[w][i] = [0.0] * 12
            for j in range(12):
                if j == w:
                    res_total_matrix[w][i][j] = 1.0
        word_train[w], word_test[w], res_train[w], res_test[w] = split(image_matrix[w], res_total_matrix[w])
    return word_train, word_test, res_train, res_test


if __name__ == '__main__':
    word_train, word_test, res_train, res_test = get_train_data()
    for i in range(12):
        savetxt("split train&test\\word_train_" + str(i), word_train[i], fmt="%0i")
        savetxt("split train&test\\word_test_" + str(i), word_test[i], fmt="%0i")
        savetxt("split train&test\\res_train_" + str(i), res_train[i], fmt="%0i")
        savetxt("split train&test\\res_test_" + str(i), res_test[i], fmt="%0i")


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
#
#
# input_nodes = 784
# hidden_nodes = 200
# output_nodes = 10
#
# # learning rate is 0.3
# learning_rate = 0.1
#
# # create instance of neural network
# n = neuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)
#
# # train the neural network
#
# # load the mnist training data csv file into a list
# training_data_file = open("mnist_dataset/mnist_train.csv", 'r')
# training_data_list = training_data_file.readlines()
# training_data_file.close()
#
# # epochs is the number of times the training data set is used for training
# epochs = 5
# for e in range(epochs):
#     # go through all records in the training data set
#     for record in training_data_list:
#         all_values = record.split(',')
#         # scale and shift the inputs
#         inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
#         # create the target output values (all 0.01, except the desired label which is 0.99)
#         targets = numpy.zeros(output_nodes) + 0.01
#         # all_values[0] is the target label for this record
#         targets[int(all_values[0])] = 0.99
#         n.train(inputs, targets)
#         pass
