import numpy


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
#
