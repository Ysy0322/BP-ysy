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
        res_total_matrix[w] = [w] * 620
        word_train[w], word_test[w], res_train[w], res_test[w] = split(image_matrix[w], res_total_matrix[w])
    x_train, x_test, y_train, y_test = split(image_matrix[0], res_total_matrix[0])
    # print("words for train:")
    # print(x_train)
    # print("words for test:")
    # print(x_test)
    # print("res for train:")
    # print(y_train)
    # print("res for test:")
    # print(y_train)
    # savetxt()
    return word_train, word_test, res_train, res_test


get_train_data()
# if __name__ == '__main__':
#     train_data, train_label, test_data, test_label = get_data()
#     savetxt("train_data", train_data, fmt="&.0i")
#     savetxt("train_label", train_label, fmt="&.0i")
#     savetxt("test_data", test_data, fmt="&.0i")
#     savetxt("test_label", test_label, fmt="&.0i")
