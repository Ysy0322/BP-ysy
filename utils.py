# -*- coding: utf-8 -*-
import os
import numpy
import pylab
from PIL import Image
from sklearn.model_selection import train_test_split
from pylab import savetxt

'''
生成高斯分布的概率密度随机数
loc：float
    此概率分布的均值（对应着整个分布的中心centre）
scale：float
    此概率分布的标准差（对应于分布的宽度，scale越大越矮胖，scale越小，越瘦高）
size：int or tuple of ints
    输出的shape，默认为None，只输出一个值
'''

numpy.random.seed(0)


def generate_random(x, y):
    return numpy.random.normal(loc=0.0, scale=numpy.power(y, -0.5), size=(x, y))


'''
读取特定文件夹下的bmp格式图片文件列表
'''


def get_file_list(path):
    files = os.listdir(path)
    files.sort(key=lambda x: int(x.split('.')[0]))
    return [os.path.join(path, f) for f in files if f.endswith('.bmp')]


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
        # savetxt("word image\\word" + str(1 + w) + '.txt', image_bit_matrix[w], fmt="%0i")
    return image_bit_matrix


'''
读取测试文件夹下面的所有图片
'''


def get_test_image_matrix():
    image_file_matrix = get_file_list("test")
    image_bit_matrix = [0.0] * len(image_file_matrix)
    for i in range(len(image_file_matrix)):
        image = numpy.array(Image.open(image_file_matrix[i]))
        image_bit_matrix[i] = numpy.ndarray.flatten(image)
        numpy.array(image.reshape(1, 784))

    test = image_bit_matrix

    return test


# savetxt("test\\test_data", get_test_image_matrix(), fmt="%0i")

'''
分割训练集和测试集
'''


def split(x, y, scale=0.20):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=scale, random_state=0)
    return x_train, x_test, y_train, y_test


def get_all_data():
    image_matrix = get_image_matrix()

    i = 0
    train = [0.0] * 620 * 12
    train_label = [0.0] * 620 * 12
    label = [0.0] * 12
    for x in range(620):
        for y in range(12):
            train[i] = image_matrix[y][x]
            train[i] = numpy.array(train[i]).astype(int)
            label = [0.0] * 12
            label[y] = 1
            train_label[i] = label
            i += 1

    return train, train_label


'''

'''


def get_data(scale=0.20):
    res_total_matrix = [0.0] * 12
    word_train = [0.0] * 12
    res_train = [0.0] * 12
    word_test = [0.0] * 12
    res_test = [0.0] * 12
    image_matrix = get_image_matrix()
    train = [0.0] * int(12 * 620 * (1 - scale))
    train_label = [0.0] * int(12 * 620 * (1 - scale))
    test = [0.0] * int(12 * 620 * scale)
    test_label = [0.0] * int(12 * 620 * scale)

    for w in range(12):
        res_total_matrix[w] = [0.0] * 620
        for i in range(620):
            res_total_matrix[w][i] = [0.0] * 12
            for j in range(12):
                if j == w:
                    res_total_matrix[w][i][j] = 1
                else:
                    res_total_matrix[w][i][j] = 0
        word_train[w], word_test[w], res_train[w], res_test[w] = split(image_matrix[w], res_total_matrix[w], scale)

    i = 0
    j = 0
    for x in range(len(word_train[0])):
        for y in range(12):
            train[i] = word_train[y][x]
            train[i] = numpy.array(train[i]).astype(int)
            train_label[i] = res_train[y][x]
            i += 1

    for x in range(len(word_test[0])):
        for y in range(12):
            test[j] = word_test[y][x]
            test[j] = numpy.array(test[j]).astype(int)
            test_label[j] = res_test[y][x]
            j += 1

    return train, test, train_label, test_label


'''
画图
'''


def draw(train_corrects, test_corrects, hid, learn):
    x = numpy.arange(1, len(train_corrects) + 1, 1)
    pylab.plot(x, train_corrects, label="train correction ")
    if len(test_corrects) != 0:
        pylab.plot(x, test_corrects, label="test correction")
    pylab.xlabel("train times")
    pylab.ylabel("correction")
    pylab.title("hid nodes: " + str(hid) + ", learn rate: " + str(learn))
    pylab.legend(loc='best')
    pylab.show()


'''
预测结果写入文件
'''
'''
predict_label = []
predict_label.append(1)
predict_label.append(2)
predict_label.append(3)
save_predict(predict_label, "out\\test_predict")
'''


def save_predict(predict_label, path):
    savetxt(path, predict_label, fmt="%0i")


'''
数据写入文件
'''


def write_train_data():
    # word_train, word_test, res_train, res_test = get_data()
    train, train_label = get_all_data()
    savetxt("split train&test\\train", train, fmt="%0i")
    savetxt("split train&test\\train_label", train_label, fmt="%0i")
    # savetxt("split train&test\\word_train", word_train, fmt="%0i")
    # savetxt("split train&test\\word_test", word_test, fmt="%0i")
    # savetxt("split train&test\\res_train", res_train, fmt="%0i")
    # savetxt("split train&test\\res_test", res_test, fmt="%0i")

# write_train_data()
