import numpy as np
import pylab
import matplotlib.pyplot as plt

# 绘制标准Sinx函数图像
x = np.arange(0, 2 * np.pi, 0.01)
x = x.reshape((len(x), 1))
y = np.sin(x)

pylab.plot(x, y, label='sinx')
pylab.legend(loc='upper right')
pylab.plot(x, y-1, label='sinx-1', linestyle='--', color='r')
pylab.legend(loc='upper right')
plt.axhline(linewidth=1, color='r')
plt.axvline(x=np.pi, linestyle='--', linewidth=1, color='r')

# pylab.plot(test_x_ndarray, test_y_ndarray, ‘–‘, label = str(times) + ‘ times’)


plt.show()
