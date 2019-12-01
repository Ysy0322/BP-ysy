from sympy import *


def function():
    def innerF(x):
        return x * 2 + 1

    return innerF


# 函数求导
def diffTest(function, arg):
    x = Symbol('x')
    y = function()
    print(diff(y, x))
    print(diff(y, x).subs(x, arg))
    return diff(y, x).subs(x, arg)


def test(function):
    x = 1
    print(diff(function, x))


diffTest(function, 2)
