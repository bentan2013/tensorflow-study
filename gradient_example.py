# x**3 + 2 *x + e**3 -3 = 0, x=?
# https://ctmakro.github.io/site/on_learning/gd.html

def expression(x):
    e = 2.718
    return x ** 3 + 2 * x + e ** 3 - 3


def error(x):
    return expression(x) ** 2


def d(x):
    detal = 0.00000001
    return (error(x + detal) - error(x - detal)) / (2 * detal)


def gradient_descent(x):
    rate = 0.001
    print(d(x))
    x = x - d(x) * rate
    return x


x = 0.0
for i in range(1000):
    x = gradient_descent(x)
    if error(x) < 0.01 and error(x) > -0.01:
        print("finish", x, error(x))
        break
    else:
        print(x, error(x))
