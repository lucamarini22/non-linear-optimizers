import numpy as np

def paraboloid(x1, x2):
    r_x = (x1 ** 2) + (x2 ** 2)
    return r_x


def grad_paraboloid(x1, x2):
    grad = [2*x1, 2*x2]
    #grad = [400 * np.power(x1, 3) - 400 * x1 * x2 + 2 * x1 - 2, 200 * (x2 - np.power(x1, 2))]
    return np.asarray(grad, dtype=np.float64)