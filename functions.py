import numpy as np


# ________________________________________paraboloid________________________________________
def paraboloid(x1, x2):
    r_x = (x1 ** 2) + (x2 ** 2)
    return r_x


def grad_paraboloid(x1, x2):
    dx1 = 2 * x1
    dx2 = 2 * x2
    grad = [dx1, dx2]
    return np.asarray(grad, dtype=np.float64)


# ________________________________________matyas________________________________________
def matyas_function(x1, x2):
    return 0.26 * (x1 ** 2 + x2 ** 2) - 0.48 * x1 * x2


def grad_matyas_function(x1, x2):
    dx1 = 0.52 * x1 - 0.48 * x2
    dx2 = 0.52 * x2 - 0.48 * x1
    grad = [dx1, dx2]
    return np.asarray(grad, dtype=np.float64)


# ________________________________________easom________________________________________
def easom_function(x1, x2):
    return -np.cos(x1) * np.cos(x2) * np.exp(-(x1 - np.pi) ** 2 - (x2 - np.pi) ** 2)


def grad_easom_function(x1, x2):
    dx1 = 2 * np.exp(-(-np.pi + x1) ** 2 - (-np.pi + x2) ** 2) * (-np.pi + x1) * np.cos(x1) * np.cos(x2) + np.exp(-(-np.pi + x1) ** 2 - (-np.pi + x2) ** 2) * np.cos(x2) * np.sin(x1)
    dx2 = 2 * np.exp(-(-np.pi + x1) ** 2 - (-np.pi + x2) ** 2) * (-np.pi + x2) * np.cos(x1) * np.cos(x2) + np.exp(-(-np.pi + x1) ** 2 - (-np.pi + x2) ** 2) * np.cos(x1) * np.sin(x2)
    grad = [dx1, dx2]
    return np.asarray(grad, dtype=np.float64)


# ________________________________________bukin________________________________________
def bukin_function(x1, x2):
    return 100 * np.sqrt(np.abs(x2 - 0.01 * x1 ** 2)) + 0.01 * np.abs(x1 + 10)


def grad_bukin_function(x1, x2):
    dx1 = (0.01 * (10 + x1)) / np.abs(10 + x1) - (x1 * (-0.01 * x1 ** 2 + x2)) / np.abs(-0.01 * x1 ** 2 + x2) ** (3 / 2)
    dx2 = (50 * (-0.01 * x1 ** 2 + x2)) / np.abs(-0.01 * x1 ** 2 + x2) ** (3 / 2)
    grad = [dx1, dx2]
    return np.asarray(grad, dtype=np.float64)


# ________________________________________three_hump_camel_function________________________________________
def three_hump_camel_function(x1, x2):
    return 2 * (x1 ** 2) - 1.05 * (x1 ** 4) + ((x1 ** 6) / 6) + x1 * x2 + (x2 ** 2)


def grad_three_hump_camel_function(x1, x2):
    dx1 = (x1 ** 5) - 4.2 * (x1 ** 3) + 4 * x1 + x2
    dx2 = x1 + 2 * x2
    grad = [dx1, dx2]
    return np.asarray(grad, dtype=np.float64)
