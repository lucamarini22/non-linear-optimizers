import math
import numpy as np
import matplotlib.pyplot as plt
import adam


# function
def f(x):
    return np.power(x, 5) - 3 * x + 2


# gradient of function
def g(x):
    return np.power(x, 4) * 5 - 3


# minimum of the considered function
y = 0.9096

if __name__ == '__main__':
    plt.style.use('ggplot')

    theta_values, loss_values = adam.optimize(max_iter=1000)

    # plt.title('theta values')
    plt.xlabel('iterations')
    plt.ylabel('theta values')
    plt.plot(theta_values)
    plt.show()

    plt.xlabel('iterations')
    plt.ylabel('loss values')
    plt.plot(loss_values)
    plt.show()
