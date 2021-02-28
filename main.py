import math
import numpy as np
import matplotlib.pyplot as plt

# function
def f(x):
    return np.power(x, 5) - 3 * x + 2


# gradient of function
def g(x):
    return np.power(x, 4) * 5 - 3


if __name__ == '__main__':
    plt.style.use('ggplot')

    # constants
    beta_1 = 0.9
    beta_2 = 0.999

    theta = 0
    g_t = 0
    s_t = 0
    # num of iteration
    t = 1
    # learning rate
    lr = 0.01
    eps = 1e-9
    # max iteration stopping criterion
    max_iter = 1000
    # list that contains all theta values
    theta_values = []
    # list containing all loss values
    loss_values = []

    # true value
    y = 0.9096

    while True:
        theta_values.append(theta)
        loss_values.append(np.power(theta - y, 2))

        grad = g(theta)
        g_t = beta_1 * g_t + (1 - beta_1) * grad
        s_t = beta_2 * s_t + (1 - beta_2) * grad * grad

        g_deb = g_t / (1 - np.power(beta_1, t))
        s_deb = s_t / (1 - np.power(beta_2, t))

        theta_prev = theta
        theta = theta - (lr / (np.sqrt(s_deb) + eps)) * g_deb
        t += 1

        if f(theta) == 0 or t > max_iter:
            print('exit')
            print(theta)
            break

    plt.plot(theta_values)
    plt.show()
    plt.plot(loss_values)
    plt.show()


