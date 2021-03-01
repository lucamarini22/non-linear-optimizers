import numpy as np
import main
import utils


def optimize(beta_1=0.9, beta_2=0.999, lr=0.01, eps=1e-8, max_iter=10000, tol=1e-5):
    print('adam:')
    # constants
    beta_1 = beta_1
    beta_2 = beta_2

    theta = 0
    g_t = 0
    s_t = 0
    # num of iteration
    t = 1
    # learning rate
    lr = lr
    eps = eps
    # max iteration stopping criterion
    max_iter = max_iter
    # list that contains all theta values
    theta_values = []
    # list containing all loss values
    # loss_values = []

    while True:
        theta_values.append(theta)
        # loss_values.append(np.power(theta - main.y, 2))

        grad = main.g(theta)
        g_t = beta_1 * g_t + (1 - beta_1) * grad
        s_t = beta_2 * s_t + (1 - beta_2) * grad * grad

        g_deb = g_t / (1 - np.power(beta_1, t))
        s_deb = s_t / (1 - np.power(beta_2, t))

        theta_prev = theta
        theta = theta - (lr / (np.sqrt(s_deb) + eps)) * g_deb
        t += 1

        if np.abs(theta - theta_prev) <= tol:
            print(' - tol')
            utils.print_result(t, theta)
            break
        if t > max_iter:
            print(' - max iterations reached')
            utils.print_result(t, theta)
            break
    return theta_values  # , loss_values
