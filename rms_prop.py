import numpy as np
import utils


def optimize(func_grad, beta=0.9, lr=0.01, eps=1e-8, max_iter=10000, tol=1e-6):
    print('rms_prop')
    # constants
    beta = beta

    theta = 0
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

    theta_prev = -10

    while np.abs(theta - theta_prev) > tol and t <= max_iter:
        theta_values.append(theta)
        # loss_values.append(np.power(theta - main.y, 2))

        grad = func_grad(theta)
        s_t = beta * s_t + (1 - beta) * grad * grad

        theta_prev = theta
        theta = theta - (lr / (np.sqrt(s_t) + eps)) * grad
        t += 1

    if np.abs(theta - theta_prev) <= tol:
        print(' - tol')
        utils.print_result(t, theta)
    if t > max_iter:
        print(' - max iterations reached')
        utils.print_result(t, theta)
    return theta_values  # , loss_values
