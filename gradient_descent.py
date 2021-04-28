import numpy as np
import utils
from random import randrange


def optimize(func_grad, lr=0.01, max_iter=10000, tol=1e-6):
    print('gradient_descent')
    # constants
    theta_1 = randrange(10)
    theta_2 = randrange(10)

    # lists that contain theta_1, theta_2 values and the iterations
    theta_values_1, theta_values_2, count = np.empty(0), np.empty(0), np.empty(0)
    # num of iteration
    t = 1
    # learning rate
    lr = lr
    # max iteration stopping criterion
    max_iter = max_iter


    # list containing all loss values
    # loss_values = []
    theta_prev_1 = -10
    theta_prev_2 = -10

    while (np.abs(theta_1 - theta_prev_1) > tol
           and np.abs(theta_2 - theta_prev_2) > tol) \
            and t <= max_iter:
        theta_values_1 = np.append(theta_values_1, theta_1)
        theta_values_2 = np.append(theta_values_2, theta_2)
        count = np.append(count, t)
        # loss_values.append(np.power(theta - main.y, 2))

        grad_1, grad_2 = func_grad(theta_1, theta_2)

        theta_prev_1 = theta_1
        theta_prev_2 = theta_2
        theta_1 = theta_1 - lr * grad_1
        theta_2 = theta_2 - lr * grad_2
        t += 1

    if (np.abs(theta_1 - theta_prev_1) <= tol
           or np.abs(theta_2 - theta_prev_2) <= tol):
        print(' - tol')
        utils.print_result(t, (theta_1, theta_2))
    if t > max_iter:
        print(' - max iterations reached')
        utils.print_result(t, (theta_1, theta_2))
    return theta_values_1, theta_values_2, count  # , loss_values
