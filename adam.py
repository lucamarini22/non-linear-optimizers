import numpy as np
import utils


def optimize(func_grad, init_theta_1, init_theta_2, beta_1=0.9, beta_2=0.999, lr=0.01, eps=1e-8, max_iter=10000,
             tol=1e-6):
    print('adam')
    # constants
    theta_1 = init_theta_1
    theta_2 = init_theta_2

    g_t_1 = 0
    g_t_2 = 0

    s_t_1 = 0
    s_t_2 = 0
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

        g_t_1 = beta_1 * g_t_1 + (1 - beta_1) * grad_1
        g_t_2 = beta_1 * g_t_2 + (1 - beta_1) * grad_2

        s_t_1 = beta_2 * s_t_1 + (1 - beta_2) * grad_1 * grad_1
        s_t_2 = beta_2 * s_t_2 + (1 - beta_2) * grad_2 * grad_2

        g_deb_1 = g_t_1 / (1 - np.power(beta_1, t))
        g_deb_2 = g_t_2 / (1 - np.power(beta_1, t))

        s_deb_1 = s_t_1 / (1 - np.power(beta_2, t))
        s_deb_2 = s_t_2 / (1 - np.power(beta_2, t))

        theta_prev_1 = theta_1
        theta_prev_2 = theta_2
        theta_1 = theta_1 - (lr / (np.sqrt(s_deb_1) + eps)) * g_deb_1
        theta_2 = theta_2 - (lr / (np.sqrt(s_deb_2) + eps)) * g_deb_2
        t += 1

    if (np.abs(theta_1 - theta_prev_1) <= tol
           or np.abs(theta_2 - theta_prev_2) <= tol):
        print(' - tol')
        utils.print_result(t, (theta_1, theta_2))
    if t > max_iter:
        print(' - max iterations reached')
        utils.print_result(t, (theta_1, theta_2))
    return theta_values_1, theta_values_2, count  # , loss_values
