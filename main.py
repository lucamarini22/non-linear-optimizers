import matplotlib.pyplot as plt

import adam
import momentum
import nesterov_momentum
import rms_prop
import adagrad


# function
def f(x):
    return x * x * x * x - x * x * x - 4 * x


# gradient of function
def g(x):
    return 4 * x * x * x - 3 * x * x - 4


if __name__ == '__main__':
    lr = 0.1
    max_iter = 10000
    plt.style.use('ggplot')

    momentum_theta_values = momentum.optimize(g, eps=1e-8, lr=lr, max_iter=max_iter)
    nesterov_momentum_theta_values = nesterov_momentum.optimize(g, eps=1e-8, lr=lr, max_iter=max_iter)
    adagrad_theta_values = adagrad.optimize(g, eps=1e-8, lr=lr, max_iter=max_iter)
    rms_prop_theta_values = rms_prop.optimize(g, eps=1e-8, lr=lr, max_iter=max_iter)
    adam_theta_values = adam.optimize(g, eps=1e-8, lr=lr, max_iter=max_iter)

    # plt.title('')
    plt.xlabel('iterations')
    plt.ylabel('theta values')
    plt.plot(momentum_theta_values, color='violet', label='momentum')
    plt.plot(nesterov_momentum_theta_values, color='red', label='Nesterov momentum')
    plt.plot(adagrad_theta_values, color='green', label='adagrad')
    plt.plot(rms_prop_theta_values, color='blue', label='rms_prop')
    plt.plot(adam_theta_values, color='orange', label='adam')
    plt.legend()
    plt.show()

    # plt.title('')
    # plt.xlabel('iterations')
    # plt.ylabel('loss values')
    # plt.plot(adam_loss_values, color='orange', label='adam')
    # plt.plot(rms_prop_loss_values, color='blue', label='rms_prop')
    # plt.legend()
    # plt.show()
