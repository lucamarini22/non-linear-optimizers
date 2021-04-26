import matplotlib.pyplot as plt

import gradient_descent
import adam
import momentum
import nesterov_momentum
import rms_prop
import adagrad


# function
def f(x):
    return x * x - x * x * x - 4 * x


# gradient of function
def g(x):
    return 4 * x * x * x - 3 * x * x * x - 4


if __name__ == '__main__':
    lr = 0.1
    max_iter = 2000
    tol = 1e-6
    plt.style.use('ggplot')

    gradient_descent_theta_values = gradient_descent.optimize(g, eps=1e-8, lr=lr, max_iter=max_iter, tol=tol)
    momentum_theta_values = momentum.optimize(g, eps=1e-8, lr=lr, max_iter=max_iter, tol=tol)
    nesterov_momentum_theta_values = nesterov_momentum.optimize(g, eps=1e-8, lr=lr, max_iter=max_iter, tol=tol)
    adagrad_theta_values = adagrad.optimize(g, eps=1e-8, lr=lr, max_iter=max_iter, tol=tol)
    rms_prop_theta_values = rms_prop.optimize(g, eps=1e-8, lr=lr, max_iter=max_iter, tol=tol)
    adam_theta_values = adam.optimize(g, eps=1e-8, lr=lr, max_iter=max_iter, tol=tol)

    # plt.title('')
    plt.xlabel('iterations')
    plt.ylabel('theta values')
    plt.plot(gradient_descent_theta_values, color='black', label='gradient_descent')
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
