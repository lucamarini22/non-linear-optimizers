import matplotlib.pyplot as plt

import adagrad_2
import adam_2
import gradient_descent
import adam
import momentum
import momentum_2
import nesterov_momentum
import nesterov_momentum_2
import rms_prop
import adagrad
import numpy as np

import functions as func
from random import randrange

import rms_prop_2

if __name__ == '__main__':
    # constants
    lr = 0.1
    max_iter = 1000
    tol = 1e-6

    # initial values of theta_1 and theta_2
    theta_1 = randrange(10)
    theta_2 = randrange(10)

    plt.style.use('ggplot')
    # ________________________________________optimizers________________________________________
    # gradient descent
    gd_theta_values_1, gd_theta_values_2, gd_count = gradient_descent.optimize(func.grad_paraboloid,
                                                                               init_theta_1=theta_1,
                                                                               init_theta_2=theta_2,
                                                                               lr=lr, max_iter=max_iter,
                                                                               tol=tol)
    # momentum
    momentum_theta_values_1, momentum_theta_values_2, \
    momentum_count = momentum_2.optimize(func.grad_paraboloid,
                                         init_theta_1=theta_1,
                                         init_theta_2=theta_2,
                                         beta=0.9,
                                         lr=lr, max_iter=max_iter,
                                         tol=tol)
    # Nesterov momentum
    nesterov_mom_theta_values_1, nesterov_mom_theta_values_2, \
        nesterov_mom_count = nesterov_momentum_2.optimize(func.grad_paraboloid,
                                                          init_theta_1=theta_1,
                                                          init_theta_2=theta_2,
                                                          beta=0.9,
                                                          lr=lr, max_iter=max_iter,
                                                          tol=tol)
    # adagrad
    adagrad_theta_values_1, adagrad_theta_values_2, \
        adagrad_count = adagrad_2.optimize(func.grad_paraboloid,
                                           init_theta_1=theta_1,
                                           init_theta_2=theta_2,
                                           lr=lr, eps=1e-8, max_iter=max_iter,
                                           tol=tol)
    # rmsprop
    rms_prop_theta_values_1, rms_prop_theta_values_2, \
        rms_prop_count = rms_prop_2.optimize(func.grad_paraboloid,
                                             init_theta_1=theta_1,
                                             init_theta_2=theta_2,
                                             beta=0.9,
                                             lr=lr, eps=1e-8, max_iter=max_iter,
                                             tol=tol)
    # adam
    adam_theta_values_1, adam_theta_values_2, \
        adam_count = adam_2.optimize(func.grad_paraboloid,
                                     init_theta_1=theta_1,
                                     init_theta_2=theta_2,
                                     beta_1=0.9, beta_2=0.999,
                                     lr=lr, eps=1e-8, max_iter=max_iter,
                                     tol=tol)

    # defining the 3D space where to compute the function
    x = np.linspace(-2, 4, 250)
    y = np.linspace(-2, 4, 250)
    X, Y = np.meshgrid(x, y)
    Z = func.paraboloid(X, Y)

    # Angles needed for quiver plot
    # gradient descent
    anglesx = gd_theta_values_1[1:] - gd_theta_values_1[:-1]
    anglesy = gd_theta_values_2[1:] - gd_theta_values_2[:-1]

    # create figure
    fig = plt.figure(figsize=(16, 8))
    # 3D plot
    ax = fig.add_subplot(1, 2, 1, projection='3d')
    # plot 3D function
    ax.plot_surface(X, Y, Z, rstride=5, cstride=5, cmap='jet', alpha=.4, edgecolor='none')

    # ________________________________________plotting optimizers________________________________________
    # gradient descent 3D plot
    ax.plot(gd_theta_values_1, gd_theta_values_2,
            func.paraboloid(gd_theta_values_1, gd_theta_values_2),
            color='r', marker='.', alpha=.4, label='gradient_descent')
    # momentum 3D plot
    ax.plot(momentum_theta_values_1, momentum_theta_values_2,
            func.paraboloid(momentum_theta_values_1, momentum_theta_values_2),
            color='purple', marker='.', alpha=.4, label='momentum')
    # Nesterov momentum 3D plot
    ax.plot(nesterov_mom_theta_values_1, nesterov_mom_theta_values_2,
            func.paraboloid(nesterov_mom_theta_values_1, nesterov_mom_theta_values_2),
            color='black', marker='.', alpha=.4, label='Nesterov_momentum')
    # adagrad 3D plot
    ax.plot(adagrad_theta_values_1, adagrad_theta_values_2,
            func.paraboloid(adagrad_theta_values_1, adagrad_theta_values_2),
            color='violet', marker='.', alpha=.4, label='adagrad')
    # rms_prop 3D plot
    ax.plot(rms_prop_theta_values_1, rms_prop_theta_values_2,
            func.paraboloid(rms_prop_theta_values_1, rms_prop_theta_values_2),
            color='b', marker='.', alpha=.4, label='rms_prop')
    # adam 3D plot
    ax.plot(adam_theta_values_1, adam_theta_values_2,
            func.paraboloid(adam_theta_values_1, adam_theta_values_2),
            color='orange', marker='.', alpha=.4, label='adam')
    ax.legend()
    ax.view_init(45, 280)
    ax.set_xlabel('x')
    ax.set_ylabel('y')

    ax = fig.add_subplot(1, 2, 2)
    ax.contour(X, Y, Z, 50, cmap='jet')
    # Plotting the iterations and intermediate values

    # gradient descent
    ax.scatter(gd_theta_values_1, gd_theta_values_2, color='r', marker='*')
    # momentum
    ax.scatter(momentum_theta_values_1, momentum_theta_values_2, color='b', marker='*')


    ax.quiver(gd_theta_values_1[:-1], gd_theta_values_2[:-1], anglesx, anglesy,
              scale_units='xy', angles='xy', scale=1,
              color='r', alpha=.3)
    # ax.set_title('Gradient Descent with {} iterations'.format(len(gd_count)))

    plt.show()
