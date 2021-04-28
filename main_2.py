import matplotlib.pyplot as plt

import gradient_descent
import adam
import momentum
import nesterov_momentum
import rms_prop
import adagrad
import numpy as np

import functions as func



if __name__ == '__main__':
    lr = 0.1
    max_iter = 100
    tol = 1e-6
    plt.style.use('ggplot')

    theta_values_1, theta_values_2, count = gradient_descent.optimize(func.grad_paraboloid,
                                                                      lr=lr, max_iter=max_iter,
                                                                      tol=tol)
    theta_values_3, theta_values_4, count_2 = gradient_descent.optimize(func.grad_paraboloid,
                                                                      lr=lr, max_iter=max_iter,
                                                                      tol=tol)

    x = np.linspace(-2, 4, 250)
    y = np.linspace(-2, 4, 250)
    X, Y = np.meshgrid(x, y)
    Z = func.paraboloid(X, Y)

    # Angles needed for quiver plot
    anglesx = theta_values_1[1:] - theta_values_1[:-1]
    anglesy = theta_values_2[1:] - theta_values_2[:-1]

    fig = plt.figure(figsize=(16, 8))
    ax = fig.add_subplot(1, 2, 1, projection='3d')
    ax.plot_surface(X, Y, Z, rstride=5, cstride=5, cmap='jet', alpha=.4, edgecolor='none')
    ax.plot(theta_values_1, theta_values_2,
            func.paraboloid(theta_values_1, theta_values_2),
            color='r', marker='*', alpha=.4)

    '''
    ax.plot(theta_values_3, theta_values_4,
            func.paraboloid(theta_values_3, theta_values_4),
            color='b', marker='x', alpha=.4)
            '''
    ax.view_init(45, 280)
    ax.set_xlabel('x')
    ax.set_ylabel('y')

    ax = fig.add_subplot(1, 2, 2)
    ax.contour(X, Y, Z, 50, cmap='jet')
    # Plotting the iterations and intermediate values
    ax.scatter(theta_values_1, theta_values_2, color='r', marker='*')
    '''
    ax.scatter(theta_values_3, theta_values_4, color='b', marker='x')
    '''
    ax.quiver(theta_values_1[:-1], theta_values_2[:-1], anglesx, anglesy, scale_units='xy', angles='xy', scale=1, color='r', alpha=.3)
    ax.set_title('Gradient Descent with {} iterations'.format(len(count)))

    plt.show()
