import matplotlib.pyplot as plt

import gradient_descent
import adam
import momentum
import nesterov_momentum
import rms_prop
import adagrad
import numpy as np


def rosenbrock(x):

    r_x = 100 * (x[1] - x[0] ** 2)**2 + (1 - x[0]) ** 2
    return r_x


def grad_r(x1, x2):
    grad = [400 * np.power(x1, 3) - 400 * x1 * x2 + 2*x1 - 2, 200 * (x2 - np.power(x1, 2))]
    return np.asarray(grad)


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

    gradient_descent_theta_values = gradient_descent.optimize(grad_r, eps=1e-8, lr=lr, max_iter=max_iter, tol=tol)
    momentum_theta_values = momentum.optimize(g, eps=1e-8, lr=lr, max_iter=max_iter, tol=tol)
    nesterov_momentum_theta_values = nesterov_momentum.optimize(g, eps=1e-8, lr=lr, max_iter=max_iter, tol=tol)
    adagrad_theta_values = adagrad.optimize(g, eps=1e-8, lr=lr, max_iter=max_iter, tol=tol)
    rms_prop_theta_values = rms_prop.optimize(g, eps=1e-8, lr=lr, max_iter=max_iter, tol=tol)
    adam_theta_values = adam.optimize(g, eps=1e-8, lr=lr, max_iter=max_iter, tol=tol)

    functions = {
        "Rosenbrock": [[], [], []]
    }

    for i in range(-60, 60):
        for j in range(-60, 60):
            functions['Rosenbrock'][0].append(i)
            functions['Rosenbrock'][1].append(j)
            functions['Rosenbrock'][2].append(rosenbrock((i / 10, j / 10)))

    xs = functions['Rosenbrock'][0]
    ys = functions['Rosenbrock'][1]
    zs = functions['Rosenbrock'][2]

    fig = plt.figure( figsize = (20,10))
    ax1 = fig.add_subplot(111, projection='3d', title = 'Rosenbrock')
    ax1.scatter(xs, ys, zs, marker='x', color = 'green', label = "data points")

    plt.show()
    fig = plt.figure(figsize=(20, 10))
    results = {
        "Rosenbrock": []
    }
    results['Rosenbrock'].append(gradient_descent_theta_values)
    Rosen_points = np.array(results['Rosenbrock'][0])
    #Rosen_energy = np.array(results['Rosenbrock'][1])
    #Rosen_temp = np.array(results['Rosenbrock'][2])

    #fig = plt.figure( figsize = (20,10))

    # Rosenbrock plot
    #ax1 = fig.add_subplot(111, projection='3d', title='Rosenbrock test')
    print(gradient_descent_theta_values)
    xs = gradient_descent_theta_values[0]
    ys = gradient_descent_theta_values[1]
    xv, yv = np.meshgrid(xs, ys)
    zs = rosenbrock((xv, yv))
    #zs = Rosen_energy
    ax1.scatter(xs, ys, zs, marker='o', color='green', label="data points")
    ax1.scatter(xs[0], ys[0], zs[0], color='red', marker='*', s=100, label="initial point")
    ax1.scatter(1., 1., 0., marker='^', s=100, color='black', label="global minimum")
    # plt.title('')
    plt.xlabel('iterations')
    plt.ylabel('theta values')
    plt.show()




    #plt.plot(gradient_descent_theta_values, color='black', label='gradient_descent')
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
