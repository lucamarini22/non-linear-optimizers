import math
import numpy as np

def func(x):
    return np.power(x, 2) - 4 * x + 4


def df(x):  # calculates the gradient
    return 2 * x - 4


# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    print(df(2))



