import numpy as np
import numpy.linalg as la

"""Implementeaza o serie de kernel-uri bine cunoscute: liniar, RBF
"""

def linear():
    """
    Intoarce o functie anonima, in doi parametrii, x si y, ce calculeaza produsul scalar a doi vectori x si y
    :return:
    """
    return lambda x, y: np.dot(x, y)


def radial_basis(gamma=10):
    """
    Intoarce o functie anonima, in doi parametrii, x si y, ce implementeaza forma de Radial Basis Function,
    avand parametrul \gamma
    :param gamma: parametrul de ponderare a normei diferentei vectorilor x si y
    :return:
    """
    return lambda x, y: np.exp( -np.linalg.norm(np.subtract(x, y)) * np.linalg.norm(np.subtract(x, y)) * gamma)
    pass
x = np.array([2, 3])
y = np.array([5, 6])
print(radial_basis()(x, y))