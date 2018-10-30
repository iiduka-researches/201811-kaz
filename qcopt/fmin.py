#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import itertools
import numpy as np
from qcopt.projections import *
from typing import Optional, Union, Callable, Iterable
__all__ = ['lcon2nonexp', 'fmin']


def lcon2nonexp(
        A: np.ndarray = np.array([]),
        b: np.ndarray = np.array([]),
        lb: Optional[Union[np.ndarray, float]] = None,
        ub: Optional[Union[np.ndarray, float]] = None,
        alpha: float=0.5,
) -> Callable[[np.ndarray], np.ndarray]:
    p = average(*[half_space(w, b) for w, b in zip(A, b)])
    if lb is not None or ub is not None:
        bounds = box(lb, ub)
        p = compose(bounds, p)
    p = firm_up(p, alpha)
    return p


def fmin(
        func: Callable[[np.ndarray], float],
        x0: np.ndarray,
        grad: Callable[[np.ndarray], np.ndarray],
        A: np.ndarray=np.array([]),
        b: np.ndarray=np.array([]),
        lb: Optional[Union[np.ndarray, float]]=None,
        ub: Optional[Union[np.ndarray, float]]=None,
        v_seq: Optional[Iterable[float]]=None,
        a_seq: Optional[Iterable[float]]=None,
        alpha: float=0.5,
        maxiter: int=1000,
        injection: Optional[Callable[[int, np.ndarray], None]]=None
) -> np.ndarray:
    """minimizes a quasi-convex function with linear constraints.

    :param func:      the objective function to be minimized
    :param x0:        initial point
    :param grad:      a function which returns a nonzero subgradient of the objective function
    :param A:         a matrix defining the constraint, (numpy.inner(A, x) <= b).all()
    :param b:         an upper bounds on the constraint, (numpy.inner(A, x) <= b).all()
    :param lb:        lower bounds on independent variables
    :param ub:        upper bounds on independent variables
    :param v_seq:     a sequence of step-sizes
    :param a_seq:     a sequence of Krasnosel'skii-Mann parameters
    :param alpha:     a parameter for firm_up
    :param maxiter:   maximum number of iterations
    :param injection: a procedure called at each iteration
    :return:          the acquired optimum
    """
    if v_seq is None:
        v_seq = map(lambda k: 1. / k, itertools.count(1))
    if a_seq is None:
        a_seq = itertools.repeat(0.5)
    x = x0.copy()
    p = lcon2nonexp(A, b, lb, ub, alpha)
    if not injection:
        def injection(*args):
            return True
    if injection(0, x) is False:
        return x
    for k, v, a in zip(range(maxiter), v_seq, a_seq):
        u = grad(x)
        u *= -v / np.linalg.norm(u)
        u += x
        u = p(u)
        x *= a
        u *= 1 - a
        x += u
        if injection(k + 1, x) is False:
            break
    return x


if __name__ == '__main__':
    def func(x: np.ndarray) -> float:
        return np.sum(np.abs(x))


    def grad(x: np.ndarray) -> np.ndarray:
        if np.count_nonzero(x) == 0:
            return np.random.rand()
        return np.sign(x)


    def inject(k: int, x: np.ndarray) -> None:
        print("%3d: %.8f at %s" % (k, func(x), x))


    fmin(func, 2 * np.ones(2), grad,
         np.array([[-1., -1.]]), np.array([-1.]),
         maxiter=10, injection=inject)
