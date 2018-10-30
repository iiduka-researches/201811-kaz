#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import itertools
import numpy as np, scipy.optimize
from typing import Optional, Callable, Iterable
__all__ = ['fmin_kiwiel2001']


def _project(
        x: np.ndarray,
        A: np.ndarray=np.array([]),
        b: np.ndarray=np.array([]),
        bounds: Optional[scipy.optimize.Bounds] = None,
        tol: Optional[float]=None
) -> np.ndarray:
    c = x * 2.
    e = np.eye(x.size) * 2.

    def f(u: np.ndarray) -> float:
        return np.inner(u, u) - np.inner(x, u) * 2.

    def g(u: np.ndarray) -> np.ndarray:
        return u * 2. - c

    def h(u: np.ndarray) -> np.ndarray:
        return e

    result = scipy.optimize.minimize(
        f, x, method='trust-constr', jac=g, hess=h, bounds=bounds, tol=tol,
        constraints=scipy.optimize.LinearConstraint(A, -np.inf, b))

    return result.x


def fmin_kiwiel2001(
        func: Callable[[np.ndarray], float],
        x0: np.ndarray,
        grad: Callable[[np.ndarray], np.ndarray],
        A: np.ndarray=np.array([]),
        b: np.ndarray=np.array([]),
        lb: Optional[np.ndarray]=None,
        ub: Optional[np.ndarray]=None,
        t_seq: Optional[Iterable[float]]=None,
        maxiter: int=1000,
        injection: Optional[Callable[[int, np.ndarray], None]]=None
) -> np.ndarray:
    """minimizes a quasi-convex function subject to linear constraints using Algorithm (14) of K. C. Kiwiel: Convergence
    and efficiency of subgradient methods for quasiconvex minimization (2001), Mathematical Programming Series A 90 (pp.1-25).

    :param func:      the objective function to be minimized
    :param x0:        initial point
    :param grad:      a function which returns a nonzero subgradient of the objective function
    :param A:         a matrix defining the constraint, (numpy.inner(A, x) <= b).all()
    :param b:         an upper bounds on the constraint, (numpy.inner(A, x) <= b).all()
    :param lb:        lower bounds on independent variables
    :param ub:        upper bounds on independent variables
    :param t_seq:     a sequence of step-sizes
    :param maxiter:   maximum number of iterations
    :param injection: a procedure called at each iteration
    :return:          the acquired optimum
    """
    if t_seq is None:
        t_seq = map(lambda k: 1. / k, itertools.count(1))
    x = x0.copy()
    if lb is not None or ub is not None:
        x = np.clip(x, lb, ub)
    if lb is None:
        lb = -np.inf
    if ub is None:
        ub = np.inf
    bounds = scipy.optimize.Bounds(lb, ub)
    if not injection:
        def injection(*args):
            return True
    if injection(0, x) is False:
        return x
    for k, t in zip(range(maxiter), t_seq):
        g = grad(x)
        g *= t / np.linalg.norm(g)
        x -= g
        x = _project(x, A, b, bounds, tol=(t / 10.))
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

    fmin_kiwiel2001(func, 2 * np.ones(2), grad,
                    np.array([[-1., -1.]]), np.array([-1.]),
                    0., None, maxiter=10, injection=inject)
