#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
from typing import Optional, Union, Tuple, Callable
__all__ = ['box', 'half_space', 'ball', 'sum_up', 'average', 'compose', 'firm_up']


def box(lb: Optional[Union[np.ndarray, float]]=None, ub: Optional[Union[np.ndarray, float]]=None) -> Callable[[np.ndarray], np.ndarray]:
    """makes a metric projection onto box defined by its lower bound lb and upper bound ub.
    """

    def _t(x: np.ndarray) -> np.ndarray:
        return np.clip(x, lb, ub)

    return _t


def half_space(w: np.ndarray, d: float) -> Callable[[np.ndarray], np.ndarray]:
    """makes a metric projection onto halfspace defined by numpy.inner(w, x) <= d.
    """
    l = np.linalg.norm(w)
    assert l != 0, 'Parameter w have to be a nonzero vector.'
    d = d / l
    w = w / l

    def _t(x: np.ndarray) -> np.ndarray:
        det = d - np.inner(w, x)
        if det >= 0:
            y = x.copy()
        else:
            y = det * w
            y += x
        return y

    return _t


def ball(c: np.ndarray, r: float) -> Callable[[np.ndarray], np.ndarray]:
    """makes a metric projection onto the closed ball with center c and radius r.
    """
    c = c.copy()
    assert r >= 0, 'Parameter r must be a positive real.'

    def _t(x: np.ndarray) -> np.ndarray:
        v = x - c
        d = np.linalg.norm(v)
        if d <= r:
            y = x.copy()
        else:
            y = c.copy()
            v *= r / d
            y += v
        return y

    return _t

def sum_up(*args: Tuple[Callable[[np.ndarray], np.ndarray]]) -> Callable[[np.ndarray], np.ndarray]:
    """creates the sum of given mappings.
    """

    def _t(x: np.ndarray) -> np.ndarray:
        y = np.zeros_like(x)
        for t in args:
            y += t(x)
        return y

    return _t


def average(*args: Tuple[Callable[[np.ndarray], np.ndarray]]) -> Callable[[np.ndarray], np.ndarray]:
    """makes a mapping which calculates the barycenter of all points transformed by given mappings.
    """
    assert len(args) > 0, 'Function average requires at least one mapping as its parameter.'

    def _t(x: np.ndarray) -> np.ndarray:
        y = np.zeros_like(x)
        for t in args:
            y += t(x)
        y /= len(args)
        return y

    return _t


def compose(*args: Tuple[Callable[[np.ndarray], np.ndarray]]) -> Callable[[np.ndarray], np.ndarray]:
    """composes given mappings
    """
    assert len(args) > 0, 'Function compose requires at least one mapping as its parameter.'

    def _t(x: np.ndarray) -> np.ndarray:
        y = x
        for t in reversed(args):
            y = t(y)
        return y

    return _t


def firm_up(t: Callable[[np.ndarray], np.ndarray], alpha: float=0.5) -> Callable[[np.ndarray], np.ndarray]:
    """converts given nonexpansive mapping into a firmly nonexpansive mapping whose fixed point coincides with the given one.
    This implementation is based on the property of the firm nonexpansiveness presented in Remark 4.37 of H. H. Bauschke,
    P. L. Combettes: Convex Analysis and Monotone Operator Theory in Hilbert Spaces (2017), Springer International Publishing.
    """
    assert 0 < alpha <= 0.5, 'The value of alpha must be between 0 and 0.5.'

    def _t(x: np.ndarray) -> np.ndarray:
        y = t(x)
        assert y is not x, 'The nonexpansive mapping t did not create a new instance as its return value.'
        y *= alpha
        y += (1. - alpha) * x
        return y

    return _t
