#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
from unittest import TestCase
from qcopt.projections import *


class TestBox(TestCase):
    def test_behavior(self):
        p = box(np.array([-1., -2.]), np.array([3., 4.]))
        np.testing.assert_equal(
            p(np.array([0., 0.])), np.array([0., 0.]))
        np.testing.assert_equal(
            p(np.array([-1., -2.])), np.array([-1., -2.]))
        np.testing.assert_equal(
            p(np.array([3., 4.])), np.array([3., 4.]))
        np.testing.assert_equal(
            p(np.array([-3., -4.])), np.array([-1., -2.]))
        np.testing.assert_equal(
            p(np.array([5., 6.])), np.array([3., 4.]))

    def test_lb_only(self):
        p = box(lb=np.ones(2))
        np.testing.assert_equal(
            p(np.zeros(2)), np.ones(2))
        np.testing.assert_equal(
            p(np.ones(2)), np.ones(2))
        np.testing.assert_equal(
            p(np.full(2, 2)), np.full(2, 2))

    def test_ub_only(self):
        p = box(ub=np.ones(2))
        np.testing.assert_equal(
            p(np.zeros(2)), np.zeros(2))
        np.testing.assert_equal(
            p(np.ones(2)), np.ones(2))
        np.testing.assert_equal(
            p(np.full(2, 2)), np.ones(2))

    def test_scalar(self):
        p = box(0, 1)
        np.testing.assert_equal(
            p(np.full(10, -1)), np.full(10, 0))
        np.testing.assert_equal(
            p(np.full(10, 0)), np.full(10, 0))
        np.testing.assert_equal(
            p(np.full(10, 1)), np.full(10, 1))
        np.testing.assert_equal(
            p(np.full(10, 2)), np.full(10, 1))

    def test_reallocation(self):
        p = box(np.zeros(2), np.ones(2))
        x = np.array([0.5, 0.5])
        self.assertIsNot(p(x), x)


class TestHalfSpace(TestCase):
    def test_behavior(self):
        p = half_space(np.array([1., 2.]), 3.)
        np.testing.assert_array_equal(
            p(np.array([0., 0.])), np.array([0., 0.]))
        np.testing.assert_array_equal(
            p(np.array([0., 1.5])), np.array([0., 1.5]))
        np.testing.assert_array_equal(
            p(np.array([3., 0.])), np.array([3., 0.]))
        np.testing.assert_array_equal(
            p(np.array([3., 0.])), np.array([3., 0.]))
        np.testing.assert_array_almost_equal(
            p(np.array([4., 2.])), np.array([3., 0.]))

    def test_behavior_negative(self):
        p = half_space(np.array([-1., -1.]), -1.)
        np.testing.assert_array_almost_equal(
            p(np.array([0., 0.])), np.array([0.5, 0.5]))
        np.testing.assert_array_equal(
            p(np.array([0.5, 0.5])), np.array([0.5, 0.5]))
        np.testing.assert_array_equal(
            p(np.array([1., 1.])), np.array([1., 1.]))
        np.testing.assert_array_equal(
            p(np.array([0., 1.])), np.array([0., 1.]))

    def test_w_changed(self):
        w = np.array([1., 1.])
        p = half_space(w, 3.)
        w[0] = 3.
        np.testing.assert_equal(
            p(np.array([3., 0.])), np.array([3., 0.]))

    def test_reallocation(self):
        p = half_space(np.array([1., 1.]), 3.)
        x = np.array([1., 1.])
        self.assertIsNot(p(x), x)
        x = np.array([3., 3.])
        self.assertIsNot(p(x), x)


class TestBall(TestCase):
    def test_behavior(self):
        p = ball(np.array([2., -3.]), 1.)
        np.testing.assert_equal(
            p(np.array([2., -3.])), np.array([2., -3.]))
        np.testing.assert_equal(
            p(np.array([1., -3.])), np.array([1., -3.]))
        np.testing.assert_equal(
            p(np.array([3., -3.])), np.array([3., -3.]))
        np.testing.assert_equal(
            p(np.array([2., -4.])), np.array([2., -4.]))
        np.testing.assert_equal(
            p(np.array([2., -2.])), np.array([2., -2.]))
        np.testing.assert_almost_equal(
            p(np.array([2., -1.])), np.array([2., -2.]))
        np.testing.assert_almost_equal(
            p(np.array([5., -3.])), np.array([3., -3.]))
        np.testing.assert_almost_equal(
            p(np.array([3., -2.])), np.array([2., -3.]) + 2 ** -0.5)

    def test_c_changed(self):
        c = np.array([1., 1.])
        p = ball(c, 1.)
        c[0] = 3.
        np.testing.assert_equal(
            p(np.array([1., 1.])), np.array([1., 1.]))

    def test_reallocation(self):
        p = ball(np.array([2., -3.]), 1.)
        x = np.array([2., -3.])
        self.assertIsNot(p(x), x)
        x = np.array([3., -2.])
        self.assertIsNot(p(x), x)


class TestSumUp(TestCase):
    def test_behavior(self):
        def u(x: np.ndarray) -> np.ndarray:
            return x + 1.
        def v(x: np.ndarray) -> np.ndarray:
            return x + 2.
        w = sum_up(u, v)
        np.testing.assert_almost_equal(
            w(np.array([0., 0.])), np.array([3., 3.]))


class TestAverage(TestCase):
    def test_behavior(self):
        def u(x: np.ndarray) -> np.ndarray:
            return x + 1.
        def v(x: np.ndarray) -> np.ndarray:
            return x + 2.
        w = average(u, v)
        np.testing.assert_almost_equal(
            w(np.array([0., 0.])), np.array([1.5, 1.5]))


class TestCompose(TestCase):
    def test_behavior(self):
        def u(x: np.ndarray) -> np.ndarray:
            return x + 1.
        def v(x: np.ndarray) -> np.ndarray:
            return x + 2.
        w = compose(u, v)
        np.testing.assert_almost_equal(
            w(np.array([0., 0.])), np.array([3., 3.]))

    def test_order(self):
        def u(x: np.ndarray) -> np.ndarray:
            return x - 1.
        def v(x: np.ndarray) -> np.ndarray:
            return np.fmax(0., x)
        w = compose(v, u)
        np.testing.assert_equal(
            w(np.array([0., 0.])), np.array([0., 0.]))


class TestFirmUp(TestCase):
    def test_behavior(self):
        def u(x: np.ndarray) -> np.ndarray:
            return x / 2.
        v = firm_up(u)
        np.testing.assert_almost_equal(
            v(np.array([1., 2.])), np.array([0.75, 1.5]))

    def test_behavior_with_alpha(self):
        def u(x: np.ndarray) -> np.ndarray:
            return x / 2.
        v = firm_up(u, 0.25)
        np.testing.assert_array_almost_equal(
            v(np.array([1., 2.])), np.array([0.875, 1.75]))
