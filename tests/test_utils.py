#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import time
from qcopt.utils import *
from unittest import TestCase


class TestStopwatch(TestCase):
    @staticmethod
    def _sleep(secs: float) -> None:
        b = time.perf_counter()
        while time.perf_counter() < b + secs:
            pass

    def test_behavior(self):
        sw = Stopwatch()
        self.assertFalse(sw.running)
        self.assertEqual(sw.elapsed, 0.)
        sw.start()
        self._sleep(0.1)
        self.assertAlmostEqual(sw.stop(), 0.1, delta=1e-3)
        self.assertAlmostEqual(sw.elapsed, 0.1, delta=1e-3)
        self._sleep(0.15)
        sw.start()
        self._sleep(0.05)
        self.assertAlmostEqual(sw.record(), 0.15, delta=1e-3)
        self.assertAlmostEqual(sw.elapsed, 0.15, delta=1e-3)
        self.assertTrue(sw.running)
        self._sleep(0.1)
        self.assertAlmostEqual(sw.record(), 0.25, delta=1e-3)
        self.assertAlmostEqual(sw.elapsed, 0.25, delta=1e-3)
        self._sleep(0.05)
        self.assertAlmostEqual(sw.peek(), 0.3, delta=1e-3)
        self.assertAlmostEqual(sw.elapsed, 0.25, delta=1e-3)
        sw.reset()
        self.assertFalse(sw.running)
        self.assertEqual(sw.elapsed, 0.)
