#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from time import process_time
__all__ = ['Stopwatch']

class Stopwatch(object):
    def __init__(self):
        self.elapsed = 0.
        self.running = False

    def reset(self) -> None:
        self.elapsed = 0.
        self.running = False

    def start(self) -> None:
        if self.running:
            return
        self.running = process_time()

    def peek(self) -> float:
        if self.running:
            return self.elapsed + (process_time() - self.running)
        return self.elapsed

    def record(self) -> float:
        if self.running:
            now = process_time()
            self.elapsed += now - self.running
            self.running = now
        return self.elapsed

    def stop(self) -> float:
        if self.running:
            self.elapsed += process_time() - self.running
            self.running = False
        return self.elapsed
