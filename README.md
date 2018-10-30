# An implementation of the fixed point quasiconvex subgradient method
This repository provides an implementation of the fixed point quasiconvex subgradient method.
This implementation is based on the treatise,

> K. Hishinuma, H. Iiduka: Fixed Point Quasiconvex Subgradient Method, (submitted)

and is provided officially by the authors of the paper.
All numerical examples presented in the paper are used this implementation.


## Requirements
  * [Python 3.6.6](https://www.python.org) or later
  * [NumPy 1.15.0](http://www.numpy.org)
  * [SciPy 1.1.0](https://www.scipy.org)


## Usage
```python
from qcopt import fmin

fmin(
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
    maxiter: int=1000
) -> np.ndarray
```
  * `func` is an objective function to be minimized.
  * `x0` is an initial point.
  * `grad` is a method for taking a nonzero subgradient of the function `func` at given point.
  * `A` is a NumPy matrix which expresses the constraint such that `np.dot(A, x) <= b`.
  * `b` is a NumPy vector which expresses the constraint such that `np.dot(A, x) <= b`.
  * `lb` is a lower bound on variables.
  * `ub` is a upper bound on variables.
  * `v_seq` is a sequence of stepsizes.
  * `a_seq` is a sequence of coefficients of the Krasnosel'skii-Mann iterator.
  * `alpha` is a parameter for the `firm_up` method.
  * `maxiter` is a maximum number of iterations.


# License
MIT License

Copyright (c) 2018 Kazuhiro HISHINUMA and Hideaki IIDUKA

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.


# Authors
  * [Kazuhiro HISHINUMA](https://arnip.org)
  * [Hideaki IIDUKA](https://iiduka.net)
