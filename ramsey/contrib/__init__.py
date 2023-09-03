from ramsey._src.contrib.gaussian_process.gaussian_process import GP
from ramsey._src.contrib.gaussian_process.sparse_gaussian_process import \
    SparseGP
from ramsey._src.contrib.gaussian_process.train_gaussian_process import \
    train_gaussian_process, train_sparse_gaussian_process
from ramsey._src.contrib.gaussian_process.kernel.non_stationary import (
    Linear,
    linear
)
from ramsey._src.contrib.gaussian_process.kernel.stationary import (
    ExponentiatedQuadratic,
    Periodic,
    exponentiated_quadratic,
    periodic
)

SquaredExponential = ExponentiatedQuadratic

__all__ = [
    "ExponentiatedQuadratic",
    "GP",
    "Linear",
    "Periodic",
    "SparseGP",
    "SquaredExponential",
    "exponentiated_quadratic",
    "linear",
    "periodic",
    "train_gaussian_process",
    "train_sparse_gaussian_process",
]

