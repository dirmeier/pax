"""
ramsey: probabilistic modelling using JAX
"""


from ramsey._src.neural_process.attentive_neural_process import ANP
from ramsey._src.neural_process.doubly_attentive_neural_process import DANP
from ramsey._src.neural_process.neural_process import NP


__version__ = "0.1.0"

__all__ = ["NP", "ANP", "DANP"]
