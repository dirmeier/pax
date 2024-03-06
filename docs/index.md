# 👋 Welcome to Ramsey!

*Probabilistic deep learning using JAX*

Ramsey is a library for probabilistic modelling using [`JAX`](https://github.com/google/jax),
[`Flax`](https://github.com/google/flax) and [`NumPyro`](https://github.com/pyro-ppl/numpyro).
It offers high quality implementations of neural processes, Gaussian processes, Bayesian time series and state-space models, clustering processes,
and everything else Bayesian.

Ramsey makes use of

- [`Flax's`](https://github.com/google/flax)s module system for models with trainable parameters (such as neural or Gaussian processes),
- [`NumPyro`](https://github.com/pyro-ppl/numpyro) for models where parameters are endowed with prior distributions (such as Gaussian processes, Bayesian neural networks, etc.)

and is hence aimed at being fully compatible with both of them.

## Example usage

You can, for instance, construct a simple neural process like this:

``` py
from jax import random as jr

from ramsey import NP
from ramsey.nn import MLP
from ramsey.data import sample_from_sine_function

def get_neural_process():
    dim = 128
    np = NP(
        decoder=MLP([dim] * 3 + [2]),
        latent_encoder=(
            MLP([dim] * 3), MLP([dim, dim * 2])
        )
    )
    return np

key = jr.PRNGKey(23)
data = sample_from_sine_function(key)

neural_process = get_neural_process()
params = neural_process.init(
    key, x_context=data.x, y_context=data.y, x_target=data.x
)
```

The neural process takes a decoder and a set of two latent encoders as arguments. All of these are typically MLPs, but
Ramsey is flexible enough that you can change them, for instance, to CNNs or RNNs. Once the model is defined, you can initialize
its parameters just like in Flax.

## Why Ramsey

Just as the names of other probabilistic languages are inspired by researchers in the field
(e.g., Stan, Edward, Turing), Ramsey takes its name from one of my favourite philosophers/mathematicians,
[Frank Ramsey](https://plato.stanford.edu/entries/ramsey/).

## Installation

To install from PyPI, call:

```sh
pip install ramsey
```

To install the latest GitHub <RELEASE>, just call the following on the
command line:

```shell
pip install git+https://github.com/ramsey-devs/ramsey@<RELEASE>
```

See also the installation instructions for [`JAX`](https://github.com/google/jax), if
you plan to use Ramsey on GPU/TPU.

## Contributing

Contributions in the form of pull requests are more than welcome. A good way to start is to check out issues labelled
["good first issue"](https://github.com/ramsey-devs/ramsey/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22).

In order to contribute:

1. Clone Ramsey, and install it and its dev dependencies via `pip install -e '.[dev]'`,
2. create a new branch locally `git checkout -b feature/my-new-feature` or `git checkout -b issue/fixes-bug`,
3. implement your contribution,
4. test it by calling `tox` on the (Unix) command line,
5. submit a PR 🙂

## License

Ramsey is licensed under the Apache 2.0 License.
