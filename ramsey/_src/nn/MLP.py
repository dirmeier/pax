import dataclasses
from collections.abc import Callable, Iterable

import jax
from flax import nnx
from flax.linen import initializers
from flax.linen.linear import default_kernel_init
from jax import Array


class MLP(nnx.Module):
    """A multi-layer perceptron.

    Attributes
    ----------
    output_sizes: Iterable[int]
        number of hidden nodes per layer
    dropout: Optional[float]
        dropout rate to apply after each hidden layer
    kernel_init: initializers.Initializer
        initializer for weights of hidden layers
    bias_init: initializers.Initializer
        initializer for bias of hidden layers
    use_bias: bool
        boolean if hidden layers should use bias nodes
    activation: Callable
        activation function to apply after each hidden layer. Default is relu.
    activate_final: bool
        if true, activate last layer
    """

    def __init__(
            self,
            output_sizes: Iterable[int],
            *,
            dropout: float | None = None,
            kernel_init: initializers.Initializer = default_kernel_init,
            bias_init: initializers.Initializer = initializers.zeros_init(),
            use_bias: bool = True,
            activation: Callable = jax.nn.relu,
            activate_final: bool = False,
            rngs: nnx.Rngs,
    ):
        self.activation = activation
        self.activate_final = activate_final
        self.dropout = dropout

        layers = []
        for din, dout in zip(output_sizes[:1], output_sizes[1:]):
            layers.append(
                nnx.Linear(
                    in_features=din,
                    out_features=dout,
                    kernel_init=kernel_init,
                    bias_init=bias_init,
                    use_bias=use_bias,
                    rngs=rngs
                )
            )
        self.layers = tuple(layers)
        if dropout is not None:
            self.dropout_layer = nnx.Dropout(dropout, rngs=rngs)

    # pylint: disable=too-many-function-args
    def __call__(self, inputs: Array):
        """Transform the inputs through the MLP.

        Parameters
        ----------
        inputs: Array
            input data of dimension (*batch_dims, spatial_dims..., feature_dims)
        is_training: boolean
            if true, uses training mode (i.e., dropout)

        Returns
        -------
        Array
            returns the transformed inputs
        """
        num_layers = len(self.layers)
        out = inputs
        for i, layer in enumerate(self.layers):
            out = layer(out)
            if i < (num_layers - 1) or self.activate_final:
                if self.dropout is not None:
                    out = self.dropout_layer(out)
                out = self.activation(out)
        return out
