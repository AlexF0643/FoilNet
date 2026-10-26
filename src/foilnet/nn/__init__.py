"""Neural-network surrogates."""

from foilnet.nn.numpy_mlp import Adam, NumpyMLP, fit, gradient_check

__all__ = ["Adam", "NumpyMLP", "fit", "gradient_check"]
