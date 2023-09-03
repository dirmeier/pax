# pylint: skip-file

import pytest

from ramsey import ANP, DANP, MLP, NP, MultiHeadAttention


def __lnp():
    np = NP(
        decoder=MLP([3, 2]),
        latent_encoder=(MLP([3, 3]), MLP([3, 6])),
    )
    return np


def __np():
    np = NP(
        decoder=MLP([3, 2]),
        deterministic_encoder=MLP([4, 4]),
        latent_encoder=(MLP([3, 3]), MLP([3, 6])),
    )
    return np


def __anp():
    np = ANP(
        decoder=MLP([3, 2]),
        deterministic_encoder=(
            MLP([4, 4]),
            MultiHeadAttention(8, 8, MLP([8, 8])),
        ),
        latent_encoder=(MLP([3, 3]), MLP([3, 6])),
    )
    return np


def __danp():
    np = DANP(
        decoder=MLP([3, 2]),
        deterministic_encoder=(
            MLP([4, 4]),
            MultiHeadAttention(8, 8, MLP([8, 8])),
            MultiHeadAttention(8, 8, MLP([8, 8])),
        ),
        latent_encoder=(
            MLP([3, 3]),
            MultiHeadAttention(8, 8, MLP([8, 8])),
            MLP([3, 6]),
        ),
    )
    return np


@pytest.fixture(params=[__lnp, __np, __anp, __danp])
def module(request):
    yield request.param
