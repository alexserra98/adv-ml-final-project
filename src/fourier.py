import torch as t
from torch import Tensor
import torch.nn.functional as F
import numpy as np

from pathlib import Path
import os
import sys

import plotly.express as px
import plotly.graph_objects as go

from functools import *
import gdown
from typing import List, Tuple, Union, Optional
from fancy_einsum import einsum
import einops
from jaxtyping import Float, Int
from tqdm import tqdm

from transformer_lens import utils, ActivationCache, HookedTransformer, HookedTransformerConfig
from transformer_lens.hook_points import HookPoint
from transformer_lens.components import LayerNorm
from my_utils import *

device = t.device("cuda" if t.cuda.is_available() else "cpu")

t.set_grad_enabled(False)

def make_fourier_basis(p: int) -> Tuple[Tensor, List[str]]:
    '''
    Returns a pair `fourier_basis, fourier_basis_names`, where `fourier_basis` is
    a `(p, p)` tensor whose rows are Fourier components and `fourier_basis_names`
    is a list of length `p` containing the names of the Fourier components (e.g.
    `["const", "cos 1", "sin 1", ...]`). You may assume that `p` is odd.
    '''
    # SOLUTION
    # Define a grid for the Fourier basis vecs (we'll normalize them all at the end)
    # Note, the first vector is just the constant wave
    fourier_basis = t.ones(p, p)
    fourier_basis_names = ['Const']
    for i in range(1, p // 2 + 1):
        # Define each of the cos and sin terms
        fourier_basis[2*i-1] = t.cos(2*t.pi*t.arange(p)*i/p)
        fourier_basis[2*i] = t.sin(2*t.pi*t.arange(p)*i/p)
        fourier_basis_names.extend([f'cos {i}', f'sin {i}'])
    # Normalize vectors, and return them
    fourier_basis /= fourier_basis.norm(dim=1, keepdim=True)
    return fourier_basis.to(device), fourier_basis_names

def fft1d(x: t.Tensorm, fourier_basis : t.Tensor) -> t.Tensor:
    '''
    Returns the 1D Fourier transform of `x`,
    which can be a vector or a batch of vectors.

    x.shape = (..., p)
    '''
    # SOLUTION
    return x @ fourier_basis.T

def fourier_2d_basis_term(i: int, j: int, fourier_basis: t.Tensor) -> Float[Tensor, "p p"]:
    '''
    Returns the 2D Fourier basis term corresponding to the outer product of the
    `i`-th component of the 1D Fourier basis in the `x` direction and the `j`-th
    component of the 1D Fourier basis in the `y` direction.

    Returns a 2D tensor of length `(p, p)`.
    '''
    # SOLUTION
    return (fourier_basis[i][:, None] * fourier_basis[j][None, :])


def fft2d(tensor: t.Tensor, fourier_basis: t.Tensor) -> t.Tensor:
    '''
    Retuns the components of `tensor` in the 2D Fourier basis.

    Asumes that the input has shape `(p, p, ...)`, where the
    last dimensions (if present) are the batch dims.
    Output has the same shape as the input.
    '''
    # SOLUTION
    # fourier_basis[i] is the i-th basis vector, which we want to multiply along
    return einops.einsum(
        tensor, fourier_basis, fourier_basis,
        "px py ..., i px, j py -> i j ..."
    )
    
def fft1d_given_dim(tensor: t.Tensor, dim: int) -> t.Tensor:
    '''
    Performs 1D FFT along the given dimension (not necessarily the last one).
    '''
    return fft1d(tensor.transpose(dim, -1)).transpose(dim, -1)



