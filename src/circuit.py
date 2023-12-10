import torch as t
from torch import Tensor
import torch.nn.functional as F
import numpy as np

from pathlib import Path
import os
import sys


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

def arrange_by_2d_freqs(tensor):
    '''
    Takes a tensor of shape (p, p, ...) and returns a tensor of shape
    (p//2 - 1, 3, 3, ...) representing the Fourier coefficients sorted by
    frequency (each slice contains const, linear and quadratic terms).

    In other words, if the first two dimensions of the original tensor
    correspond to indexing by 2D Fourier frequencies as follows:

        1           cos(w_1*x)            sin(w_1*x)           ...
        cos(w_1*y)  cos(w_1*x)cos(w_1*y)  sin(w_1*x)cos(w_1*y) ...
        sin(w_1*y)  cos(w_1*x)sin(w_1*y)  sin(w_1*x)sin(w_1*y) ...
        cos(w_2*y)  cos(w_1*x)cos(w_2*y)  sin(w_1*x)cos(w_2*y) ...
        ...

    Then the (k-1)-th slice of the new tensor are the terms corresponding to
    the following 2D Fourier frequencies:

        1           cos(w_k*x)            sin(w_k*x)           ...
        cos(w_k*y)  cos(w_k*x)cos(w_k*y)  sin(w_k*x)cos(w_k*y) ...
        sin(w_k*y)  cos(w_k*x)sin(w_k*y)  sin(w_k*x)sin(w_k*y) ...

    for k = 1, 2, ..., p//2.

    Note we omit the constant term, i.e. the 0th slice has frequency k=1.
    '''
    idx_2d_y_all = []
    idx_2d_x_all = []
    for freq in range(1, p//2):
        idx_1d = [0, 2*freq-1, 2*freq]
        idx_2d_x_all.append([idx_1d for _ in range(3)])
        idx_2d_y_all.append([[i]*3 for i in idx_1d])
    return tensor[idx_2d_y_all, idx_2d_x_all]


def find_neuron_freqs(
    fourier_neuron_acts: Float[Tensor, "p p d_mlp"]
) -> Tuple[Float[Tensor, "d_mlp"], Float[Tensor, "d_mlp"]]:
    '''
    Returns the tensors `neuron_freqs` and `neuron_frac_explained`,
    containing the frequencies that explain the most variance of each
    neuron and the fraction of variance explained, respectively.
    '''
    fourier_neuron_acts_by_freq = arrange_by_2d_freqs(fourier_neuron_acts)
    assert fourier_neuron_acts_by_freq.shape == (p//2-1, 3, 3, d_mlp)

    # SOLUTION
    # Sum squares of all frequency coeffs, for each neuron
    square_of_all_terms = einops.reduce(
        fourier_neuron_acts.pow(2),
        "x_coeff y_coeff neuron -> neuron",
        "sum"
    )

    # Sum squares just corresponding to const+linear+quadratic terms,
    # for each frequency, for each neuron
    square_of_each_freq = einops.reduce(
        fourier_neuron_acts_by_freq.pow(2),
        "freq x_coeff y_coeff neuron -> freq neuron",
        "sum"
    )

    # Find the freq explaining most variance for each neuron
    # (and the fraction of variance explained)
    neuron_variance_explained, neuron_freqs = square_of_each_freq.max(0)
    neuron_frac_explained = neuron_variance_explained / square_of_all_terms

    # The actual frequencies count up from k=1, not 0!
    neuron_freqs += 1

    return neuron_freqs, neuron_frac_explained


def project_onto_direction(batch_vecs: t.Tensor, v: t.Tensor) -> t.Tensor:
    '''
    Returns the component of each vector in `batch_vecs` in the direction of `v`.

    batch_vecs.shape = (n, ...)
    v.shape = (n,)
    '''

    # Get tensor of components of each vector in v-direction
    components_in_v_dir = einops.einsum(
        batch_vecs, v,
        "n ..., n -> ..."
    )

    # Use these components as coefficients of v in our projections
    return einops.einsum(
        components_in_v_dir, v,
        "..., n -> n ..."
    )
    
def project_onto_frequency(batch_vecs: t.Tensor, freq: int, fourier_2d_basis_term: t.Tensor) -> t.Tensor:
    '''
    Returns the projection of each vector in `batch_vecs` onto the
    2D Fourier basis directions corresponding to frequency `freq`.

    batch_vecs.shape = (p**2, ...)
    '''
    assert batch_vecs.shape[0] == p**2
    # SOLUTION

    return sum([
        project_onto_direction(
            batch_vecs,
            fourier_2d_basis_term(i, j).flatten(),
        )
        for i in [0, 2*freq-1, 2*freq] for j in [0, 2*freq-1, 2*freq]
    ])
    
    
def get_trig_sum_directions(k: int, fourier_2d_basis_term: t.Tensor) -> Tuple[Float[Tensor, "p p"], Float[Tensor, "p p"]]:
    '''
    Given frequency k, returns the normalized vectors in the 2D Fourier basis
    representing the directions:

        cos(ω_k * (x + y))
        sin(ω_k * (x + y))

    respectively.
    '''
    # SOLUTION
    cosx_cosy_direction = fourier_2d_basis_term(2*k-1, 2*k-1)
    sinx_siny_direction = fourier_2d_basis_term(2*k, 2*k)
    sinx_cosy_direction = fourier_2d_basis_term(2*k, 2*k-1)
    cosx_siny_direction = fourier_2d_basis_term(2*k-1, 2*k)

    cos_xplusy_direction = (cosx_cosy_direction - sinx_siny_direction) / np.sqrt(2)
    sin_xplusy_direction = (sinx_cosy_direction + cosx_siny_direction) / np.sqrt(2)

    return cos_xplusy_direction, sin_xplusy_direction