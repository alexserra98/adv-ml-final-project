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
from typing import List, Tuple, Union, Optional,Dict
from fancy_einsum import einsum
import einops
from jaxtyping import Float, Int
from tqdm import tqdm

from transformer_lens import utils, ActivationCache, HookedTransformer, HookedTransformerConfig
from transformer_lens.hook_points import HookPoint
from transformer_lens.components import LayerNorm
from my_utils import *
from circuit import *

def get_metrics(model: HookedTransformer, metric_cache, metric_fn, name, reset=False, full_run_data=Dict):
    '''
    Define a metric (by metric_fn) and add it to the cache, with the name `name`.

    If `reset` is True, then the metric will be recomputed, even if it is already in the cache.
    '''
    if reset or (name not in metric_cache) or (len(metric_cache[name])==0):
        metric_cache[name]=[]
        for c, sd in enumerate(tqdm((full_run_data['state_dicts']))):
            model = load_in_state_dict(model, sd)
            out = metric_fn(model)
            if type(out)==t.Tensor:
                out = utils.to_numpy(out)
            metric_cache[name].append(out)
        model = load_in_state_dict(model, full_run_data['state_dicts'][400])
        try:
            metric_cache[name] = t.tensor(metric_cache[name])
        except:
            metric_cache[name] = t.tensor(np.array(metric_cache[name]))


def test_loss(model):
    logits = model(all_data)[:, -1, :-1]
    return test_logits(logits, False, mode='test')


def train_loss(model):
    logits = model(all_data)[:, -1, :-1]
    return test_logits(logits, False, mode='train')

def excl_loss(model: HookedTransformer, key_freqs: list) -> list:
    '''
    Returns the excluded loss (i.e. subtracting the components of logits corresponding to
    cos(w_k(x+y)) and sin(w_k(x+y)), for each frequency k in key_freqs.
    '''
    excl_loss_list = []
    logits = model(all_data)[:, -1, :-1]
    # SOLUTION

    for freq in key_freqs:
        cos_xplusy_direction, sin_xplusy_direction = get_trig_sum_directions(freq)

        logits_cos_xplusy = project_onto_direction(
            logits,
            cos_xplusy_direction.flatten()
        )

        logits_sin_xplusy = project_onto_direction(
            logits,
            sin_xplusy_direction.flatten()
        )

        logits_excl = logits - logits_cos_xplusy - logits_sin_xplusy

        loss = test_logits(logits_excl, bias_correction=False, mode='train').item()

        excl_loss_list.append(loss)

    return excl_loss_list

def fourier_embed(model: HookedTransformer):
    '''
    Returns norm of Fourier transform of the model's embedding matrix.
    '''
    # SOLUTION
    W_E_fourier = fourier_basis.T @ model.W_E[:-1]
    return einops.reduce(W_E_fourier.pow(2), 'vocab d_model -> vocab', 'sum')

def embed_SVD(model: HookedTransformer) -> t.Tensor:
    '''
    Returns vector S, where W_E = U @ diag(S) @ V.T in singular value decomp.
    '''
    # SOLUTION
    U, S, V = t.svd(model.W_E[:, :-1])
    return S

def tensor_trig_ratio(model: HookedTransformer, mode: str, key_freqs: list):
    '''
    Returns the fraction of variance of the (centered) activations which
    is explained by the Fourier directions corresponding to cos(ω(x+y))
    and sin(ω(x+y)) for all the key frequencies.
    '''
    logits, cache = model.run_with_cache(all_data)
    logits = logits[:, -1, :-1]
    if mode == "neuron_pre":
        tensor = cache['pre', 0][:, -1]
    elif mode == "neuron_post":
        tensor = cache['post', 0][:, -1]
    elif mode == "logit":
        tensor = logits
    else:
        raise ValueError(f"{mode} is not a valid mode")

    tensor_centered = tensor - einops.reduce(tensor, 'xy index -> 1 index', 'mean')
    tensor_var = einops.reduce(tensor_centered.pow(2), 'xy index -> index', 'sum')
    tensor_trig_vars = []

    for freq in key_freqs:
        cos_xplusy_direction, sin_xplusy_direction = get_trig_sum_directions(freq)
        cos_xplusy_projection_var = project_onto_direction(
            tensor_centered, cos_xplusy_direction.flatten()
        ).pow(2).sum(0)
        sin_xplusy_projection_var = project_onto_direction(
            tensor_centered, sin_xplusy_direction.flatten()
        ).pow(2).sum(0)

        tensor_trig_vars.extend([cos_xplusy_projection_var, sin_xplusy_projection_var])

    return utils.to_numpy(sum(tensor_trig_vars)/tensor_var)

def get_frac_explained(model: HookedTransformer, key_freqs_plus: list, neuron_freqs: list):
    _, cache = model.run_with_cache(all_data, return_type=None)

    returns = []

    for neuron_type in ['pre', 'post']:
        neuron_acts = cache[neuron_type, 0][:, -1].clone().detach()
        neuron_acts_centered = neuron_acts - neuron_acts.mean(0)
        neuron_acts_fourier = fft2d(
            einops.rearrange(neuron_acts_centered, "(x y) neuron -> x y neuron", x=p)
        )

        # Calculate the sum of squares over all inputs, for each neuron
        square_of_all_terms = einops.reduce(
            neuron_acts_fourier.pow(2), "x y neuron -> neuron", "sum"
        )

        frac_explained = t.zeros(d_mlp).to(device)
        frac_explained_quadratic_terms = t.zeros(d_mlp).to(device)

        for freq in key_freqs_plus:
            # Get Fourier activations for neurons in this frequency cluster
            # We arrange by frequency (i.e. each freq has a 3x3 grid with const, linear & quadratic terms)
            acts_fourier = arrange_by_2d_freqs(neuron_acts_fourier[..., neuron_freqs==freq])

            # Calculate the sum of squares over all inputs, after filtering for just this frequency
            # Also calculate the sum of squares for just the quadratic terms in this frequency
            if freq==-1:
                squares_for_this_freq = squares_for_this_freq_quadratic_terms = einops.reduce(
                    acts_fourier[:, 1:, 1:].pow(2), "freq x y neuron -> neuron", "sum"
                )
            else:
                squares_for_this_freq = einops.reduce(
                    acts_fourier[freq-1].pow(2), "x y neuron -> neuron", "sum"
                )
                squares_for_this_freq_quadratic_terms = einops.reduce(
                    acts_fourier[freq-1, 1:, 1:].pow(2), "x y neuron -> neuron", "sum"
                )

            frac_explained[neuron_freqs==freq] = squares_for_this_freq / square_of_all_terms[neuron_freqs==freq]
            frac_explained_quadratic_terms[neuron_freqs==freq] = squares_for_this_freq_quadratic_terms / square_of_all_terms[neuron_freqs==freq]

        returns.extend([frac_explained, frac_explained_quadratic_terms])

    frac_active = (neuron_acts > 0).float().mean(0)

    return t.nan_to_num(t.stack(returns + [neuron_freqs, frac_active], axis=0))

def avg_attn_pattern(model: HookedTransformer):
    _, cache = model.run_with_cache(all_data, return_type=None)
    return utils.to_numpy(einops.reduce(
        cache['pattern', 0][:, :, 2],
        'batch head pos -> head pos', 'mean')
    )
    
def trig_loss(model: HookedTransformer,  key_freqs: list, mode: str = 'all'):
    logits = model(all_data)[:, -1, :-1]

    trig_logits = []
    for freq in key_freqs:
        cos_xplusy_dir, sin_xplusy_dir = get_trig_sum_directions(freq)
        cos_xplusy_proj = project_onto_direction(logits, cos_xplusy_dir.flatten())
        sin_xplusy_proj = project_onto_direction(logits, sin_xplusy_dir.flatten())
        trig_logits.extend([cos_xplusy_proj, sin_xplusy_proj])
    trig_logits = sum(trig_logits)

    return test_logits(
        trig_logits, bias_correction=True, original_logits=logits, mode=mode
    )

def sum_sq_weights(model):
    return [param.pow(2).sum().item() for name, param in model.named_parameters()]