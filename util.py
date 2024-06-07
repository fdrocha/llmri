import os
import random

import matplotlib.pyplot as plt
import pandas as pd
import torch

import util


def decode_line(line):
    return line.replace("\\n", "\n")


def encode_line(line):
    return line.replace("\n", "\\n")


class ActivationLoader:

    def __init__(self, path):
        self.path = path
        self.input_fn = os.path.join(self.path, "inputs.txt")
        self.activation_fn_template = os.path.join(self.path,
                                                   "activations%06d.pt")
        self.reload()

    def reload(self):
        self.inputs = [
            util.decode_line(line.strip())
            for line in open(self.input_fn, "r").readlines()
        ]
        self.n_sentences = len(self.inputs)

    def get_sentence_activations(self, i):
        if i == "random":
            i = self.get_random_ind()
        acts = torch.load(self.activation_fn_template % i).to(
            device="cpu", dtype=torch.float32)
        attn = acts[::2]
        mlp = acts[1::2]
        t = torch.stack((attn, mlp))
        return self.inputs[i], t

    def get_random_ind(self):
        return random.randint(0, self.n_sentences - 1)


def tranks(tensor):
    # Flatten and sort
    flattened, indices = tensor.flatten().sort()

    # Create ranks tensor (long for indexing)
    ranks = torch.zeros_like(flattened, dtype=torch.long)
    ranks[indices] = torch.arange(len(flattened))

    # Convert to float for normalization
    ranks = ranks.float()

    # Normalize to [0, 1]
    ranks = ranks / (len(flattened) - 1)

    # Reshape to original shape
    return ranks.reshape(tensor.shape)


def batched_covariance(x):
    """Takes in a matrix of shape (N, L, K) and returns L, mean (with shape (N,K)) and cov (with shape (N, K, K))"""
    L = x.shape[1]
    xmean = torch.mean(x, dim=1)
    xdemeaned = x - xmean.unsqueeze(1)
    xcov = torch.matmul(xdemeaned.mT, xdemeaned) / (L - 1)
    return L, xmean, xcov


def sample_from_tensor(t, nsamples):
    tf = t.flatten()
    inds = torch.randint(0, tf.shape[0], (nsamples, ))
    return tf[inds]


def imshow(img, title=None, **kv):
    fig, ax = plt.subplots()
    if title:
        ax.set_title(title)
    im = ax.imshow(img, cmap="jet", aspect="auto", **kv)
    fig.colorbar(im, ax=ax)


def tshow(acts, title=None, trank=True, color_bar=False, **kv):
    fig, axs = plt.subplots(1, 2, sharey=True, figsize=(24, 8))
    if title:
        fig.suptitle(title, fontsize=16)
    if trank:
        im0 = axs[0].imshow(tranks(acts[0]), cmap="jet", aspect="auto", **kv)
        im1 = axs[1].imshow(tranks(acts[1]), cmap="jet", aspect="auto", **kv)
    else:
        im0 = axs[0].imshow(acts[0], cmap="jet", aspect="auto", **kv)
        im1 = axs[1].imshow(acts[1], cmap="jet", aspect="auto", **kv)
    axs[0].set_title("Self-Attention")
    axs[1].set_title("MLP")
    if color_bar:
        fig.colorbar(im1, ax=axs[1])
    return fig


def rescale_absmean(t):
    return t / t.abs().mean()


def rescale_interquartile(t):
    return t / (t.quantile(0.75) - t.quantile(0.25))


def cat_acts(loader, inds):
    ts = []
    for ind in inds:
        _, acts = loader.get_sentence_activations(ind)
        ts.append(acts)
    return torch.cat(ts, dim=2)


def arrays_to_labelled_df(data, var_name="x"):
    return pd.concat([
        pd.DataFrame({
            var_name: xs
        }).assign(label=label) for (label, xs) in data.items()
    ])


def interquart(xs, dim=None):
    return xs.quantile(0.75, dim=dim) - xs.quantile(0.25, dim=dim)
