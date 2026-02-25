"""
An adaptation of Andrej Karpathy's nanoGPT implementation in PyTorch.
Original source: https://github.com/karpathy/nanoGPT

Original License:
MIT License

Copyright (c) 2022 Andrej Karpathy

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

Original comments:
Full definition of a GPT Language Model, all of it in this single file.
References:
1) the official GPT-2 TensorFlow implementation released by OpenAI:
https://github.com/openai/gpt-2/blob/master/src/model.py
2) huggingface/transformers PyTorch implementation:
https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py
"""

import os
import sys

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

import pickle as pkl
import mprl.util as util


# @torch.jit.script # good to enable when not using torch.compile, disable when using (our default)
def new_gelu(x):
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT).
    Reference: Gaussian Error Linear Units (GELU) paper: https://arxiv.org/abs/1606.08415
    """
    return (
        0.5
        * x
        * (
            1.0
            + torch.tanh(
                math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))
            )
        )
    )


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        # causal mask to ensure that attention in only applied to the left in the input sequence
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(config.block_size, config.block_size)).view(
                1, 1, config.block_size, config.block_size
            ),
        )
        self.dropout = config.dropout
        # flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0
        self.flash = hasattr(torch.nn.functional, "scaled_dot_product_attention")
        self.n_head = config.n_head
        self.n_embd = config.n_embd

    def forward(self, x):
        (B, T, C) = x.size()    # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)     # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)     # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)     # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True)
        else:
            # manual implementation of attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2). contiguous().view(B, T, C)   # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 2 * config.n_embd)
        self.c_proj = nn.Linear(2 * config.n_embd, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = new_gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        # x = x + self.attn(x)
        # x = x + self.mlp(x)
        return x


@dataclass
class GPTConfig:
    block_size: int = 1024
    input_dim: int = 256
    output_dim: int = 256
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.1
    gpt_name: str = 'nanogpt'


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.input_dim is not None
        assert config.output_dim is not None
        assert config.block_size is not None
        self.config = config

        self.input_dim = config.input_dim
        self.output_dim = config.output_dim
        self.block_size = config.block_size
        self.gpt_name = config.gpt_name

        self.transformer = nn.ModuleDict(
            dict(
                state_encoder=nn.Linear(config.input_dim, config.n_embd, bias=False),
                action_encoder=nn.Linear(config.output_dim, config.n_embd, bias=False),
                wpe=nn.Embedding(config.block_size, config.n_embd),
                drop=nn.Dropout(config.dropout),
                h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
                ln_f=nn.LayerNorm(config.n_embd),
            )
        )
        self.lm_head = nn.Linear(config.n_embd, 1, bias=False)

        # init all weights, and apply a special scaled init tot the residual projections, per GPT-2 paper
        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                torch.nn.init.normal_(
                    p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer)
                )

        # report number of parameters
        n_params = sum(p.numel() for p in self.parameters())
        print("number of parameters: %.2fm" % (n_params / 1e6,))

    def forward(self, state, action ,targets=None):
        device = state.device
        b, t, _ = action.size()
        assert (t + 1 <= self.config.block_size), \
            f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        pos = torch.arange(0, t+1, dtype=torch.long, device=device).unsqueeze(0)  # shape (1, t)

        # forward the GPT model itself
        state_emd = self.transformer.state_encoder(state).unsqueeze(1)
        action_emd = self.transformer.action_encoder(action)
        tok_emd = torch.cat([state_emd, action_emd], dim=1)
        pos_emd = self.transformer.wpe(pos)     # position embeddings of shape (1, t, n_embd)
        x = self.transformer.drop(tok_emd + pos_emd)
        # x = self.transformer.drop(tok_emd)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        # x = new_gelu(x)
        logits = self.lm_head(x)
        return logits

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def crop_block_size(self, block_size):
        assert block_size <= self.config.block_size
        self.config.block_size = block_size
        self.transformer.wps.weight = nn.Parameter(
            self.transformer.wpr.weight[:block_size]
        )
        for block in self.transformer.h:
            block.attn.bias = block.attm.bias[:, :, :block_size, :block_size]

    def configure_optimizers(self, weight_decay, learning_rate, betas):
        """
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """
        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear,)
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = "%s.%s" % (mn, pn) if mn else pn  # full param name
                if pn.endswith("bias"):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

        # validate that we considered evert parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert (
            len(inter_params) == 0
        ), "parameters %s made it into both decay/no_decay sets" % (str(inter_params),)
        assert (
            len(param_dict.keys() - union_params) == 0
        ), "parameters %s were not separated into either decay/no_decay set" % (
            str(param_dict.keys() - union_params),
        )

        # create the pytorch optimizer object
        optim_groups = [
            {
                "params": [param_dict[pn] for pn in sorted(list(decay))],
                "weight_decay": weight_decay,
            },
            {
                "params": [param_dict[pn] for pn in sorted(list(no_decay))],
                "weight_decay": 0.0,
            },
        ]
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas)
        return optimizer

    def save(self, log_dir: str, epoch: int):
        """
        Save NN structure and weights to file
        Args:
            log_dir: directory to save weights to
            epoch: training epoch

        Returns:
            None
        """
        # Get paths to structure parameters and weights respectively
        s_path, w_path = util.get_nn_save_paths(log_dir, self.gpt_name, epoch)

        # Store structure parameters
        with open(s_path, "wb") as f:
            parameters = {
                "input_dim": self.input_dim,
                "output_dim": self.output_dim,
                "block_size": self.block_size,
            }
            pkl.dump(parameters, f)

        # Store NN weights
        with open(w_path, "wb") as f:
            torch.save(self.state_dict(), f)

    def load(self, log_dir: str, epoch: int):
        """
        Load NN structure and weights from file
        Args:
            log_dir: directory stored weights
            epoch: training epoch

        Returns:
            None
        """
        # Get paths to structure parameters and weighs respectively
        s_path, w_path = util.get_nn_save_paths(log_dir, self.gpt_name, epoch)

        # Check structure parameters
        with open(s_path, "rb") as f:
            parameters = pkl.load(f)
            assert self.input_dim == parameters["input_dim"] \
                   and self.output_dim == parameters["output_dim"] \
                   and self.block_size == parameters["block_size"], \
                   "NN structure parameters do not match"

        # Load NN weights
        self.load_state_dict(torch.load(w_path))