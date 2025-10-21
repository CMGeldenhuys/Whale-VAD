from typing import List, Optional, Callable

from functools import partial

import torch
from torch import Tensor
from torch.nn import Module
from torch.nn.utils.rnn import unpad_sequence


def unpad_mean(
    input: Tensor, *, dim: Optional[int] = None, keepdim: bool = False, lengths
):

    input_seq = unpad_sequence(input, lengths=lengths, batch_first=True)
    if keepdim:
        input_seq = [seq.unsqueeze(0) for seq in input_seq]

    seq_mean = [torch.mean(seq, dim=dim, keepdim=keepdim) for seq in input_seq]
    return torch.vstack(seq_mean)


def padding_mask(
    lengths: List[int] | Tensor,
    *,
    dtype: torch.dtype = torch.bool,
    device: torch.device | None = None,
    max_len: Optional[int] = None
) -> Tensor:
    if isinstance(lengths, Tensor):
        lengths = lengths.tolist()
    batch_size = len(lengths)
    time = max_len if max_len is not None else max(lengths)

    mask = torch.zeros((batch_size, time), dtype=dtype, device=device)

    for i, l in enumerate(lengths):
        assert l <= time, "index of bounds, increate max_len"
        mask[i, l:] = True
    return mask
