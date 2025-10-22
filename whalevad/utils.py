from dataclasses import dataclass
from typing import List, Optional

import io
import torchaudio

import torch
from torch import Tensor
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
    max_len: Optional[int] = None,
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


def load_remote_audio(url: str, filename: str, **kwargs):
    from remotezip import RemoteZip

    with RemoteZip(url) as zf:
        with zf.open(filename) as file:
            buffer = io.BytesIO(file.read())
            return torchaudio.load(buffer, **kwargs)


@dataclass
class ATBFLExamplar:
    filename: str
    start: int
    end: int


ATBFL_REPO_URL = (
    "https://zenodo.org/records/15092732/files/biodcase_development_set.zip?download=1"
)
ATBFL_EXAMPLARS = {
    "train": ATBFLExamplar(
        "biodcase_development_set/train/audio/kerguelen2014/2014-06-29T23-00-00_000.wav",
        0,
        2500,
    )
}


def get_atbfl_examplar(
    filename: Optional[str] = None,
    start: Optional[int] = None,
    end: Optional[int] = None,
    **kwargs,
):
    if filename is None:
        default = ATBFL_EXAMPLARS["train"]
        filename = default.filename
        start = start if start is not None else default.start
        end = end if end is not None else default.end
    audio, sr = load_remote_audio(ATBFL_REPO_URL, filename, **kwargs)
    assert sr == 250, f"expected sample rate of ATBFL to be 250Hz got {sr}Hz instead"

    if start is not None or end is not None:
        audio = audio[:, start:end]

    return audio, sr
