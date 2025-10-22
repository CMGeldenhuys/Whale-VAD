from dataclasses import dataclass
from typing import List, Literal, Optional, Tuple

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
class ATBFLExemplar:
    filename: str
    start_s: float
    end_s: float
    sample_rate: int = 250

    def fetch_and_load(self, **kwargs):
        start = int(self.start_s * self.sample_rate)
        end = int(self.end_s * self.sample_rate)
        return get_atbfl_exemplar(self.filename, start, end, **kwargs)


ATBFL_REPO_URLS = {
    "train": "https://zenodo.org/records/15092732/files/biodcase_development_set.zip?download=1",
    "val": "https://zenodo.org/records/15092732/files/biodcase_development_set.zip?download=1",
}
ATBFL_EXAMPLARS = {
    "val": ATBFLExemplar(
        "biodcase_development_set/validation/audio/kerguelen2014/2014-06-29T23-00-00_000.wav",
        2755.0,
        2765.0,
    )
}


def get_atbfl_exemplar(
    filename: Optional[str] = None,
    start: Optional[int] = None,
    end: Optional[int] = None,
    split: Literal["train", "val"] | None = None,
    **kwargs,
) -> Tuple[Tensor, int]:
    """
    Fetches and loads an ATBFL exemplar from the repository. This operation only downloads the required chunks from the repo not the entire zip file.

    Args:
        filename: The filename of the audio file to load.
        start: The start time in seconds.
        end: The end time in seconds.
        split: The split to use ("train" or "val")

    Returns:
        A tuple containing the audio tensor and the sample rate.
    """
    if filename is None:
        assert start is None, "start must be None when using pre-configured exemplars"
        assert end is None, "end must be None when using pre-configured exemplars"

        split = split or "train"
        exemplar = ATBFL_EXAMPLARS[split]

        return exemplar.fetch_and_load(**kwargs)
    assert split is not None, "split must be specified"
    audio, sr = load_remote_audio(ATBFL_REPO_URLS[split], filename, **kwargs)
    assert sr == 250, f"expected sample rate of ATBFL to be 250Hz got {sr}Hz instead"

    if start is not None or end is not None:
        audio = audio[:, start:end]

    return audio, sr
