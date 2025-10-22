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
    """
    Computes the mean of a sequence of tensors along a specified dimension.

    Args:
        input (Tensor): The input tensor.
        dim (Optional[int]): The dimension along which to compute the mean.
        keepdim (bool): Whether to keep the dimension of the input tensor.
        lengths (List[int] | Tensor): The lengths of the sequences.

    Returns:
        Tensor: The mean of the sequence of tensors.
    """

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
    """
    Creates a padding mask for a batch of sequences.

    Args:
        lengths (List[int] | Tensor): The lengths of the sequences.
        dtype (torch.dtype): The data type of the mask.
        device (torch.device | None): The device on which to create the mask.
        max_len (Optional[int]): The maximum length of the sequences.

    Returns:
        Tensor: The padding mask.
    """
    if isinstance(lengths, Tensor):
        lengths = lengths.tolist()
    batch_size = len(lengths)
    time = max_len if max_len is not None else max(lengths)

    mask = torch.zeros((batch_size, time), dtype=dtype, device=device)

    for i, l in enumerate(lengths):
        assert l <= time, "index of bounds, increate max_len"
        mask[i, l:] = True
    return mask


def load_remote_audio(
    url: str,
    filename: str,
    *,
    frame_offset: Optional[int] = None,
    num_frames: Optional[int] = None,
    start_s: Optional[float] = None,
    end_s: Optional[float] = None,
    sample_rate: Optional[int] = None,
    **kwargs,
):
    from remotezip import RemoteZip

    if start_s is not None:
        assert frame_offset is None, "offset and start_s cannot be both specified"
        assert (
            sample_rate is not None
        ), "sample_rate must be specified when using start_s"
        frame_offset = int(start_s * sample_rate)
    frame_offset = frame_offset or 0
    del start_s

    if end_s is not None:
        assert num_frames is None, "num_frames and end_s cannot be both specified"
        assert sample_rate is not None, "sample_rate must be specified when using end_s"
        end = int(end_s * sample_rate)
        num_frames = end - frame_offset
        assert num_frames > 0, "num_frames must be positive"
    num_frames = num_frames or -1
    del end_s

    with RemoteZip(url) as zf:
        with zf.open(filename) as file:
            buffer = io.BytesIO(file.read())
            audio, sr = torchaudio.load(
                buffer,
                frame_offset=frame_offset,
                num_frames=num_frames,
                **kwargs,
            )
    assert (
        sr == sample_rate
    ), f"sample rate mismatch: {sr}Hz (file) != {sample_rate}Hz (expected)"
    return audio, sr


@dataclass
class ATBFLExemplar:
    filename: str
    start_s: float
    end_s: float
    sample_rate: int = 250

    def fetch_and_load(self, **kwargs):
        return get_atbfl_exemplar(
            self.filename,
            start_s=self.start_s,
            end_s=self.end_s,
            sample_rate=self.sample_rate,
            **kwargs,
        )


ATBFL_REPO_URLS = {
    "train": "https://zenodo.org/records/15092732/files/biodcase_development_set.zip?download=1",
    "val": "https://zenodo.org/records/15092732/files/biodcase_development_set.zip?download=1",
}
ATBFL_EXAMPLARS = {
    "train": ATBFLExemplar(
        "biodcase_development_set/train/audio/kerguelen2005/2005-06-21T18-00-00_000.wav",
        960.0,
        970.0,
    ),
    "val": ATBFLExemplar(
        "biodcase_development_set/validation/audio/kerguelen2014/2014-06-29T23-00-00_000.wav",
        2755.0,
        2765.0,
    ),
}


def get_atbfl_exemplar(
    filename: Optional[str] = None,
    *,
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
        split = split or "train"
        exemplar = ATBFL_EXAMPLARS[split]

        return exemplar.fetch_and_load(**kwargs, split=split)
    assert split is not None, "split must be specified"

    return load_remote_audio(ATBFL_REPO_URLS[split], filename, **kwargs)
