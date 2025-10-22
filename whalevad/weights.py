from enum import Enum
from typing import Any, Callable, Dict, Mapping, Optional, Self
from dataclasses import dataclass, field

from torch import Tensor
from torch.nn import Module
from torch.hub import load_state_dict_from_url

from whalevad.model import WhaleVADClassifier, WhaleVADModel
from whalevad.spectrogram import SpectrogramExtractor

__all__ = [
    "WhaleVAD_Weights",
    "whalevad",
]


# Based on https://github.com/pytorch/vision/blob/d5df0d67dc43db85a3963795903b51c57a6146c1/torchvision/models/_api.py
@dataclass
class Weights:
    """
    Dataclass containing configuration related to model weights.

    Args:
        url (str): The URL to download the pre-trained weights from.
        transform (Optional[Callable | Module]): The transformation to apply to the input data.
        meta (Dict[str, Any]): Additional metadata about the weights.
        model_config (Dict[str, Any]): Configuration parameters for the model.
    """

    url: str
    transform: Optional[Callable | Module] = None
    meta: Dict[str, Any] = field(default_factory=dict)
    model_config: Dict[str, Any] = field(default_factory=dict)


class WeightsEnum(Enum):
    """
    This class is the parent class of all model weights. Each model building method receives an optional `weights`
    parameter with its associated pre-trained weights. It inherits from `Enum` and its values should be of type
    `Weights`.

    Args:
        value (Weights): The data class entry with the weight information.
    """

    value: Weights  # type: ignore

    def get_state_dict(self, *args: Any, **kwargs: Any) -> Mapping[str, Any]:
        return load_state_dict_from_url(self.url, *args, **kwargs)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}.{self._name_}"

    @property
    def url(self):
        return self.value.url

    @property
    def transform(self):
        return self.value.transform

    @property
    def meta(self):
        return self.value.meta

    @property
    def model_config(self):
        return self.value.model_config

    @classmethod
    def verify(cls, inst: str | Self) -> Self:
        if isinstance(inst, WeightsEnum):
            return inst

        return getattr(cls, inst)


class WhaleVAD_Weights(WeightsEnum):
    ATBFL_DCASE_3P_V1 = Weights(
        url="https://github.com/CMGeldenhuys/Whale-VAD/releases/download/v0.1.0/WhaleVAD_ATBFL_3P-c6f6a07a.pt",
        model_config=dict(
            num_classes=7,
            feat_channels=3,
        ),
        transform=SpectrogramExtractor(
            sample_rate=250,
            n_fft=256,
            win_length=256,
            hop_length=5,
            norm_features="demean",
            power=None,
            complex_repr="trig",
        ),
    )
    DEFAULT = ATBFL_DCASE_3P_V1


def whalevad(
    weights: Optional[WhaleVAD_Weights | str] = None,
    progress: bool = True,
    transform: Optional[Module | Callable] = None,
    eval: bool = True,
    **kwargs,
) -> WhaleVADModel:
    """
    Create a WhaleVAD model with pre-trained weights.

    Args:
        weights (Optional[WhaleVAD_Weights | str]): The weights to use for the model.
        progress (bool): Whether to show a progress bar while downloading the weights.
        transform (Optional[Module | Callable]): The transformation to apply to the input data.
        **kwargs: Additional keyword arguments to pass to the model constructor.

    Returns:
        WhaleVADModel

    Example:
        >>> model = whalevad(weights='DEFAULT')
        >>> model.eval()
        >>> classifier, transform = model
    """
    if weights is None:
        clf = WhaleVADClassifier(**kwargs)
        return WhaleVADModel(clf, transform)
    weights = WhaleVAD_Weights.verify(weights)
    state = weights.get_state_dict(progress=progress, check_hash=True)
    clf = WhaleVADClassifier(**weights.model_config, **kwargs)
    clf.load_state_dict(state)

    if transform is None:
        transform = weights.transform

    model = WhaleVADModel(clf, transform)

    if eval:
        model = model.eval()

    return model
