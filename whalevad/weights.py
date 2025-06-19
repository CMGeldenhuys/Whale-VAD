from enum import Enum
from typing import Any, Callable, Dict, Mapping, Optional
from dataclasses import dataclass, field

from torch import Tensor
from torch.nn import Module
from torch.hub import load_state_dict_from_url

from whalevad.model import WhaleVADClassifier, WhaleVADModel
from whalevad.specgrogram import SpectrogramExtractor

__all__ = [
    "WhaleVAD_Weights",
    "whalevad",
]


# Based on https://github.com/pytorch/vision/blob/d5df0d67dc43db85a3963795903b51c57a6146c1/torchvision/models/_api.py
@dataclass
class Weights:
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


class WhaleVAD_Weights(WeightsEnum):
    ATBFL_DCASE_3P_V1 = Weights(
        url="https://github.com/CMGeldenhuys/Whale-VAD/tag/v0.1.0/WhaleVAD_ATBFL_3P-8f25a81b.pt",
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
    weights: Optional[WhaleVAD_Weights] = None,
    progress: bool = True,
    transform: Optional[Module | Callable] = None,
    **kwargs,
) -> WhaleVADModel:
    if weights is None:
        clf = WhaleVADClassifier(**kwargs)
        return WhaleVADModel(clf, transform)
    state = weights.get_state_dict(progress=progress, check_hash=True)
    clf = WhaleVADClassifier(**weights.model_config, **kwargs)
    clf.load_state_dict(state)

    if transform is None:
        transform = weights.transform

    model = WhaleVADModel(clf, transform)

    return model
