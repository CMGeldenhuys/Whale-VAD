from typing import Callable, Dict, List, Literal, Optional, Tuple

from torch import Tensor, adaptive_avg_pool1d
import torch
from torch.nn import (
    GELU,
    LSTM,
    AdaptiveAvgPool1d,
    BatchNorm2d,
    Conv2d,
    Flatten,
    Linear,
    MaxPool2d,
    Module,
    ModuleList,
    Sequential,
    Dropout,
    Dropout2d,
)


class WhaleVADClassifier(Module):

    _class_mapping = ["bmabz", "d", "bp"]

    def __init__(
        self,
        *,
        num_classes: Optional[int] = None,
        feat_channels=1,
        include_intermediate_features: bool = False,
        include_bounding_boxes: bool = False,
        include_bottleneck_layers: bool = True,
        include_aggregation_layers: bool = True,
        num_anchors: int = 64,
    ) -> None:
        super().__init__()
        num_classes = (
            num_classes if num_classes is not None else len(self._class_mapping)
        )
        self.num_anchors = num_anchors

        # Act as a learnable mel
        self.fbank = Conv2d(
            in_channels=feat_channels,
            out_channels=64,
            kernel_size=(7, 1),  # only convolve over spectrum not time
            stride=(3, 1),
            padding=0,  # we can drop some spec features
        )

        conv_block_feat_extractor = Sequential(
            # LAYER 1
            Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=(5, 5),
                stride=(3, 1),
                padding=5 // 2,  # needed to keep output resolution the same
            ),
            BatchNorm2d(128),
            GELU(),
            MaxPool2d(
                kernel_size=(5, 1),
                stride=1,
                padding=0,
            ),
            # ------------------
            # LAYER 2
            Conv2d(
                in_channels=128,
                out_channels=128,
                kernel_size=(3, 3),
                stride=(2, 1),
                padding=3 // 2,
            ),
            BatchNorm2d(128),
            GELU(),
            MaxPool2d(
                kernel_size=(3, 1),
                stride=1,
                padding=0,
            ),
        )
        conv_block_bottleneck = Sequential(
            # ------------------
            # LAYER 3
            Conv2d(
                in_channels=128,
                out_channels=64,
                kernel_size=(1, 1),
                stride=(1, 1),
                padding=1 // 2,
            ),
            GELU(),
            Dropout(0.1),
            # ------------------
            # LAYER 4
            Conv2d(
                in_channels=64,
                out_channels=64,
                kernel_size=(3, 3),
                stride=(1, 1),
                padding=3 // 2,
            ),
            GELU(),
            Dropout(0.1),
            # ------------------
            # LAYER 5
            Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=(1, 1),
                stride=(1, 1),
                padding=1 // 2,
            ),
            BatchNorm2d(128),
            GELU(),
            Dropout(0.1),
        )

        conv_block_feat_agg = Sequential(
            # Spatial dropout: https://arxiv.org/pdf/1411.4280
            Dropout2d(0.2),
            # ------------------
            # LAYER 6
            Conv2d(
                in_channels=128,
                out_channels=128,
                kernel_size=(3, 3),
                stride=(1, 1),
                padding=3 // 2,
                groups=128,
            ),
            BatchNorm2d(128),
            GELU(),
            # ------------------
            # LAYER 7
            Conv2d(
                in_channels=128,
                out_channels=128,
                kernel_size=(3, 3),
                stride=(1, 1),
                padding=3 // 2,
                groups=128,
            ),
            BatchNorm2d(128),
            GELU(),
            # ------------------
            # LAYER 8
            Conv2d(
                in_channels=128,
                out_channels=128,
                kernel_size=(3, 3),
                stride=(1, 1),
                padding=3 // 2,
                groups=128,
            ),
            BatchNorm2d(128),
            GELU(),
        )

        cnn_blocks: List[Module] = []
        if include_bottleneck_layers:
            cnn_blocks.append(conv_block_bottleneck)

        if include_aggregation_layers:
            cnn_blocks.append(conv_block_feat_agg)
        # Residual blocks
        # TODO: make residule module
        self.cnn_blocks = Sequential(
            conv_block_feat_extractor,
            ResidualBlock(  # Expected residual shape: (batch, channels=128, feat=3, time)
                *cnn_blocks,
                output_residuals=include_intermediate_features,
            ),
        )
        # flatten to shape (batch, time, channel*feat=3*128)

        # Eigen dim reduction
        if include_intermediate_features:
            # three residual/intermediate layer with 3 channels each
            # num_res*num_chan*num_feat=3*128*3
            self.feat_proj = Linear(3 * 128 * 3, 64)
            self.bb_proj = Linear(3 * 128 * 3, 64)
        else:
            # channels*num_features=128*3
            self.feat_proj = Linear(128 * 3, 64)
            self.bb_proj = Linear(128 * 3, 64)

        self.lstm = LSTM(
            input_size=64,  # output of feat_proj
            hidden_size=128,
            bidirectional=True,
            num_layers=2,
            dropout=0.5,
        )

        self.classifier = Linear(
            in_features=128 * 2,  # hidden_size * 2 (bidirectional)
            out_features=num_classes,
        )

        self.include_bounding_boxes = include_bounding_boxes
        self.bounding_box_mlp = (
            Sequential(
                Linear(64, 128),
                Dropout(0.05),
                GELU(),
                Linear(128, 128),
                Dropout(0.05),
                GELU(),
            )
            if include_bounding_boxes
            else None
        )
        self.bounding_box_reg = (
            Linear(128, 4) if include_bounding_boxes else None  # ([XYXY])
        )

        self.bounding_box_conf = (
            Sequential(
                Linear(128, 1),
                Flatten(),
            )
            if include_bounding_boxes
            else None
        )

    def forward(
        self,
        features: Tensor,  # shape(batch_dim, *, [channel], time, embedding_size)
        **opts,
    ) -> Tuple[Tensor, Tensor, Dict]:
        return_opts = dict()
        # Swap time and feature dim
        features = features.transpose(-1, -2)
        # add empty channel dim
        if features.ndim < 4:
            features = features.unsqueeze(1)
        # shape: (batch, chanel=1, dim, frames)

        # Think mel...
        filt_feat = self.fbank(features)
        # shape: (batch, channel=64, features=58, time)

        # Think cepstrum
        cepstrum = self.cnn_blocks(filt_feat)
        # shape: (batch, channel=128, features=3, time)
        cepstrum = cepstrum.flatten(start_dim=1, end_dim=2)
        cepstrum = cepstrum.transpose(-1, -2)
        # shape: (batch, time, channel*feat=384)

        # Down project
        latent = self.feat_proj(cepstrum)
        # shape: (batch, time, proj=128)

        # RNN
        logits, pred, clf_opts = self.lstm(latent, **opts)
        return_opts.update(clf_opts)

        if self.include_bounding_boxes:
            assert self.bounding_box_conf is not None
            assert self.bounding_box_reg is not None
            assert self.bounding_box_mlp is not None

            bb_latent = self.bb_proj(cepstrum)
            bb_latent = bb_latent.transpose(-1, -2)
            # shape: batch, proj, time
            bb_latent = adaptive_avg_pool1d(bb_latent, self.num_anchors)
            bb_latent = bb_latent.transpose(-1, -2)
            # shape: batch, num_anchors, proj
            bb_latent = self.bounding_box_mlp(bb_latent)

            bounding_box_conf = self.bounding_box_conf(bb_latent)
            bounding_box = self.bounding_box_reg(bb_latent)

            return_opts["bounding_box_reg"] = bounding_box
            return_opts["bounding_box_conf"] = bounding_box_conf
        # shape: (batch, time, proj=128)

        return logits, pred, return_opts


class ResidualBlock(Module):
    def __init__(
        self,
        *blocks: Module,
        connection: Literal["parallel", "series"] = "series",
        residual: Literal["concat", "sum"] = "sum",
        output_residuals: bool = False,
    ):
        super().__init__()
        assert residual == "sum"
        assert connection == "series"

        self.blocks = ModuleList(blocks)
        self.output_residuals = output_residuals

    def forward(self, X: Tensor):
        if self.output_residuals:
            residuals = [X]

        for block in self.blocks:
            X = X + block(X)

            if self.output_residuals:
                residuals.append(X)

        if self.output_residuals:
            return torch.concat(residuals, dim=-2)
        return X


class WhaleVADModel(Module):
    def __init__(
        self,
        classifier: WhaleVADClassifier,
        transform: Optional[Module | Callable[[Tensor], Tensor]],
    ):
        super().__init__()
        self.classifier = classifier
        self.transform = transform

    def forward(self, audio: Tensor):
        spec = audio
        if self.transform:
            spec = self.transform(audio)
        return self.classifier(spec)

    def __iter__(self):
        yield self.classifier
        yield self.transform
