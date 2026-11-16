"""Text recognition building blocks adapted from clovaai/deep-text-recognition-benchmark.

Source    https://github.com/clovaai/deep-text-recognition-benchmark
Commit    e2117f2fb882b3c6085030500a260c113be27a63
License   Apache-2.0, Copyright (c) 2019-present NAVER Corp. Full text in
          third_party/deep-text-recognition-benchmark/LICENSE.
Adapted   modules/feature_extraction.py (VGG_FeatureExtractor, ResNet_FeatureExtractor with
          ResNet and BasicBlock), modules/sequence_modeling.py (BidirectionalLSTM) and
          modules/prediction.py (Attention, AttentionCell).

Modifications (Apache-2.0 section 4b, changes stated)
1. Classes renamed to PEP8 (VGG_FeatureExtractor to VGGFeatureExtractor, and so on).
   Python 2 style super(Class, self) calls replaced by super().
2. Dropped RCNN_FeatureExtractor and GRCL (unused) and the TPS transformation (our digit
   strips are straight, so rectification adds nothing).
3. Attention API fixes for PyTorch 2.x. The module level global device is removed. Tensors
   are created on the device of batch_H. torch.FloatTensor(...).zero_().to(device) is
   replaced by torch.zeros, and the one-hot helper by torch.nn.functional.one_hot.
4. Attention.forward takes explicit tokens. Teacher forcing reads text_in (starts with our
   BOS) and inference starts from go_index (our BOS) instead of the original token 0.
   The number of steps is an argument, not batch_max_length + 1.
5. Attention returns the attention weights (B, steps, T) next to the logits. The original
   discards them. The weights sum to 1 over T.
6. AttentionCell projects batch_H once per sequence, not once per step (same maths).
7. BidirectionalLSTM drops flatten_parameters (not needed on torch 2.x).
8. ResNet forward is unchanged except it is written as a loop over small stages.
"""

import torch
import torch.nn.functional as fn
from torch import nn


class VGGFeatureExtractor(nn.Module):
    """FeatureExtractor of CRNN (https://arxiv.org/pdf/1507.05717.pdf).

    Input (B, C, 32, W) gives (B, output_channel, 1, W / 4 - 1).
    """

    def __init__(self, input_channel: int, output_channel: int = 512) -> None:
        super().__init__()
        oc = [output_channel // 8, output_channel // 4, output_channel // 2, output_channel]
        self.ConvNet = nn.Sequential(
            nn.Conv2d(input_channel, oc[0], 3, 1, 1),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(oc[0], oc[1], 3, 1, 1),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(oc[1], oc[2], 3, 1, 1),
            nn.ReLU(True),
            nn.Conv2d(oc[2], oc[2], 3, 1, 1),
            nn.ReLU(True),
            nn.MaxPool2d((2, 1), (2, 1)),
            nn.Conv2d(oc[2], oc[3], 3, 1, 1, bias=False),
            nn.BatchNorm2d(oc[3]),
            nn.ReLU(True),
            nn.Conv2d(oc[3], oc[3], 3, 1, 1, bias=False),
            nn.BatchNorm2d(oc[3]),
            nn.ReLU(True),
            nn.MaxPool2d((2, 1), (2, 1)),
            nn.Conv2d(oc[3], oc[3], 2, 1, 0),
            nn.ReLU(True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ConvNet(x)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(
        self, inplanes: int, planes: int, stride: int = 1, downsample: nn.Module | None = None
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x if self.downsample is None else self.downsample(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + residual)


class ResNet(nn.Module):
    """ResNet of FAN (Cheng et al., ICCV 2017) as used in the benchmark."""

    def __init__(
        self, input_channel: int, output_channel: int, layers: tuple[int, ...] = (1, 2, 5, 3)
    ) -> None:
        super().__init__()
        ocb = [output_channel // 4, output_channel // 2, output_channel, output_channel]
        self.inplanes = output_channel // 8
        self.conv0_1 = nn.Conv2d(input_channel, output_channel // 16, 3, 1, 1, bias=False)
        self.bn0_1 = nn.BatchNorm2d(output_channel // 16)
        self.conv0_2 = nn.Conv2d(output_channel // 16, self.inplanes, 3, 1, 1, bias=False)
        self.bn0_2 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)

        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.layer1 = self._make_layer(ocb[0], layers[0])
        self.conv1 = nn.Conv2d(ocb[0], ocb[0], 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(ocb[0])

        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.layer2 = self._make_layer(ocb[1], layers[1])
        self.conv2 = nn.Conv2d(ocb[1], ocb[1], 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(ocb[1])

        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=(2, 1), padding=(0, 1))
        self.layer3 = self._make_layer(ocb[2], layers[2])
        self.conv3 = nn.Conv2d(ocb[2], ocb[2], 3, 1, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(ocb[2])

        self.layer4 = self._make_layer(ocb[3], layers[3])
        self.conv4_1 = nn.Conv2d(ocb[3], ocb[3], 2, (2, 1), (0, 1), bias=False)
        self.bn4_1 = nn.BatchNorm2d(ocb[3])
        self.conv4_2 = nn.Conv2d(ocb[3], ocb[3], 2, 1, 0, bias=False)
        self.bn4_2 = nn.BatchNorm2d(ocb[3])

    def _make_layer(self, planes: int, blocks: int, stride: int = 1) -> nn.Sequential:
        downsample = None
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes, 1, stride, bias=False), nn.BatchNorm2d(planes)
            )
        layers = [BasicBlock(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes
        layers += [BasicBlock(planes, planes) for _ in range(1, blocks)]
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.bn0_1(self.conv0_1(x)))
        x = self.relu(self.bn0_2(self.conv0_2(x)))
        stages = [
            (self.maxpool1, self.layer1, self.conv1, self.bn1),
            (self.maxpool2, self.layer2, self.conv2, self.bn2),
            (self.maxpool3, self.layer3, self.conv3, self.bn3),
        ]
        for pool, layer, conv, bn in stages:
            x = self.relu(bn(conv(layer(pool(x)))))
        x = self.relu(self.bn4_1(self.conv4_1(self.layer4(x))))
        return self.relu(self.bn4_2(self.conv4_2(x)))


class ResNetFeatureExtractor(nn.Module):
    """FeatureExtractor of FAN. Input (B, C, 32, W) gives (B, output_channel, 1, W / 4 + 1)."""

    def __init__(self, input_channel: int, output_channel: int = 512) -> None:
        super().__init__()
        self.ConvNet = ResNet(input_channel, output_channel)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ConvNet(x)


class BidirectionalLSTM(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int) -> None:
        super().__init__()
        self.rnn = nn.LSTM(input_size, hidden_size, bidirectional=True, batch_first=True)
        self.linear = nn.Linear(hidden_size * 2, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Visual features (B, T, input_size) to contextual features (B, T, output_size)."""
        recurrent, _ = self.rnn(x)
        return self.linear(recurrent)


class AttentionCell(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_embeddings: int) -> None:
        super().__init__()
        self.i2h = nn.Linear(input_size, hidden_size, bias=False)
        self.h2h = nn.Linear(hidden_size, hidden_size)  # either i2h or h2h should have bias
        self.score = nn.Linear(hidden_size, 1, bias=False)
        self.rnn = nn.LSTMCell(input_size + num_embeddings, hidden_size)

    def forward(
        self,
        prev_hidden: tuple[torch.Tensor, torch.Tensor],
        batch_h: torch.Tensor,
        batch_h_proj: torch.Tensor,
        char_onehots: torch.Tensor,
    ) -> tuple[tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        """One decoder step. batch_h_proj is self.i2h(batch_h), computed once by the caller.

        Returns the new (h, c) and the attention weights alpha (B, T, 1).
        """
        prev_hidden_proj = self.h2h(prev_hidden[0]).unsqueeze(1)
        e = self.score(torch.tanh(batch_h_proj + prev_hidden_proj))  # (B, T, 1)
        alpha = fn.softmax(e, dim=1)
        context = torch.bmm(alpha.permute(0, 2, 1), batch_h).squeeze(1)  # (B, channels)
        cur_hidden = self.rnn(torch.cat([context, char_onehots], dim=1), prev_hidden)
        return cur_hidden, alpha


class Attention(nn.Module):
    """Attention decoder of the benchmark (Bahdanau style, LSTM cell, one-hot inputs)."""

    def __init__(self, input_size: int, hidden_size: int, num_classes: int) -> None:
        super().__init__()
        self.attention_cell = AttentionCell(input_size, hidden_size, num_classes)
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.generator = nn.Linear(hidden_size, num_classes)

    def forward(
        self,
        batch_h: torch.Tensor,
        text_in: torch.Tensor | None,
        num_steps: int,
        go_index: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Decode num_steps tokens from contextual features batch_h (B, T, channels).

        With text_in (B, num_steps) the decoder is teacher forced. With text_in=None it feeds
        back its own argmax, starting from go_index. Returns logits (B, num_steps, classes)
        and attention weights (B, num_steps, T).
        """
        batch, device = batch_h.shape[0], batch_h.device
        hidden = (
            batch_h.new_zeros(batch, self.hidden_size),
            batch_h.new_zeros(batch, self.hidden_size),
        )
        proj = self.attention_cell.i2h(batch_h)
        prev = torch.full((batch,), go_index, dtype=torch.long, device=device)
        outputs, alphas = [], []
        for i in range(num_steps):
            if text_in is not None:
                prev = text_in[:, i]
            onehot = fn.one_hot(prev, self.num_classes).to(batch_h.dtype)
            hidden, alpha = self.attention_cell(hidden, batch_h, proj, onehot)
            step = self.generator(hidden[0])
            outputs.append(step)
            alphas.append(alpha.squeeze(-1))
            prev = step.argmax(dim=-1)
        return torch.stack(outputs, dim=1), torch.stack(alphas, dim=1)
