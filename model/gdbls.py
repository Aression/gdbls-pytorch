import torch
from torch import nn
from torch import Tensor
from typing import Type, Any, Callable, Union, List, Optional

from torch._jit_internal import ignore
from torchvision.utils import _log_api_usage_once


def conv3x3(
        in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1
) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class FeatureBlock(nn.Module):
    def __init__(
            self,
            inplanes: int,
            planes: int,
            stride: int = 1,
            divn: int = 4,
            dropout_rate: float = 0.1,
            downsample: Optional[nn.Module] = None,
            norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if downsample is None:
            downsample = nn.MaxPool2d(kernel_size=2)

        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes // 2, stride)
        self.bn1 = norm_layer(planes // 2)
        self.conv2 = conv3x3(planes // 2, planes // 2)
        self.bn2 = norm_layer(planes // 2)
        self.conv3 = conv3x3(planes // 2, planes)
        self.bn3 = norm_layer(planes)

        self.relu = nn.ReLU(inplace=True)

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(planes, planes // divn)
        self.fc2 = nn.Linear(planes // divn, planes)

        self.reshape = torch.reshape
        self.multiply = torch.multiply

        self.dropout = nn.Dropout(dropout_rate)
        self.downsample = downsample
        self.planes = planes

    def forward(self, x: Tensor) -> Tensor:
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.dropout(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.relu(out)

        # se block
        identity = out
        # only for [N,W,H,C], this is globalaveragepooling2d
        squeeze = self.avgpool(out)
        se_res = self.fc1(squeeze)
        se_res = self.fc2(se_res)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.multiply(self.reshape(se_res, (1, 1, self.planes)), identity)
        out = self.relu(out)
        out = self.dropout(out)
        return out


# gdbls base
# GrandDescentBoardLearningSystem
class GDBLS(nn.Module):
    def __init__(
            self,
            num_classes: int = 10,
            input_shape: List[int] = None,
            overall_dropout=0.5,
            zero_init_residual: bool = True,
            filters: List[int] = None,
            divns: List[int] = None,
            dropout_rate: List[float] = None,
            norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super(GDBLS, self).__init__()
        _log_api_usage_once(self)
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if input_shape is None:
            input_shape = [32, 32, 3]

        self._norm_layer = norm_layer

        self.inplanes = 3
        self.num_classes = num_classes

        self.fb1 = FeatureBlock(
            inplanes=self.inplanes,
            planes=filters[0],
            divn=divns[0],
            dropout_rate=dropout_rate[0],
        )
        self.fb2 = FeatureBlock(
            inplanes=filters[0],
            planes=filters[1],
            divn=divns[1],
            dropout_rate=dropout_rate[0],
        )
        self.fb3 = FeatureBlock(
            inplanes=filters[1],
            planes=filters[2],
            divn=divns[2],
            dropout_rate=dropout_rate[0],
        )

        self.flatten = torch.flatten
        self.fc = nn.Linear
        self.concat = torch.concat

        self.dropout = nn.Dropout(overall_dropout)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, FeatureBlock):
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def _forward_impl(self, x: Tensor) -> Tensor:
        p1 = self.fb1(x)
        p2 = self.fb2(p1)
        p3 = self.fb3(p2)

        t1 = self.flatten(p1)
        t2 = self.flatten(p2)
        t3 = self.flatten(p3)

        merged = self.concat([t1, t2, t3])
        merged = self.dropout(merged)

        out = self.fc(merged.shape[-1], self.num_classes)(merged)
        return out

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)
