from dataclasses import dataclass
from typing import Any, Callable, Dict, List

import torchvision.models as tv_models


@dataclass(frozen=True)
class RGBBackboneSpec:
    model_fn: Callable[..., Any]
    weights: Any
    out_features: int


RGB_BACKBONES: Dict[str, RGBBackboneSpec] = {}


def register_rgb_backbone(name: str, model_fn: Callable[..., Any], weights: Any, out_features: int):
    if name in RGB_BACKBONES:
        raise ValueError(f"RGB backbone already registered: {name}")
    RGB_BACKBONES[name] = RGBBackboneSpec(
        model_fn=model_fn,
        weights=weights,
        out_features=out_features,
    )


def get_rgb_backbone_spec(name: str) -> RGBBackboneSpec:
    if name not in RGB_BACKBONES:
        raise ValueError(
            f"Unsupported RGB backbone: {name}. "
            f"Choose from: {list_rgb_backbones()}"
        )
    return RGB_BACKBONES[name]


def list_rgb_backbones() -> List[str]:
    return sorted(RGB_BACKBONES.keys())


# Default RGB backbones.
register_rgb_backbone(
    "resnet18",
    tv_models.resnet18,
    tv_models.ResNet18_Weights.IMAGENET1K_V1,
    out_features=512,
)
register_rgb_backbone(
    "resnet34",
    tv_models.resnet34,
    tv_models.ResNet34_Weights.IMAGENET1K_V1,
    out_features=512,
)
register_rgb_backbone(
    "resnet50",
    tv_models.resnet50,
    tv_models.ResNet50_Weights.IMAGENET1K_V1,
    out_features=2048,
)
register_rgb_backbone(
    "resnet101",
    tv_models.resnet101,
    tv_models.ResNet101_Weights.IMAGENET1K_V1,
    out_features=2048,
)
