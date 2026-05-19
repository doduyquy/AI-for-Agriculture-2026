from dataclasses import dataclass
from typing import Any, Callable, Dict, List

import torchvision.models as tv_models

from src.models.resnet18_rgb import Resnet18
from src.models.resnet_34_rgb import Resnet34


@dataclass(frozen=True)
class RGBBackboneSpec:
    model_fn: Callable[..., Any]
    weights: Any
    out_features: int


RGB_BACKBONES: Dict[str, RGBBackboneSpec] = {}
MODEL_REGISTRY: Dict[str, Callable[..., Any]] = {}


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


def register_model(name: str, model_cls: Callable[..., Any]):
    if name in MODEL_REGISTRY:
        raise ValueError(f"Model already registered: {name}")
    MODEL_REGISTRY[name] = model_cls


def get_model_builder(name: str) -> Callable[..., Any] | None:
    return MODEL_REGISTRY.get(name)


def list_models() -> List[str]:
    return sorted(MODEL_REGISTRY.keys())


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


# Custom full models.
register_model("resnet18_rgb", Resnet18)
register_model("resnet34_rgb", Resnet34)