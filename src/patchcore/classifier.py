import logging
from typing import Tuple, List, Optional

import torch
import torch.nn as nn
import timm


LOGGER = logging.getLogger(__name__)


def _normalize_alias(name: str) -> str:
    """Fix common typos/aliases for timm model names.

    Example: 'tf_efficient_b3_ns' -> 'tf_efficientnet_b3_ns'
    """
    aliases = {
        "tf_efficient_b3_ns": "tf_efficientnet_b3_ns",
        "tf_efficient_b0_ns": "tf_efficientnet_b0_ns",
        "tf_efficient_b2_ns": "tf_efficientnet_b2_ns",
        "tf_efficient_b4_ns": "tf_efficientnet_b4_ns",
    }
    return aliases.get(name, name)


def _try_create_model(name: str, pretrained: bool):
    return timm.create_model(name, pretrained=pretrained, num_classes=0, global_pool="avg")


class AnomalyClassifier(nn.Module):
    """
    Transfer-learning classifier for 8-class anomaly types.

    Defaults to tf_efficientnet_b3_ns for strong accuracy on small datasets.
    """

    def __init__(self, num_classes: int = 8, model_name: str = "tf_efficientnet_b3_ns", pretrained: bool = True, dropout: float = 0.2):
        super().__init__()
        # Resolve model name and fallbacks for broader timm compatibility
        preferred = _normalize_alias(model_name)
        fallbacks: List[str] = [
            preferred,
            "tf_efficientnet_b2_ns",
            "tf_efficientnet_b0_ns",
            "resnet50",
            "resnet34",
        ]

        last_err = None
        self.backbone = None
        for name in fallbacks:
            try:
                self.backbone = _try_create_model(name, pretrained)
                if name != preferred:
                    LOGGER.warning(f"Requested model '{model_name}' not available. Using fallback '{name}' instead.")
                break
            except Exception as e:
                last_err = e
                continue
        if self.backbone is None:
            # Surface the original error context
            raise RuntimeError(f"Failed to construct backbone for '{model_name}'. Last error: {last_err}")
        
        in_features = self.backbone.num_features
        self.head = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(x)
        logits = self.head(feats)
        return logits

    @torch.no_grad()
    def predict(self, x: torch.Tensor, density: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        logits = self.forward(x, density)
        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1)
        return preds, probs
