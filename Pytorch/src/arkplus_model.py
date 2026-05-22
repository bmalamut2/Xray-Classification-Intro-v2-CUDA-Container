from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import timm


class ArkPlusModel(nn.Module):
    """Shared encoder/projector with one task head per Ark+ dataset."""

    def __init__(
        self,
        backbone: str,
        num_classes_list: List[int],
        pretrained: bool = True,
        projector_features: Optional[int] = 1376,
        use_mlp: bool = False,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.encoder = timm.create_model(backbone, pretrained=pretrained, num_classes=0)
        encoder_features = int(getattr(self.encoder, "num_features", 0))
        if encoder_features <= 0:
            raise ValueError(f"Could not infer feature size for backbone: {backbone}")

        self.projector = None
        self.num_features = encoder_features
        if projector_features is not None and int(projector_features) > 0:
            self.num_features = int(projector_features)
            if use_mlp:
                self.projector = nn.Sequential(
                    nn.Linear(encoder_features, self.num_features),
                    nn.ReLU(inplace=True),
                    nn.Linear(self.num_features, self.num_features),
                )
            else:
                self.projector = nn.Linear(encoder_features, self.num_features)

        self.omni_heads = nn.ModuleList(
            [nn.Linear(self.num_features, int(num_classes)) for num_classes in num_classes_list]
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        features = self.encoder(x)
        if isinstance(features, (tuple, list)):
            features = features[0]
        if features.dim() == 4:
            features = features.mean(dim=(2, 3))
        elif features.dim() == 3:
            features = features.mean(dim=1)
        if self.projector is not None:
            features = self.projector(features)
        return features

    def forward(
        self, x: torch.Tensor, head_index: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.encode(x)
        if head_index is None:
            return features, [head(features) for head in self.omni_heads]
        return features, self.omni_heads[int(head_index)](features)


def build_arkplus_model(
    backbone: str,
    num_classes_list: List[int],
    pretrained: bool = True,
    projector_features: Optional[int] = 1376,
    use_mlp: bool = False,
) -> ArkPlusModel:
    return ArkPlusModel(
        backbone=backbone,
        num_classes_list=num_classes_list,
        pretrained=pretrained,
        projector_features=projector_features,
        use_mlp=use_mlp,
    )


def unwrap_model(model: nn.Module) -> nn.Module:
    return model.module if hasattr(model, "module") else model
