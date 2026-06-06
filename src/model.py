"""Model utilities for transfer learning on chest X-ray classification."""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import Adam
from torchvision import models

# Allow imports from the project root when executing this module directly.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
	sys.path.append(str(PROJECT_ROOT))


def _load_densenet121(pretrained: bool) -> nn.Module:
	"""Load DenseNet121 with compatibility for modern/legacy torchvision APIs."""
	try:
		weights = models.DenseNet121_Weights.DEFAULT if pretrained else None
		return models.densenet121(weights=weights)
	except AttributeError:
		return models.densenet121(pretrained=pretrained)


def _load_resnet50(pretrained: bool) -> nn.Module:
	"""Load ResNet50 with compatibility for modern/legacy torchvision APIs."""
	try:
		weights = models.ResNet50_Weights.DEFAULT if pretrained else None
		return models.resnet50(weights=weights)
	except AttributeError:
		return models.resnet50(pretrained=pretrained)


def build_model(
	architecture: str,
	num_classes: int,
	pretrained: bool,
	dropout: float,
	freeze_backbone: bool,
) -> nn.Module:
	"""Build and return a transfer-learning model for binary chest X-ray classification."""
	architecture = architecture.lower()

	if architecture == "densenet121":
		model = _load_densenet121(pretrained=pretrained)

		if freeze_backbone:
			for param in model.parameters():
				param.requires_grad = False

		model.classifier = nn.Sequential(
			nn.Linear(1024, 512),
			nn.BatchNorm1d(512),
			nn.ReLU(inplace=True),
			nn.Dropout(p=dropout),
			nn.Linear(512, 256),
			nn.BatchNorm1d(256),
			nn.ReLU(inplace=True),
			nn.Dropout(p=dropout / 2),
			nn.Linear(256, num_classes),
		)

		if freeze_backbone:
			for param in model.classifier.parameters():
				param.requires_grad = True

		return model

	if architecture == "resnet50":
		model = _load_resnet50(pretrained=pretrained)

		if freeze_backbone:
			for param in model.parameters():
				param.requires_grad = False

			for param in model.layer4.parameters():
				param.requires_grad = True

			for param in model.fc.parameters():
				param.requires_grad = True

		model.fc = nn.Sequential(
			nn.Linear(2048, 512),
			nn.BatchNorm1d(512),
			nn.ReLU(inplace=True),
			nn.Dropout(p=dropout),
			nn.Linear(512, 256),
			nn.BatchNorm1d(256),
			nn.ReLU(inplace=True),
			nn.Dropout(p=dropout / 2),
			nn.Linear(256, num_classes),
		)

		if freeze_backbone:
			for param in model.fc.parameters():
				param.requires_grad = True

		return model

	raise ValueError("Unsupported architecture. Choose 'densenet121' or 'resnet50'.")


def unfreeze_backbone(model: nn.Module, architecture: str) -> None:
	"""Unfreeze all layers to enable full-model fine-tuning."""
	architecture = architecture.lower()
	if architecture not in {"densenet121", "resnet50"}:
		raise ValueError("Unsupported architecture. Choose 'densenet121' or 'resnet50'.")

	for param in model.parameters():
		param.requires_grad = True

	print("Backbone unfrozen for fine-tuning")


def get_optimizer(model: nn.Module, lr: float, weight_decay: float):
	"""Create Adam optimizer with slower backbone LR and normal head LR."""
	head_names = ["classifier", "fc"]
	backbone_params = []
	head_params = []

	for name, param in model.named_parameters():
		if not param.requires_grad:
			continue
		if any(name.startswith(head_name) for head_name in head_names):
			head_params.append(param)
		else:
			backbone_params.append(param)

	param_groups = []
	if backbone_params:
		param_groups.append({"params": backbone_params, "lr": lr * 0.1})
	if head_params:
		param_groups.append({"params": head_params, "lr": lr})

	return Adam(param_groups, weight_decay=weight_decay)


def count_parameters(model: nn.Module) -> None:
	"""Print total, trainable, and frozen parameter statistics."""
	total_params = sum(param.numel() for param in model.parameters())
	trainable_params = sum(param.numel() for param in model.parameters() if param.requires_grad)
	frozen_params = total_params - trainable_params
	trainable_percent = (100.0 * trainable_params / total_params) if total_params > 0 else 0.0

	print(f"Total parameters: {total_params:,}")
	print(f"Trainable parameters: {trainable_params:,}")
	print(f"Frozen parameters: {frozen_params:,}")
	print(f"Trainable percentage: {trainable_percent:.2f}%")


if __name__ == "__main__":
	model = build_model(
		architecture="densenet121",
		num_classes=2,
		pretrained=True,
		dropout=0.5,
		freeze_backbone=True,
	)

	count_parameters(model)

	x = torch.randn(4, 3, 224, 224)
	with torch.no_grad():
		y = model(x)

	print(f"{tuple(x.shape)} -> {tuple(y.shape)}")
	print("Classifier head:")
	print(model.classifier)
