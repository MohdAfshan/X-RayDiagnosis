"""Dataset and dataloader utilities for chest X-ray binary classification."""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

# Allow imports from the project root when executing this module directly.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
	sys.path.append(str(PROJECT_ROOT))


class ChestXrayDataset(Dataset):
	"""PyTorch dataset for chest X-ray images with NORMAL/PNEUMONIA labels."""

	class_to_idx = {"NORMAL": 0, "PNEUMONIA": 1}

	def __init__(self, root_dir: str, split: str, transform=None) -> None:
		"""Initialize dataset by scanning split/class folders for image files.

		Args:
			root_dir: Path to chest_xray directory.
			split: One of train, val, or test.
			transform: Optional torchvision transform pipeline.
		"""
		split = split.lower()
		if split not in {"train", "val", "test"}:
			raise ValueError("split must be one of: 'train', 'val', 'test'")

		self.root_dir = Path(root_dir)
		self.split = split
		self.transform = transform

		self.image_paths: List[Path] = []
		self.labels: List[int] = []

		split_dir = self.root_dir / self.split

		for class_name, label in self.class_to_idx.items():
			class_dir = split_dir / class_name
			jpeg_paths = sorted(class_dir.glob("*.jpeg"))
			jpg_paths = sorted(class_dir.glob("*.jpg"))
			paths = jpeg_paths + jpg_paths

			for image_path in paths:
				self.image_paths.append(image_path)
				self.labels.append(label)

	def __len__(self) -> int:
		"""Return the total number of samples in the dataset."""
		return len(self.image_paths)

	def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, str]:
		"""Load one sample and return image tensor, label, and file path."""
		image_path = self.image_paths[idx]
		label = self.labels[idx]

		image = Image.open(image_path).convert("RGB")

		if self.transform is not None:
			image = self.transform(image)

		return image, label, str(image_path)


def get_transforms(split: str, img_size: int = 224) -> transforms.Compose:
	"""Create augmentation pipelines for training or deterministic evaluation."""
	mean = [0.485, 0.456, 0.406]
	std = [0.229, 0.224, 0.225]

	split = split.lower()
	if split == "train":
		transform_list = [
			transforms.Resize((img_size + 32, img_size + 32)),
			transforms.RandomCrop(img_size),
			transforms.RandomHorizontalFlip(p=0.5),
			transforms.RandomRotation(degrees=15),
			transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.1),
			transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
			transforms.ToTensor(),
			transforms.Normalize(mean=mean, std=std),
		]
	else:
		transform_list = [
			transforms.Resize((img_size + 32, img_size + 32)),
			transforms.CenterCrop(img_size),
			transforms.ToTensor(),
			transforms.Normalize(mean=mean, std=std),
		]

	return transforms.Compose(transform_list)


def get_dataloaders(data_dir: str, batch_size: int = 32, num_workers: int = 2):
	"""Build train/val/test dataloaders and compute class weights from training labels."""
	train_dataset = ChestXrayDataset(
		root_dir=data_dir,
		split="train",
		transform=get_transforms("train"),
	)
	val_dataset = ChestXrayDataset(
		root_dir=data_dir,
		split="val",
		transform=get_transforms("val"),
	)
	test_dataset = ChestXrayDataset(
		root_dir=data_dir,
		split="test",
		transform=get_transforms("test"),
	)

	train_loader = DataLoader(
		train_dataset,
		batch_size=batch_size,
		shuffle=True,
		num_workers=num_workers,
		pin_memory=True,
	)
	val_loader = DataLoader(
		val_dataset,
		batch_size=batch_size,
		shuffle=False,
		num_workers=num_workers,
		pin_memory=True,
	)
	test_loader = DataLoader(
		test_dataset,
		batch_size=batch_size,
		shuffle=False,
		num_workers=num_workers,
		pin_memory=True,
	)

	num_classes = len(ChestXrayDataset.class_to_idx)
	labels_tensor = torch.tensor(train_dataset.labels, dtype=torch.long)
	counts = torch.bincount(labels_tensor, minlength=num_classes).float()
	total_samples = float(len(train_dataset))

	class_weights = torch.zeros(num_classes, dtype=torch.float32)
	nonzero_mask = counts > 0
	class_weights[nonzero_mask] = total_samples / (num_classes * counts[nonzero_mask])

	return train_loader, val_loader, test_loader, class_weights


def show_batch(loader, class_names, n: int = 8) -> None:
	"""Display and save a denormalized image grid from a dataloader batch."""
	images, labels, _ = next(iter(loader))

	n = min(n, images.size(0))
	cols = min(4, n)
	rows = math.ceil(n / cols)

	mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
	std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

	figure = plt.figure(figsize=(4 * cols, 4 * rows))

	for i in range(n):
		ax = figure.add_subplot(rows, cols, i + 1)
		image = images[i].cpu() * std + mean
		image = image.clamp(0, 1).permute(1, 2, 0).numpy()

		ax.imshow(image)
		ax.set_title(class_names[labels[i].item()])
		ax.axis("off")

	output_path = PROJECT_ROOT / "outputs" / "plots" / "sample_batch.png"
	output_path.parent.mkdir(parents=True, exist_ok=True)

	plt.tight_layout()
	plt.savefig(output_path, dpi=150)
	plt.close(figure)
	print(f"Saved sample batch plot to: {output_path}")


if __name__ == "__main__":
	train_loader, val_loader, test_loader, class_weights = get_dataloaders("data/raw/chest_xray")

	class_names = ["NORMAL", "PNEUMONIA"]

	for split_name, loader in [
		("train", train_loader),
		("val", val_loader),
		("test", test_loader),
	]:
		labels = torch.tensor(loader.dataset.labels, dtype=torch.long)
		counts = torch.bincount(labels, minlength=len(class_names)).tolist()
		distribution = {class_names[i]: int(counts[i]) for i in range(len(class_names))}
		print(f"{split_name} class distribution: {distribution}")

	batch_images, batch_labels, batch_paths = next(iter(train_loader))
	print(f"Batch image tensor shape: {batch_images.shape}")
	print(f"Batch label tensor shape: {batch_labels.shape}")
	print(f"First sample path: {batch_paths[0]}")
	print(f"Class weights: {class_weights}")

	show_batch(train_loader, class_names, n=8)
