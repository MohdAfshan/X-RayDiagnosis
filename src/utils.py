"""Utility helpers for configuration, reproducibility, and training metrics."""

import random
from pathlib import Path

import numpy as np
import torch
import yaml


def load_config(config_path: str) -> dict:
	"""Load a YAML configuration file and return it as a dictionary."""
	with open(config_path, "r", encoding="utf-8") as file:
		return yaml.safe_load(file)


def set_seed(seed: int = 42) -> None:
	"""Set random seeds for reproducible results across supported libraries."""
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
	"""Return the active torch device and print device details."""
	if torch.cuda.is_available():
		device = torch.device("cuda")
		gpu_name = torch.cuda.get_device_name(0)
		print(f"Using device: {device} ({gpu_name})")
	else:
		device = torch.device("cpu")
		print(f"Using device: {device}")
	return device


class AverageMeter:
	"""Track and update running average statistics for a metric."""

	def __init__(self) -> None:
		"""Initialize meter values."""
		self.reset()

	def reset(self) -> None:
		"""Reset all tracked statistics to zero."""
		self.val = 0.0
		self.avg = 0.0
		self.sum = 0.0
		self.count = 0

	def update(self, val: float, n: int = 1) -> None:
		"""Update running statistics with a new value and sample count."""
		self.val = val
		self.sum += val * n
		self.count += n
		self.avg = self.sum / self.count if self.count != 0 else 0.0


if __name__ == "__main__":
	config_file = Path(__file__).resolve().parents[1] / "config.yaml"
	config = load_config(str(config_file))
	print(config)
	get_device()
