"""Training script for chest X-ray binary classification."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
	sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset import get_dataloaders
from src.model import build_model, get_optimizer, unfreeze_backbone
from src.utils import AverageMeter, load_config, set_seed


def _train_one_epoch(model, loader, criterion, optimizer, device, epoch: int, num_epochs: int):
	"""Run one training epoch and return average loss and accuracy."""
	model.train()
	loss_meter = AverageMeter()

	correct = 0
	total = 0

	for batch_idx, (images, labels, _) in enumerate(loader):
		print(f"Epoch {epoch+1}/{num_epochs} - Batch {batch_idx+1}/{len(loader)} running...")
		images = images.to(device, non_blocking=True)
		labels = labels.to(device, non_blocking=True)

		optimizer.zero_grad()
		outputs = model(images)
		loss = criterion(outputs, labels)
		loss.backward()
		torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
		optimizer.step()

		preds = torch.argmax(outputs, dim=1)
		batch_size = labels.size(0)
		loss_meter.update(loss.item(), batch_size)
		correct += (preds == labels).sum().item()
		total += batch_size

	train_acc = correct / total if total > 0 else 0.0
	return loss_meter.avg, train_acc


def _validate_one_epoch(model, loader, criterion, device):
	"""Run one validation epoch and return loss, accuracy, AUC, and weighted F1."""
	model.eval()
	loss_meter = AverageMeter()

	all_labels = []
	all_preds = []
	all_probs = []
	all_logits = []

	with torch.no_grad():
		for images, labels, _ in loader:
			images = images.to(device, non_blocking=True)
			labels = labels.to(device, non_blocking=True)

			outputs = model(images)
			loss = criterion(outputs, labels)
			probs = torch.softmax(outputs, dim=1)[:, 1]
			preds = torch.argmax(outputs, dim=1)
			loss_meter.update(loss.item(), labels.size(0))

			all_logits.append(outputs.detach().cpu())
			all_labels.append(labels.detach().cpu())
			all_preds.append(preds.detach().cpu())
			all_probs.append(probs.detach().cpu())

	labels_tensor = torch.cat(all_labels)
	preds_tensor = torch.cat(all_preds)
	probs_tensor = torch.cat(all_probs)
	_ = torch.cat(all_logits)

	val_loss = loss_meter.avg
	val_acc = (preds_tensor == labels_tensor).float().mean().item()

	labels_np = labels_tensor.numpy()
	preds_np = preds_tensor.numpy()
	probs_np = probs_tensor.numpy()

	try:
		val_auc = roc_auc_score(labels_np, probs_np)
	except ValueError:
		val_auc = 0.5

	val_f1 = f1_score(labels_np, preds_np, average="weighted")
	return val_loss, val_acc, val_auc, val_f1


def _plot_history(history, output_path: Path, phase2_start_epoch: int) -> None:
	"""Create and save training curves with a phase-2 marker."""
	epochs = list(range(1, len(history["train_loss"]) + 1))

	fig, axes = plt.subplots(2, 2, figsize=(14, 10))
	ax1, ax2, ax3, ax4 = axes.flatten()

	ax1.plot(epochs, history["train_loss"], label="Train Loss")
	ax1.plot(epochs, history["val_loss"], label="Val Loss")
	ax1.axvline(x=phase2_start_epoch, color="k", linestyle="--", linewidth=1)
	ax1.set_title("Loss")
	ax1.set_xlabel("Epoch")
	ax1.set_ylabel("Loss")
	ax1.legend()

	ax2.plot(epochs, history["train_acc"], label="Train Acc")
	ax2.plot(epochs, history["val_acc"], label="Val Acc")
	ax2.axvline(x=phase2_start_epoch, color="k", linestyle="--", linewidth=1)
	ax2.set_title("Accuracy")
	ax2.set_xlabel("Epoch")
	ax2.set_ylabel("Accuracy")
	ax2.legend()

	ax3.plot(epochs, history["val_auc"], label="Val AUC")
	ax3.axvline(x=phase2_start_epoch, color="k", linestyle="--", linewidth=1)
	ax3.set_title("Validation AUC")
	ax3.set_xlabel("Epoch")
	ax3.set_ylabel("AUC")
	ax3.legend()

	ax4.plot(epochs, history["val_f1"], label="Val F1")
	ax4.axvline(x=phase2_start_epoch, color="k", linestyle="--", linewidth=1)
	ax4.set_title("Validation F1")
	ax4.set_xlabel("Epoch")
	ax4.set_ylabel("F1")
	ax4.legend()

	fig.tight_layout()
	output_path.parent.mkdir(parents=True, exist_ok=True)
	fig.savefig(output_path, dpi=150)
	plt.close(fig)


def main(fresh: bool = False) -> None:
	"""Run full two-phase training, evaluation logging, and artifact saving.
	
	Args:
		fresh: If True, delete existing checkpoints/plots/results and start from epoch 1.
	"""
	config = load_config("config.yaml")
	set_seed(42)
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	print("Using device:", device)

	data_cfg = config["data"]
	model_cfg = config["model"]
	train_cfg = config["training"]
	paths_cfg = config["paths"]

	checkpoint_dir = PROJECT_ROOT / paths_cfg["checkpoint_dir"]
	plots_dir = PROJECT_ROOT / paths_cfg["plots_dir"]
	results_dir = PROJECT_ROOT / paths_cfg["results_dir"]

	if fresh:
		print("Starting fresh training (ignoring checkpoints)")
		for directory in [checkpoint_dir, plots_dir, results_dir]:
			if directory.exists():
				shutil.rmtree(directory)

	checkpoint_dir.mkdir(parents=True, exist_ok=True)
	plots_dir.mkdir(parents=True, exist_ok=True)
	results_dir.mkdir(parents=True, exist_ok=True)

	train_loader, val_loader, test_loader, class_weights = get_dataloaders(
		data_dir=data_cfg["data_dir"],
		batch_size=data_cfg["batch_size"],
		num_workers=data_cfg["num_workers"],
	)
	_ = test_loader

	model = build_model(
		architecture=model_cfg["architecture"],
		num_classes=data_cfg["num_classes"],
		pretrained=model_cfg["pretrained"],
		dropout=model_cfg["dropout"],
		freeze_backbone=True,
	).to(device)

	base_lr = float(train_cfg["learning_rate"])
	weight_decay = float(train_cfg["weight_decay"])
	optimizer = get_optimizer(model, lr=base_lr, weight_decay=weight_decay)

	criterion = nn.CrossEntropyLoss(
		weight=class_weights.to(device),
		label_smoothing=float(train_cfg["label_smoothing"]),
	)

	history = {
		"train_loss": [],
		"train_acc": [],
		"val_loss": [],
		"val_acc": [],
		"val_auc": [],
		"val_f1": [],
	}

	best_auc = -1.0
	epochs_no_improve = 0
	phase2_start_epoch = 11
	total_epochs = int(train_cfg["epochs"])
	start_epoch = 1

	if not fresh:
		last_checkpoint_path = checkpoint_dir / "last_model.pth"
		if last_checkpoint_path.exists():
			checkpoint = torch.load(last_checkpoint_path, map_location=device)
			model.load_state_dict(checkpoint["model_state_dict"])
			start_epoch = int(checkpoint["epoch"]) + 1
			best_auc = float(checkpoint.get("val_auc", best_auc))

			if start_epoch >= phase2_start_epoch:
				unfreeze_backbone(model, architecture=model_cfg["architecture"])
				optimizer = get_optimizer(model, lr=base_lr * 0.1, weight_decay=weight_decay)
			else:
				optimizer = get_optimizer(model, lr=base_lr, weight_decay=weight_decay)

			try:
				optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
			except ValueError:
				print("Warning: Optimizer state mismatch. Continuing with fresh optimizer state.")

			print(f"Resuming training from epoch {start_epoch}/{total_epochs}")

	history_path = results_dir / "history.json"
	if not fresh and history_path.exists():
		with open(history_path, "r", encoding="utf-8") as file:
			loaded_history = json.load(file)
		for key in history:
			if key in loaded_history and isinstance(loaded_history[key], list):
				history[key] = loaded_history[key]

		if history["val_auc"]:
			best_auc = max(best_auc, max(history["val_auc"]))

	if start_epoch > total_epochs:
		print(f"Training already completed up to epoch {start_epoch - 1}. Nothing to run.")
		return

	scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
		optimizer,
		mode="max",
		patience=int(train_cfg["lr_patience"]),
		factor=float(train_cfg["lr_factor"]),
		min_lr=1e-7,
	)

	start_time = time.time()

	for epoch in range(start_epoch, total_epochs + 1):
		if epoch == phase2_start_epoch:
			print("--- Phase 2: Fine-tuning entire network ---")
			unfreeze_backbone(model, architecture=model_cfg["architecture"])
			optimizer = get_optimizer(model, lr=base_lr * 0.1, weight_decay=weight_decay)
			scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
				optimizer,
				mode="max",
				patience=int(train_cfg["lr_patience"]),
				factor=float(train_cfg["lr_factor"]),
				min_lr=1e-7,
			)

		train_loss, train_acc = _train_one_epoch(
			model,
			train_loader,
			criterion,
			optimizer,
			device,
			epoch - 1,
			total_epochs,
		)
		val_loss, val_acc, val_auc, val_f1 = _validate_one_epoch(model, val_loader, criterion, device)
		scheduler.step(val_auc)

		history["train_loss"].append(train_loss)
		history["train_acc"].append(train_acc)
		history["val_loss"].append(val_loss)
		history["val_acc"].append(val_acc)
		history["val_auc"].append(val_auc)
		history["val_f1"].append(val_f1)

		current_lr = max(group["lr"] for group in optimizer.param_groups)
		print(
			f"Epoch {epoch:02d}/{total_epochs} | "
			f"Train Loss: {train_loss:.4f} | "
			f"Train Acc: {train_acc * 100:.1f}% | "
			f"Val Loss: {val_loss:.4f} | "
			f"Val Acc: {val_acc * 100:.1f}% | "
			f"Val AUC: {val_auc:.4f} | "
			f"Val F1: {val_f1:.3f} | "
			f"LR: {current_lr:.6f}"
		)

		if val_auc > best_auc:
			best_auc = val_auc
			epochs_no_improve = 0
			torch.save(
				{
					"epoch": epoch,
					"model_state_dict": model.state_dict(),
					"optimizer_state_dict": optimizer.state_dict(),
					"val_auc": best_auc,
					"config": config,
				},
				checkpoint_dir / "best_model.pth",
			)
		else:
			epochs_no_improve += 1

		torch.save(
			{
				"epoch": epoch,
				"model_state_dict": model.state_dict(),
				"optimizer_state_dict": optimizer.state_dict(),
				"val_auc": val_auc,
				"config": config,
			},
			checkpoint_dir / "last_model.pth",
		)

		if epochs_no_improve >= int(train_cfg["early_stop_patience"]):
			print(f"Early stopping triggered at epoch {epoch}. Best val AUC: {best_auc:.4f}")
			break

	with open(history_path, "w", encoding="utf-8") as file:
		json.dump(history, file, indent=2)

	curves_path = plots_dir / "training_curves.png"
	_plot_history(history, curves_path, phase2_start_epoch=phase2_start_epoch)

	total_minutes = (time.time() - start_time) / 60.0
	print(f"Total training time: {total_minutes:.2f} minutes")


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="Train chest X-ray classification model.")
	parser.add_argument(
		"--fresh",
		action="store_true",
		help="Delete existing checkpoints/plots/results and start training from scratch.",
	)
	args = parser.parse_args()
	main(fresh=args.fresh)
