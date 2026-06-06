"""
Grad-CAM (Gradient-weighted Class Activation Mapping) implementation from scratch.
Uses PyTorch hooks to visualize model predictions on X-ray images.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image
from src.utils import load_config, get_device
from src.dataset import get_dataloaders
from src.model import build_model


class GradCAM:
    """
    Grad-CAM: Gradient-weighted Class Activation Mapping
    Captures feature maps and gradients via PyTorch hooks to generate attention maps.
    """
    
    def __init__(self, model: nn.Module, target_layer_name: str):
        """
        Initialize Grad-CAM with a model and target layer.
        
        Args:
            model: PyTorch model (nn.Module)
            target_layer_name: Name of layer to hook (e.g., 'features.denseblock4')
        """
        self.model = model
        self.target_layer_name = target_layer_name
        self.feature_maps = None
        self.gradients = None
        
        # Find and hook the target layer
        self._find_and_hook_layer()
    
    def _find_and_hook_layer(self):
        """Find target layer by name and register forward/backward hooks."""
        target_layer = None
        for name, module in self.model.named_modules():
            if name == self.target_layer_name:
                target_layer = module
                break
        
        if target_layer is None:
            raise ValueError(f"Layer '{self.target_layer_name}' not found in model")
        
        # Forward hook: save feature maps
        def forward_hook(module, input, output):
            self.feature_maps = output.detach()
        
        # Backward hook: save gradients
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()
        
        target_layer.register_forward_hook(forward_hook)
        target_layer.register_full_backward_hook(backward_hook)
    
    def generate(self, input_tensor: torch.Tensor, class_idx: int = None) -> np.ndarray:
        """
        Generate Grad-CAM heatmap for input tensor.
        
        Args:
            input_tensor: Input image tensor, shape [1, 3, 224, 224]
            class_idx: Target class index. If None, use argmax prediction.
        
        Returns:
            Grad-CAM heatmap as numpy array, shape [224, 224], values in [0, 1]
        """
        # Forward pass
        self.model.eval()
        # Enable gradients for input even in eval mode
        input_tensor = input_tensor.clone().detach().requires_grad_(True)
        
        with torch.enable_grad():
            logits = self.model(input_tensor)  # [1, num_classes]
        
        # Determine target class
        if class_idx is None:
            class_idx = logits.argmax(dim=1).item()
        
        # Zero gradients
        self.model.zero_grad()
        
        # Backward pass
        target_logit = logits[0, class_idx]
        target_logit.backward()
        
        # Compute Grad-CAM
        gradients = self.gradients[0]  # [C, H, W]
        feature_maps = self.feature_maps[0]  # [C, H, W]
        
        # Global average pooling of gradients
        weights = gradients.mean(dim=(1, 2))  # [C]
        
        # Weighted sum of feature maps
        cam = (weights[:, None, None] * feature_maps).sum(dim=0)  # [H, W]
        
        # ReLU and normalize
        cam = F.relu(cam)
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        
        # Convert to numpy and resize
        cam = cam.detach().cpu().numpy()
        cam = cv2.resize(cam, (224, 224))
        
        return cam.astype(np.float32)


def overlay_heatmap(original_pil: Image.Image, heatmap: np.ndarray, alpha: float = 0.5) -> Image.Image:
    """
    Overlay Grad-CAM heatmap on original image.
    
    Args:
        original_pil: Original image as PIL Image
        heatmap: Grad-CAM heatmap, shape [H, W], values in [0, 1]
        alpha: Blending factor (0-1), higher = more heatmap visible
    
    Returns:
        Blended PIL Image
    """
    # Resize original to 224x224
    original_pil = original_pil.resize((224, 224))
    original_rgb = np.array(original_pil.convert('RGB'), dtype=np.uint8)
    
    # Convert heatmap to uint8 and apply colormap
    heatmap_uint8 = (heatmap * 255).astype(np.uint8)
    colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)  # BGR
    colored_rgb = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
    
    # Blend images
    blended = (alpha * colored_rgb + (1 - alpha) * original_rgb).astype(np.uint8)
    
    return Image.fromarray(blended)


def visualize_gradcam(model, test_loader, device, n=8, save_path='outputs/plots/gradcam_results.png'):
    """
    Visualize Grad-CAM for correct and incorrect predictions.
    
    Args:
        model: Trained PyTorch model
        test_loader: Test data loader
        device: Device (cuda or cpu)
        n: Number of images per category (total = 2*n)
        save_path: Path to save visualization
    """
    # Initialize Grad-CAM
    gradcam = GradCAM(model, target_layer_name='features.denseblock4')
    model.to(device)
    model.eval()
    
    correct_images = []
    incorrect_images = []
    correct_data = []
    incorrect_data = []
    
    # Collect n correct and n incorrect predictions
    with torch.no_grad():
        for images, labels, filepaths in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            preds = outputs.argmax(dim=1)
            confs = torch.softmax(outputs, dim=1).max(dim=1)[0]
            
            for i in range(images.size(0)):
                is_correct = preds[i] == labels[i]
                conf_pct = confs[i].item() * 100
                
                if is_correct and len(correct_images) < n:
                    correct_images.append(images[i:i+1])
                    correct_data.append({
                        'true': labels[i].item(),
                        'pred': preds[i].item(),
                        'conf': conf_pct,
                        'filepath': filepaths[i]
                    })
                
                elif not is_correct and len(incorrect_images) < n:
                    incorrect_images.append(images[i:i+1])
                    incorrect_data.append({
                        'true': labels[i].item(),
                        'pred': preds[i].item(),
                        'conf': conf_pct,
                        'filepath': filepaths[i]
                    })
                
                if len(correct_images) >= n and len(incorrect_images) >= n:
                    break
            
            if len(correct_images) >= n and len(incorrect_images) >= n:
                break
    
    # Combine all images and metadata
    all_images = correct_images + incorrect_images
    all_data = correct_data + incorrect_data
    all_correct = [True] * len(correct_images) + [False] * len(incorrect_images)
    
    # Generate Grad-CAM visualizations
    original_images = []
    gradcam_images = []
    class_names = ['NORMAL', 'PNEUMONIA']
    
    for input_tensor, metadata, is_correct in zip(all_images, all_data, all_correct):
        # Generate Grad-CAM (requires gradients, so no no_grad context)
        heatmap = gradcam.generate(input_tensor.clone(), class_idx=metadata['pred'])
        
        # Load original image
        original_pil = Image.open(metadata['filepath']).convert('RGB')
        
        # Create overlaid image
        overlaid_pil = overlay_heatmap(original_pil, heatmap, alpha=0.5)
        
        original_images.append(original_pil)
        gradcam_images.append(overlaid_pil)
    
    # Create visualization figure
    fig, axes = plt.subplots(2, n, figsize=(20, 8))
    
    for col in range(n):
        # Row 1: Original images with titles
        ax = axes[0, col]
        ax.imshow(original_images[col])
        ax.axis('off')
        
        metadata = all_data[col]
        is_correct = all_correct[col]
        true_name = class_names[metadata['true']]
        pred_name = class_names[metadata['pred']]
        conf_pct = metadata['conf']
        
        title = f"True: {true_name}\nPred: {pred_name}\nConf: {conf_pct:.1f}%"
        title_color = 'green' if is_correct else 'red'
        ax.set_title(title, color=title_color, fontsize=10, fontweight='bold')
        
        # Row 2: Grad-CAM overlaid images
        ax = axes[1, col]
        ax.imshow(gradcam_images[col])
        ax.axis('off')
        ax.set_title('Grad-CAM', fontsize=10)
    
    # Add legend
    correct_patch = mpatches.Patch(color='green', label='Correct')
    incorrect_patch = mpatches.Patch(color='red', label='Incorrect')
    fig.legend(handles=[correct_patch, incorrect_patch], loc='lower center', ncol=2, bbox_to_anchor=(0.5, -0.02))
    
    plt.tight_layout()
    
    # Create save directory
    save_path_obj = Path(save_path)
    save_path_obj.parent.mkdir(parents=True, exist_ok=True)
    
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Grad-CAM visualization saved to {save_path}")
    plt.close()


if __name__ == '__main__':
    # Load configuration
    config = load_config('config.yaml')
    
    # Get device
    device = get_device()
    
    # Build and load model
    model = build_model(
        architecture=config['model']['architecture'],
        num_classes=config['data']['num_classes'],
        pretrained=True,
        dropout=config['model']['dropout'],
        freeze_backbone=False
    )
    
    # Load checkpoint
    checkpoint_path = Path('outputs/results/best_model.pth')
    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded checkpoint from epoch {checkpoint['epoch']} (val_auc: {checkpoint['val_auc']:.4f})")
    else:
        print(f"Warning: Checkpoint not found at {checkpoint_path}")
    
    model = model.to(device)
    
    # Load test data
    
    _, _, test_loader, _ = get_dataloaders(
        data_dir=config['data']['data_dir'],
        batch_size=config['data']['batch_size'],
        num_workers=config['data']['num_workers']
    )
    # Generate Grad-CAM visualization
    visualize_gradcam(model, test_loader, device, n=4, save_path='outputs/plots/gradcam_results.png')
