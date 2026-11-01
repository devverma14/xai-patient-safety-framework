"""
Grad-CAM (Gradient-weighted Class Activation Mapping) implementation for PyTorch CNN architectures.

References:
    Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization",
    IEEE International Conference on Computer Vision (ICCV), 2017.
"""

from typing import Optional, Tuple, Dict, Any
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def get_default_target_layer(model: nn.Module) -> nn.Module:
    """
    Auto-detect the final 2D convolutional layer in a PyTorch CNN model.

    Args:
        model: PyTorch classification model.

    Returns:
        The last instance of nn.Conv2d found in the model module tree.
    """
    last_conv = None
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            last_conv = module

    if last_conv is None:
        raise ValueError("No nn.Conv2d layer found in the provided model for Grad-CAM target layer.")

    return last_conv


class GradCAM:
    """
    Computes Grad-CAM heatmaps for a given PyTorch CNN model and target layer.
    """

    def __init__(
        self,
        model: nn.Module,
        target_layer: Optional[nn.Module] = None,
        device: Optional[torch.device] = None,
    ):
        """
        Initialize GradCAM explainer.

        Args:
            model: PyTorch model (e.g. DenseNet121).
            target_layer: Specific target convolutional layer. If None, auto-detects final Conv2d layer.
            device: Computing device (cpu or cuda). If None, uses model's current device.
        """
        self.model = model
        self.model.eval()

        if target_layer is None:
            self.target_layer = get_default_target_layer(self.model)
        else:
            self.target_layer = target_layer

        if device is None:
            # Infer device from model parameters
            first_param = next(self.model.parameters(), None)
            self.device = first_param.device if first_param is not None else torch.device("cpu")
        else:
            self.device = device

        self.activations: Optional[torch.Tensor] = None
        self.gradients: Optional[torch.Tensor] = None
        self.forward_handle = None
        self.backward_handle = None

        self._register_hooks()

    def _register_hooks(self):
        """Register forward and backward hooks on the target convolutional layer."""

        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_in, grad_out):
            # grad_out is a tuple containing gradients w.r.t layer output
            self.gradients = grad_out[0].detach()

        self.forward_handle = self.target_layer.register_forward_hook(forward_hook)
        self.backward_handle = self.target_layer.register_full_backward_hook(backward_hook)

    def remove_hooks(self):
        """Remove registered PyTorch hooks to avoid memory leaks."""
        if self.forward_handle is not None:
            self.forward_handle.remove()
            self.forward_handle = None
        if self.backward_handle is not None:
            self.backward_handle.remove()
            self.backward_handle = None

    def generate_heatmap(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None,
    ) -> Tuple[np.ndarray, int, float]:
        """
        Generate normalized Grad-CAM heatmap for an input image tensor.

        Args:
            input_tensor: Image tensor of shape (1, C, H, W) or (C, H, W).
            target_class: Target class index. If None, uses top predicted class index.

        Returns:
            Tuple of:
                - heatmap: 2D numpy array normalized to [0, 1] with shape (H, W).
                - predicted_class: Class index evaluated.
                - class_probability: Softmax probability score for target class.
        """
        if input_tensor.dim() == 3:
            input_tensor = input_tensor.unsqueeze(0)

        if input_tensor.dim() != 4:
            raise ValueError(f"Expected 4D input tensor (N, C, H, W), got shape {input_tensor.shape}")

        batch_size, channels, height, width = input_tensor.shape
        if batch_size != 1:
            raise ValueError(f"Grad-CAM expects batch_size=1, got batch_size={batch_size}")

        input_tensor = input_tensor.to(self.device)
        input_tensor = input_tensor.clone().detach().requires_grad_(True)

        # Forward pass
        self.model.zero_grad()
        outputs = self.model(input_tensor)

        probs = F.softmax(outputs, dim=1)
        if target_class is None:
            target_class = torch.argmax(probs, dim=1).item()

        target_class = int(target_class)
        class_prob = float(probs[0, target_class].item())

        # Target class output score
        score = outputs[0, target_class]
        score.backward(retain_graph=True)

        if self.activations is None or self.gradients is None:
            raise RuntimeError("Failed to capture activations or gradients from target layer.")

        # Compute global average pooled gradients (channel importance weights alpha_k)
        # activations shape: (1, C_feat, H_feat, W_feat)
        # gradients shape:   (1, C_feat, H_feat, W_feat)
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)
        cam = F.relu(cam)

        # Interpolate heatmap to original input resolution (H, W)
        cam_upsampled = F.interpolate(
            cam,
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        )

        heatmap_np = cam_upsampled.squeeze().cpu().numpy()

        # Normalize heatmap to [0, 1] range safely
        max_val = np.max(heatmap_np)
        min_val = np.min(heatmap_np)

        if max_val > min_val:
            heatmap_norm = (heatmap_np - min_val) / (max_val - min_val)
        else:
            heatmap_norm = np.zeros_like(heatmap_np, dtype=np.float32)

        return heatmap_norm.astype(np.float32), target_class, class_prob

    def __del__(self):
        """Cleanup hooks on garbage collection."""
        self.remove_hooks()
