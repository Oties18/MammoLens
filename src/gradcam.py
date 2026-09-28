"""Grad-CAM visualization.

Produces a heatmap of which regions drove the model's prediction. For a
medical model this is essential: it lets you sanity-check that the network
attends to the lesion rather than to imaging artifacts or text labels.

Minimal self-contained implementation (no extra dependency) using forward
and backward hooks on the last convolutional feature map.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model.eval()
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, inp, out):
        self.activations = out.detach()

    def _save_gradient(self, module, grad_in, grad_out):
        self.gradients = grad_out[0].detach()

    def __call__(self, input_tensor, class_idx=None):
        """input_tensor: (1, C, H, W). Returns a HxW heatmap in [0, 1]."""
        logits = self.model(input_tensor)
        if class_idx is None:
            class_idx = int(logits.argmax(dim=1))
        self.model.zero_grad()
        logits[0, class_idx].backward()

        # Global-average-pool the gradients to get per-channel weights.
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(
            cam, size=input_tensor.shape[2:],
            mode="bilinear", align_corners=False,
        )
        cam = cam.squeeze().cpu().numpy()
        cam -= cam.min()
        if cam.max() > 0:
            cam /= cam.max()
        return cam


def last_conv_layer(model):
    """Best-effort lookup of the final conv layer for a timm EfficientNet."""
    # timm EfficientNet exposes the final 1x1 conv as `conv_head`.
    if hasattr(model, "conv_head"):
        return model.conv_head
    # Fallback: last nn.Conv2d in the module tree.
    conv = None
    for module in model.modules():
        if isinstance(module, torch.nn.Conv2d):
            conv = module
    return conv


def overlay_heatmap(image_np, cam, alpha=0.5):
    """Blend a [0,1] heatmap over an RGB image (both HxWx3, uint8/float)."""
    import matplotlib.cm as cm

    heatmap = cm.jet(cam)[..., :3]  # drop alpha channel
    image = np.asarray(image_np, dtype=np.float32) / 255.0
    blended = (1 - alpha) * image + alpha * heatmap
    return np.clip(blended, 0, 1)
