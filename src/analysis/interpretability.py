import torch
import torch.nn.functional as F
import numpy as np
from scipy.ndimage import gaussian_filter

def smooth_heatmap_gaussian(heatmap, kernel_size=11, sigma=2):
    """Applies 1D Gaussian smoothing to a heatmap."""
    # Create 1D Gaussian kernel
    x = torch.arange(kernel_size) - kernel_size // 2
    gauss = torch.exp(-x**2 / (2 * sigma**2))
    gauss = gauss / gauss.sum()

    # Reshape for conv1d: (out_channels, in_channels, kernel_size)
    gauss = gauss.view(1, 1, -1).to(heatmap.device)

    # Add batch/channel dimensions
    heatmap = heatmap.view(1, 1, -1)

    # Apply smoothing
    smoothed = F.conv1d(heatmap, gauss, padding=kernel_size // 2)
    return smoothed.view(-1).cpu().numpy()



def compute_gradcam_1d(model, input_tensor, target_layer, target_class, smooth=True):
    model.eval()
    activations = []
    gradients = []

    def forward_hook(module, input, output):
        activations.append(output)

    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0])

    # Register hooks
    handle_forward = target_layer.register_forward_hook(forward_hook)
    handle_backward = target_layer.register_full_backward_hook(backward_hook)

    # Forward and backward
    output = model(input_tensor)
    model.zero_grad()
    output[0, target_class].backward()

    # Collect
    act = activations[0].detach()  # shape: (1, C, L, 1)
    grad = gradients[0].detach()   # shape: (1, C, L, 1)
    weights = grad.mean(dim=2, keepdim=True)
    cam = (weights * act).sum(dim=1)  # shape: (1, L, 1)
    cam = torch.relu(cam).squeeze()

    # Normalize
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

    if smooth:
        cam = smooth_heatmap_gaussian(cam)

    # Cleanup hooks
    handle_forward.remove()
    handle_backward.remove()

    return cam

def compute_gradcam(model, input_tensor, target_layer):
    # Set model to evaluation mode
    model.eval()

    # Define hooks for activations and gradients
    activations = None
    gradients = None

    def forward_hook(module, input, output):
        nonlocal activations
        activations = output

    def backward_hook(module, grad_input, grad_output):
        nonlocal gradients
        gradients = grad_output[0]

    # Register hooks
    forward_handle = target_layer.register_forward_hook(forward_hook)
    backward_handle = target_layer.register_full_backward_hook(backward_hook)

    # Forward pass
    output = model(input_tensor)

    # Backward pass
    model.zero_grad()
    output.backward()

    # Remove hooks
    forward_handle.remove()
    backward_handle.remove()

    # Compute Grad-CAM
    weights = torch.mean(gradients, dim=(2, 3), keepdim=True)
    cam = torch.sum(weights * activations, dim=1, keepdim=True)
    cam = torch.relu(cam)
    cam -= torch.min(cam)
    cam /= torch.max(cam)

    return cam

