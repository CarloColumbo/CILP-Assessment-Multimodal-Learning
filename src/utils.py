import os

# Set environment variables for reproducibility BEFORE importing torch
os.environ['PYTHONHASHSEED'] = '51'
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

import numpy as np
import torch
import random
from typing import Callable, Tuple, List, Any

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seeds(seed=51):
    """
    Set seeds for complete reproducibility across all libraries and operations.

    Args:
        seed (int): Random seed value
    """
    # Set environment variables before other imports
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

    # Python random module
    random.seed(seed)

    # NumPy
    np.random.seed(seed)

    # PyTorch CPU
    torch.manual_seed(seed)

    # PyTorch GPU (all devices)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # For multi-GPU setups

        # CUDA deterministic operations
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # PyTorch deterministic algorithms (may impact performance)
    try:
        torch.use_deterministic_algorithms(True)
    except RuntimeError:
        # Some operations don't have deterministic implementations
        print("Warning: Some operations may not be deterministic")

    print(f"All random seeds set to {seed} for reproducibility")


def create_deterministic_training_dataloader(dataset, batch_size, shuffle=True, **kwargs):
    """
    Create a DataLoader with deterministic behavior.

    Args:
        dataset: PyTorch Dataset instance
        batch_size: Batch size
        shuffle: Whether to shuffle data
        **kwargs: Additional DataLoader arguments

    Returns:
        Training DataLoader with reproducible behavior
    """
    # Use a generator with fixed seed for reproducible shuffling
    generator = torch.Generator()
    generator.manual_seed(51)

    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator if shuffle else None,
        **kwargs
    )


def get_xyza(lidar_depth, azimuth, zenith):
    """
    Convert LiDAR polar coordinates to Cartesian coordinates (x, y, z) and a binary mask (a).

    Args:
        lidar_depth (np.ndarray): Depth values from LiDAR
        azimuth (np.ndarray): Azimuth angles
        zenith (np.ndarray): Zenith angles

    Returns:
        np.ndarray: Stacked array of shape (4, H, W) containing x, y, z, and a
    """
    x = lidar_depth * np.sin(-azimuth[:, None]) * np.cos(-zenith[None, :])
    y = lidar_depth * np.cos(-azimuth[:, None]) * np.cos(-zenith[None, :])
    z = lidar_depth * np.sin(-zenith[None, :])
    a = np.where(lidar_depth < 50.0, np.ones_like(lidar_depth), np.zeros_like(lidar_depth))
    xyza = np.stack((x, y, z, a))
    return xyza


def infer_model(
        model: torch.nn.Module,
        dataloader: torch.utils.data.DataLoader,
        input_fn: Callable[[Any], Tuple[torch.Tensor, ...]],
    ) -> tuple[float, List[np.ndarray]]:
    """
    Inference loop for a given model. We assume binary classification with
    one output neuron.

    Args:
        model (torch.nn.Module): The model to use for inference.
        dataloader (torch.utils.data.DataLoader): The data loader for inference data.
        input_fn (Callable): A function that takes a batch and returns the model inputs.

    Returns:
        tuple[float, List[np.ndarray]]: Outputs and groud truths.
    """
    model.eval()
    outputs = []
    ground_truth = []

    with torch.no_grad():
        for batch in dataloader:
            logits = model(*input_fn(batch))
            probs = torch.sigmoid(logits).squeeze(1)
            preds = (probs >= 0.5).long().cpu().numpy()
            labels = batch[2].squeeze(1).long().cpu().numpy()

            outputs.extend(preds)
            ground_truth.extend(labels)

    outputs = np.vstack(outputs)
    ground_truth = np.vstack(ground_truth)

    return outputs, ground_truth


def get_rgb_input(batch):
    """
    Get RGB input from the batch.

    Args:
        batch: A batch of data from the DataLoader. It is expected to be a list or tuple where:
            - batch[0]: RGB inputs
    Returns:
        A list containing RGB inputs moved to the specified device.
    """
    return [batch[0].to(device)]


def get_lidar_input(batch):
    """
    Get Lidar input from the batch.

    Args:
        batch: A batch of data from the DataLoader. It is expected to be a list or tuple where:
            - batch[1]: Lidar inputs
    Returns:
        A list containing Lidar inputs moved to the specified device.
    """
    return [batch[1].to(device)]


def get_mm_intermediate_inputs(batch):
    """
    Get intermediate inputs for multi-modal fusion.
    
    Args:
        batch: A batch of data from the DataLoader. It is expected to be a list or tuple where:
            - batch[0]: RGB inputs
            - batch[1]: XYZ inputs
    
    Returns:
        A list containing RGB and XYZ inputs moved to the specified device.
    """
    inputs_rgb = batch[0].to(device)
    inputs_xyz = batch[1].to(device)
    return [inputs_rgb, inputs_xyz]


def get_mm_late_inputs(batch):
    """
    Get late fusion inputs for multi-modal fusion.

    Args:
        batch: A batch of data from the DataLoader. It is expected to be a list or tuple where:
            - batch[0]: RGB inputs
            - batch[1]: XYZ inputs

    Returns:
        A list containing RGB and XYZ inputs moved to the specified device.
    """
    inputs_rgb = batch[0].to(device)
    inputs_xyz = batch[1].to(device)
    return [inputs_rgb, inputs_xyz]
