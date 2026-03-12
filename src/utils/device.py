"""Utility functions for device management.

This module is intentionally lightweight; seeding helpers have been moved to
:mod:`utils.seeding` to avoid duplicate definitions.
"""

import os
from typing import Dict

import torch


def get_device() -> str:
    """Return the best available device for computation.

    Checks for CUDA first, then Apple's Metal Performance Shaders (MPS), and
    falls back to the CPU.
    """
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


def get_device_info() -> Dict[str, object]:
    """Gather a small dictionary of device-related information.

    Returns:
        A dictionary describing the selected device and relevant metadata.
    """
    device = get_device()
    info = {"device": device}
    
    if device == "cuda":
        info.update({
            "cuda_available": True,
            "cuda_version": torch.version.cuda,
            "gpu_count": torch.cuda.device_count(),
            "current_gpu": torch.cuda.current_device(),
            "gpu_name": torch.cuda.get_device_name(),
        })
    elif device == "mps":
        info.update({
            "mps_available": True,
            "mps_version": torch.backends.mps.version(),
        })
    else:
        info.update({
            "cpu_only": True,
            "cpu_count": os.cpu_count(),
        })
    
    return info
