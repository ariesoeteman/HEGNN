import gc
import pdb

import torch


def print_active_tensors():
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) and obj.is_cuda:
                size = obj.element_size() * obj.numel() / 1024**2
                if size > 1:
                    print(
                        f"Tensor: {obj.shape}, dtype: {obj.dtype}, device: {obj.device}, memory: {obj.element_size() * obj.numel() / 1024**2:.2f} MB"
                    )
        except Exception as e:
            pass  # Some objects might not be accessible


def print_memory(stage=""):
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        reserved = torch.cuda.memory_reserved() / 1024**2
        print(f"{stage} Reserved: {reserved:.2f} MB")
    else:
        print(f"{stage}: CUDA is not available; no memory print.")
