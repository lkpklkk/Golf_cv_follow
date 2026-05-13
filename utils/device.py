def select_torch_device(preferred="auto"):
    """
    Return the best available torch device string for model inference.

    The app should still run without torch import success in modules that only
    need config-time device selection, so import torch lazily here.
    """
    if preferred != "auto":
        return preferred

    try:
        import torch
    except ImportError:
        return "cpu"

    if torch.cuda.is_available():
        return "cuda:0"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def use_half_precision(device, enabled=True):
    return bool(enabled and str(device).startswith("cuda"))
