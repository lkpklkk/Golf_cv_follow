import cv2
import config


def _probe_cameras(max_index=None):
    """Return a list of (index, label) for every camera that opens successfully."""
    if max_index is None:
        max_index = config.CAMERA_SCAN_MAX_INDEX

    available = []
    for i in range(max_index):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            label = f"Camera {i}"
            # On macOS/Linux the backend name gives a hint about the device type
            backend = cap.getBackendName()
            if backend:
                label += f" ({backend})"
            available.append((i, label))
            cap.release()
    return available


def pick_camera(default=0):
    """
    Probe available cameras and let the user pick one from the terminal.

    If only one camera is found it is selected automatically.
    Falls back to `default` if no cameras are found.

    Returns the chosen camera index (int).
    """
    print("\nScanning for cameras...")
    cameras = _probe_cameras()

    if not cameras:
        print(f"No cameras found. Falling back to index {default}.")
        return default

    if len(cameras) == 1:
        idx, label = cameras[0]
        print(f"Found one camera: {label} — using it.")
        return idx

    print(f"\nFound {len(cameras)} camera(s):")
    for i, (idx, label) in enumerate(cameras):
        marker = " (default)" if idx == default else ""
        print(f"  [{i}] {label}{marker}")

    while True:
        try:
            raw = input(
                f"\nSelect camera [0-{len(cameras)-1}] (Enter = default {default}): "
            ).strip()
            if raw == "":
                return default
            choice = int(raw)
            if 0 <= choice < len(cameras):
                return cameras[choice][0]
            print(f"Please enter a number between 0 and {len(cameras)-1}.")
        except ValueError:
            print("Invalid input — enter a number.")
