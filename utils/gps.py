from __future__ import annotations

"""
GPS coordinate retrieval on macOS.

Primary source: CoreLocationCLI (brew install corelocationcli).
  - It is a proper signed app bundle, so macOS honours the location permission.
  - Grant permission once: System Settings → Privacy & Security →
    Location Services → enable CoreLocationCLI (or Terminal if listed).

Fallback: PyObjC CoreLocation (pip install pyobjc-framework-CoreLocation).
  - Works when the Python process already has location permission.

Both paths fall back to (None, None) so the app always starts cleanly.

GpsPoller runs in a background thread and caches the latest fix.
"""

import json
import shutil
import subprocess
import threading
import time


# ---------------------------------------------------------------------------
# Primary: CoreLocationCLI


def _corelocationcli(timeout: float) -> tuple[float | None, float | None]:
    """Call CoreLocationCLI --json with a subprocess timeout."""
    cli = shutil.which("CoreLocationCLI")
    if cli is None:
        raise FileNotFoundError("CoreLocationCLI not found")

    proc = subprocess.run(
        [cli, "--json"],
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    if proc.returncode != 0 or not proc.stdout.strip():
        return None, None

    data = json.loads(proc.stdout)
    lat = data.get("latitude")
    lon = data.get("longitude")
    if lat is not None and lon is not None:
        return float(lat), float(lon)
    return None, None


# ---------------------------------------------------------------------------
# Fallback: PyObjC CoreLocation


def _corelocation_pyobjc(timeout: float) -> tuple[float | None, float | None]:
    """Drive NSRunLoop manually so callbacks fire from any thread."""
    import objc  # noqa: F401
    from CoreFoundation import CFRunLoopRunInMode, kCFRunLoopDefaultMode
    from CoreLocation import CLLocationManager
    from Foundation import NSObject

    result: list[tuple[float, float] | None] = [None]
    done = threading.Event()

    class _Delegate(NSObject):
        def locationManager_didUpdateLocations_(self, manager, locations):
            loc = locations.lastObject()
            if loc is not None:
                c = loc.coordinate()
                result[0] = (c.latitude, c.longitude)
            done.set()

        def locationManager_didFailWithError_(self, manager, error):
            done.set()

        def locationManagerDidChangeAuthorization_(self, manager):
            status = manager.authorizationStatus()
            if status in (2, 3):  # denied / restricted
                done.set()

    delegate = _Delegate.alloc().init()
    manager = CLLocationManager.alloc().init()
    manager.setDelegate_(delegate)
    manager.setDesiredAccuracy_(10.0)
    manager.requestAlwaysAuthorization()
    manager.startUpdatingLocation()

    deadline = time.monotonic() + timeout
    while not done.is_set() and time.monotonic() < deadline:
        CFRunLoopRunInMode(kCFRunLoopDefaultMode, 0.1, False)

    manager.stopUpdatingLocation()
    return result[0] if result[0] else (None, None)


# ---------------------------------------------------------------------------
# Public API


def get_location(timeout: float = 4.0) -> tuple[float | None, float | None]:
    """
    Return (latitude, longitude) or (None, None) if GPS is unavailable.
    Tries CoreLocationCLI first, then PyObjC, then gives up.
    """
    try:
        lat, lon = _corelocationcli(timeout)
        if lat is not None:
            return lat, lon
    except Exception:
        pass

    try:
        return _corelocation_pyobjc(timeout)
    except Exception:
        pass

    return None, None


class GpsPoller:
    """
    Non-blocking GPS poller. Polls in a background thread every `poll_interval`
    seconds and caches the latest fix. Call `.location` to read without blocking.

    Usage:
        poller = GpsPoller()
        lat, lon = poller.location   # always instant
        poller.stop()
    """

    def __init__(self, poll_interval: float = 5.0):
        self._lat: float | None = None
        self._lon: float | None = None
        self._updated_at: float | None = None
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._poll_interval = poll_interval
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    @property
    def location(self) -> tuple[float | None, float | None]:
        with self._lock:
            return self._lat, self._lon

    @property
    def age_seconds(self) -> float | None:
        """Seconds since the last successful fix, or None if never fixed."""
        with self._lock:
            if self._updated_at is None:
                return None
            return time.monotonic() - self._updated_at

    def stop(self) -> None:
        self._stop_event.set()

    def _loop(self) -> None:
        while not self._stop_event.is_set():
            lat, lon = get_location()
            if lat is not None:
                with self._lock:
                    self._lat, self._lon = lat, lon
                    self._updated_at = time.monotonic()
            self._stop_event.wait(self._poll_interval)
