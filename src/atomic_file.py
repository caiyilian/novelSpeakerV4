"""Atomic file publication tolerant of short-lived Windows file locks."""

import os
import time


def replace_with_retry(source, target, timeout=10.0):
    deadline = time.monotonic() + timeout
    attempt = 0
    while True:
        try:
            os.replace(source, target)
            return
        except OSError as exc:
            if getattr(exc, "winerror", None) not in {5, 32, 33}:
                raise
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise
            if attempt == 0:
                print(
                    f"[File retry] Cannot replace {target} (Windows {exc.winerror}); "
                    f"retrying for up to {timeout:g}s. Pending file: {source}",
                    flush=True,
                )
            time.sleep(min(0.05 * (2 ** min(attempt, 4)), remaining))
            attempt += 1
