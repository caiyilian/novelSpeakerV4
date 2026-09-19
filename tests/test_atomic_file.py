import ctypes
import json
import os
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from atomic_file import replace_with_retry
from volume_review import _write_json_atomic


class AtomicFileTests(unittest.TestCase):
    @unittest.skipUnless(sys.platform == "win32", "Windows sharing semantics")
    def test_checkpoint_survives_actual_windows_reader_lock(self):
        from ctypes import wintypes

        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        create = kernel.CreateFileW
        create.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD,
                           ctypes.c_void_p, wintypes.DWORD, wintypes.DWORD,
                           wintypes.HANDLE]
        create.restype = wintypes.HANDLE
        close = kernel.CloseHandle
        close.argtypes = [wintypes.HANDLE]
        close.restype = wintypes.BOOL
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "state.json"
            _write_json_atomic(target, {"completed": 10})
            # A reader that shares read/write access, but not deletion.
            handle = create(str(target), 0x80000000, 3, None, 3, 0x80, None)
            self.assertNotEqual(ctypes.c_void_p(-1).value, handle)
            timer = threading.Timer(0.2, lambda: close(handle))
            timer.start()
            try:
                _write_json_atomic(target, {"completed": 11})
            finally:
                timer.join()
            self.assertEqual({"completed": 11}, json.loads(target.read_text()))
            self.assertEqual([], list(Path(directory).glob("*.tmp")))

    def test_permanent_lock_keeps_old_and_new_versions(self):
        error = PermissionError("locked")
        error.winerror = 5
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "state.json"
            _write_json_atomic(target, {"completed": 10})
            with patch("atomic_file.os.replace", side_effect=error), patch(
                "atomic_file.time.monotonic", side_effect=[0, 0, 11]
            ), patch("atomic_file.time.sleep"):
                with self.assertRaises(PermissionError):
                    _write_json_atomic(target, {"completed": 11})
            self.assertEqual({"completed": 10}, json.loads(target.read_text()))
            pending = list(Path(directory).glob("*.tmp"))
            self.assertEqual(1, len(pending))
            self.assertEqual({"completed": 11}, json.loads(pending[0].read_text()))

    def test_non_lock_errors_are_not_retried(self):
        with patch("atomic_file.os.replace", side_effect=FileNotFoundError), patch(
            "atomic_file.time.sleep"
        ) as sleep:
            with self.assertRaises(FileNotFoundError):
                replace_with_retry("missing", "target")
            sleep.assert_not_called()

    def test_failed_serialization_does_not_damage_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "state.json"
            _write_json_atomic(target, {"completed": 10})
            with self.assertRaises(TypeError):
                _write_json_atomic(target, {"bad": object()})
            self.assertEqual({"completed": 10}, json.loads(target.read_text()))
            self.assertEqual([], list(Path(directory).glob("*.tmp")))
