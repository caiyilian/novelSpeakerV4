import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from control_center_core import (  # noqa: E402
    MODEL_FILTER,
    VolumeSpec,
    backup_volume,
    build_process_environment,
    build_run_arguments,
    format_eta,
    find_project_root,
    parse_duration_seconds,
    parse_progress_line,
    read_recent_failure,
    read_progress,
    read_review_progress,
    volume_specs,
)


class ControlCenterCoreTests(unittest.TestCase):
    def test_project_root_search_skips_empty_nested_data_directory(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "project"
            nested = root / "dist" / "v1.0.1"
            (root / "data").mkdir(parents=True)
            (root / "data" / "novel.txt").write_text("novel", encoding="utf-8")
            (nested / "data").mkdir(parents=True)

            detected = find_project_root([nested])

        self.assertEqual(root, detected)

    def test_volume_specs_use_root_data_for_first_volume(self):
        root = Path("C:/workspace")
        specs = volume_specs(root)
        self.assertEqual(root / "data", specs[0].data_dir)
        self.assertEqual(root / "data" / "volume5", specs[4].data_dir)

    def test_progress_matches_dialogue_and_label_line_counts(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            data_dir = Path(temp_dir)
            (data_dir / "novel.txt").write_text(
                "旁白「第一句」\n一行有「第二句」和「第三句」\n",
                encoding="utf-8",
            )
            (data_dir / "labeled.txt").write_text("甲\n乙\n", encoding="utf-8")
            progress = read_progress(VolumeSpec(1, "第1卷", data_dir))

        self.assertEqual(2, progress.labeled)
        self.assertEqual(3, progress.total)
        self.assertEqual(67, progress.percent)

    def test_continue_and_reset_arguments_are_distinct(self):
        spec = VolumeSpec(3, "第3卷", Path("C:/workspace/data/volume3"))
        continued = build_run_arguments(spec, reset=False)
        restarted = build_run_arguments(spec, reset=True)
        self.assertNotIn("--reset-state", continued)
        self.assertIn("--reset-state", restarted)
        self.assertEqual(MODEL_FILTER, continued[continued.index("--api-model") + 1])
        self.assertEqual("2", continued[continued.index("--api-round-robin-offset") + 1])

    def test_packaged_arguments_reenter_executable_worker(self):
        spec = VolumeSpec(2, "第2卷", Path("C:/workspace/data/volume2"))
        arguments = build_run_arguments(spec, reset=True, frozen=True)

        self.assertEqual("--worker", arguments[0])
        self.assertNotIn("run_label.py", " ".join(arguments))
        self.assertIn("--reset-state", arguments)

    def test_first_pass_and_review_eta_lines_are_parsed(self):
        first_pass = parse_progress_line(
            "  [154/1561] L472 -> 人物 | tools=290 avg=23s "
            "elapsed=58m31s remaining=8h54m"
        )
        review = parse_progress_line(
            "  [Review mirrored-jury 27/120 22.5%] D31 juror | calls=8 ETA=1h05m"
        )

        self.assertEqual("first-pass", first_pass.phase)
        self.assertEqual(32040, first_pass.eta_seconds)
        self.assertEqual("mirrored-jury", review.phase)
        self.assertEqual(27, review.completed)
        self.assertEqual(3900, review.eta_seconds)

    def test_eta_duration_formatting(self):
        self.assertEqual(3900, parse_duration_seconds("1h05m"))
        self.assertEqual(93780, parse_duration_seconds("1d2h3m"))
        self.assertIsNone(parse_duration_seconds("later"))
        self.assertEqual("预计剩余 1 小时 5 分钟", format_eta(3900))
        self.assertEqual("预计时间计算中", format_eta(None))

    def test_review_progress_reads_durable_checkpoint(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            data_dir = Path(temp_dir)
            (data_dir / "volume_review_state.json").write_text(
                '{"progress":{"phase":"scene-review","status":"running",'
                '"completed":9,"total":30,"eta_seconds":321.5}}',
                encoding="utf-8",
            )
            progress = read_review_progress(VolumeSpec(1, "第1卷", data_dir))

        self.assertEqual("场景复审", progress.phase_label)
        self.assertEqual(30, progress.percent)
        self.assertTrue(progress.active)
        self.assertEqual(321.5, progress.eta_seconds)

    def test_process_environment_uses_selected_key_file_and_direct_connections(self):
        key_path = Path("C:/keys/accounts.txt")
        environment = build_process_environment(
            key_path,
            {"PATH": os.environ.get("PATH", "")},
            use_sensenova_pool=False,
        )
        self.assertEqual(str(key_path.resolve()), environment["SENSENOVA_API_KEYS_FILE"])
        self.assertEqual("0", environment["SENSENOVA_POOL_ENABLED"])
        self.assertEqual("0", environment["SENSENOVA_USE_ENV_PROXY"])
        self.assertEqual("0", environment["AGNES_USE_ENV_PROXY"])
        self.assertEqual("utf-8:backslashreplace", environment["PYTHONIOENCODING"])
        self.assertTrue(environment["NOVELSPEAKER_PROJECT_ROOT"])

    def test_process_environment_can_use_pool_without_a_key_file(self):
        environment = build_process_environment(
            None,
            {
                "PATH": os.environ.get("PATH", ""),
                "SENSENOVA_API_KEYS_FILE": "stale-key-file",
            },
        )

        self.assertEqual("1", environment["SENSENOVA_POOL_ENABLED"])
        self.assertNotIn("SENSENOVA_API_KEYS_FILE", environment)

    def test_backup_copies_runtime_files_without_moving_sources(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            data_dir = root / "data"
            data_dir.mkdir()
            (data_dir / "novel.txt").write_text("novel", encoding="utf-8")
            (data_dir / "answers.txt").write_text("answers", encoding="utf-8")
            (data_dir / "labeled.txt").write_text("speaker\n", encoding="utf-8")
            (data_dir / "label_log.jsonl").write_text("{}\n", encoding="utf-8")
            spec = VolumeSpec(1, "第1卷", data_dir)

            with patch("control_center_core.datetime") as mocked_datetime:
                mocked_datetime.now.return_value.strftime.return_value = "20260910_220000"
                mocked_datetime.now.return_value.isoformat.return_value = "2026-09-10T22:00:00"
                target, copied = backup_volume(spec, root / "backup")

            self.assertEqual(2, copied)
            self.assertTrue((target / "labeled.txt").exists())
            self.assertTrue((target / "label_log.jsonl").exists())
            self.assertFalse((target / "novel.txt").exists())
            self.assertTrue((data_dir / "labeled.txt").exists())

    def test_unchanged_volume_is_not_backed_up_twice(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            data_dir = root / "data"
            data_dir.mkdir()
            (data_dir / "labeled.txt").write_text("speaker\n", encoding="utf-8")
            spec = VolumeSpec(1, "第1卷", data_dir)
            backup_root = root / "backup"

            first_target, first_count = backup_volume(spec, backup_root)
            second_target, second_count = backup_volume(spec, backup_root)

            self.assertEqual(1, first_count)
            self.assertEqual(0, second_count)
            self.assertEqual(first_target, second_target)
            self.assertEqual(1, len(list(backup_root.glob("*/volume1*"))))

            (data_dir / "labeled.txt").write_text("speaker\nother\n", encoding="utf-8")
            third_target, third_count = backup_volume(spec, backup_root)
            self.assertEqual(1, third_count)
            self.assertNotEqual(first_target, third_target)

    def test_recent_failure_reads_last_transaction_event(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            data_dir = Path(temp_dir)
            log_path = data_dir / "label_log.temp.jsonl"
            log_path.write_text(
                '{"event":"round_complete"}\n'
                '{"event":"round_failed","error":{"message":"HTTP 429"}}\n',
                encoding="utf-8",
            )
            spec = VolumeSpec(1, "第1卷", data_dir)
            self.assertEqual("HTTP 429", read_recent_failure(spec))


if __name__ == "__main__":
    unittest.main()
