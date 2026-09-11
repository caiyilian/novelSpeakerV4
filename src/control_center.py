"""PyQt6 desktop control center for five-volume annotation runs."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from collections import deque
from datetime import datetime
from pathlib import Path

from PyQt6.QtCore import (
    QProcess,
    QProcessEnvironment,
    QRunnable,
    QSettings,
    QSize,
    Qt,
    QThreadPool,
    QTimer,
    QUrl,
    pyqtSignal,
    QObject,
)
from PyQt6.QtGui import (
    QAction,
    QColor,
    QCloseEvent,
    QDesktopServices,
    QFont,
    QIcon,
    QPainter,
    QPixmap,
    QTextCursor,
)
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMenu,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QStyle,
    QSystemTrayIcon,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from control_center_core import (
    PROJECT_ROOT,
    ProgressHint,
    VolumeSpec,
    backup_volume,
    build_process_environment,
    build_run_arguments,
    child_python_executable,
    format_eta,
    parse_progress_line,
    read_progress,
    read_recent_failure,
    read_review_progress,
    volume_specs,
)


APP_NAME = "NovelSpeakerControlCenter"
APP_VERSION = "1.0.1"
AUTOSTART_NAME = "NovelSpeaker Control Center"
SETTINGS_PATH = PROJECT_ROOT / "config" / "control_center.ini"
DEFAULT_LOG_DIR = PROJECT_ROOT / "runtime_logs"


def app_icon() -> QIcon:
    pixmap = QPixmap(64, 64)
    pixmap.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    painter.setPen(Qt.PenStyle.NoPen)
    painter.setBrush(QColor("#127c78"))
    painter.drawRoundedRect(4, 4, 56, 56, 10, 10)
    painter.setPen(QColor("#ffffff"))
    font = QFont("Segoe UI", 30, QFont.Weight.DemiBold)
    painter.setFont(font)
    painter.drawText(pixmap.rect(), Qt.AlignmentFlag.AlignCenter, "N")
    painter.end()
    return QIcon(pixmap)


def autostart_command() -> str:
    executable = Path(sys.executable)
    if getattr(sys, "frozen", False):
        return subprocess.list2cmdline([str(executable), "--minimized"])
    if executable.stem.lower() == "python":
        pythonw = executable.with_name("pythonw.exe")
        if pythonw.exists():
            executable = pythonw
    return subprocess.list2cmdline(
        [str(executable), "-X", "utf8", str(Path(__file__).resolve()), "--minimized"]
    )


def set_windows_autostart(enabled: bool) -> None:
    if sys.platform != "win32":
        raise RuntimeError("开机启动目前仅支持 Windows")
    import winreg

    key_path = r"Software\Microsoft\Windows\CurrentVersion\Run"
    with winreg.OpenKey(
        winreg.HKEY_CURRENT_USER,
        key_path,
        0,
        winreg.KEY_SET_VALUE,
    ) as key:
        if enabled:
            winreg.SetValueEx(key, AUTOSTART_NAME, 0, winreg.REG_SZ, autostart_command())
        else:
            try:
                winreg.DeleteValue(key, AUTOSTART_NAME)
            except FileNotFoundError:
                pass


class BackupSignals(QObject):
    finished = pyqtSignal(int, str, int, str)


class BackupTask(QRunnable):
    def __init__(self, spec: VolumeSpec, backup_root: Path):
        super().__init__()
        self.spec = spec
        self.backup_root = backup_root
        self.signals = BackupSignals()

    def run(self) -> None:
        try:
            target, copied = backup_volume(self.spec, self.backup_root)
            self.signals.finished.emit(self.spec.number, str(target), copied, "")
        except Exception as exc:
            self.signals.finished.emit(self.spec.number, "", 0, str(exc))


class VolumeProcess(QObject):
    output = pyqtSignal(int, str)
    state_changed = pyqtSignal(int, str, str)
    finished = pyqtSignal(int, int)

    def __init__(self, spec: VolumeSpec, parent: QObject | None = None):
        super().__init__(parent)
        self.spec = spec
        self.process = QProcess(self)
        self.process.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        self.process.readyReadStandardOutput.connect(self._read_output)
        self.process.started.connect(self._started)
        self.process.finished.connect(self._finished)
        self.process.errorOccurred.connect(self._process_error)
        self.log_handle = None
        self.latest_log_path: Path | None = None
        self._stopping = False
        self._output_line_buffer = ""
        self.latest_progress_hint: ProgressHint | None = None
        self.run_started_at = 0.0
        self.run_start_labeled = 0

    @property
    def running(self) -> bool:
        return self.process.state() != QProcess.ProcessState.NotRunning

    def start(self, key_file: Path, log_dir: Path, reset: bool = False) -> bool:
        if self.running:
            return False

        log_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.latest_log_path = log_dir / f"volume{self.spec.number}_{stamp}.log"
        self.log_handle = self.latest_log_path.open("a", encoding="utf-8", buffering=1)
        self._stopping = False
        self._output_line_buffer = ""
        self.latest_progress_hint = None
        self.run_started_at = time.monotonic()
        self.run_start_labeled = 0 if reset else read_progress(self.spec).labeled

        environment = QProcessEnvironment.systemEnvironment()
        for name, value in build_process_environment(key_file).items():
            environment.insert(name, value)
        self.process.setProcessEnvironment(environment)
        self.process.setWorkingDirectory(str(PROJECT_ROOT))
        self.process.setProgram(child_python_executable())
        self.process.setArguments(build_run_arguments(self.spec, reset=reset))
        self.state_changed.emit(
            self.spec.number,
            "starting",
            "正在重置并启动" if reset else "正在从断点启动",
        )
        self.process.start()
        return True

    def stop(self) -> None:
        if not self.running:
            return
        self._stopping = True
        self.state_changed.emit(self.spec.number, "stopping", "正在停止")
        self.process.terminate()
        QTimer.singleShot(5000, self._kill_if_running)

    def _kill_if_running(self) -> None:
        if self.running:
            self.process.kill()

    def _started(self) -> None:
        self.state_changed.emit(self.spec.number, "running", "标注进程已启动")

    def _read_output(self) -> None:
        data = bytes(self.process.readAllStandardOutput())
        text = data.decode("utf-8", errors="replace")
        if self.log_handle:
            self.log_handle.write(text)
        previous_hint = self.latest_progress_hint
        self._parse_progress_output(text)
        self.output.emit(self.spec.number, text)
        if "[API pool]" in text or "temporarily unavailable" in text:
            self.state_changed.emit(self.spec.number, "waiting", "模型池冷却中")
        elif "Traceback (most recent call last)" in text:
            self.state_changed.emit(self.spec.number, "finishing_error", "正在收集错误信息")
        elif "[Review retry" in text:
            self.state_changed.emit(self.spec.number, "waiting", "复审任务重试等待")
        elif "[Review " in text or "Starting resumable full-volume" in text:
            self.state_changed.emit(self.spec.number, "reviewing", "正在进行全卷复审")
        elif "Annotation complete" in text or "Full-volume second-pass complete" in text:
            self.state_changed.emit(self.spec.number, "running", "正在完成收尾")
        elif (
            self.latest_progress_hint is not None
            and self.latest_progress_hint is not previous_hint
            and self.latest_progress_hint.phase == "first-pass"
        ):
            self.state_changed.emit(self.spec.number, "running", "正在进行首轮标注")

    def _parse_progress_output(self, text: str, *, flush: bool = False) -> None:
        combined = self._output_line_buffer + text
        lines = combined.split("\n")
        self._output_line_buffer = "" if flush else lines.pop()
        if flush and lines and not lines[-1]:
            lines.pop()
        for line in lines:
            hint = parse_progress_line(line.rstrip("\r"))
            if hint is not None:
                self.latest_progress_hint = hint

    def estimated_first_pass_eta(
        self,
        labeled: int,
        remaining: int,
    ) -> float | None:
        hint = self.latest_progress_hint
        if hint is not None and hint.phase == "first-pass":
            return hint.eta_seconds
        if not self.running or self.run_started_at <= 0:
            return None
        remaining = max(0, int(remaining))
        if remaining <= 0:
            return 0.0
        completed_here = labeled - self.run_start_labeled
        elapsed = time.monotonic() - self.run_started_at
        if completed_here <= 0 or elapsed < 15:
            return None
        return elapsed / completed_here * remaining

    def _finished(self, exit_code: int, _exit_status) -> None:
        self._read_output()
        self._parse_progress_output("", flush=True)
        if self.log_handle:
            self.log_handle.close()
            self.log_handle = None
        if self._stopping:
            self.state_changed.emit(self.spec.number, "idle", "已停止，可继续")
        elif exit_code == 0:
            self.state_changed.emit(self.spec.number, "complete", "本次任务已完成")
        else:
            self.state_changed.emit(
                self.spec.number,
                "error",
                f"进程异常退出，代码 {exit_code}",
            )
        self.finished.emit(self.spec.number, exit_code)

    def _process_error(self, error) -> None:
        if error == QProcess.ProcessError.FailedToStart:
            self.state_changed.emit(self.spec.number, "error", self.process.errorString())


class VolumeRow(QFrame):
    continue_clicked = pyqtSignal(int)
    restart_clicked = pyqtSignal(int)
    backup_clicked = pyqtSignal(int)
    stop_clicked = pyqtSignal(int)
    log_clicked = pyqtSignal(int)

    STATUS = {
        "idle": ("#667085", "待继续"),
        "starting": ("#2563a6", "启动中"),
        "running": ("#127c78", "标注中"),
        "reviewing": ("#2563a6", "复审中"),
        "waiting": ("#b26a00", "等待额度"),
        "stopping": ("#b26a00", "停止中"),
        "finishing_error": ("#c2413b", "检测到错误"),
        "error": ("#c2413b", "出错"),
        "complete": ("#2f855a", "已完成"),
        "backup": ("#6b4fa1", "备份中"),
    }

    def __init__(self, spec: VolumeSpec, style: QStyle, parent: QWidget | None = None):
        super().__init__(parent)
        self.spec = spec
        self.state = "idle"
        self.setObjectName("volumeRow")
        self.setFixedHeight(88)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(18, 8, 14, 8)
        layout.setSpacing(14)

        identity = QVBoxLayout()
        self.name_label = QLabel(spec.name)
        self.name_label.setObjectName("volumeName")
        self.path_label = QLabel(str(spec.data_dir.relative_to(PROJECT_ROOT)))
        self.path_label.setObjectName("muted")
        identity.addWidget(self.name_label)
        identity.addWidget(self.path_label)
        identity.addStretch()
        layout.addLayout(identity)
        layout.setStretchFactor(identity, 0)

        status_box = QVBoxLayout()
        self.status_label = QLabel()
        self.status_label.setFixedWidth(92)
        self.status_detail = QLabel("")
        self.status_detail.setObjectName("muted")
        self.status_detail.setFixedWidth(150)
        self.status_detail.setToolTip("")
        status_box.addWidget(self.status_label)
        status_box.addWidget(self.status_detail)
        status_box.addStretch()
        layout.addLayout(status_box)

        progress_box = QVBoxLayout()
        progress_header = QHBoxLayout()
        self.count_label = QLabel("0 / 0")
        self.count_label.setObjectName("progressCount")
        self.percent_label = QLabel("0%")
        self.percent_label.setObjectName("muted")
        progress_header.addWidget(self.count_label)
        progress_header.addStretch()
        progress_header.addWidget(self.percent_label)
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setTextVisible(False)
        self.progress.setFixedSize(270, 10)
        self.eta_label = QLabel("")
        self.eta_label.setObjectName("etaLabel")
        self.eta_label.setFixedHeight(17)
        progress_box.addLayout(progress_header)
        progress_box.addWidget(self.progress)
        progress_box.addWidget(self.eta_label)
        progress_box.addStretch()
        layout.addLayout(progress_box)

        layout.addStretch(1)

        self.continue_button = QPushButton("继续")
        self.continue_button.setIcon(style.standardIcon(QStyle.StandardPixmap.SP_MediaPlay))
        self.continue_button.setObjectName("primaryButton")
        self.continue_button.clicked.connect(lambda: self.continue_clicked.emit(spec.number))

        self.restart_button = QPushButton("重新标注")
        self.restart_button.setIcon(style.standardIcon(QStyle.StandardPixmap.SP_BrowserReload))
        self.restart_button.clicked.connect(lambda: self.restart_clicked.emit(spec.number))

        self.backup_button = QPushButton("备份")
        self.backup_button.setIcon(style.standardIcon(QStyle.StandardPixmap.SP_DialogSaveButton))
        self.backup_button.clicked.connect(lambda: self.backup_clicked.emit(spec.number))

        self.stop_button = QToolButton()
        self.stop_button.setIcon(style.standardIcon(QStyle.StandardPixmap.SP_MediaStop))
        self.stop_button.setToolTip("停止本卷")
        self.stop_button.setFixedSize(34, 34)
        self.stop_button.clicked.connect(lambda: self.stop_clicked.emit(spec.number))

        self.log_button = QToolButton()
        self.log_button.setIcon(style.standardIcon(QStyle.StandardPixmap.SP_FileDialogDetailedView))
        self.log_button.setToolTip("打开本卷最新运行日志")
        self.log_button.setFixedSize(34, 34)
        self.log_button.clicked.connect(lambda: self.log_clicked.emit(spec.number))

        for button in (self.continue_button, self.restart_button, self.backup_button):
            button.setFixedHeight(34)
            layout.addWidget(button)
        layout.addWidget(self.stop_button)
        layout.addWidget(self.log_button)

        self.set_status("idle", "")

    def update_progress(
        self,
        labeled: int,
        total: int,
        percent: int,
        stage: str = "首轮",
    ) -> None:
        self.count_label.setText(f"{labeled:,} / {total:,}")
        self.percent_label.setText(f"{stage} {percent}%")
        self.progress.setValue(percent)

    def set_eta(self, stage: str = "", seconds: float | None = None) -> None:
        if not stage:
            self.eta_label.clear()
            self.eta_label.setToolTip("")
            return
        text = f"{stage} · {format_eta(seconds)}"
        self.eta_label.setText(text)
        self.eta_label.setToolTip(text)

    def set_status(self, state: str, detail: str = "") -> None:
        self.state = state
        color, text = self.STATUS.get(state, self.STATUS["idle"])
        self.status_label.setText(f'<span style="color:{color}; font-size:16px;">●</span>&nbsp; {text}')
        compact_detail = detail.replace("\n", " ").strip()
        if len(compact_detail) > 30:
            compact_detail = compact_detail[:29] + "…"
        self.status_detail.setText(compact_detail)
        self.status_detail.setToolTip(detail)

        busy = state in {
            "starting",
            "running",
            "reviewing",
            "waiting",
            "stopping",
            "finishing_error",
            "backup",
        }
        self.continue_button.setEnabled(not busy)
        self.restart_button.setEnabled(not busy)
        self.backup_button.setEnabled(not busy)
        self.stop_button.setEnabled(
            state in {"starting", "running", "reviewing", "waiting", "finishing_error"}
        )
        self.continue_button.setText("重试" if state == "error" else "继续")


class ControlCenter(QMainWindow):
    def __init__(self, start_minimized: bool = False):
        super().__init__()
        SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
        self.settings = QSettings(str(SETTINGS_PATH), QSettings.Format.IniFormat)
        self.specs = volume_specs()
        self.spec_by_number = {spec.number: spec for spec in self.specs}
        self.rows: dict[int, VolumeRow] = {}
        self.controllers: dict[int, VolumeProcess] = {}
        self.output_buffers = {spec.number: deque(maxlen=500) for spec in self.specs}
        self.progress_cache = {}
        self.latest_logs: dict[int, Path] = {}
        self.pending_restart: set[int] = set()
        self.thread_pool = QThreadPool.globalInstance()
        self.thread_pool.setMaxThreadCount(2)
        self._allow_quit = False
        self._tray_notice_shown = False

        self.setWindowTitle("小说对话角色标注控制台")
        self.setWindowIcon(app_icon())
        self.resize(1320, 930)
        self.setMinimumSize(1120, 720)
        self._build_ui()
        self._build_tray()
        self._load_settings()
        self._create_controllers()
        self._refresh_all_progress(initial=True)

        self.refresh_timer = QTimer(self)
        self.refresh_timer.setInterval(2000)
        self.refresh_timer.timeout.connect(self._refresh_all_progress)
        self.refresh_timer.start()

        if start_minimized and self.tray.isSystemTrayAvailable():
            QTimer.singleShot(0, self.hide)

    def _build_ui(self) -> None:
        central = QWidget()
        central.setObjectName("central")
        root = QVBoxLayout(central)
        root.setContentsMargins(24, 20, 24, 18)
        root.setSpacing(14)

        title_row = QHBoxLayout()
        title_box = QVBoxLayout()
        title = QLabel("小说对话角色标注控制台")
        title.setObjectName("title")
        subtitle = QLabel("五卷任务中心")
        subtitle.setObjectName("subtitle")
        title_box.addWidget(title)
        title_box.addWidget(subtitle)
        title_row.addLayout(title_box)
        title_row.addStretch()
        self.pool_label = QLabel("模型池  SenseNova × 0  ·  Agnes × 3")
        self.pool_label.setObjectName("poolLabel")
        title_row.addWidget(self.pool_label)
        root.addLayout(title_row)

        settings_band = QFrame()
        settings_band.setObjectName("settingsBand")
        settings_layout = QVBoxLayout(settings_band)
        settings_layout.setContentsMargins(16, 14, 16, 14)
        settings_layout.setSpacing(9)

        key_row = QHBoxLayout()
        key_label = QLabel("API Key 文件")
        key_label.setFixedWidth(100)
        self.key_path = QLineEdit()
        self.key_path.setReadOnly(True)
        self.key_path.setPlaceholderText("需要选择文件")
        self.key_button = QPushButton("选择文件")
        self.key_button.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_DialogOpenButton))
        self.key_button.clicked.connect(self._choose_key_file)
        self.key_count = QLabel("未选择")
        self.key_count.setObjectName("muted")
        self.key_count.setFixedWidth(74)
        key_row.addWidget(key_label)
        key_row.addWidget(self.key_path, 1)
        key_row.addWidget(self.key_button)
        key_row.addWidget(self.key_count)
        settings_layout.addLayout(key_row)

        log_row = QHBoxLayout()
        log_label = QLabel("运行日志路径")
        log_label.setFixedWidth(100)
        self.log_path = QLineEdit()
        self.log_path.setReadOnly(True)
        self.log_button = QPushButton("选择目录")
        self.log_button.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_DirOpenIcon))
        self.log_button.clicked.connect(self._choose_log_directory)
        self.open_log_dir_button = QToolButton()
        self.open_log_dir_button.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_DirOpenIcon))
        self.open_log_dir_button.setToolTip("打开运行日志目录")
        self.open_log_dir_button.setFixedSize(34, 34)
        self.open_log_dir_button.clicked.connect(self._open_log_directory)
        self.autostart = QCheckBox("开机启动")
        self.autostart.setObjectName("switch")
        self.autostart.toggled.connect(self._toggle_autostart)
        log_row.addWidget(log_label)
        log_row.addWidget(self.log_path, 1)
        log_row.addWidget(self.log_button)
        log_row.addWidget(self.open_log_dir_button)
        log_row.addSpacing(8)
        log_row.addWidget(self.autostart)
        settings_layout.addLayout(log_row)

        self.key_warning = QLabel("请选择 API Key 文件后再启动标注任务")
        self.key_warning.setObjectName("warning")
        settings_layout.addWidget(self.key_warning)
        root.addWidget(settings_band)

        actions = QHBoxLayout()
        section_title = QLabel("卷任务")
        section_title.setObjectName("sectionTitle")
        actions.addWidget(section_title)
        actions.addStretch()
        self.all_continue = QPushButton("全部继续")
        self.all_continue.setObjectName("primaryButton")
        self.all_continue.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_MediaPlay))
        self.all_continue.clicked.connect(self._continue_all)
        self.all_backup = QPushButton("全部备份")
        self.all_backup.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_DialogSaveButton))
        self.all_backup.clicked.connect(self._backup_all)
        self.all_stop = QPushButton("全部停止")
        self.all_stop.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_MediaStop))
        self.all_stop.clicked.connect(self._stop_all)
        actions.addWidget(self.all_continue)
        actions.addWidget(self.all_backup)
        actions.addWidget(self.all_stop)
        root.addLayout(actions)

        volume_list = QFrame()
        volume_list.setObjectName("volumeList")
        volume_list.setMinimumHeight(5 * 88 + 4)
        volume_layout = QVBoxLayout(volume_list)
        volume_layout.setContentsMargins(0, 0, 0, 0)
        volume_layout.setSpacing(1)
        for spec in self.specs:
            row = VolumeRow(spec, self.style())
            row.continue_clicked.connect(self._continue_volume)
            row.restart_clicked.connect(self._restart_volume)
            row.backup_clicked.connect(self._backup_volume)
            row.stop_clicked.connect(self._stop_volume)
            row.log_clicked.connect(self._open_volume_log)
            self.rows[spec.number] = row
            volume_layout.addWidget(row)
        root.addWidget(volume_list)

        console_header = QHBoxLayout()
        console_title = QLabel("实时输出")
        console_title.setObjectName("sectionTitle")
        self.console_volume = QComboBox()
        for spec in self.specs:
            self.console_volume.addItem(spec.name, spec.number)
        self.console_volume.currentIndexChanged.connect(self._switch_console)
        self.clear_console_button = QToolButton()
        self.clear_console_button.setIcon(
            self.style().standardIcon(QStyle.StandardPixmap.SP_DialogResetButton)
        )
        self.clear_console_button.setToolTip("清空当前显示")
        self.clear_console_button.clicked.connect(self.console_clear)
        self.console_toggle = QToolButton()
        self.console_toggle.setText("显示输出")
        self.console_toggle.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.console_toggle.setIcon(
            self.style().standardIcon(QStyle.StandardPixmap.SP_ArrowDown)
        )
        self.console_toggle.clicked.connect(self._toggle_console)
        console_header.addWidget(console_title)
        console_header.addWidget(self.console_volume)
        console_header.addStretch()
        console_header.addWidget(self.clear_console_button)
        console_header.addWidget(self.console_toggle)
        root.addLayout(console_header)

        self.console = QPlainTextEdit()
        self.console.setReadOnly(True)
        self.console.setMaximumBlockCount(500)
        self.console.setFixedHeight(110)
        self.console.setObjectName("console")
        self.console.setVisible(False)
        self.clear_console_button.setVisible(False)
        root.addWidget(self.console)

        footer = QHBoxLayout()
        self.summary = QLabel("")
        self.summary.setObjectName("muted")
        footer.addWidget(self.summary)
        footer.addStretch()
        tray_note = QLabel("关闭窗口后继续在系统托盘运行")
        tray_note.setObjectName("muted")
        footer.addWidget(tray_note)
        root.addLayout(footer)

        self.setCentralWidget(central)
        self.setStyleSheet(STYLE_SHEET)

    def _build_tray(self) -> None:
        self.tray = QSystemTrayIcon(app_icon(), self)
        self.tray.setToolTip("小说对话角色标注控制台")
        menu = QMenu()
        show_action = QAction("显示控制台", self)
        show_action.triggered.connect(self._show_window)
        continue_action = QAction("全部继续", self)
        continue_action.triggered.connect(self._continue_all)
        stop_action = QAction("全部停止", self)
        stop_action.triggered.connect(self._stop_all)
        quit_action = QAction("退出", self)
        quit_action.triggered.connect(self._quit_application)
        menu.addAction(show_action)
        menu.addSeparator()
        menu.addAction(continue_action)
        menu.addAction(stop_action)
        menu.addSeparator()
        menu.addAction(quit_action)
        self.tray.setContextMenu(menu)
        self.tray.activated.connect(self._tray_activated)
        self.tray.show()

    def _create_controllers(self) -> None:
        for spec in self.specs:
            controller = VolumeProcess(spec, self)
            controller.output.connect(self._on_output)
            controller.state_changed.connect(self._on_state_changed)
            controller.finished.connect(self._on_finished)
            self.controllers[spec.number] = controller

    def _load_settings(self) -> None:
        self.key_path.setText(str(self.settings.value("key_file", "")))
        self.log_path.setText(str(self.settings.value("log_dir", str(DEFAULT_LOG_DIR))))
        self.autostart.blockSignals(True)
        self.autostart.setChecked(self.settings.value("autostart", False, type=bool))
        self.autostart.blockSignals(False)
        geometry = self.settings.value("geometry")
        if geometry:
            self.restoreGeometry(geometry)
        self._validate_key_file()

    def _save_settings(self) -> None:
        self.settings.setValue("key_file", self.key_path.text())
        self.settings.setValue("log_dir", self.log_path.text())
        self.settings.setValue("autostart", self.autostart.isChecked())
        self.settings.setValue("geometry", self.saveGeometry())
        self.settings.sync()

    def _validate_key_file(self) -> bool:
        path = Path(self.key_path.text().strip()) if self.key_path.text().strip() else None
        valid = bool(path and path.is_file())
        count = 0
        if valid and path:
            try:
                keys = {
                    line.strip()
                    for line in path.read_text(encoding="utf-8").splitlines()
                    if line.strip() and not line.strip().startswith("#")
                }
                count = len(keys)
            except OSError:
                valid = False
        self.key_warning.setVisible(not valid)
        self.key_count.setText(f"{count} 个 Key" if valid else "未选择")
        self.pool_label.setText(f"模型池  SenseNova × {count}  ·  Agnes × 3")
        self.all_continue.setEnabled(valid)
        for row in self.rows.values():
            if row.state not in {
                "starting",
                "running",
                "reviewing",
                "waiting",
                "stopping",
                "finishing_error",
                "backup",
            }:
                row.continue_button.setEnabled(valid)
                row.restart_button.setEnabled(valid)
        return valid

    def _choose_key_file(self) -> None:
        current = self.key_path.text() or str(PROJECT_ROOT / "config")
        selected, _filter = QFileDialog.getOpenFileName(
            self,
            "选择 API Key 文件",
            current,
            "所有文件 (*)",
        )
        if selected:
            self.key_path.setText(selected)
            self._save_settings()
            self._validate_key_file()

    def _choose_log_directory(self) -> None:
        selected = QFileDialog.getExistingDirectory(
            self,
            "选择运行日志目录",
            self.log_path.text() or str(DEFAULT_LOG_DIR),
        )
        if selected:
            self.log_path.setText(selected)
            self._save_settings()

    def _toggle_autostart(self, checked: bool) -> None:
        try:
            set_windows_autostart(checked)
            self._save_settings()
        except Exception as exc:
            self.autostart.blockSignals(True)
            self.autostart.setChecked(not checked)
            self.autostart.blockSignals(False)
            QMessageBox.critical(self, "开机启动设置失败", str(exc))

    def _start_volume(self, number: int, reset: bool = False) -> None:
        if not self._validate_key_file():
            QMessageBox.warning(self, "缺少 API Key 文件", "请先选择 API Key 文件。")
            return
        controller = self.controllers[number]
        if controller.running:
            return
        controller.start(
            Path(self.key_path.text()),
            Path(self.log_path.text() or DEFAULT_LOG_DIR),
            reset=reset,
        )

    def _continue_volume(self, number: int) -> None:
        self._start_volume(number, reset=False)

    def _restart_volume(self, number: int) -> None:
        answer = QMessageBox.question(
            self,
            f"重新标注第{number}卷",
            "将先备份当前标注和日志，然后清空本卷状态并从头开始。是否继续？",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        self.pending_restart.add(number)
        self._backup_volume(number)

    def _backup_volume(self, number: int) -> None:
        controller = self.controllers.get(number)
        if controller and controller.running:
            QMessageBox.information(self, "任务正在运行", "请先停止本卷，再创建一致的备份。")
            self.pending_restart.discard(number)
            return
        row = self.rows[number]
        row.set_status("backup", "正在复制标注和日志")
        task = BackupTask(
            self.spec_by_number[number],
            PROJECT_ROOT / "backup" / "control_center",
        )
        task.signals.finished.connect(self._backup_finished)
        self.thread_pool.start(task)

    def _backup_finished(self, number: int, target: str, copied: int, error: str) -> None:
        if error:
            self.rows[number].set_status("error", f"备份失败：{error}")
            self.pending_restart.discard(number)
            return
        if copied:
            self.rows[number].set_status("idle", f"已备份 {copied} 个文件")
            self.statusBar().showMessage(f"第{number}卷备份完成：{target}", 10000)
        else:
            self.rows[number].set_status("idle", "内容未变化")
            self.statusBar().showMessage(
                f"第{number}卷与上次备份相同，未重复创建备份",
                10000,
            )
        if number in self.pending_restart:
            self.pending_restart.discard(number)
            self._start_volume(number, reset=True)

    def _backup_all(self) -> None:
        for spec in self.specs:
            if not self.controllers[spec.number].running:
                self._backup_volume(spec.number)

    def _continue_all(self) -> None:
        for spec in self.specs:
            self._start_volume(spec.number, reset=False)

    def _stop_volume(self, number: int) -> None:
        self.controllers[number].stop()

    def _stop_all(self) -> None:
        for controller in self.controllers.values():
            controller.stop()

    def _on_output(self, number: int, text: str) -> None:
        lines = text.splitlines()
        self.output_buffers[number].extend(lines)
        if self.console_volume.currentData() == number:
            self.console.moveCursor(QTextCursor.MoveOperation.End)
            self.console.insertPlainText(text)
            self.console.ensureCursorVisible()

    def _on_state_changed(self, number: int, state: str, detail: str) -> None:
        self.rows[number].set_status(state, detail)
        if state in {"idle", "error", "complete"}:
            self.rows[number].set_eta()
        self._validate_key_file()
        self._update_summary()

    def _on_finished(self, number: int, _exit_code: int) -> None:
        controller = self.controllers[number]
        if controller.latest_log_path:
            self.latest_logs[number] = controller.latest_log_path
        self._refresh_progress(number)

    def _refresh_progress(self, number: int) -> None:
        spec = self.spec_by_number[number]
        progress = read_progress(spec)
        self.progress_cache[number] = progress
        row = self.rows[number]
        controller = self.controllers.get(number)
        review = read_review_progress(spec)

        if controller and controller.running and progress.complete and review and review.active:
            hint = controller.latest_progress_hint
            review_completed = review.completed
            review_total = review.total
            if (
                review_total <= 0
                and hint is not None
                and hint.phase != "first-pass"
            ):
                review_completed = hint.completed
                review_total = hint.total
            row.update_progress(
                review_completed,
                review_total,
                0
                if review_total <= 0
                else min(100, round(review_completed * 100 / review_total)),
                stage="复审",
            )
            eta = review.eta_seconds
            if eta is None and hint is not None and hint.phase != "first-pass":
                eta = hint.eta_seconds
            row.set_eta(review.phase_label, eta)
            if row.state not in {"waiting", "stopping", "finishing_error"}:
                row.set_status("reviewing", review.phase_label)
            return

        row.update_progress(progress.labeled, progress.total, progress.percent)
        if controller and controller.running and not progress.complete:
            row.set_eta(
                "首轮标注",
                controller.estimated_first_pass_eta(
                    progress.labeled,
                    max(0, progress.total - progress.labeled),
                ),
            )
            if row.state == "reviewing":
                row.set_status("running", "正在进行首轮标注")
        elif not controller or not controller.running:
            row.set_eta()

    def _refresh_all_progress(self, initial: bool = False) -> None:
        for spec in self.specs:
            self._refresh_progress(spec.number)
            row = self.rows[spec.number]
            if initial:
                failure = read_recent_failure(spec)
                progress = self.progress_cache[spec.number]
                if failure and not progress.complete:
                    row.set_status("error", failure)
                elif progress.complete:
                    row.set_status("idle", "首轮已完成，可继续复审")
                elif progress.labeled:
                    row.set_status("idle", "已有断点")
                else:
                    row.set_status("idle", "尚未开始")
        self._validate_key_file()
        self._update_summary()

    def _update_summary(self) -> None:
        running = sum(controller.running for controller in self.controllers.values())
        errors = sum(row.state == "error" for row in self.rows.values())
        completed = sum(
            self.progress_cache.get(spec.number, read_progress(spec)).complete
            for spec in self.specs
        )
        self.summary.setText(f"运行中 {running}  ·  出错 {errors}  ·  首轮完成 {completed}/5")

    def _switch_console(self) -> None:
        number = self.console_volume.currentData()
        self.console.setPlainText("\n".join(self.output_buffers[number]))
        self.console.moveCursor(QTextCursor.MoveOperation.End)

    def console_clear(self) -> None:
        number = self.console_volume.currentData()
        self.output_buffers[number].clear()
        self.console.clear()

    def _toggle_console(self) -> None:
        visible = not self.console.isVisible()
        self.console.setVisible(visible)
        self.clear_console_button.setVisible(visible)
        self.console_toggle.setText("收起输出" if visible else "显示输出")
        self.console_toggle.setIcon(
            self.style().standardIcon(
                QStyle.StandardPixmap.SP_ArrowUp
                if visible
                else QStyle.StandardPixmap.SP_ArrowDown
            )
        )

    def _open_log_directory(self) -> None:
        path = Path(self.log_path.text() or DEFAULT_LOG_DIR)
        path.mkdir(parents=True, exist_ok=True)
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(path)))

    def _open_volume_log(self, number: int) -> None:
        path = self.latest_logs.get(number)
        if not path:
            log_dir = Path(self.log_path.text() or DEFAULT_LOG_DIR)
            matches = sorted(log_dir.glob(f"volume{number}_*.log"), key=lambda item: item.stat().st_mtime)
            path = matches[-1] if matches else self.spec_by_number[number].annotation_log_path
        if path.exists():
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(path)))
        else:
            QMessageBox.information(self, "没有日志", "本卷还没有可打开的日志文件。")

    def _show_window(self) -> None:
        self.showNormal()
        self.raise_()
        self.activateWindow()

    def _tray_activated(self, reason) -> None:
        if reason in {
            QSystemTrayIcon.ActivationReason.Trigger,
            QSystemTrayIcon.ActivationReason.DoubleClick,
        }:
            self._show_window()

    def closeEvent(self, event: QCloseEvent) -> None:
        self._save_settings()
        if self._allow_quit:
            event.accept()
            return
        event.ignore()
        self.hide()
        if not self._tray_notice_shown:
            self.tray.showMessage(
                "小说对话角色标注控制台",
                "任务仍在后台运行，可从系统托盘重新打开。",
                QSystemTrayIcon.MessageIcon.Information,
                3000,
            )
            self._tray_notice_shown = True

    def _quit_application(self) -> None:
        running = [controller for controller in self.controllers.values() if controller.running]
        if running:
            answer = QMessageBox.question(
                self,
                "退出控制台",
                "退出会停止当前标注进程。是否继续？",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if answer != QMessageBox.StandardButton.Yes:
                return
        self._allow_quit = True
        self._save_settings()
        for controller in running:
            controller.process.terminate()
            if not controller.process.waitForFinished(2000):
                controller.process.kill()
                controller.process.waitForFinished(1000)
        self.tray.hide()
        QApplication.instance().quit()


STYLE_SHEET = """
QWidget#central {
    background: #f3f5f7;
    color: #20262d;
    font-family: "Segoe UI", "Microsoft YaHei UI";
    font-size: 13px;
}
QLabel#title { font-size: 25px; font-weight: 650; color: #182026; }
QLabel#subtitle { color: #667085; font-size: 13px; }
QLabel#sectionTitle { font-size: 16px; font-weight: 650; color: #20262d; }
QLabel#volumeName { font-size: 17px; font-weight: 650; min-width: 72px; }
QLabel#progressCount { font-weight: 600; color: #344054; }
QLabel#muted { color: #667085; font-size: 12px; }
QLabel#etaLabel { color: #235b59; font-size: 12px; font-weight: 600; }
QLabel#poolLabel {
    color: #235b59;
    background: #e8f4f2;
    border: 1px solid #c7e3df;
    border-radius: 5px;
    padding: 7px 10px;
    font-weight: 600;
}
QLabel#warning {
    color: #9a4d00;
    background: #fff3df;
    border-left: 3px solid #d98200;
    padding: 7px 10px;
}
QFrame#settingsBand, QFrame#volumeList {
    background: #ffffff;
    border: 1px solid #dce1e6;
    border-radius: 6px;
}
QFrame#volumeRow {
    background: #ffffff;
    border: 0;
    border-bottom: 1px solid #e7eaee;
}
QFrame#volumeRow:hover { background: #fafcfc; }
QLineEdit, QComboBox {
    background: #ffffff;
    border: 1px solid #cfd6dc;
    border-radius: 5px;
    padding: 7px 9px;
    min-height: 20px;
    selection-background-color: #127c78;
}
QLineEdit:focus, QComboBox:focus { border: 1px solid #127c78; }
QPushButton, QToolButton {
    background: #ffffff;
    color: #28323b;
    border: 1px solid #cfd6dc;
    border-radius: 5px;
    padding: 6px 11px;
    min-height: 20px;
}
QPushButton:hover, QToolButton:hover { background: #f1f5f5; border-color: #9cadb0; }
QPushButton:pressed, QToolButton:pressed { background: #e4eceb; }
QPushButton:disabled, QToolButton:disabled { color: #a5adb5; background: #f4f5f6; }
QPushButton#primaryButton {
    color: #ffffff;
    background: #127c78;
    border-color: #127c78;
    font-weight: 600;
}
QPushButton#primaryButton:hover { background: #0f6c68; }
QProgressBar { background: #e1e6e9; border: 0; border-radius: 4px; }
QProgressBar::chunk { background: #127c78; border-radius: 4px; }
QPlainTextEdit#console {
    background: #20262d;
    color: #d9e3e6;
    border: 1px solid #151a1f;
    border-radius: 5px;
    padding: 8px;
    font-family: "Cascadia Mono", "Consolas";
    font-size: 12px;
}
QCheckBox#switch { spacing: 9px; font-weight: 600; }
QCheckBox#switch::indicator { width: 36px; height: 20px; }
QCheckBox#switch::indicator:unchecked {
    image: none;
    background: #b8c0c7;
    border: 1px solid #aab2b9;
    border-radius: 10px;
}
QCheckBox#switch::indicator:checked {
    image: none;
    background: #127c78;
    border: 1px solid #127c78;
    border-radius: 10px;
}
QStatusBar { background: #f3f5f7; color: #52606b; }
"""


def configure_worker_stdio() -> None:
    """Make the packaged worker's pipe output independent of Windows ACP."""
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        reconfigure = getattr(stream, "reconfigure", None)
        if not callable(reconfigure):
            continue
        try:
            reconfigure(
                encoding="utf-8",
                errors="backslashreplace",
                write_through=True,
            )
        except (OSError, TypeError, ValueError):
            pass


def run_packaged_worker(arguments: list[str]) -> int:
    configure_worker_stdio()
    if arguments == ["--encoding-smoke-test"]:
        from run_label import _writeline

        _writeline("UTF-8 packaged worker: • 中文输出正常")
        return 0

    sys.argv = ["run_label.py", *arguments]
    from run_label import main as run_label_main

    result = run_label_main()
    return int(result or 0)


def main() -> int:
    if "--worker" in sys.argv[1:]:
        worker_index = sys.argv.index("--worker")
        return run_packaged_worker(sys.argv[worker_index + 1 :])

    parser = argparse.ArgumentParser(description="NovelSpeaker desktop control center")
    parser.add_argument("--minimized", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--version", action="version", version=APP_VERSION)
    args = parser.parse_args()

    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )
    app = QApplication(sys.argv[:1])
    app.setApplicationName(APP_NAME)
    app.setOrganizationName("NovelSpeaker")
    app.setQuitOnLastWindowClosed(False)
    app.setWindowIcon(app_icon())
    window = ControlCenter(start_minimized=args.minimized)
    if args.smoke_test:
        window._allow_quit = True
        QTimer.singleShot(500, app.quit)
    elif not args.minimized:
        window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
