#!/usr/bin/env python3
"""
yt_downloader_all_fixes_ffmpeg_config_full.py

Full script with two ffmpeg options:
  1) Default path on Windows: <script_dir>/ffmpeg/bin/ffmpeg.exe
  2) User-configurable path via Settings (persisted in config.json)

This version:
 - Resolves ffmpeg for yt-dlp using `ffmpeg_location` so merging works in frozen builds
 - Allows user to set ffmpeg path in Settings
 - Attempts to use bundled icon for main window and settings dialog (works for frozen onedir)
 - Keeps .part resume files and other behavior from earlier edits
"""
import sys
import threading
import json
import queue as pyqueue
import time
import uuid
import argparse
import os
import tempfile
import shutil
import unicodedata
import platform
import re
from pathlib import Path
from collections import deque
from yt_dlp import YoutubeDL
import subprocess
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

# -------------------------
# Determine sensible default ffmpeg location
# -------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
if platform.system() == "Windows":
    DEFAULT_FFMPEG = str((SCRIPT_DIR / "ffmpeg" / "bin" / "ffmpeg.exe").resolve())
else:
    DEFAULT_FFMPEG = "ffmpeg"  # rely on PATH on non-Windows by default

# -------------------------
# Config / persistence
# -------------------------
APP_DIR = Path.cwd().resolve() / "config"
APP_DIR.mkdir(parents=True, exist_ok=True)
QUEUE_FILE = APP_DIR / "queue.json"

DEFAULT_CONFIG = {
    "audio_bitrate": "192k",
    "merge_output_format": "mp4",
    "theme": "light",
    "window_geometry": None,
    "ffmpeg_path": DEFAULT_FFMPEG
}
CONFIG_FILE = APP_DIR / "config.json"

# On Windows hide child console windows launched by subprocesses
if platform.system() == "Windows":
    _orig_popen = subprocess.Popen

    def _popen_no_window(*args, **kwargs):
        si = subprocess.STARTUPINFO()
        si.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        si.wShowWindow = subprocess.SW_HIDE
        kwargs.setdefault("startupinfo", si)
        kwargs.setdefault("creationflags", getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000))
        return _orig_popen(*args, **kwargs)

    subprocess.Popen = _popen_no_window

# -------------------------
# bundled resource helper
# -------------------------
def bundled_resource_path(rel_path: str) -> Path:
    """
    Resolve a path to a bundled resource. If running frozen (PyInstaller),
    base is sys._MEIPASS, else it's the script folder.
    """
    try:
        if getattr(sys, 'frozen', False) and hasattr(sys, '_MEIPASS'):
            base = Path(sys._MEIPASS)
        else:
            base = Path(__file__).resolve().parent
        return (base / rel_path).resolve()
    except Exception:
        return (Path(__file__).resolve().parent / rel_path).resolve()

def load_config():
    if CONFIG_FILE.exists():
        try:
            cfg = json.load(CONFIG_FILE.open("r", encoding="utf-8"))
            for k, v in DEFAULT_CONFIG.items():
                if k not in cfg:
                    cfg[k] = v
            return cfg
        except Exception:
            return DEFAULT_CONFIG.copy()
    else:
        save_config(DEFAULT_CONFIG.copy())
        return DEFAULT_CONFIG.copy()

def save_config(cfg):
    try:
        with CONFIG_FILE.open("w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2)
    except Exception as e:
        print("Failed to save config:", e)

CONFIG = load_config()

# -------------------------
# Resolve ffmpeg executable for yt-dlp
# -------------------------
def resolve_ffmpeg_exe() -> str:
    """
    Return the path to an ffmpeg executable to tell yt-dlp about.
    Priority:
      1. CONFIG['ffmpeg_path'] if set and exists (accepts absolute or relative paths)
      2. If frozen and a bundled path exists under sys._MEIPASS, use that
      3. If the above looks like a path but doesn't exist, try SCRIPT_DIR relative resolution
      4. Finally return "ffmpeg" (rely on PATH)
    """
    ff = None
    try:
        if isinstance(CONFIG, dict):
            ff = CONFIG.get("ffmpeg_path")
    except Exception:
        ff = None
    if not ff:
        ff = DEFAULT_FFMPEG

    # 1) if absolute/existing path -> return
    try:
        p = Path(ff)
        if p.exists():
            return str(p)
    except Exception:
        pass

    # 2) if frozen, try bundled path
    try:
        if getattr(sys, "frozen", False):
            candidate = bundled_resource_path(ff)
            if candidate.exists():
                return str(candidate)
    except Exception:
        pass

    # 3) if path-like but not found, try relative to script dir
    try:
        if (os.path.sep in ff) or (os.path.altsep and os.path.altsep in ff):
            rel_candidate = (SCRIPT_DIR / ff).resolve()
            if rel_candidate.exists():
                return str(rel_candidate)
    except Exception:
        pass

    # 4) fallback to plain "ffmpeg"
    return "ffmpeg"

# -------------------------
# Small helpers & logger
# -------------------------
_ANSI_RE = re.compile(r'\x1b\[[0-9;]*m')
def strip_ansi(s: str) -> str:
    if not s:
        return ""
    return _ANSI_RE.sub("", str(s))

class _YTDLLogger:
    def debug(self, msg): pass
    def warning(self, msg): pass
    def error(self, msg): pass

def bytes_to_human(n):
    if n is None:
        return "Unknown"
    try:
        n = float(n)
    except Exception:
        return "Unknown"
    symbols = ('B','KB','MB','GB','TB')
    prefix = 0
    while n >= 1024 and prefix < len(symbols)-1:
        n /= 1024.0
        prefix += 1
    return f"{n:.2f} {symbols[prefix]}"

def sanitize_path_component(s: str) -> str:
    s = (s or "")
    s = s.strip()
    invalid = r'\/:*?"<>|'
    return "".join(c for c in s if c not in invalid).strip()

def run_cmd(cmd):
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return proc.returncode, proc.stdout, proc.stderr

def normalize_res_input(res):
    if not res:
        return None
    r = str(res).strip()
    if not r:
        return None
    if r.lower().endswith('p'):
        return r.lower()
    if r.isdigit():
        return r + 'p'
    return r.lower()

def is_plausible_youtube_url(url: str) -> bool:
    if not url:
        return False
    u = str(url).strip()
    if 'youtube.com/watch' in u and 'v=' in u:
        try:
            vid = u.split('v=')[1].split('&')[0]
            return len(vid) >= 8
        except Exception:
            return False
    return True

# -------------------------
# Partial cleanup helper (preserve .part for resume)
# -------------------------
def cleanup_partial_files(dest_dir):
    removed = []
    try:
        p = Path(dest_dir)
        if not p.exists() or not p.is_dir():
            return removed
        patterns = ["*.part~", "*.part.tmp", "*.download", "*.aria2", "*~"]
        for pat in set(patterns):
            for f in p.glob(pat):
                try:
                    f.unlink()
                    removed.append(str(f))
                except Exception:
                    pass
        return removed
    except Exception:
        return removed

# -------------------------
# Task and TaskQueue
# -------------------------
class DownloadTask:
    def __init__(self, url, dest='.', resolution=None, audio_only=False, audio_format='mp3', tid=None, status='queued', title=None):
        self.id = tid or str(uuid.uuid4())
        self.url = url
        self.dest = str(Path(dest).expanduser().resolve())
        self.resolution = resolution
        self.audio_only = bool(audio_only)
        self.audio_format = audio_format
        self.status = status
        self.title = title

    def to_dict(self):
        return {
            'id': self.id,
            'url': self.url,
            'dest': self.dest,
            'resolution': self.resolution,
            'audio_only': self.audio_only,
            'audio_format': self.audio_format,
            'status': self.status,
            'title': self.title,
        }

    @staticmethod
    def from_dict(d):
        return DownloadTask(
            url=d['url'],
            dest=d.get('dest','.'), resolution=d.get('resolution'),
            audio_only=d.get('audio_only', False),
            audio_format=d.get('audio_format','mp3'),
            tid=d.get('id'), status=d.get('status','queued'), title=d.get('title')
        )

class TaskQueue:
    def __init__(self):
        self._dq = deque()
        self._cv = threading.Condition()

    def put(self, item):
        with self._cv:
            self._dq.append(item)
            self._cv.notify()

    def put_left(self, item):
        with self._cv:
            self._dq.appendleft(item)
            self._cv.notify()

    def get(self, timeout=None):
        with self._cv:
            if timeout is None:
                while not self._dq:
                    self._cv.wait()
            else:
                end = time.time() + timeout
                while not self._dq:
                    remaining = end - time.time()
                    if remaining <= 0:
                        raise pyqueue.Empty
                    self._cv.wait(remaining)
            return self._dq.popleft()

    def qsize(self):
        with self._cv:
            return len(self._dq)

    def empty(self):
        with self._cv:
            return len(self._dq) == 0

    def clear(self):
        with self._cv:
            count = len(self._dq)
            self._dq.clear()
            return count

    def to_list(self):
        with self._cv:
            return list(self._dq)

    def extend(self, items):
        with self._cv:
            for it in items:
                self._dq.append(it)
            self._cv.notify_all()

    def remove_by_id(self, tid):
        with self._cv:
            for i, t in enumerate(self._dq):
                if t.id == tid:
                    del self._dq[i]
                    return True
            return False

    def move_up(self, tid):
        with self._cv:
            for i, t in enumerate(self._dq):
                if t.id == tid and i > 0:
                    self._dq[i], self._dq[i-1] = self._dq[i-1], self._dq[i]
                    return True
            return False

    def move_down(self, tid):
        with self._cv:
            n = len(self._dq)
            for i, t in enumerate(self._dq):
                if t.id == tid and i < n-1:
                    self._dq[i], self._dq[i+1] = self._dq[i+1], self._dq[i]
                    return True
            return False

# -------------------------
# yt-dlp helpers & conversion (uses ffmpeg_location)
# -------------------------
class DownloadCancelled(Exception):
    pass

def fetch_info(url):
    ffexe = resolve_ffmpeg_exe()
    opts = {'quiet': True, 'no_warnings': True, 'logger': _YTDLLogger()}
    if ffexe:
        opts['ffmpeg_location'] = ffexe
    with YoutubeDL(opts) as ydl:
        return ydl.extract_info(url, download=False)

def list_available_resolutions(info):
    res = set()
    for f in info.get('formats', []):
        if f.get('vcodec') and f.get('vcodec') != 'none' and f.get('height'):
            res.add(f"{f['height']}p")
    return sorted(res, key=lambda r: int(r[:-1]))

def estimate_size_for_resolution(info, desired_res=None, audio_only=False):
    formats = info.get('formats', [])
    video_candidates = [f for f in formats if f.get('vcodec') and f.get('vcodec') != 'none']
    audio_candidates = [f for f in formats if f.get('acodec') and f.get('acodec') != 'none']

    chosen_video = None
    chosen_audio = None

    if audio_only:
        if not audio_candidates:
            return (None, None, None)
        chosen_audio = sorted(audio_candidates, key=lambda x: (x.get('filesize') or x.get('filesize_approx') or 0), reverse=True)[0]
        audio_bytes = chosen_audio.get('filesize') or chosen_audio.get('filesize_approx')
        return (audio_bytes, None, chosen_audio)

    if desired_res:
        try:
            target_h = int(desired_res[:-1])
        except Exception:
            target_h = None
        if target_h:
            exact = [f for f in video_candidates if f.get('height') == target_h]
            if exact:
                chosen_video = sorted(exact, key=lambda x: (x.get('filesize') or x.get('filesize_approx') or 0), reverse=True)[0]

    if not chosen_video and video_candidates:
        chosen_video = sorted(video_candidates, key=lambda x: (x.get('height') or 0, x.get('tbr') or 0), reverse=True)[0]

    if audio_candidates:
        chosen_audio = sorted(audio_candidates, key=lambda x: (x.get('abr') or 0, x.get('filesize') or x.get('filesize_approx') or 0), reverse=True)[0]

    video_bytes = chosen_video.get('filesize') or chosen_video.get('filesize_approx') if chosen_video else 0
    audio_bytes = chosen_audio.get('filesize') or chosen_audio.get('filesize_approx') if chosen_audio else 0

    total = None
    try:
        total = int(video_bytes or 0) + int(audio_bytes or 0)
    except Exception:
        total = None

    chosen_res_str = f"{chosen_video.get('height')}p" if chosen_video and chosen_video.get('height') else None
    return (total, chosen_res_str, (chosen_video, chosen_audio))

def ytdl_download(url, out_dir, format_spec, progress_hook=None, quiet=True, cancel_event: threading.Event = None, merge_output_format=None):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    outtmpl = str(out_dir / "%(title)s.%(ext)s")

    def _wrap_hook(d):
        if cancel_event and cancel_event.is_set():
            raise DownloadCancelled("Download cancelled by user")
        if progress_hook:
            try:
                progress_hook(d)
            except DownloadCancelled:
                raise
            except Exception:
                pass

    ffexe = resolve_ffmpeg_exe()
    opts = {
        'format': format_spec,
        'outtmpl': outtmpl,
        'noplaylist': True,
        'merge_output_format': merge_output_format or CONFIG.get("merge_output_format", "mp4"),
        'progress_hooks': [_wrap_hook],
        'quiet': quiet,
        'no_warnings': True,
        'logger': _YTDLLogger(),
    }
    if ffexe:
        opts['ffmpeg_location'] = ffexe

    with YoutubeDL(opts) as ydl:
        try:
            info = ydl.extract_info(url, download=True)
            filename = ydl.prepare_filename(info)
            p = Path(filename)
            if p.exists():
                return p
            alt = p.with_suffix('.' + (merge_output_format or CONFIG.get("merge_output_format", "mp4")))
            if alt.exists():
                return alt
            title = info.get('title') or ''
            safe = sanitize_path_component(title)
            candidates = list(out_dir.glob(safe + '*'))
            return candidates[0] if candidates else Path(filename)
        except DownloadCancelled:
            try:
                cleanup_partial_files(out_dir)
            except Exception:
                pass
            raise
        except Exception:
            try:
                cleanup_partial_files(out_dir)
            except Exception:
                pass
            raise

def make_ascii_safe(s: str, max_len: int = 80) -> str:
    if not s:
        s = "audio"
    nk = unicodedata.normalize("NFKD", s)
    out_chars = []
    for ch in nk:
        if ch.isalnum() or ch in (' ', '.', '-', '_'):
            out_chars.append(ch)
        else:
            out_chars.append('_')
    out = ''.join(out_chars).strip()
    out = out.replace(' ', '_')
    if len(out) > max_len:
        out = out[:max_len]
    if not out:
        out = "audio"
    return out

def postconvert_audio(input_file: Path, output_format: str, audio_bitrate=None):
    input_file = Path(input_file)
    if audio_bitrate is None:
        audio_bitrate = CONFIG.get("audio_bitrate", "192k")

    if input_file.suffix.lower() == f".{output_format.lower()}":
        return input_file

    final_out = input_file.with_suffix('.' + output_format)
    base_safe = make_ascii_safe(input_file.stem)
    tmp_fd, tmp_path = tempfile.mkstemp(prefix=f"{base_safe}_conv_", suffix='.' + output_format, dir=str(input_file.parent))
    os.close(tmp_fd)

    try:
        ffmpeg_exe = resolve_ffmpeg_exe()

        # If ffmpeg_exe looks like a path (contains a separator) but doesn't exist, fall back to 'ffmpeg'
        try:
            if (os.path.sep in ffmpeg_exe or (os.path.altsep and os.path.altsep in ffmpeg_exe)) and not Path(ffmpeg_exe).exists():
                rel_candidate = (SCRIPT_DIR / ffmpeg_exe).resolve()
                if rel_candidate.exists():
                    ffmpeg_exe = str(rel_candidate)
                else:
                    ffmpeg_exe = "ffmpeg"
        except Exception:
            ffmpeg_exe = "ffmpeg"

        cmd = [ffmpeg_exe, '-y', '-i', str(input_file), '-vn']
        if output_format == 'mp3':
            cmd += ['-c:a', 'libmp3lame', '-b:a', audio_bitrate]
        elif output_format == 'aac':
            cmd += ['-c:a', 'aac', '-b:a', audio_bitrate]
        elif output_format == 'opus':
            cmd += ['-c:a', 'libopus', '-b:a', audio_bitrate]
        else:
            raise ValueError("Unsupported audio format: " + str(output_format))
        cmd.append(tmp_path)

        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if proc.returncode != 0:
            stderr = proc.stderr or proc.stdout
            try:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
            except Exception:
                pass
            raise RuntimeError(f"ffmpeg conversion failed: {stderr.strip()}")

        try:
            shutil.move(tmp_path, str(final_out))
        except Exception as e:
            try:
                shutil.copy(tmp_path, str(final_out))
                os.remove(tmp_path)
            except Exception:
                raise RuntimeError(f"Failed to move converted file to final destination: {e}")

        try:
            if input_file.exists():
                input_file.unlink()
        except Exception:
            pass

        return final_out

    finally:
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass

# -------------------------
# Persistence functions
# -------------------------
def _backup_queue_if_exists():
    try:
        if QUEUE_FILE.exists():
            ts = datetime_now_str()
            backup = APP_DIR / f"queue_backup_{ts}.json"
            shutil.copy2(str(QUEUE_FILE), str(backup))
            return backup
    except Exception:
        pass
    return None

def datetime_now_str():
    from datetime import datetime
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def save_queue_to_disk(tasks_list):
    try:
        try:
            _backup_queue_if_exists()
        except Exception:
            pass
        arr = [t.to_dict() for t in tasks_list]
        with open(QUEUE_FILE, 'w', encoding='utf-8') as f:
            json.dump(arr, f, indent=2)
    except Exception as e:
        print("Failed to save queue:", e)

def load_queue_from_disk():
    if not QUEUE_FILE.exists():
        return []
    try:
        with open(QUEUE_FILE, 'r', encoding='utf-8') as f:
            arr = json.load(f)
        return [DownloadTask.from_dict(d) for d in arr]
    except Exception as e:
        print("Failed to load queue:", e)
        return []

def cleanup_queue_files():
    try:
        if QUEUE_FILE.exists():
            try:
                QUEUE_FILE.unlink()
            except Exception:
                pass
        for bk in APP_DIR.glob("queue_backup_*.json"):
            try:
                bk.unlink()
            except Exception:
                pass
    except Exception:
        pass

# -------------------------
# Worker
# -------------------------
class DownloaderWorker(threading.Thread):
    def __init__(self, task_queue: TaskQueue, ui_queue: pyqueue.Queue, stop_event: threading.Event):
        super().__init__(daemon=True)
        self.task_queue = task_queue
        self.ui_queue = ui_queue
        self.stop_event = stop_event
        self.cancel_event = threading.Event()
        self.current_task_id = None
        self._lock = threading.Lock()

    def request_cancel(self):
        self.cancel_event.set()

    def ui_log(self, *args):
        self.ui_queue.put(('log', ' '.join(str(a) for a in args)))

    def ui_status(self, task_id, status):
        self.ui_queue.put(('task_status', task_id, status))

    def ui_progress(self, task_id, downloaded, total):
        self.ui_queue.put(('task_progress', task_id, downloaded, total))

    def run_yt_dlp(self, url, dest, format_spec, task_id):
        def hook(d):
            status = d.get('status')
            if status == 'downloading':
                downloaded = d.get('downloaded_bytes') or 0
                total = d.get('total_bytes') or d.get('total_bytes_estimate') or None
                self.ui_progress(task_id, downloaded, total)
            elif status == 'finished':
                downloaded = d.get('downloaded_bytes') or 0
                total = d.get('total_bytes') or d.get('total_bytes_estimate') or downloaded
                self.ui_progress(task_id, downloaded, total)
        return ytdl_download(url, dest, format_spec, progress_hook=hook, quiet=True, cancel_event=self.cancel_event, merge_output_format=CONFIG.get("merge_output_format"))

    def run(self):
        self.ui_log("Worker started")
        while not self.stop_event.is_set():
            try:
                task = self.task_queue.get(timeout=0.5)
            except pyqueue.Empty:
                continue
            if task is None:
                break

            try:
                info = fetch_info(task.url)
            except Exception as e:
                reason = strip_ansi(str(e))
                self.ui_log("Failed to fetch info for", task.url, " — ", reason)
                self.ui_status(task.id, 'failed')
                self.ui_queue.put(('task_failed_reason', task.id, reason))
                self.ui_queue.put(('persist_queue_now',))
                continue

            # playlist expansion
            try:
                if info.get('_type') == 'playlist' or info.get('entries'):
                    entries = info.get('entries') or []
                    playlist_title = sanitize_path_component(info.get('title') or "playlist")
                    playlist_folder = Path(task.dest) / playlist_title
                    playlist_folder.mkdir(parents=True, exist_ok=True)
                    self.ui_log(f"Playlist detected ({len(entries)} entries). Queuing into folder: {playlist_folder}")
                    for e in entries:
                        video_url = e.get('webpage_url') or (("https://www.youtube.com/watch?v=" + e.get('id')) if e.get('id') else None)
                        if video_url:
                            subtask = DownloadTask(video_url, dest=str(playlist_folder), resolution=task.resolution,
                                                   audio_only=task.audio_only, audio_format=task.audio_format)
                            self.task_queue.put(subtask)
                            self.ui_queue.put(('task_added', subtask.to_dict()))
                    self.ui_queue.put(('task_removed', task.id))
                    continue
            except Exception:
                pass

            # single video processing
            self.ui_log("Starting:", task.url)
            self.ui_status(task.id, 'downloading')
            with self._lock:
                self.current_task_id = task.id
                self.cancel_event.clear()

            # try metadata/title
            try:
                info_meta = fetch_info(task.url)
                title = info_meta.get('title') or ''
            except Exception as e:
                reason = strip_ansi(str(e))
                self.ui_log("Failed to fetch info for", task.url, " — ", reason)
                self.ui_status(task.id, 'failed')
                self.ui_queue.put(('task_failed_reason', task.id, reason))
                self.ui_queue.put(('persist_queue_now',))
                with self._lock:
                    self.current_task_id = None
                    self.cancel_event.clear()
                continue

            self.ui_queue.put(('task_update_title', task.id, title))

            try:
                if task.audio_only:
                    self.ui_log("Downloading audio-only for", task.url)
                    got = self.run_yt_dlp(task.url, task.dest, 'bestaudio', task.id)
                    self.ui_log("Audio downloaded:", got)
                    if task.audio_format:
                        self.ui_log("Converting audio to", task.audio_format)
                        conv = postconvert_audio(got, task.audio_format, audio_bitrate=CONFIG.get("audio_bitrate"))
                        self.ui_log("Converted:", conv)
                else:
                    fmt = None
                    if task.resolution and isinstance(task.resolution, str) and task.resolution.endswith('p'):
                        try:
                            height = int(task.resolution[:-1])
                            fmt = f"bestvideo[height={height}]+bestaudio/best"
                        except Exception:
                            fmt = None
                    if fmt:
                        try:
                            got = self.run_yt_dlp(task.url, task.dest, fmt, task.id)
                            self.ui_log("Downloaded (requested res):", got)
                        except DownloadCancelled:
                            self.ui_log("Download cancelled for", task.url)
                            try:
                                cleanup_partial_files(task.dest)
                            except Exception:
                                pass
                            self.ui_status(task.id, 'canceled')
                            self.ui_queue.put(('persist_queue_now',))
                            with self._lock:
                                self.current_task_id = None
                                self.cancel_event.clear()
                            continue
                        except Exception as e:
                            self.ui_log("Requested resolution not available or failed:", e)
                            try:
                                got = self.run_yt_dlp(task.url, task.dest, 'bestvideo+bestaudio/best', task.id)
                                self.ui_log("Downloaded (fallback):", got)
                            except DownloadCancelled:
                                self.ui_log("Download cancelled for", task.url)
                                try:
                                    cleanup_partial_files(task.dest)
                                except Exception:
                                    pass
                                self.ui_status(task.id, 'canceled')
                                self.ui_queue.put(('persist_queue_now',))
                                with self._lock:
                                    self.current_task_id = None
                                    self.cancel_event.clear()
                                continue
                    else:
                        try:
                            got = self.run_yt_dlp(task.url, task.dest, 'bestvideo+bestaudio/best', task.id)
                            self.ui_log("Downloaded:", got)
                        except DownloadCancelled:
                            self.ui_log("Download cancelled for", task.url)
                            try:
                                cleanup_partial_files(task.dest)
                            except Exception:
                                pass
                            self.ui_status(task.id, 'canceled')
                            self.ui_queue.put(('persist_queue_now',))
                            with self._lock:
                                self.current_task_id = None
                                self.cancel_event.clear()
                            continue

                self.ui_status(task.id, 'done')
            except Exception as e:
                self.ui_log("Download failed:", e)
                try:
                    cleanup_partial_files(task.dest)
                except Exception:
                    pass
                self.ui_status(task.id, 'failed')
            finally:
                with self._lock:
                    self.current_task_id = None
                    self.cancel_event.clear()
                self.ui_progress(task.id, 0, 1)
                self.ui_queue.put(('persist_queue_now',))

        self.ui_log("Worker stopped")

# -------------------------
# GUI App
# -------------------------
class App:
    STATUS_COLORS = {
        'queued': {'bg': "#f9e8a9", 'fg': '#000000'},
        'downloading': {'bg': "#aad0fd", 'fg': '#000000'},
        'done': {'bg': "#a8eca8", 'fg': '#000000'},
        'failed': {'bg': "#f48b8b", 'fg': '#000000'},
        'canceled': {'bg': "#F7CB99", 'fg': '#000000'},
    }

    def __init__(self, root):
        self.root = root
        root.title("YouTube Video and Audio Downloader v1.0")
        self.style = ttk.Style()

        # Attempt to set application icon (handles frozen bundle too)
        try:
            # prefer bundled icon in frozen state
            ico_path = None
            try:
                # try png first
                candidate = bundled_resource_path("icon.png")
                if candidate.exists():
                    ico_path = candidate
                else:
                    candidate = bundled_resource_path("icon.ico")
                    if candidate.exists():
                        ico_path = candidate
            except Exception:
                pass

            # fallback to icon next to script
            if ico_path is None:
                local = Path(__file__).resolve().parent / "icon.png"
                if local.exists():
                    ico_path = local
                else:
                    local = Path(__file__).resolve().parent / "icon.ico"
                    if local.exists():
                        ico_path = local

            if ico_path is not None and ico_path.exists():
                if ico_path.suffix.lower() == ".ico":
                    try:
                        self.root.iconbitmap(str(ico_path))
                    except Exception:
                        # iconbitmap may fail in some environments; also try PhotoImage
                        try:
                            self._app_icon_image = tk.PhotoImage(file=str(ico_path))
                            self.root.iconphoto(False, self._app_icon_image)
                        except Exception:
                            pass
                else:
                    try:
                        self._app_icon_image = tk.PhotoImage(file=str(ico_path))
                        self.root.iconphoto(False, self._app_icon_image)
                    except Exception:
                        pass
        except Exception:
            pass

        # queues and worker
        self.task_queue = TaskQueue()
        self.ui_queue = pyqueue.Queue()
        self.stop_event = threading.Event()
        self.worker = DownloaderWorker(self.task_queue, self.ui_queue, self.stop_event)
        self.worker.start()

        # mapping id->task
        self.tasks_map = {}

        # menu created after frames set up (but we create now)
        self._create_menu()

        main_pane = ttk.Panedwindow(root, orient='vertical')
        main_pane.grid(row=0, column=0, sticky='nsew')
        root.columnconfigure(0, weight=1)
        root.rowconfigure(0, weight=1)

        container = ttk.Frame(main_pane)
        main_pane.add(container, weight=1)

        self.canvas = tk.Canvas(container, highlightthickness=0)
        self.vscroll = ttk.Scrollbar(container, orient='vertical', command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=self.vscroll.set)
        self.canvas.grid(row=0, column=0, sticky='nsew')
        self.vscroll.grid(row=0, column=1, sticky='ns')
        container.columnconfigure(0, weight=1)
        container.rowconfigure(0, weight=1)

        self.frm = ttk.Frame(self.canvas, padding=10)
        self.canvas_window = self.canvas.create_window((0,0), window=self.frm, anchor='nw')

        for i in range(5):
            self.frm.columnconfigure(i, weight=1)
        self.frm.rowconfigure(6, weight=3)
        self.frm.rowconfigure(8, weight=0)
        self.frm.rowconfigure(10, weight=1)

        self.frm.bind('<Configure>', lambda e: self.canvas.configure(scrollregion=self.canvas.bbox('all')))

        self._platform = platform.system()
        if self._platform in ('Windows','Darwin'):
            self.canvas.bind_all('<MouseWheel>', self._on_mousewheel)
        else:
            self.canvas.bind_all('<Button-4>', self._on_mousewheel)
            self.canvas.bind_all('<Button-5>', self._on_mousewheel)

        # build UI pieces
        self._build_inputs()
        self._build_buttons()
        self._build_tree()
        self._build_tree_controls()
        self._build_progress_and_log()

        # log control state (after building log area)
        self._log_row = 10
        self._log_grid_opts = {'row': self._log_row, 'column': 0, 'columnspan': 5, 'pady': (8,0), 'sticky': 'nsew'}
        self._log_prev_weight = 1
        self._log_visible = True
        self._saved_root_geom = None

        # initial sizing
        self.root.update_idletasks()
        try:
            self.canvas.configure(scrollregion=self.canvas.bbox('all'))
            self.canvas.itemconfigure(self.canvas_window, width=self.canvas.winfo_width(), height=self.frm.winfo_reqheight())
        except Exception:
            pass

        root.bind('<Configure>', self._on_root_resize)

        # restore geometry if saved; otherwise set a roomy default first-run
        try:
            geom = CONFIG.get('window_geometry')
            if geom:
                try:
                    self.root.geometry(geom)
                except Exception:
                    pass
            else:
                try:
                    default_w, default_h = 1200, 800
                    sw = self.root.winfo_screenwidth()
                    sh = self.root.winfo_screenheight()
                    x = max(0, int((sw - default_w) / 2))
                    y = max(0, int((sh - default_h) / 8))
                    self.root.geometry(f"{default_w}x{default_h}+{x}+{y}")
                except Exception:
                    try:
                        self.root.geometry("1200x800")
                    except Exception:
                        pass
        except Exception:
            pass
        try:
            self.root.minsize(800, 520)
        except Exception:
            pass

        # load persisted queue and populate tree; re-enqueue non-done tasks
        persisted = load_queue_from_disk()
        for t in persisted:
            if t.id in self.tasks_map:
                existing = self.tasks_map[t.id]
                existing.url = t.url
                existing.dest = t.dest
                existing.resolution = t.resolution
                existing.audio_only = t.audio_only
                existing.audio_format = t.audio_format
                existing.status = t.status
                existing.title = t.title
                try:
                    self.tree_insert_task(existing)
                except Exception:
                    pass
                continue

            self.tasks_map[t.id] = t
            self.tree_insert_task(t)
            if t.status in ('queued', 'failed', 'canceled', 'downloading'):
                if is_plausible_youtube_url(t.url):
                    t.status = 'queued'
                    try:
                        self.task_queue.put(t)
                    except Exception:
                        pass
                else:
                    t.status = 'failed'
                    self.append_log(f"Skipping invalid persisted URL (marked failed): {t.url}")

        self.apply_theme(CONFIG.get('theme','light'))
        # small debug line to show resolved ffmpeg in log
        try:
            self.append_log("Resolved ffmpeg:", resolve_ffmpeg_exe())
        except Exception:
            pass
        self.root.after(200, self.poll_ui)

    # ---------------- Menu ----------------
    def _create_menu(self):
        menubar = tk.Menu(self.root)
        filem = tk.Menu(menubar, tearoff=0)
        filem.add_command(label='Settings', command=self.open_settings)
        filem.add_separator()
        filem.add_command(label='Exit', command=self._on_exit)
        menubar.add_cascade(label='File', menu=filem)

        self.view_menu = tk.Menu(menubar, tearoff=0)
        self.view_menu.add_command(label='Toggle Theme', command=self._toggle_theme)
        self.view_menu.add_command(label='Hide Log Panel', command=self.toggle_log_panel)
        self._toggle_log_menu_index = self.view_menu.index('end')
        menubar.add_cascade(label='View', menu=self.view_menu)

        helpm = tk.Menu(menubar, tearoff=0)
        helpm.add_command(label='About', command=self._show_about)
        menubar.add_cascade(label='Help', menu=helpm)

        try:
            self.root.config(menu=menubar)
        except Exception:
            pass

    def _show_about(self):
        ff = CONFIG.get('ffmpeg_path', DEFAULT_FFMPEG)
        messagebox.showinfo('About', f'YouTube Video and Audio Downloader v1.0\n\nDeveloped with yt-dlp and Tkinter.\n\nSupports persistent download queue with resume on restart.')

    def _on_exit(self):
        try:
            geom = self.root.geometry()
            if isinstance(CONFIG, dict):
                CONFIG['window_geometry'] = geom
                save_config(CONFIG)
        except Exception:
            pass
        try:
            self.save_queue()
        except Exception:
            pass

        self.stop_event.set()
        try:
            self.task_queue.put(None)
        except Exception:
            pass
        self.root.quit()

    def _toggle_theme(self):
        current = CONFIG.get('theme','light')
        new = 'dark' if current=='light' else 'light'
        CONFIG['theme'] = new
        save_config(CONFIG)
        self.apply_theme(new)

    # ---------------- UI builders (inputs/buttons/tree/log) ----------------
    def _build_inputs(self):
        ttk.Label(self.frm, text="YouTube URL (video or playlist):").grid(row=0, column=0, sticky='w')
        self.url_var = tk.StringVar()
        ttk.Entry(self.frm, textvariable=self.url_var).grid(row=0, column=1, columnspan=4, sticky='ew')

        ttk.Label(self.frm, text="Destination folder:").grid(row=1, column=0, sticky='w')
        self.dest_var = tk.StringVar(value=str(Path.cwd()))
        ttk.Entry(self.frm, textvariable=self.dest_var).grid(row=1, column=1, sticky='ew')

        self.btn_browse = ttk.Button(self.frm, text="Browse...", command=self.browse_dest)
        self.btn_browse.grid(row=1, column=2, sticky='w')
        self.btn_browse.configure(width=14)

        # labeled frame with title "FFmpeg"
        '''
        lf = ttk.LabelFrame(self.frm, text="FFmpeg: Use Settings to change the executable path.")
        lf.grid(row=2, column=3, sticky='w', padx=(6,0))

        tk.Label(
            lf,
            text=f"{CONFIG.get('ffmpeg_path', DEFAULT_FFMPEG)}",
            bg="#1505f1",   # light blue background
            fg="#F7F5F5",   # black text
            padx=6,
            pady=3
        ).grid(row=0, column=0, sticky='w')
        '''
        # ---------- Create a bold style for the LabelFrame title ----------
        style = ttk.Style()
        style.configure(
            "Bold.TLabelframe.Label", 
            font=("Segoe UI", 10, "bold")  # bold frame title
        )

        # ---------- Create the LabelFrame with the bold style ----------
        lf = ttk.LabelFrame(
            self.frm, 
            text="FFmpeg - Use Settings to change the executable path:",
            style="Bold.TLabelframe"
        )
        lf.grid(row=2, column=3, sticky='w', padx=(6,0))

        # ---------- Create the bold label inside ----------
        tk.Label(
            lf,
            text=f"{CONFIG.get('ffmpeg_path', DEFAULT_FFMPEG)}",
            font=("calibri", 10, "bold"),   # bold text
            bg="#e0fffc",
            fg="#004b47",
            padx=6,
            pady=3
        ).grid(row=0, column=0, sticky='w')


        ttk.Label(self.frm, text="Preferred resolution (e.g. 1080p) or blank for highest:").grid(row=2, column=0, sticky='w')

        self.res_var = tk.StringVar()
        self.res_combo = ttk.Combobox(self.frm, textvariable=self.res_var, width=12, state="readonly")
        self.res_combo.grid(row=2, column=1, sticky='w')
        self.res_combo['values'] = []

        self.btn_resolutions = ttk.Button(self.frm, text="Get Resolutions", command=self.get_resolutions)
        self.btn_resolutions.grid(row=2, column=2, sticky='w')
        self.btn_resolutions.configure(width=14)

        self.available_lbl = ttk.Label(self.frm, text="")
        self.available_lbl.grid(row=3, column=0, columnspan=5, sticky='w')

        self.audio_only_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(self.frm, text="Download Audio only", variable=self.audio_only_var).grid(row=4, column=0, sticky='w')
        ttk.Label(self.frm, text="Select Audio format:").grid(row=4, column=1, sticky='w')
        self.audio_format_var = tk.StringVar(value='mp3')
        ttk.OptionMenu(self.frm, self.audio_format_var, 'mp3', 'mp3', 'aac', 'opus').grid(row=4, column=2, sticky='w')

    def _build_buttons(self):
        self.btn_frame = ttk.Frame(self.frm)
        self.btn_frame.grid(row=5, column=0, columnspan=5, sticky='ew', pady=(6,6))
        for i in range(5):
            self.btn_frame.columnconfigure(i, weight=1)
        self.btn_estimate = ttk.Button(self.btn_frame, text="Estimate Size", command=self.estimate_size)
        self.btn_queue = ttk.Button(self.btn_frame, text="Queue URL / Playlist", command=self.queue_current)
        self.btn_now = ttk.Button(self.btn_frame, text="Download Now (front)", command=self.download_now)
        self.btn_retry_all = ttk.Button(self.btn_frame, text="Retry Failed (all)", command=self.retry_all_failed)
        self.btn_settings = ttk.Button(self.btn_frame, text="Settings", command=self.open_settings)
        self.btn_estimate.grid(row=0, column=0, sticky='ew', padx=3)
        self.btn_queue.grid(row=0, column=1, sticky='ew', padx=3)
        self.btn_now.grid(row=0, column=2, sticky='ew', padx=3)
        self.btn_retry_all.grid(row=0, column=3, sticky='ew', padx=3)
        self.btn_settings.grid(row=0, column=4, sticky='ew', padx=3)

    def _build_tree(self):
        cols = ('idx','title','url','dest','res','type','status')
        tree_frame = ttk.Frame(self.frm)
        tree_frame.grid(row=6, column=0, columnspan=5, sticky='nsew', pady=(6,0))
        tree_frame.columnconfigure(0, weight=1)
        tree_frame.rowconfigure(0, weight=1)

        self.tree = ttk.Treeview(tree_frame, columns=cols, show='headings')
        for c,label in zip(cols,('#','Title','URL','Destination','Res','Type','Status')):
            self.tree.heading(c, text=label)
            self.tree.column(c, anchor='w', width=120 if c!='title' else 340)
        self.tree.grid(row=0, column=0, sticky='nsew')

        tree_scroll = ttk.Scrollbar(tree_frame, orient='vertical', command=self.tree.yview)
        self.tree.configure(yscrollcommand=tree_scroll.set)
        tree_scroll.grid(row=0, column=1, sticky='ns')

        for st,col in self.STATUS_COLORS.items():
            try:
                self.tree.tag_configure(st, background=col['bg'], foreground=col['fg'])
            except Exception:
                pass

    def _build_tree_controls(self):
        ctrl_frame = ttk.Frame(self.frm)
        ctrl_frame.grid(row=7, column=0, columnspan=5, sticky='ew', pady=(6,0))
        num_cols = 6
        for i in range(num_cols):
            ctrl_frame.columnconfigure(i, weight=1)

        btn_move_up = ttk.Button(ctrl_frame, text="Move Up", command=lambda: self.move_selected('up'))
        btn_move_down = ttk.Button(ctrl_frame, text="Move Down", command=lambda: self.move_selected('down'))
        btn_retry = ttk.Button(ctrl_frame, text="Retry Selected", command=self.retry_selected)
        btn_remove = ttk.Button(ctrl_frame, text="Remove Selected", command=self.remove_selected)
        btn_cancel = ttk.Button(ctrl_frame, text="Cancel Download", command=self.cancel_download)
        btn_clear = ttk.Button(ctrl_frame, text="Clear Queue", command=self.clear_queue)

        btn_move_up.grid(row=0, column=0, sticky='ew', padx=8)
        btn_move_down.grid(row=0, column=1, sticky='ew', padx=8)
        btn_retry.grid(row=0, column=2, sticky='ew', padx=8)
        btn_remove.grid(row=0, column=3, sticky='ew', padx=8)
        btn_cancel.grid(row=0, column=4, sticky='ew', padx=8)
        btn_clear.grid(row=0, column=5, sticky='ew', padx=8)

    def _build_progress_and_log(self):
        self.progress = ttk.Progressbar(self.frm, mode='determinate')
        self.progress.grid(row=8, column=0, columnspan=5, sticky='ew', pady=(8,0))
        self.status_lbl = ttk.Label(self.frm, text='Idle')
        self.status_lbl.grid(row=9, column=0, columnspan=5, sticky='w')

        self.log_frame = ttk.Frame(self.frm)
        self.log_frame.grid(row=10, column=0, columnspan=5, pady=(8,0), sticky='nsew')
        self.frm.rowconfigure(10, weight=1)

        self.log_text = tk.Text(self.log_frame, height=18, wrap='word')
        self.log_text.grid(row=0, column=0, sticky='nsew')
        self.log_frame.columnconfigure(0, weight=1)
        self.log_frame.rowconfigure(0, weight=1)

        log_scroll = ttk.Scrollbar(self.log_frame, orient='vertical', command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=log_scroll.set)
        log_scroll.grid(row=0, column=1, sticky='ns')

    # ---------------- mouse & resize ----------------
    def _on_mousewheel(self, event):
        try:
            if self._platform == 'Windows':
                self.canvas.yview_scroll(int(-1*(event.delta/120)), 'units')
            elif self._platform == 'Darwin':
                delta = int(-1*(event.delta))
                if delta == 0:
                    delta = -1
                self.canvas.yview_scroll(delta, 'units')
            else:
                if getattr(event,'num',None) == 4:
                    self.canvas.yview_scroll(-1,'units')
                elif getattr(event,'num',None) == 5:
                    self.canvas.yview_scroll(1,'units')
        except Exception:
            pass

    def _on_root_resize(self, event):
        try:
            self.canvas.itemconfigure(self.canvas_window, width=self.canvas.winfo_width())
        except Exception:
            pass

    # ---------------- theme ----------------
    def apply_theme(self, theme_name: str):
        t = theme_name or 'light'
        try:
            if t == 'dark':
                self.style.theme_use('clam')
                bg, fg, entry_bg, text_bg = '#2b2b2b','#eaeaea','#3a3a3a','#1e1e1e'
            else:
                self.style.theme_use('default')
                bg, fg, entry_bg, text_bg = '#f7f7f7','#000000','#ffffff','#ffffff'
        except Exception:
            bg, fg, entry_bg, text_bg = '#f7f7f7','#000000','#ffffff','#ffffff'
        try:
            self.root.configure(bg=bg)
            self.style.configure('TLabel', background=bg, foreground=fg)
            self.style.configure('TFrame', background=bg)
            self.style.configure('TButton', padding=6)
            self.style.configure('TCheckbutton', background=bg, foreground=fg)
            self.style.configure('Treeview', background=entry_bg, fieldbackground=entry_bg, foreground=fg)
        except Exception:
            pass
        try:
            self.log_text.configure(bg=text_bg, fg=fg, insertbackground=fg)
        except Exception:
            pass

    # ---------------- UI actions (queue/save/clear) ----------------
    def browse_dest(self):
        folder = filedialog.askdirectory()
        if folder:
            self.dest_var.set(folder)

    def get_resolutions(self):
        url = self.url_var.get().strip()
        if not url:
            messagebox.showinfo("Info", "Enter URL first")
            return
        try:
            info = fetch_info(url)
            res = list_available_resolutions(info)
            if not res:
                self.available_lbl.config(text="No video resolutions found")
                self.res_combo['values'] = []
                return
            self.available_lbl.config(text="Available: " + ", ".join(res))
            self.res_combo['values'] = res
            self.res_var.set(res[-1])
            self.append_log("Available resolutions loaded: " + ", ".join(res))
        except Exception as e:
            messagebox.showerror("Error", f"Failed to fetch resolutions: {strip_ansi(str(e))}")

    def estimate_size(self):
        url = self.url_var.get().strip()
        if not url:
            messagebox.showinfo("Info", "Enter URL first")
            return
        try:
            info = fetch_info(url)
            audio_only = self.audio_only_var.get()
            desired_input = self.res_var.get().strip()
            desired = normalize_res_input(desired_input) or None
            est, chosen, _ = estimate_size_for_resolution(info, desired_res=desired, audio_only=audio_only)
            self.append_log(f"Estimated size: {bytes_to_human(est)} (chosen: {chosen})")
        except Exception as e:
            messagebox.showerror("Error", f"Estimate failed: {strip_ansi(str(e))}")

    def queue_current(self):
        url = self.url_var.get().strip()
        if not url:
            messagebox.showinfo("Info", "Enter URL first")
            return
        dest = self.dest_var.get().strip() or '.'
        res_input = self.res_var.get().strip()
        res = normalize_res_input(res_input) or None
        audio_only = self.audio_only_var.get()
        audio_format = self.audio_format_var.get()

        try:
            info = fetch_info(url)
            if info.get('_type') == 'playlist' or info.get('entries'):
                entries = info.get('entries') or []
                playlist_title = sanitize_path_component(info.get('title') or "playlist")
                playlist_folder = Path(dest) / playlist_title
                playlist_folder.mkdir(parents=True, exist_ok=True)
                added = 0
                for e in entries:
                    video_url = e.get('webpage_url') or (("https://www.youtube.com/watch?v=" + e.get('id')) if e.get('id') else None)
                    if video_url:
                        t = DownloadTask(video_url, dest=str(playlist_folder), resolution=res, audio_only=audio_only, audio_format=audio_format)
                        self.task_queue.put(t)
                        self.tasks_map[t.id] = t
                        self.tree_insert_task(t)
                        added += 1
                self.append_log(f"Queued {added} entries from playlist into folder: {playlist_folder}")
                self.save_queue()
                return
        except Exception as e:
            self.append_log("Playlist expansion locally failed, will queue the playlist URL for worker expansion:", strip_ansi(str(e)))

        t = DownloadTask(url, dest=dest, resolution=res, audio_only=audio_only, audio_format=audio_format)
        self.task_queue.put(t)
        self.tasks_map[t.id] = t
        self.tree_insert_task(t)
        self.append_log("Queued:", url)
        self.save_queue()

    def download_now(self):
        url = self.url_var.get().strip()
        if not url:
            messagebox.showinfo("Info", "Enter URL first")
            return
        dest = self.dest_var.get().strip() or '.'
        res_input = self.res_var.get().strip()
        res = normalize_res_input(res_input) or None
        audio_only = self.audio_only_var.get()
        audio_format = self.audio_format_var.get()

        t = DownloadTask(url, dest=dest, resolution=res, audio_only=audio_only, audio_format=audio_format)
        self.task_queue.put_left(t)
        self.tasks_map[t.id] = t
        self.tree_insert_task(t, at_top=True)
        self.append_log("Download Now queued (front):", url)
        self.save_queue()
        try:
            self.worker.request_cancel()
            self.append_log("Requested cancel of current download (preempt).")
        except Exception as e:
            self.append_log("Failed to request cancel:", e)

    def retry_all_failed(self):
        count = 0
        for tid, t in list(self.tasks_map.items()):
            if t.status == 'failed':
                t.status = 'queued'
                self.task_queue.put(t)
                if tid in self.tree.get_children():
                    vals = list(self.tree.item(tid, 'values'))
                    vals[6] = 'queued'
                    self.tree.item(tid, values=vals)
                    self.tree.item(tid, tags=('queued',))
                count += 1
        self.append_log(f"Retried {count} failed tasks.")
        self.save_queue()

    def retry_selected(self):
        sel = self.tree.selection()
        if not sel:
            messagebox.showinfo("Info", "Select a queued/failed task in the list first.")
            return
        for tid in sel:
            t = self.tasks_map.get(tid)
            if not t:
                continue
            t.status = 'queued'
            self.task_queue.put(t)
            vals = list(self.tree.item(tid, 'values'))
            vals[6] = 'queued'
            self.tree.item(tid, values=vals)
            self.tree.item(tid, tags=('queued',))
        self.append_log("Retried selected tasks.")
        self.save_queue()

    def remove_selected(self):
        sel = self.tree.selection()
        if not sel:
            messagebox.showinfo("Info", "Select task(s) to remove.")
            return
        for tid in sel:
            removed = self.task_queue.remove_by_id(tid)
            if tid in self.tasks_map:
                del self.tasks_map[tid]
            if tid in self.tree.get_children():
                self.tree.delete(tid)
        self.refresh_tree_indexes()
        self.append_log(f"Removed {len(sel)} selected tasks.")
        self.save_queue()

    def move_selected(self, direction):
        sel = self.tree.selection()
        if not sel:
            messagebox.showinfo("Info", "Select one task to move.")
            return
        tid = sel[0]
        if direction == 'up':
            ok = self.task_queue.move_up(tid)
            if ok:
                items = list(self.tree.get_children())
                idx = items.index(tid)
                if idx > 0:
                    self.tree.move(tid, '', idx-1)
            else:
                self.append_log("Cannot move up (either first item or not in queue).")
        else:
            ok = self.task_queue.move_down(tid)
            if ok:
                items = list(self.tree.get_children())
                idx = items.index(tid)
                if idx < len(items)-1:
                    self.tree.move(tid, '', idx+1)
            else:
                self.append_log("Cannot move down (either last item or not in queue).")
        self.refresh_tree_indexes()
        self.save_queue()

    def clear_queue(self):
        if not messagebox.askyesno("Clear queue", "Clear the queue and remove persisted queue files?"):
            return
        count = self.task_queue.clear()
        self.tasks_map.clear()
        for iid in list(self.tree.get_children()):
            self.tree.delete(iid)
        self.append_log(f"Cleared {count} queued tasks")
        try:
            cleanup_queue_files()
        except Exception:
            pass

    def cancel_download(self):
        try:
            sel = self.tree.selection()
        except Exception:
            sel = ()

        if not sel:
            try:
                if hasattr(self, 'worker') and self.worker is not None:
                    self.worker.request_cancel()
                    self.append_log("Requested cancel of current download.")
                else:
                    self.append_log("No worker available to cancel.")
            except Exception as e:
                self.append_log("Cancel request failed:", e)
            return

        canceled_count = 0
        tried_request_cancel = False
        for tid in list(sel):
            t = self.tasks_map.get(tid)
            try:
                cur = getattr(self.worker, 'current_task_id', None)
            except Exception:
                cur = None

            if tid == cur:
                try:
                    if hasattr(self, 'worker') and self.worker is not None:
                        self.worker.request_cancel()
                        tried_request_cancel = True
                        self.append_log(f"Requested cancel of active task {tid}.")
                except Exception as e:
                    self.append_log("Failed to request cancel for active task:", e)

                if t:
                    t.status = 'canceled'
                if tid in self.tree.get_children():
                    vals = list(self.tree.item(tid, 'values'))
                    vals[6] = 'canceled'
                    self.tree.item(tid, values=vals)
                    try:
                        self.tree.item(tid, tags=('canceled',))
                    except Exception:
                        pass
                canceled_count += 1
                if t:
                    try:
                        cleanup_partial_files(t.dest)
                    except Exception:
                        pass
                continue

            try:
                removed = False
                try:
                    removed = self.task_queue.remove_by_id(tid)
                except Exception:
                    removed = False

                if removed:
                    if t:
                        t.status = 'canceled'
                    if tid in self.tree.get_children():
                        vals = list(self.tree.item(tid, 'values'))
                        vals[6] = 'canceled'
                        self.tree.item(tid, values=vals)
                        try:
                            self.tree.item(tid, tags=('canceled',))
                        except Exception:
                            pass
                    canceled_count += 1
                    if t:
                        try:
                            cleanup_partial_files(t.dest)
                        except Exception:
                            pass
                    continue

                if t and t.status not in ('done',):
                    t.status = 'canceled'
                    if tid in self.tree.get_children():
                        vals = list(self.tree.item(tid, 'values'))
                        vals[6] = 'canceled'
                        self.tree.item(tid, values=vals)
                        try:
                            self.tree.item(tid, tags=('canceled',))
                        except Exception:
                            pass
                    canceled_count += 1
                    try:
                        cleanup_partial_files(t.dest)
                    except Exception:
                        pass
            except Exception as e:
                self.append_log("Error while cancelling task", tid, ":", e)

        try:
            self.save_queue()
        except Exception:
            pass

        self.append_log(f"Cancelled {canceled_count} selected task(s).")

    # ---------------- tree helpers ----------------
    def tree_insert_task(self, t: DownloadTask, at_top: bool = False):
        values = (
            0,
            t.title or t.url,
            t.url,
            t.dest,
            t.resolution or "best",
            "audio" if t.audio_only else "video",
            t.status,
        )
        try:
            if hasattr(self, 'tree') and self.tree.exists(t.id):
                self.tree.item(t.id, values=values)
                try:
                    self.tree.item(t.id, tags=(t.status,))
                except Exception:
                    pass
                if at_top:
                    try:
                        self.tree.move(t.id, '', 0)
                    except Exception:
                        pass
            else:
                if at_top:
                    self.tree.insert("", 0, iid=t.id, values=values, tags=(t.status,))
                else:
                    self.tree.insert("", "end", iid=t.id, values=values, tags=(t.status,))
        except Exception:
            try:
                if hasattr(self, 'tree') and self.tree.exists(t.id):
                    self.tree.delete(t.id)
                if at_top:
                    self.tree.insert("", 0, iid=t.id, values=values, tags=(t.status,))
                else:
                    self.tree.insert("", "end", iid=t.id, values=values, tags=(t.status,))
            except Exception:
                pass

        try:
            self.refresh_tree_indexes()
        except Exception:
            pass

    def refresh_tree_indexes(self):
        for idx, iid in enumerate(self.tree.get_children(), start=1):
            vals = list(self.tree.item(iid, 'values'))
            vals[0] = idx
            self.tree.item(iid, values=vals)

    def reload_tree_from_taskqueue(self):
        for iid in list(self.tree.get_children()):
            self.tree.delete(iid)
        for t in self.task_queue.to_list():
            self.tasks_map[t.id] = t
            self.tree_insert_task(t)

    def open_settings(self):
        cfg = load_config()
        dlg = SettingsDialog(self.root, cfg)
        # attempt to set icon for settings dialog (bundled or local)
        try:
            ico_path = None
            try:
                candidate = bundled_resource_path("icon.png")
                if candidate.exists():
                    ico_path = candidate
                else:
                    candidate = bundled_resource_path("icon.ico")
                    if candidate.exists():
                        ico_path = candidate
            except Exception:
                pass
            if ico_path is None:
                local = Path(__file__).resolve().parent / "icon.png"
                if local.exists():
                    ico_path = local
                else:
                    local = Path(__file__).resolve().parent / "icon.ico"
                    if local.exists():
                        ico_path = local
            if ico_path and ico_path.exists():
                try:
                    if ico_path.suffix.lower() == ".ico":
                        dlg.top.iconbitmap(str(ico_path))
                    else:
                        img = tk.PhotoImage(file=str(ico_path))
                        dlg.top.iconphoto(False, img)
                        # keep a reference to prevent GC
                        dlg._icon_img = img
                except Exception:
                    pass
        except Exception:
            pass

        self.root.wait_window(dlg.top)
        if dlg.result:
            cfg_new = dlg.result
            save_config(cfg_new)
            global CONFIG
            CONFIG = cfg_new
            self.apply_theme(cfg_new.get('theme', 'light'))
            self.append_log("Saved settings:", cfg_new)

    def toggle_log_panel(self):
        row = getattr(self, '_log_row', 10)
        visible = getattr(self, '_log_visible', True)

        self.root.update_idletasks()
        before_root_h = self.root.winfo_height()
        before_frame_req_h = self.frm.winfo_reqheight()
        before_log_h = self.log_frame.winfo_height() if visible else 0

        if visible:
            try:
                was_maximized = self.root.state() == 'zoomed'
                if was_maximized:
                    self.root.state('normal')
                    self.root.update_idletasks()
                geom = self.root.geometry()
                parts = geom.split('+', 1)[0].split('x')
                if len(parts) >= 2:
                    w = int(parts[0]); h = int(parts[1])
                    new_h = max(120, h - int(before_log_h or 0))
                    if '+' in geom:
                        xy = '+' + geom.split('+',1)[1]
                    else:
                        xy = ''
                    self.root.geometry(f"{w}x{new_h}{xy}")
                if was_maximized:
                    self.root.state('zoomed')
            except Exception:
                pass
            try:
                self.log_frame.grid_forget()
            except Exception:
                try:
                    self.log_frame.grid_remove()
                except Exception:
                    pass
            try:
                self.frm.rowconfigure(row, weight=0, minsize=0)
            except Exception:
                pass
            try:
                self._saved_root_geom = self.root.geometry()
            except Exception:
                self._saved_root_geom = None

            self._log_visible = False
            try:
                self.view_menu.entryconfig(self._toggle_log_menu_index, label='View Log Panel')
            except Exception:
                pass
        else:
            try:
                opts = getattr(self, '_log_grid_opts', None)
                if opts:
                    self.log_frame.grid(row=opts['row'], column=opts['column'],
                                        columnspan=opts['columnspan'], pady=opts['pady'],
                                        sticky=opts['sticky'])
                else:
                    self.log_frame.grid(row=row, column=0, columnspan=5, pady=(8,0), sticky='nsew')
                self.frm.rowconfigure(row, weight=(getattr(self, '_log_prev_weight', 1) or 1), minsize=0)
            except Exception:
                pass

            try:
                was_maximized_before = self.root.state() == 'zoomed'
                if self._saved_root_geom and not was_maximized_before:
                    self.root.geometry(self._saved_root_geom)
                elif was_maximized_before:
                    self.root.state('zoomed')
            except Exception:
                pass

            self._log_visible = True
            try:
                self.view_menu.entryconfig(self._toggle_log_menu_index, label='Hide Log Panel')
            except Exception:
                pass

        try:
            self.root.update_idletasks()
            self.canvas.configure(scrollregion=self.canvas.bbox('all'))
            self.canvas.itemconfigure(self.canvas_window, width=self.canvas.winfo_width())
        except Exception:
            pass

    def poll_ui(self):
        while not self.ui_queue.empty():
            item = self.ui_queue.get_nowait()
            if not item:
                continue
            typ = item[0]
            if typ == 'log':
                self.append_log(item[1])
            elif typ == 'task_added':
                td = item[1]
                t = DownloadTask.from_dict(td)
                self.tasks_map[t.id] = t
                self.tree_insert_task(t)
                self.save_queue()
            elif typ == 'task_removed':
                tid = item[1]
                if tid in self.tasks_map:
                    del self.tasks_map[tid]
                if tid in self.tree.get_children():
                    self.tree.delete(tid)
                self.refresh_tree_indexes()
                self.save_queue()
            elif typ == 'task_update_title':
                tid, title = item[1], item[2]
                if tid in self.tasks_map:
                    self.tasks_map[tid].title = title
                if tid in self.tree.get_children():
                    vals = list(self.tree.item(tid, 'values'))
                    vals[1] = title or vals[1]
                    self.tree.item(tid, values=vals)
            elif typ == 'task_status':
                tid, st = item[1], item[2]
                if tid in self.tasks_map:
                    self.tasks_map[tid].status = st
                if tid in self.tree.get_children():
                    vals = list(self.tree.item(tid, 'values'))
                    vals[6] = st
                    self.tree.item(tid, values=vals)
                    try:
                        self.tree.item(tid, tags=(st,))
                    except Exception:
                        pass
                self.save_queue()
                try:
                    self._check_and_cleanup_queue_files()
                except Exception:
                    pass
            elif typ == 'task_progress':
                tid, downloaded, total = item[1], item[2], item[3]
                if total and total > 0:
                    self.progress.config(mode='determinate', maximum=total)
                    self.progress['value'] = downloaded
                    pct = (downloaded / total * 100)
                    self.status_lbl.config(text=f"Downloading... {bytes_to_human(downloaded)} / {bytes_to_human(total)} ({pct:.1f}%)")
                else:
                    self.progress.config(mode='indeterminate')
                    try:
                        self.progress.start(10)
                    except Exception:
                        pass
                    self.status_lbl.config(text=f"Downloading... {bytes_to_human(downloaded)}")
            elif typ == 'task_failed_reason':
                tid, reason = item[1], item[2]
                self.append_log(f"Task {tid} failed: {strip_ansi(reason)}")
            elif typ == 'persist_queue_now':
                try:
                    self.save_queue()
                except Exception:
                    pass
            else:
                self.append_log("Unknown UI message:", item)
        self.root.after(200, self.poll_ui)

    def append_log(self, *parts):
        text = ' '.join(str(p) for p in parts)
        try:
            self.log_text.insert('end', text + "\n")
            self.log_text.see('end')
        except Exception:
            print(text)

    def save_queue(self):
        """
        Persist queue in a deterministic order:
        1. currently-downloading task (if any)
        2. tasks in the treeview in visual order
        Only non-completed tasks are saved. This preserves the visual order
        exactly as the user sees it and avoids mixing TaskQueue and tasks_map iteration orders.
        """
        tasks_out = []
        seen = set()

        # 1) If worker has an active task, include it first (so resume order matches)
        try:
            cur_tid = getattr(self.worker, 'current_task_id', None)
            if cur_tid and cur_tid in self.tasks_map:
                tcur = self.tasks_map[cur_tid]
                if tcur.status != 'done':
                    tasks_out.append(tcur.to_dict())
                    seen.add(tcur.id)
        except Exception:
            cur_tid = None

        # 2) Walk the Treeview in visual order and add tasks (skip ones already added)
        try:
            for iid in self.tree.get_children():
                try:
                    t = self.tasks_map.get(iid)
                    if not t:
                        # fallback: attempt to reconstruct from tree values if task_map missing
                        vals = self.tree.item(iid, 'values')
                        # minimal reconstruct (url & dest may be enough to requeue)
                        t = DownloadTask(url=vals[2] if len(vals) > 2 else '', dest=vals[3] if len(vals) > 3 else '.',
                                        resolution=vals[4] if len(vals) > 4 else None,
                                        audio_only=(vals[5] == 'audio') if len(vals) > 5 else False,
                                        tid=iid, status=vals[6] if len(vals) > 6 else 'queued',
                                        title=vals[1] if len(vals) > 1 else None)
                    if t and t.id not in seen and t.status != 'done':
                        tasks_out.append(t.to_dict())
                        seen.add(t.id)
                except Exception:
                    continue
        except Exception:
            # fallback: if tree walk failed, use TaskQueue order + tasks_map
            try:
                queued = self.task_queue.to_list()
                for t in queued:
                    if t.id not in seen and t.status != 'done':
                        tasks_out.append(t.to_dict()); seen.add(t.id)
                for tid, t in self.tasks_map.items():
                    if tid not in seen and t.status != 'done':
                        tasks_out.append(t.to_dict()); seen.add(tid)
            except Exception:
                pass

        # 3) write to disk (atomic-ish)
        try:
            # backup old queue
            try:
                if QUEUE_FILE.exists():
                    backup = APP_DIR / (f"queue_backup_{datetime_now_str()}.json")
                    shutil.copy2(str(QUEUE_FILE), str(backup))
            except Exception:
                pass
            with open(QUEUE_FILE, 'w', encoding='utf-8') as f:
                json.dump(tasks_out, f, indent=2)
        except Exception as e:
            print("Failed to save queue:", e)


    def _check_and_cleanup_queue_files(self):
        try:
            worker_idle = (getattr(self.worker, 'current_task_id', None) is None)
            queue_empty = self.task_queue.empty()
            all_done = True
            for iid in self.tree.get_children():
                try:
                    status = self.tree.item(iid, "values")[6]
                except Exception:
                    status = None
                if status != "done":
                    all_done = False
                    break
            if worker_idle and queue_empty and all_done:
                try:
                    cleanup_queue_files()
                    self.append_log("All tasks done — removed persisted queue files.")
                except Exception:
                    pass
        except Exception:
            pass

# -------------------------
# Settings Dialog (includes ffmpeg path)
# -------------------------
class SettingsDialog:
    def __init__(self, parent, cfg):
        self.top = tk.Toplevel(parent)
        self.top.title("Settings")
        self.result = None

        ttk.Label(self.top, text="Audio bitrate (e.g. 192k):").grid(row=0, column=0, sticky='w')
        self.ab_var = tk.StringVar(value=cfg.get("audio_bitrate", "192k"))
        ttk.Entry(self.top, textvariable=self.ab_var).grid(row=0, column=1, sticky='ew')

        ttk.Label(self.top, text="Merge container (mp4/mkv):").grid(row=1, column=0, sticky='w')
        self.container_var = tk.StringVar(value=cfg.get("merge_output_format","mp4"))
        ttk.OptionMenu(self.top, self.container_var, self.container_var.get(), 'mp4', 'mkv').grid(row=1, column=1, sticky='w')

        ttk.Label(self.top, text="Theme:").grid(row=2, column=0, sticky='w')
        self.theme_var = tk.StringVar(value=cfg.get('theme','light'))
        ttk.OptionMenu(self.top, self.theme_var, self.theme_var.get(), 'light', 'dark').grid(row=2, column=1, sticky='w')

        # FFmpeg path row
        ttk.Label(self.top, text="FFmpeg executable path:").grid(row=3, column=0, sticky='w')
        self.ffmpeg_var = tk.StringVar(value=cfg.get('ffmpeg_path', DEFAULT_FFMPEG))
        ff_entry = ttk.Entry(self.top, textvariable=self.ffmpeg_var)
        ff_entry.grid(row=3, column=1, sticky='ew')
        btn_browse_ff = ttk.Button(self.top, text="Browse...", command=self.browse_ffmpeg)
        btn_browse_ff.grid(row=3, column=2, sticky='w')

        btn_frame = ttk.Frame(self.top)
        btn_frame.grid(row=4, column=0, columnspan=3, pady=(10,0))
        ttk.Button(btn_frame, text="Save", command=self.on_save).grid(row=0, column=0, padx=5)
        ttk.Button(btn_frame, text="Cancel", command=self.top.destroy).grid(row=0, column=1, padx=5)

        self.top.columnconfigure(1, weight=1)
    '''
    def browse_ffmpeg(self):
        if platform.system() == "Windows":
            filetypes = [("Executable", "*.exe"), ("All files", "*.*")]
        else:
            filetypes = [("All files", "*.*")]
        f = filedialog.askopenfilename(title="Select ffmpeg executable", filetypes=filetypes)
        if f:
            self.ffmpeg_var.set(f)
    '''
    def browse_ffmpeg(self):
        """
        Open a file picker correctly from a modal settings dialog.
        Ensures the file dialog is parented to the settings window, temporarily
        releases grab/topmost so the OS file dialog can show correctly, then
        restores modal behavior and focus after the dialog closes.
        """
        # filetypes depending on platform
        if platform.system() == "Windows":
            filetypes = [("Executable", "*.exe"), ("All files", "*.*")]
        else:
            filetypes = [("All files", "*.*")]

        # If we previously set a grab on self.top, release it so filedialog can function reliably.
        try:
            self.top.grab_release()
        except Exception:
            pass

        # Temporarily set topmost False to avoid file dialog appearing behind on some WMs,
        # but remember previous state if needed (we set True briefly below after restore).
        try:
            prev_topmost = self.top.attributes("-topmost")
        except Exception:
            prev_topmost = False
        try:
            self.top.attributes("-topmost", False)
        except Exception:
            pass

        # Call the file dialog using self.top as parent so it stays attached
        try:
            chosen = filedialog.askopenfilename(
                title="Select ffmpeg executable",
                parent=self.top,
                filetypes=filetypes
            )
        except Exception:
            # fallback without parent if some platform doesn't support it
            chosen = filedialog.askopenfilename(title="Select ffmpeg executable", filetypes=filetypes)

        # Restore the modal grab and focus
        try:
            if chosen:
                self.ffmpeg_var.set(chosen)
        except Exception:
            pass

        try:
            # restore topmost briefly to ensure it is above the file dialog on some platforms
            self.top.attributes("-topmost", True)
            # restore modal grab
            self.top.grab_set()
            # lift and focus the dialog so it stays above the main window
            self.top.lift()
            self.top.focus_force()
            # release the temporary topmost after a short delay so it behaves normally
            self.top.after(100, lambda: self.top.attributes("-topmost", prev_topmost))
        except Exception:
            try:
                # best-effort fallback: just focus the entry
                self.top.focus_force()
            except Exception:
                pass


    def on_save(self):
        ab = self.ab_var.get().strip() or "192k"
        container = self.container_var.get().strip() or "mp4"
        theme = self.theme_var.get().strip() or 'light'
        ff = self.ffmpeg_var.get().strip() or DEFAULT_FFMPEG
        cfg = {
            "audio_bitrate": ab,
            "merge_output_format": container,
            "theme": theme,
            "window_geometry": CONFIG.get('window_geometry'),
            "ffmpeg_path": ff
        }
        self.result = cfg
        self.top.destroy()

# -------------------------
# CLI
# -------------------------
def cli_main(args):
    task_q = TaskQueue()
    ui_q = pyqueue.Queue()
    stop_event = threading.Event()
    worker = DownloaderWorker(task_q, ui_q, stop_event)
    worker.start()

    persisted = load_queue_from_disk()
    for t in persisted:
        if t.status in ('queued','failed','canceled','downloading'):
            t.status = 'queued'
            task_q.put(t)

    if args.urls:
        for u in args.urls:
            t = DownloadTask(u, dest=args.out, resolution=args.resolution, audio_only=args.audio_only, audio_format=args.audio_format)
            task_q.put(t)

    print("Enter URLs (video or playlist) one per line. Type 'exit' to quit and wait for queue.")
    try:
        while True:
            try:
                line = input("> ").strip()
            except EOFError:
                break
            if line.lower() in ('exit','quit'):
                break
            if not line:
                continue
            t = DownloadTask(line, dest=args.out, resolution=args.resolution, audio_only=args.audio_only, audio_format=args.audio_format)
            task_q.put(t)
    except KeyboardInterrupt:
        pass

    print("Waiting for queue to finish...")
    try:
        while not task_q.empty():
            while not ui_q.empty():
                m = ui_q.get_nowait()
                if m[0] == 'log':
                    print(m[1])
            time.sleep(0.5)
    except KeyboardInterrupt:
        pass
    finally:
        stop_event.set()
        task_q.put(None)
        worker.join(timeout=3)

# -------------------------
# Entrypoint
# -------------------------
if __name__ == '__main__':
    import sys
    parser = argparse.ArgumentParser()
    parser.add_argument('--cli', action='store_true', help='Run in CLI mode')
    parser.add_argument('--out', default='.', help='Default output folder for CLI mode')
    parser.add_argument('--resolution', default=None, help='Preferred resolution like 1080p for CLI mode')
    parser.add_argument('--audio-only', action='store_true', help='CLI: download audio only')
    parser.add_argument('--audio-format', default='mp3', choices=['mp3','aac','opus'], help='CLI: audio output format')
    parser.add_argument('urls', nargs='*', help='Optional URLs to queue immediately (CLI mode)')
    args = parser.parse_args()

    if args.cli:
        cli_main(args)
        sys.exit(0)

    root = tk.Tk()
    app = App(root)
    root.mainloop()