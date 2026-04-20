import os
import sys
import queue
import socket
import getpass
import logging
import hashlib
import platform
import threading
from datetime import datetime, timedelta
from typing import Any

import numpy as np

from subjective_abstract_data_source_package import SubjectiveDataSource
from brainboost_data_source_logger_package.BBLogger import BBLogger


SAMPLE_RATE = 16000
BLOCK_MS = 100
BLOCK_SAMPLES = SAMPLE_RATE * BLOCK_MS // 1000


class SubjectiveMicDataSource(SubjectiveDataSource):
    """
    Listens to the microphone, segments audio into utterances using a
    silence-based VAD, transcribes each utterance with Whisper and
    yields a self-contained dict per utterance. The framework writes
    that dict to the canonical context file
    (YYYY_MM_DD_HH_MM_SS-Mic-[connection]-context.json) — this class
    does not write context files itself.

    By default the raw audio is discarded immediately after
    transcription. Disable the "Do Not Keep Audio" connection toggle
    to additionally archive the utterance as MP3 (or WAV when ffmpeg
    is unavailable).
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        conn = getattr(self, "_connection", {}) or {}

        self.whisper_model_size = conn.get("whisper_model_size") or "base"
        self.language = (conn.get("language") or "").strip() or None

        dev = conn.get("device_index")
        self.device_index = int(dev) if dev not in (None, "", "auto") else None

        self.silence_threshold = float(conn.get("silence_threshold") or 0.01)
        self.silence_duration = float(conn.get("silence_duration") or 0.8)
        self.min_utterance_duration = float(conn.get("min_utterance_duration") or 0.4)
        self.max_utterance_duration = float(conn.get("max_utterance_duration") or 30.0)

        self.do_not_keep_audio = bool(conn.get("do_not_keep_audio", True))
        self.audio_format = (conn.get("audio_format") or "mp3").lower()
        self.capture_speaker = bool(conn.get("capture_speaker", True))

        self.whisper_model = None
        self._ffmpeg_path = None
        self._ffmpeg_checked = False
        self._ffmpeg_unavailable = False
        self._stop_event = threading.Event()

    @classmethod
    def connection_schema(cls) -> dict:
        return {
            "whisper_model_size": {
                "type": "select",
                "label": "Whisper Model",
                "description": "Whisper model size. Larger = more accurate but slower.",
                "options": ["tiny", "base", "small", "medium", "large"],
                "default": "base",
                "required": False,
            },
            "language": {
                "type": "text",
                "label": "Language (ISO code)",
                "description": "Force a language (e.g. 'en', 'es'). Leave blank for auto-detect.",
                "required": False,
                "placeholder": "auto",
            },
            "device_index": {
                "type": "text",
                "label": "Input Device Index",
                "description": "sounddevice input device index. Leave blank for system default.",
                "required": False,
                "placeholder": "auto",
            },
            "silence_threshold": {
                "type": "number",
                "label": "Silence Threshold (RMS 0..1)",
                "description": "Below this RMS level, audio is treated as silence. Typical: 0.005–0.03.",
                "default": 0.01,
                "min": 0.0,
                "max": 1.0,
                "step": 0.001,
                "required": False,
            },
            "silence_duration": {
                "type": "number",
                "label": "Silence Duration (seconds)",
                "description": "Seconds of continuous silence that end an utterance.",
                "default": 0.8,
                "min": 0.2,
                "step": 0.1,
                "required": False,
            },
            "min_utterance_duration": {
                "type": "number",
                "label": "Min Utterance Duration (seconds)",
                "description": "Drop utterances shorter than this (noise / clicks).",
                "default": 0.4,
                "min": 0.0,
                "step": 0.1,
                "required": False,
            },
            "max_utterance_duration": {
                "type": "number",
                "label": "Max Utterance Duration (seconds)",
                "description": "Force-flush an utterance after this duration.",
                "default": 30.0,
                "min": 1.0,
                "step": 1.0,
                "required": False,
            },
            "audio_format": {
                "type": "select",
                "label": "Saved Audio Format",
                "description": "Compact format used for the saved utterance audio.",
                "options": ["mp3", "wav"],
                "default": "mp3",
                "required": False,
            },
            "do_not_keep_audio": {
                "type": "bool",
                "label": "Do Not Keep Audio",
                "description": (
                    "When checked, the raw utterance audio is never written to disk — "
                    "only the transcript survives, inside the context JSON. "
                    "Uncheck to additionally archive the audio file next to the context."
                ),
                "default": True,
                "required": False,
            },
            "capture_speaker": {
                "type": "bool",
                "label": "Also Capture Speaker Output",
                "description": (
                    "When checked, the system speaker output is captured via a "
                    "loopback device and transcribed alongside the microphone. "
                    "Each event carries a 'source' field ('microphone' or "
                    "'speaker') so you can tell them apart."
                ),
                "default": True,
                "required": False,
            },
        }

    @classmethod
    def request_schema(cls) -> dict:
        return {}

    @classmethod
    def output_schema(cls) -> dict:
        return {
            "type": {"type": "text", "label": "Event Type"},
            "datasource": {"type": "text", "label": "Datasource Class"},
            "plugin": {"type": "text", "label": "Plugin"},
            "connection_name": {"type": "text", "label": "Connection"},
            "source": {
                "type": "text",
                "label": "Source",
                "description": "'microphone' for local mic input, 'speaker' for system loopback.",
            },
            "utterance_id": {"type": "text", "label": "Utterance ID"},
            "timestamp": {
                "type": "text",
                "label": "Timestamp (spoken end)",
                "description": "When the speaker stopped talking. Use this as the canonical event time.",
            },
            "spoken_start_at": {"type": "text", "label": "Spoken Start"},
            "spoken_end_at": {"type": "text", "label": "Spoken End"},
            "transcribed_at": {"type": "text", "label": "Transcribed At"},
            "transcription_seconds": {"type": "number", "label": "Transcription Time (seconds)"},
            "duration_seconds": {"type": "number", "label": "Duration (seconds)"},
            "language": {"type": "text", "label": "Detected Language"},
            "whisper_model": {"type": "text", "label": "Whisper Model"},
            "sample_rate": {"type": "int", "label": "Sample Rate"},
            "channels": {"type": "int", "label": "Channels"},
            "input_device_index": {"type": "int", "label": "Input Device Index"},
            "hostname": {"type": "text", "label": "Hostname"},
            "os_user": {"type": "text", "label": "OS User"},
            "platform": {"type": "text", "label": "Platform"},
            "audio_kept": {"type": "bool", "label": "Audio Kept"},
            "audio_path": {"type": "text", "label": "Audio File Path"},
            "audio_format": {"type": "text", "label": "Audio Format"},
            "transcription": {"type": "textarea", "label": "Transcription"},
        }

    @classmethod
    def icon(cls) -> str:
        icon_path = os.path.join(os.path.dirname(__file__), "icon.svg")
        try:
            with open(icon_path, "r", encoding="utf-8") as f:
                return f.read()
        except Exception as exc:
            BBLogger.log(f"[Mic] Error reading icon: {exc}")
            return ""

    def supports_streaming(self) -> bool:
        return True

    def run(self, request: dict) -> dict:
        """Batch fallback: record a single utterance (or silence timeout) and return it."""
        try:
            self._load_whisper_model()
            for event in self._iter_utterances(one_shot=True):
                return event
            return self._empty_event()
        except Exception as e:
            BBLogger.log(f"[Mic] run() error: {e}")
            event = self._empty_event()
            event["error"] = str(e)
            return event

    def _empty_event(self) -> dict:
        now = datetime.now().isoformat()
        return {
            "type": "mic_transcription",
            "datasource": self.__class__.__name__,
            "plugin": "subjective_mic_datasource",
            "connection_name": getattr(self, "connection_name", "") or "",
            "source": "microphone",
            "utterance_id": "",
            "timestamp": now,
            "spoken_start_at": now,
            "spoken_end_at": now,
            "transcribed_at": now,
            "transcription_seconds": 0.0,
            "duration_seconds": 0.0,
            "language": "",
            "whisper_model": self.whisper_model_size,
            "sample_rate": SAMPLE_RATE,
            "channels": 1,
            "input_device_index": self.device_index,
            "hostname": socket.gethostname(),
            "os_user": self._safe_user(),
            "platform": f"{platform.system()} {platform.release()}",
            "audio_kept": False,
            "audio_path": None,
            "audio_format": None,
            "transcription": "",
        }

    def stream(self, request: dict):
        """Listen to the mic indefinitely, yielding one dict per transcribed utterance."""
        self._load_whisper_model()
        BBLogger.log("[Mic] Streaming started — listening for utterances")
        try:
            for event in self._iter_utterances(one_shot=False):
                yield event
        except GeneratorExit:
            self._stop_event.set()
            BBLogger.log("[Mic] Streaming stopped")
            raise

    def _iter_utterances(self, one_shot: bool):
        event_q: queue.Queue = queue.Queue()
        whisper_lock = threading.Lock()
        worker_stop = threading.Event()

        def mic_worker():
            try:
                self._mic_vad_loop(event_q, whisper_lock, worker_stop, one_shot)
            except Exception as e:
                BBLogger.log(f"[Mic] mic worker crashed: {e}")
                logging.exception("Mic worker error")

        def speaker_worker():
            try:
                self._speaker_vad_loop(event_q, whisper_lock, worker_stop, one_shot)
            except Exception as e:
                BBLogger.log(f"[Mic] speaker worker crashed: {e}")
                logging.exception("Speaker worker error")

        threads: list[threading.Thread] = []
        t_mic = threading.Thread(target=mic_worker, name="MicVAD", daemon=True)
        t_mic.start()
        threads.append(t_mic)

        if self.capture_speaker:
            t_spk = threading.Thread(target=speaker_worker, name="SpeakerVAD", daemon=True)
            t_spk.start()
            threads.append(t_spk)
            BBLogger.log("[Mic] Speaker loopback capture enabled")
        else:
            BBLogger.log("[Mic] Speaker loopback capture disabled")

        try:
            while not self._stop_event.is_set():
                if all(not t.is_alive() for t in threads) and event_q.empty():
                    break
                try:
                    event = event_q.get(timeout=0.25)
                except queue.Empty:
                    continue
                if event is None:
                    continue
                yield event
                if one_shot:
                    return
        finally:
            worker_stop.set()

    def _mic_vad_loop(self, event_q, whisper_lock, worker_stop, one_shot):
        try:
            import sounddevice as sd
        except Exception as e:
            BBLogger.log(f"[Mic] sounddevice unavailable: {e}")
            return

        audio_q: queue.Queue = queue.Queue()

        def callback(indata, frames, time_info, status):
            if status:
                BBLogger.log(f"[Mic] mic stream status: {status}")
            audio_q.put(indata.copy())

        stream_kwargs = dict(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32",
            blocksize=BLOCK_SAMPLES,
            callback=callback,
        )
        if self.device_index is not None:
            stream_kwargs["device"] = self.device_index

        def block_iter():
            with sd.InputStream(**stream_kwargs):
                while not self._stop_event.is_set() and not worker_stop.is_set():
                    try:
                        yield audio_q.get(timeout=0.25).reshape(-1)
                    except queue.Empty:
                        continue

        self._run_vad("microphone", block_iter(), event_q, whisper_lock, worker_stop, one_shot)

    def _speaker_vad_loop(self, event_q, whisper_lock, worker_stop, one_shot):
        try:
            import soundcard as sc
        except Exception as e:
            BBLogger.log(f"[Mic] soundcard unavailable — speaker capture disabled: {e}")
            return

        try:
            speaker = sc.default_speaker()
            loopback = sc.get_microphone(id=str(speaker.name), include_loopback=True)
            BBLogger.log(f"[Mic] Speaker loopback resolved: {speaker.name}")
        except Exception as e:
            BBLogger.log(f"[Mic] Could not open speaker loopback: {e}")
            return

        def block_iter():
            try:
                with loopback.recorder(samplerate=SAMPLE_RATE, channels=1, blocksize=BLOCK_SAMPLES) as rec:
                    while not self._stop_event.is_set() and not worker_stop.is_set():
                        data = rec.record(numframes=BLOCK_SAMPLES)
                        if data is None:
                            continue
                        flat = data.mean(axis=1) if data.ndim > 1 else data
                        yield flat.astype(np.float32, copy=False)
            except Exception as e:
                BBLogger.log(f"[Mic] Speaker recorder error: {e}")

        self._run_vad("speaker", block_iter(), event_q, whisper_lock, worker_stop, one_shot)

    def _run_vad(self, source: str, block_iter, event_q, whisper_lock, worker_stop, one_shot):
        utterance: list[np.ndarray] = []
        silent_blocks = 0
        in_speech = False
        silence_blocks_needed = max(1, int(self.silence_duration * 1000 / BLOCK_MS))
        max_blocks = max(1, int(self.max_utterance_duration * 1000 / BLOCK_MS))

        for block in block_iter:
            if self._stop_event.is_set() or worker_stop.is_set():
                break

            flat = block.reshape(-1)
            rms = float(np.sqrt(np.mean(flat * flat))) if flat.size else 0.0

            if rms > self.silence_threshold:
                if not in_speech:
                    in_speech = True
                utterance.append(flat)
                silent_blocks = 0
            elif in_speech:
                utterance.append(flat)
                silent_blocks += 1

            force_flush = in_speech and len(utterance) >= max_blocks
            silence_flush = in_speech and silent_blocks >= silence_blocks_needed

            if force_flush or silence_flush:
                audio = np.concatenate(utterance) if utterance else np.zeros(0, dtype=np.float32)
                utterance = []
                silent_blocks = 0
                in_speech = False

                duration = len(audio) / SAMPLE_RATE
                if duration < self.min_utterance_duration:
                    continue

                flush_at = datetime.now()
                with whisper_lock:
                    event = self._process_utterance(audio, duration, flush_at, source)
                if event:
                    event_q.put(event)
                    if one_shot:
                        return

    def _process_utterance(self, audio: np.ndarray, duration: float, flush_at: datetime, source: str = "microphone"):
        try:
            transcript, language = self._transcribe(audio)
            transcribed_at = datetime.now()
            transcript = (transcript or "").strip()

            if not transcript:
                return None

            audio_path = ""
            if not self.do_not_keep_audio:
                audio_path = self._save_audio(audio, source)

            # The VAD flushes after `silence_duration` of trailing silence,
            # so the speaker stopped talking ~that long before flush_at.
            # The `duration` includes that trailing silence too, so the
            # spoken start is flush_at - duration.
            spoken_end_dt = flush_at - timedelta(seconds=self.silence_duration)
            spoken_start_dt = flush_at - timedelta(seconds=duration)
            transcription_seconds = (transcribed_at - flush_at).total_seconds()

            timestamp = spoken_end_dt.isoformat()
            digest_src = f"{timestamp}|{source}|{transcript}".encode("utf-8")
            utterance_id = hashlib.md5(digest_src).hexdigest()[:12]

            return {
                "type": "mic_transcription",
                "datasource": self.__class__.__name__,
                "plugin": "subjective_mic_datasource",
                "connection_name": getattr(self, "connection_name", "") or "",
                "source": source,
                "utterance_id": utterance_id,
                "timestamp": timestamp,
                "spoken_start_at": spoken_start_dt.isoformat(),
                "spoken_end_at": spoken_end_dt.isoformat(),
                "transcribed_at": transcribed_at.isoformat(),
                "transcription_seconds": round(transcription_seconds, 3),
                "duration_seconds": round(duration, 3),
                "language": language or "",
                "whisper_model": self.whisper_model_size,
                "sample_rate": SAMPLE_RATE,
                "channels": 1,
                "input_device_index": self.device_index,
                "hostname": socket.gethostname(),
                "os_user": self._safe_user(),
                "platform": f"{platform.system()} {platform.release()}",
                "audio_kept": bool(audio_path),
                "audio_path": audio_path or None,
                "audio_format": (self.audio_format if audio_path else None),
                "transcription": transcript,
            }
        except Exception as e:
            BBLogger.log(f"[Mic] Error processing utterance: {e}")
            logging.exception("Mic utterance error")
            return None

    def _transcribe(self, audio: np.ndarray):
        import torch
        if self.whisper_model is None:
            self._load_whisper_model()
        kwargs = {"fp16": torch.cuda.is_available()}
        if self.language:
            kwargs["language"] = self.language
        result = self.whisper_model.transcribe(audio.astype(np.float32), **kwargs)
        return result.get("text", ""), result.get("language", "")

    def _load_whisper_model(self):
        if self.whisper_model is not None:
            return
        import whisper
        BBLogger.log(f"[Mic] Loading Whisper model '{self.whisper_model_size}'")
        self.whisper_model = whisper.load_model(self.whisper_model_size)

    def _save_audio(self, audio: np.ndarray, source: str = "microphone") -> str:
        tmp_dir = self._get_audio_dir()
        os.makedirs(tmp_dir, exist_ok=True)

        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        int16 = np.clip(audio, -1.0, 1.0)
        int16 = (int16 * 32767).astype(np.int16)

        fmt = self.audio_format
        if fmt == "mp3":
            path = os.path.join(tmp_dir, f"utterance_{source}_{ts}.mp3")
            if self._export_mp3(int16, path):
                return path
            fmt = "wav"

        path = os.path.join(tmp_dir, f"utterance_{source}_{ts}.wav")
        self._export_wav(int16, path)
        return path

    def _export_mp3(self, pcm16: np.ndarray, path: str) -> bool:
        if self._ffmpeg_unavailable:
            return False
        try:
            from pydub import AudioSegment
            self._configure_ffmpeg()
            if self._ffmpeg_unavailable:
                return False
            segment = AudioSegment(
                pcm16.tobytes(),
                frame_rate=SAMPLE_RATE,
                sample_width=2,
                channels=1,
            )
            segment.export(path, format="mp3", bitrate="64k")
            return True
        except FileNotFoundError as e:
            self._ffmpeg_unavailable = True
            BBLogger.log(
                f"[Mic] MP3 export failed ({e}) — ffmpeg binary unreachable. "
                f"Configured converter: {getattr(__import__('pydub').AudioSegment, 'converter', None)!r}. "
                f"Switching this session to WAV."
            )
            return False
        except Exception as e:
            BBLogger.log(f"[Mic] MP3 export failed ({e}) — falling back to WAV for this utterance")
            return False

    def _export_wav(self, pcm16: np.ndarray, path: str):
        import wave
        with wave.open(path, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes(pcm16.tobytes())

    def _configure_ffmpeg(self):
        from pydub import AudioSegment
        from pydub.utils import which

        if self._ffmpeg_path and os.path.exists(self._ffmpeg_path):
            AudioSegment.converter = self._ffmpeg_path
            return
        if self._ffmpeg_checked:
            return
        self._ffmpeg_checked = True

        if os.name == "nt":
            platform_sub = "windows"
            exe = "ffmpeg.exe"
        elif sys.platform == "darwin":
            platform_sub = "mac"
            exe = "ffmpeg"
        else:
            platform_sub = "linux"
            exe = "ffmpeg"

        plugin_dir = os.path.dirname(__file__)
        candidates = [
            os.path.join(plugin_dir, "deps", "bin", platform_sub, exe),
            os.path.join(
                os.path.dirname(plugin_dir),
                "subjective_transcribelocalvideo_datasource",
                "deps", "bin", platform_sub, exe,
            ),
        ]

        for c in candidates:
            if os.path.exists(c):
                AudioSegment.converter = c
                self._ffmpeg_path = c
                BBLogger.log(f"[Mic] ffmpeg resolved (local deps): {c}")
                return

        env_path = os.getenv("FFMPEG_PATH") or os.getenv("FFMPEG_BINARY")
        if env_path:
            candidate = env_path
            if os.path.isdir(candidate):
                candidate = os.path.join(candidate, exe)
            if os.path.exists(candidate):
                AudioSegment.converter = candidate
                self._ffmpeg_path = candidate
                BBLogger.log(f"[Mic] ffmpeg resolved (env): {candidate}")
                return
            BBLogger.log(f"[Mic] FFMPEG_PATH/FFMPEG_BINARY set but not found: {env_path}")

        resolved = which("ffmpeg")
        if resolved and os.path.exists(resolved):
            AudioSegment.converter = resolved
            self._ffmpeg_path = resolved
            BBLogger.log(f"[Mic] ffmpeg resolved (PATH): {resolved}")
            return

        try:
            import imageio_ffmpeg
            path = imageio_ffmpeg.get_ffmpeg_exe()
            if path and os.path.exists(path):
                AudioSegment.converter = path
                self._ffmpeg_path = path
                BBLogger.log(f"[Mic] ffmpeg resolved (imageio-ffmpeg): {path}")
                return
            BBLogger.log(f"[Mic] imageio-ffmpeg returned unusable path: {path!r}")
        except ImportError:
            BBLogger.log("[Mic] imageio-ffmpeg not installed in plugin venv")
        except Exception as e:
            BBLogger.log(f"[Mic] imageio-ffmpeg lookup failed: {e}")

        self._ffmpeg_unavailable = True
        BBLogger.log(
            "[Mic] No ffmpeg found — MP3 export disabled, using WAV. "
            "Fix by setting FFMPEG_PATH env var, placing ffmpeg at "
            f"{os.path.join(plugin_dir, 'deps', 'bin', platform_sub, exe)}, "
            "or reinstalling the plugin venv so imageio-ffmpeg is available."
        )

    @staticmethod
    def _safe_user() -> str:
        try:
            return getpass.getuser()
        except Exception:
            return os.environ.get("USERNAME") or os.environ.get("USER") or ""

    def _get_audio_dir(self) -> str:
        try:
            base = self.get_connection_temp_dir()
            if base:
                return os.path.join(base, "mic_audio")
        except Exception:
            pass
        return os.path.join(
            os.path.expanduser("~"), ".Subjective", "mic_audio",
        )

    def handle_message(self, message: Any, files: list | None = None) -> Any:
        return {"status": "ready", "model": self.whisper_model_size}
