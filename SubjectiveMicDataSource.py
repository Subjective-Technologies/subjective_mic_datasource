import os
import sys
import json
import time
import queue
import logging
import hashlib
import threading
from datetime import datetime
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
    silence-based VAD, transcribes each utterance with Whisper, writes
    the transcript to a context JSON file and emits a streaming event.

    The raw audio for each utterance is saved in compact form (MP3 when
    ffmpeg is available, otherwise WAV) to the connection temp dir so
    it can be inspected or re-transcribed, and pruned on demand.
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

        self.keep_audio = bool(conn.get("keep_audio", True))
        self.audio_format = (conn.get("audio_format") or "mp3").lower()

        default_context = os.path.join(
            os.path.expanduser("~"),
            ".Subjective", "com_subjective_userdata", "com_subjective_context",
        )
        self.context_dir = conn.get("context_dir") or default_context

        self.whisper_model = None
        self._ffmpeg_path = None
        self._stop_event = threading.Event()

    @classmethod
    def connection_schema(cls) -> dict:
        default_context = os.path.join(
            os.path.expanduser("~"),
            ".Subjective", "com_subjective_userdata", "com_subjective_context",
        )
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
            "keep_audio": {
                "type": "bool",
                "label": "Keep Audio Files",
                "description": "When disabled, audio files are deleted after successful transcription.",
                "default": True,
                "required": False,
            },
            "context_dir": {
                "type": "folder_path",
                "label": "Context Output Directory",
                "description": "Directory where transcription JSON files are saved.",
                "default": default_context,
                "required": False,
            },
        }

    @classmethod
    def request_schema(cls) -> dict:
        return {}

    @classmethod
    def output_schema(cls) -> dict:
        return {
            "transcript": {
                "type": "textarea",
                "label": "Transcript",
                "description": "Transcribed text of the utterance.",
            },
            "audio_path": {
                "type": "text",
                "label": "Audio File Path",
                "description": "Path to the saved compact audio file for this utterance.",
            },
            "output_path": {
                "type": "text",
                "label": "Context File Path",
                "description": "Path to the JSON context file written for this utterance.",
            },
            "timestamp": {
                "type": "text",
                "label": "Timestamp",
                "description": "ISO timestamp of when the utterance ended.",
            },
            "duration_seconds": {
                "type": "number",
                "label": "Duration (seconds)",
                "description": "Length of the utterance in seconds.",
            },
            "language": {
                "type": "text",
                "label": "Detected Language",
            },
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
            return {
                "transcript": "",
                "audio_path": "",
                "output_path": "",
                "timestamp": datetime.now().isoformat(),
                "duration_seconds": 0.0,
                "language": "",
            }
        except Exception as e:
            BBLogger.log(f"[Mic] run() error: {e}")
            return {"error": str(e), "transcript": "", "audio_path": "", "output_path": "",
                    "timestamp": datetime.now().isoformat(), "duration_seconds": 0.0, "language": ""}

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
        try:
            import sounddevice as sd
        except Exception as e:
            BBLogger.log(f"[Mic] sounddevice unavailable: {e}")
            raise

        audio_q: queue.Queue = queue.Queue()

        def callback(indata, frames, time_info, status):
            if status:
                BBLogger.log(f"[Mic] stream status: {status}")
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

        utterance: list[np.ndarray] = []
        silent_blocks = 0
        in_speech = False
        silence_blocks_needed = max(1, int(self.silence_duration * 1000 / BLOCK_MS))
        max_blocks = max(1, int(self.max_utterance_duration * 1000 / BLOCK_MS))

        with sd.InputStream(**stream_kwargs):
            while not self._stop_event.is_set():
                try:
                    block = audio_q.get(timeout=0.5)
                except queue.Empty:
                    continue

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

                    event = self._process_utterance(audio, duration)
                    if event:
                        yield event
                        if one_shot:
                            return

    def _process_utterance(self, audio: np.ndarray, duration: float):
        try:
            audio_path = self._save_audio(audio)
            transcript, language = self._transcribe(audio)
            transcript = (transcript or "").strip()

            if not transcript:
                if audio_path and not self.keep_audio:
                    self._safe_remove(audio_path)
                return None

            timestamp = datetime.now().isoformat()
            context_path = self._write_context_json(
                transcript=transcript,
                audio_path=audio_path,
                timestamp=timestamp,
                duration=duration,
                language=language,
            )

            if audio_path and not self.keep_audio:
                self._safe_remove(audio_path)
                audio_path = ""

            return {
                "transcript": transcript,
                "audio_path": audio_path,
                "output_path": context_path,
                "timestamp": timestamp,
                "duration_seconds": round(duration, 3),
                "language": language or "",
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

    def _save_audio(self, audio: np.ndarray) -> str:
        tmp_dir = self._get_audio_dir()
        os.makedirs(tmp_dir, exist_ok=True)

        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        int16 = np.clip(audio, -1.0, 1.0)
        int16 = (int16 * 32767).astype(np.int16)

        fmt = self.audio_format
        if fmt == "mp3":
            path = os.path.join(tmp_dir, f"utterance_{ts}.mp3")
            if self._export_mp3(int16, path):
                return path
            fmt = "wav"

        path = os.path.join(tmp_dir, f"utterance_{ts}.wav")
        self._export_wav(int16, path)
        return path

    def _export_mp3(self, pcm16: np.ndarray, path: str) -> bool:
        try:
            from pydub import AudioSegment
            self._configure_ffmpeg()
            segment = AudioSegment(
                pcm16.tobytes(),
                frame_rate=SAMPLE_RATE,
                sample_width=2,
                channels=1,
            )
            segment.export(path, format="mp3", bitrate="64k")
            return True
        except Exception as e:
            BBLogger.log(f"[Mic] MP3 export failed ({e}) — falling back to WAV")
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

        plugin_dir = os.path.dirname(__file__)
        if os.name == "nt":
            local = os.path.join(plugin_dir, "deps", "bin", "windows", "ffmpeg.exe")
        elif sys.platform == "darwin":
            local = os.path.join(plugin_dir, "deps", "bin", "mac", "ffmpeg")
        else:
            local = os.path.join(plugin_dir, "deps", "bin", "linux", "ffmpeg")

        if os.path.exists(local):
            AudioSegment.converter = local
            self._ffmpeg_path = local
            return

        env_path = os.getenv("FFMPEG_PATH") or os.getenv("FFMPEG_BINARY")
        if env_path and os.path.exists(env_path):
            AudioSegment.converter = env_path
            self._ffmpeg_path = env_path
            return

        if which("ffmpeg"):
            return

        try:
            import imageio_ffmpeg
            path = imageio_ffmpeg.get_ffmpeg_exe()
            if path and os.path.exists(path):
                AudioSegment.converter = path
                self._ffmpeg_path = path
        except Exception:
            pass

    def _write_context_json(self, transcript, audio_path, timestamp, duration, language):
        os.makedirs(self.context_dir, exist_ok=True)
        digest_src = f"{timestamp}|{audio_path}|{transcript[:120]}".encode("utf-8")
        digest = hashlib.md5(digest_src).hexdigest()[:10]
        safe_ts = timestamp.replace(":", "-").replace(".", "-")
        filename = f"mic_{safe_ts}_{digest}.json"
        path = os.path.join(self.context_dir, filename)

        payload = {
            "type": "mic_transcription",
            "connection_name": getattr(self, "connection_name", "") or "",
            "timestamp": timestamp,
            "duration_seconds": round(duration, 3),
            "language": language or "",
            "whisper_model": self.whisper_model_size,
            "audio_path": audio_path,
            "transcription": transcript,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        return path

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

    def _safe_remove(self, path: str):
        try:
            if path and os.path.exists(path):
                os.remove(path)
        except Exception as e:
            BBLogger.log(f"[Mic] Could not remove {path}: {e}")

    def handle_message(self, message: Any, files: list | None = None) -> Any:
        return {"status": "ready", "model": self.whisper_model_size}
