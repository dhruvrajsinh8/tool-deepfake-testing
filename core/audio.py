"""Audio processing manager: loading, resampling, mono conversion, validation, and encoding."""

from __future__ import annotations

import base64
from dataclasses import dataclass, field
import io
import logging
from pathlib import Path
import shutil
import subprocess
import sys
from typing import List, Optional, Union

import librosa
import numpy as np
import soundfile as sf

logger = logging.getLogger(__name__)

SUPPORTED_AUDIO_EXTENSIONS = {".wav", ".flac", ".mp3"}


def check_ffmpeg_exists(raise_error: bool = False) -> bool:
    """Verify that the ffmpeg executable exists in PATH.

    Prints a clear error message to stderr if not found.
    """
    found = shutil.which("ffmpeg") is not None
    if not found:
        error_msg = (
            "FFmpeg not found! ffmpeg is required for MP3 encoding/decoding and robustness testing. "
            "Please install ffmpeg (e.g. via 'brew install ffmpeg' or 'apt-get install ffmpeg') "
            "and ensure it is available in your PATH."
        )
        print(f"\n[ERROR] {error_msg}\n", file=sys.stderr)
        if raise_error:
            raise RuntimeError(error_msg)
        return False
    return True


@dataclass
class AudioProcessingResult:
    """Result of loading and processing an audio file."""
    data_np: np.ndarray
    sample_rate: int
    duration_sec: float
    b64_audio: str
    format: str = "wav"
    warnings: List[str] = field(default_factory=list)
    b64_wav: str = ""

    def __post_init__(self):
        # Ensure backward compatibility with code/tests expecting b64_wav
        if not self.b64_wav and self.b64_audio:
            self.b64_wav = self.b64_audio
        elif not self.b64_audio and self.b64_wav:
            self.b64_audio = self.b64_wav


def encode_audio_to_mp3(
    audio: np.ndarray,
    sample_rate: int,
    bitrate: str = "128k",
) -> bytes:
    """Encode a mono float32 numpy audio array to MP3 bytes via ffmpeg."""
    check_ffmpeg_exists(raise_error=True)

    # First write temporary 16-bit PCM WAV to in-memory buffer
    wav_buf = io.BytesIO()
    sf.write(wav_buf, audio, sample_rate, format="WAV", subtype="PCM_16")
    wav_bytes = wav_buf.getvalue()

    # Pipe WAV into ffmpeg to produce MP3
    cmd = [
        "ffmpeg",
        "-y",
        "-f", "wav",
        "-i", "pipe:0",
        "-codec:a", "libmp3lame",
        "-b:a", bitrate,
        "-ar", str(sample_rate),
        "-f", "mp3",
        "pipe:1",
    ]
    try:
        proc = subprocess.run(cmd, input=wav_bytes, capture_output=True, check=True)
        return proc.stdout
    except subprocess.CalledProcessError as exc:
        stderr_msg = exc.stderr.decode("utf-8", errors="replace")
        raise RuntimeError(f"FFmpeg MP3 encoding failed: {stderr_msg}") from exc


def reencode_wav_to_mp3(
    input_wav_path: Union[str, Path],
    output_mp3_path: Union[str, Path],
    bitrate: str = "128k",
) -> Path:
    """Re-encode an input WAV file to an MP3 file at specified bitrate via ffmpeg."""
    check_ffmpeg_exists(raise_error=True)
    in_path = Path(input_wav_path)
    out_path = Path(output_mp3_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "ffmpeg",
        "-y",
        "-i", str(in_path),
        "-codec:a", "libmp3lame",
        "-b:a", bitrate,
        str(out_path),
    ]
    try:
        subprocess.run(cmd, capture_output=True, check=True)
        return out_path
    except subprocess.CalledProcessError as exc:
        stderr_msg = exc.stderr.decode("utf-8", errors="replace")
        raise RuntimeError(f"FFmpeg re-encoding to {bitrate} MP3 failed: {stderr_msg}") from exc


def load_and_process_audio(
    file_path: Union[str, Path],
    target_sr: int = 16000,
    send_format: str = "original",
) -> AudioProcessingResult:
    """Load an audio file (wav/flac/mp3), convert to mono, validate duration,

    and return AudioProcessingResult formatted according to send_format:
    - 'original': Keep input format (if wav or mp3). For flac, convert to wav. Retain real sample rate.
    - 'wav16k': Resample to 16 kHz and encode as 16-bit PCM WAV.
    - 'mp3': Encode as MP3 via ffmpeg, retaining real sample rate.
    """
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"Audio file not found: {path}")

    ext = path.suffix.lower()
    if ext not in SUPPORTED_AUDIO_EXTENSIONS:
        raise ValueError(
            f"Unsupported audio format '{ext}' for file {path.name}. "
            f"Supported formats: {', '.join(sorted(SUPPORTED_AUDIO_EXTENSIONS))}"
        )

    # If MP3 is involved, ensure ffmpeg is available
    if ext == ".mp3" or send_format == "mp3":
        check_ffmpeg_exists(raise_error=True)

    # Use soundfile first for fast loading, fallback to librosa
    try:
        data, orig_sr = sf.read(str(path))
    except Exception as e:
        logger.debug("soundfile failed to read %s (%s); falling back to librosa", path, e)
        try:
            data, orig_sr = librosa.load(str(path), sr=None, mono=False)
            if data.ndim > 1:
                data = data.T
        except Exception as e2:
            raise ValueError(f"Unable to load audio file {path}: {e2}") from e2

    # Convert to mono if multichannel
    if data.ndim > 1:
        data = np.mean(data, axis=1)

    # Ensure float32 normalized between -1.0 and 1.0
    if np.issubdtype(data.dtype, np.integer):
        max_val = float(np.iinfo(data.dtype).max)
        data = (data / max_val).astype(np.float32)
    else:
        data = data.astype(np.float32)

    duration_sec = float(len(data)) / float(orig_sr)
    warnings: List[str] = []

    if duration_sec < 1.0:
        msg = f"Audio duration {duration_sec:.2f}s is shorter than recommended 1.0s minimum."
        logger.warning(msg)
        warnings.append(msg)
    elif duration_sec > 30.0:
        msg = f"Audio duration {duration_sec:.2f}s is longer than recommended 30.0s maximum."
        logger.warning(msg)
        warnings.append(msg)

    valid_formats = {"original", "wav16k", "mp3"}
    if send_format not in valid_formats:
        raise ValueError(f"Invalid send_format '{send_format}'. Expected one of: {sorted(valid_formats)}")

    # Format output according to send_format
    if send_format == "wav16k":
        final_sr = target_sr
        if orig_sr != target_sr:
            logger.debug("Resampling %s from %d Hz to %d Hz", path.name, orig_sr, target_sr)
            processed_data = librosa.resample(data, orig_sr=orig_sr, target_sr=target_sr)
        else:
            processed_data = data

        buffer = io.BytesIO()
        sf.write(buffer, processed_data, final_sr, format="WAV", subtype="PCM_16")
        audio_bytes = buffer.getvalue()
        b64_audio = base64.b64encode(audio_bytes).decode("ascii")
        out_format = "wav"

    elif send_format == "mp3":
        final_sr = orig_sr
        if ext == ".mp3":
            audio_bytes = path.read_bytes()
        else:
            audio_bytes = encode_audio_to_mp3(data, sample_rate=orig_sr, bitrate="128k")
        b64_audio = base64.b64encode(audio_bytes).decode("ascii")
        out_format = "mp3"

    else:
        # send_format == "original"
        final_sr = orig_sr
        if ext == ".mp3":
            audio_bytes = path.read_bytes()
            out_format = "mp3"
        elif ext == ".wav":
            # Clean mono 16-bit PCM WAV at real sample_rate
            buffer = io.BytesIO()
            sf.write(buffer, data, orig_sr, format="WAV", subtype="PCM_16")
            audio_bytes = buffer.getvalue()
            out_format = "wav"
        else:
            # FLAC input: contract requires wav or mp3, so convert to WAV at real sample_rate
            buffer = io.BytesIO()
            sf.write(buffer, data, orig_sr, format="WAV", subtype="PCM_16")
            audio_bytes = buffer.getvalue()
            out_format = "wav"

        b64_audio = base64.b64encode(audio_bytes).decode("ascii")

    return AudioProcessingResult(
        data_np=data,
        sample_rate=final_sr,
        duration_sec=duration_sec,
        b64_audio=b64_audio,
        format=out_format,
        warnings=warnings,
    )


def save_wav(file_path: Union[str, Path], audio: np.ndarray, sample_rate: int = 16000) -> Path:
    """Save a numpy audio array as a 16-bit PCM WAV file."""
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), audio, sample_rate, format="WAV", subtype="PCM_16")
    return path


def save_mp3(
    file_path: Union[str, Path],
    audio: np.ndarray,
    sample_rate: int = 16000,
    bitrate: str = "128k",
) -> Path:
    """Save a numpy audio array as an MP3 file via ffmpeg."""
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    mp3_bytes = encode_audio_to_mp3(audio, sample_rate=sample_rate, bitrate=bitrate)
    path.write_bytes(mp3_bytes)
    return path


def generate_synthetic_tone(
    duration_sec: float = 3.0,
    freq: float = 440.0,
    sample_rate: int = 16000,
) -> np.ndarray:
    """Generate a clean synthetic sine wave with a soft envelope (for testing)."""
    t = np.linspace(0, duration_sec, int(sample_rate * duration_sec), endpoint=False)
    wave = 0.5 * np.sin(2 * np.pi * freq * t)
    # Apply fade in/out envelope to avoid clicks
    envelope_len = min(int(sample_rate * 0.05), len(wave) // 2)
    if envelope_len > 0:
        fade_in = np.linspace(0, 1, envelope_len)
        fade_out = np.linspace(1, 0, envelope_len)
        wave[:envelope_len] *= fade_in
        wave[-envelope_len:] *= fade_out
    return wave.astype(np.float32)
