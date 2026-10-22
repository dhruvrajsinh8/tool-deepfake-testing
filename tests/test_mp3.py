"""Tests for MP3 support, audio manager send_format, contracts, and robustness command."""

import base64
import io
from pathlib import Path
import pytest
from typer.testing import CliRunner
import soundfile as sf
import numpy as np

from core.audio import (
    check_ffmpeg_exists,
    encode_audio_to_mp3,
    generate_synthetic_tone,
    load_and_process_audio,
    reencode_wav_to_mp3,
    save_mp3,
    save_wav,
)
from core.models import AnalyzeRequest, AudioPayload
from core.request_builder import build_request
from cli import app

runner = CliRunner()


@pytest.fixture
def temp_audio_dir(tmp_path: Path):
    sr = 22050  # non-standard sample rate to test real sample_rate preservation
    tone = generate_synthetic_tone(duration_sec=2.0, freq=440.0, sample_rate=sr)
    wav_path = save_wav(tmp_path / "test_22k.wav", tone, sample_rate=sr)
    mp3_path = save_mp3(tmp_path / "test_22k.mp3", tone, sample_rate=sr, bitrate="128k")
    return {
        "dir": tmp_path,
        "wav": wav_path,
        "mp3": mp3_path,
        "sample_rate": sr,
        "raw_tone": tone,
    }


def test_ffmpeg_check_exists():
    """Verify that ffmpeg exists check works correctly."""
    assert check_ffmpeg_exists(raise_error=False) is True


def test_mp3_encode_decode_round_trip(temp_audio_dir):
    """Test mp3 encode/decode round-trip:

    1. Take a known audio waveform.
    2. Encode to MP3 via ffmpeg.
    3. Decode MP3 back to numpy array.
    4. Assert sample rate, approximate duration, and signal correlation.
    """
    sr = temp_audio_dir["sample_rate"]
    tone = temp_audio_dir["raw_tone"]

    # 1. Encode to MP3 bytes
    mp3_bytes = encode_audio_to_mp3(tone, sample_rate=sr, bitrate="128k")
    assert len(mp3_bytes) > 0
    # MP3 header magic: ID3v2 ('ID3') or sync word 0xFF 0xFB/0xF3
    assert mp3_bytes[:3] == b"ID3" or mp3_bytes[0] == 0xFF

    # 2. Decode MP3 bytes using soundfile
    buf = io.BytesIO(mp3_bytes)
    decoded_audio, decoded_sr = sf.read(buf)

    # 3. Assert properties
    assert decoded_sr == sr
    assert decoded_audio.ndim == 1
    # Duration within 0.08s (due to MP3 padding/frames)
    orig_duration = len(tone) / sr
    dec_duration = len(decoded_audio) / decoded_sr
    assert abs(dec_duration - orig_duration) < 0.08

    # 4. Assert signal content (cross-correlation with original)
    min_len = min(len(tone), len(decoded_audio))
    corr = np.corrcoef(tone[:min_len], decoded_audio[:min_len])[0, 1]
    assert corr > 0.85, f"Expected high correlation after MP3 roundtrip, got {corr}"


def test_audio_manager_send_format_options(temp_audio_dir):
    """Test the three send_format modes: original, wav16k, mp3."""
    wav_path = temp_audio_dir["wav"]
    mp3_path = temp_audio_dir["mp3"]
    sr = temp_audio_dir["sample_rate"]

    # Mode 1: send_format="original" with WAV
    res_wav_orig = load_and_process_audio(wav_path, send_format="original")
    assert res_wav_orig.format == "wav"
    assert res_wav_orig.sample_rate == sr
    assert len(res_wav_orig.b64_audio) > 0

    # Mode 1b: send_format="original" with MP3 -> keep format="mp3" and real sample_rate
    res_mp3_orig = load_and_process_audio(mp3_path, send_format="original")
    assert res_mp3_orig.format == "mp3"
    assert res_mp3_orig.sample_rate == sr
    assert len(res_mp3_orig.b64_audio) > 0

    # Mode 2: send_format="wav16k" -> resample to 16000 Hz WAV
    res_wav16k = load_and_process_audio(mp3_path, target_sr=16000, send_format="wav16k")
    assert res_wav16k.format == "wav"
    assert res_wav16k.sample_rate == 16000

    # Mode 3: send_format="mp3" with WAV input -> encodes to MP3 keeping real sample_rate
    res_mp3_converted = load_and_process_audio(wav_path, send_format="mp3")
    assert res_mp3_converted.format == "mp3"
    assert res_mp3_converted.sample_rate == sr


def test_request_contract_audio_format_restriction():
    """Verify that AudioPayload format contract strictly enforces 'wav' or 'mp3'."""
    # Valid formats
    dummy_b64 = base64.b64encode(b"RIFFdummydata").decode("ascii")
    payload_wav = AudioPayload(format="wav", sample_rate=16000, data_b64=dummy_b64)
    assert payload_wav.format == "wav"

    payload_mp3 = AudioPayload(format="mp3", sample_rate=44100, data_b64=dummy_b64)
    assert payload_mp3.format == "mp3"

    # Invalid formats must raise ValueError / ValidationError
    with pytest.raises(Exception):
        AudioPayload(format="flac", sample_rate=16000, data_b64=dummy_b64)

    with pytest.raises(Exception):
        AudioPayload(format="ogg", sample_rate=16000, data_b64=dummy_b64)


def test_build_request_mp3_format(temp_audio_dir):
    """Verify build_request sets audio.format='mp3' and keeps real sample_rate."""
    mp3_path = temp_audio_dir["mp3"]
    req = build_request(mp3_path, send_format="original")

    assert req.audio.format == "mp3"
    assert req.audio.sample_rate == temp_audio_dir["sample_rate"]
    assert len(req.audio.data_b64) > 0


def test_robustness_command_cli(temp_audio_dir):
    """Verify cli robustness command executes re-encodings at 128k, 64k, 32k."""
    wav_path = temp_audio_dir["wav"]
    res = runner.invoke(app, ["robustness", str(wav_path)])

    assert res.exit_code == 0
    assert "MP3 Robustness Evaluation Report" in res.output
    assert "128k" in res.output
    assert "64k" in res.output
    assert "32k" in res.output
    assert "p_spoof" in res.output
    assert "baseline" in res.output
    assert "Robustness Summary" in res.output
