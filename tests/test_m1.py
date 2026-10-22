"""Milestone 1 Test Suite: Contracts, MockAdapter, Audio Manager, Runner, and CLI."""

import base64
from pathlib import Path
import pytest
from typer.testing import CliRunner

from core.audio import generate_synthetic_tone, load_and_process_audio, save_wav
from core.models import (
    AnalyzeRequest,
    AnalyzeResponse,
    AudioPayload,
    CallContext,
    ClaimedIdentity,
    ContractViolationError,
    Model1Spoof,
    Model2Speaker,
    Model3Behavior,
    RiskDecision,
    parse_and_validate_response,
)
from adapters.mock import MockAdapter
from core.request_builder import build_request
from core.runner import run_analysis
from cli import app

runner = CliRunner()


@pytest.fixture
def temp_audio_files(tmp_path: Path):
    sr = 16000
    tone_3s = generate_synthetic_tone(duration_sec=3.0, freq=440.0, sample_rate=sr)
    tone_short = generate_synthetic_tone(duration_sec=0.5, freq=440.0, sample_rate=sr)
    tone_long = generate_synthetic_tone(duration_sec=32.0, freq=440.0, sample_rate=sr)

    real_path = save_wav(tmp_path / "real_voice.wav", tone_3s, sr)
    fake_path = save_wav(tmp_path / "fake_clone.wav", tone_3s, sr)
    short_path = save_wav(tmp_path / "short.wav", tone_short, sr)
    long_path = save_wav(tmp_path / "long.wav", tone_long, sr)

    return {
        "real": real_path,
        "fake": fake_path,
        "short": short_path,
        "long": long_path,
    }


def test_audio_loader_and_duration_warnings(temp_audio_files):
    # Standard 3s audio
    res = load_and_process_audio(temp_audio_files["real"])
    assert res.sample_rate == 16000
    assert abs(res.duration_sec - 3.0) < 0.05
    assert len(res.warnings) == 0
    assert len(res.b64_wav) > 0

    # Short audio (< 1s warning)
    res_short = load_and_process_audio(temp_audio_files["short"])
    assert any("shorter than recommended 1.0s" in w for w in res_short.warnings)

    # Long audio (> 30s warning)
    res_long = load_and_process_audio(temp_audio_files["long"])
    assert any("longer than recommended 30.0s" in w for w in res_long.warnings)


def test_pydantic_contracts_validation():
    # Valid response
    valid_data = {
        "request_id": "test-req-1",
        "model1": {"p_spoof": 0.85, "decision": "spoof"},
        "model2": {"similarity": 0.91, "decision": "match"},
        "model3": {"flags": ["OTP_REQUEST"], "manipulation_score": 0.75, "transcript": "Urgent transfer"},
        "risk": {"level": "high", "action": "force_mfa_callback"},
        "latency_ms": 32.5,
    }
    resp = parse_and_validate_response(valid_data)
    assert resp.model1.p_spoof == 0.85
    assert resp.risk.action == "force_mfa_callback"

    # Contract violation: missing model1 and invalid p_spoof out of range
    invalid_data = {
        "request_id": "bad-req",
        "model1": {"p_spoof": 1.5, "decision": "spoof"},  # > 1.0 invalid
        "model2": {"similarity": 0.9, "decision": "match"},
        "model3": {"flags": [], "manipulation_score": 0.1, "transcript": ""},
        "risk": {"level": "low", "action": "allow"},
        "latency_ms": 10.0,
    }
    with pytest.raises(ContractViolationError) as exc_info:
        parse_and_validate_response(invalid_data)
    assert "Contract violation" in str(exc_info.value)


def test_mock_adapter_deterministic_scoring(temp_audio_files):
    adapter = MockAdapter(spoof_threshold=0.5, similarity_threshold=0.75)

    # Fake audio should result in high spoof probability and high risk
    req_fake = build_request(
        temp_audio_files["fake"],
        speaker_id="exec_1",
        context_data={"asks_otp": True, "urgency": "high"},
    )
    resp_fake = adapter.analyze(req_fake)
    assert resp_fake.model1.p_spoof >= 0.9
    assert resp_fake.model1.decision == "spoof"
    assert "OTP_REQUEST" in resp_fake.model3.flags
    assert resp_fake.risk.level == "high"

    # Real audio should result in low spoof probability and low risk
    req_real = build_request(
        temp_audio_files["real"],
        speaker_id="exec_1",
        context_data={"asks_otp": False, "urgency": "low"},
    )
    resp_real = adapter.analyze(req_real)
    assert resp_real.model1.p_spoof <= 0.1
    assert resp_real.model1.decision == "bonafide"
    assert resp_real.risk.level == "low"


def test_cli_single_execution(temp_audio_files):
    # Run cli command with real audio
    res = runner.invoke(app, ["single", str(temp_audio_files["real"]), "--speaker", "user_10"])
    assert res.exit_code == 0
    assert "Request Metadata" in res.output
    assert "Model Component Outputs" in res.output
    assert "BONAFIDE" in res.output
