"""Milestone 2 Test Suite: LocalAdapter, HTTPAdapter, and check-connection command."""

import base64
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest
import requests
from typer.testing import CliRunner

from adapters.local import LocalAdapter
from adapters.http import HTTPAdapter
from core.audio import generate_synthetic_tone, save_wav
from core.config import AppConfig, HttpConfig, LocalConfig
from core.models import (
    AnalyzeRequest,
    AnalyzeResponse,
    ContractViolationError,
    inspect_contract_compliance,
)
from core.request_builder import build_request
from cli import app

runner = CliRunner()


@pytest.fixture
def sample_request(tmp_path: Path):
    sr = 16000
    tone = generate_synthetic_tone(duration_sec=2.0, sample_rate=sr)
    wav_path = save_wav(tmp_path / "sample.wav", tone, sample_rate=sr)
    return build_request(wav_path, speaker_id="test_speaker")


# ============================================================================
# 1. LocalAdapter Tests
# ============================================================================

def test_local_adapter_success_with_module_and_function(sample_request):
    """Verify LocalAdapter initializes and runs callable specified by local_module and local_function."""
    adapter = LocalAdapter(
        local_module="adapters.mock",
        local_function="mock_analyze",
    )
    resp = adapter.analyze(sample_request)

    assert isinstance(resp, AnalyzeResponse)
    assert resp.request_id == sample_request.request_id
    assert 0.0 <= resp.model1.p_spoof <= 1.0
    assert resp.model1.decision in ["spoof", "bonafide"]
    assert resp.risk.level in ["low", "medium", "high"]


def test_local_adapter_success_with_module_path(sample_request):
    """Verify LocalAdapter supports module_path format 'module:function'."""
    adapter = LocalAdapter(module_path="adapters.mock:mock_analyze")
    resp = adapter.analyze(sample_request)

    assert isinstance(resp, AnalyzeResponse)
    assert resp.request_id == sample_request.request_id


def test_local_adapter_raw_output(sample_request):
    """Verify LocalAdapter analyze_raw returns dictionary or raw model representation."""
    adapter = LocalAdapter(local_module="adapters.mock", local_function="mock_analyze")
    raw = adapter.analyze_raw(sample_request)

    assert isinstance(raw, dict)
    assert "model1" in raw
    assert "p_spoof" in raw["model1"]


def test_local_adapter_import_errors():
    """Verify clear error handling when module or function cannot be resolved."""
    with pytest.raises(ImportError):
        LocalAdapter(local_module="non_existent_module_xyz", local_function="func")

    with pytest.raises(AttributeError):
        LocalAdapter(local_module="adapters.mock", local_function="non_existent_function_abc")


def test_local_adapter_contract_violation_caught(sample_request):
    """Verify that if a local function returns an invalid payload, ContractViolationError is raised."""
    # Define an invalid callable that returns invalid p_spoof > 1.0
    def bad_callable(req):
        return {
            "request_id": req.request_id,
            "model1": {"p_spoof": 99.9, "decision": "spoof"},
            "model2": {"similarity": 0.5, "decision": "match"},
            "model3": {"flags": [], "manipulation_score": 0.1, "transcript": ""},
            "risk": {"level": "low", "action": "allow"},
            "latency_ms": 10.0,
        }

    # Patch importlib to return bad callable
    with patch("adapters.mock.mock_analyze", side_effect=bad_callable):
        adapter = LocalAdapter(local_module="adapters.mock", local_function="mock_analyze")
        with pytest.raises(ContractViolationError):
            adapter.analyze(sample_request)


# ============================================================================
# 2. HTTPAdapter Tests
# ============================================================================

def test_http_adapter_json_mode(sample_request):
    """Verify HTTPAdapter sends valid JSON base64 payload and parses valid response."""
    adapter = HTTPAdapter(
        url="https://api.vishing-test.local/v1/analyze",
        api_key="secret-key-123",
        payload_format="json",
        retries=1,
    )

    mock_response_payload = {
        "request_id": sample_request.request_id,
        "model1": {"p_spoof": 0.05, "decision": "bonafide"},
        "model2": {"similarity": 0.88, "decision": "match"},
        "model3": {"flags": [], "manipulation_score": 0.15, "transcript": "Hello world"},
        "risk": {"level": "low", "action": "allow"},
        "latency_ms": 42.0,
    }

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = mock_response_payload

    with patch.object(adapter.session, "post", return_value=mock_resp) as mock_post:
        resp = adapter.analyze(sample_request)

        assert mock_post.called
        call_kwargs = mock_post.call_args[1]
        assert call_kwargs["json"]["request_id"] == sample_request.request_id
        assert call_kwargs["json"]["audio"]["format"] == sample_request.audio.format
        assert call_kwargs["headers"]["Authorization"] == "Bearer secret-key-123"
        assert call_kwargs["headers"]["X-API-Key"] == "secret-key-123"

        assert isinstance(resp, AnalyzeResponse)
        assert resp.model1.p_spoof == 0.05
        assert resp.risk.level == "low"


def test_http_adapter_multipart_mode(sample_request):
    """Verify HTTPAdapter sends multipart form-data with decoded audio bytes."""
    adapter = HTTPAdapter(
        url="https://api.vishing-test.local/v1/analyze",
        api_key="secret-key-456",
        payload_format="multipart",
        retries=1,
    )

    mock_response_payload = {
        "request_id": sample_request.request_id,
        "model1": {"p_spoof": 0.92, "decision": "spoof"},
        "model2": {"similarity": 0.12, "decision": "mismatch"},
        "model3": {"flags": ["OTP_REQUEST"], "manipulation_score": 0.85, "transcript": "Give me code"},
        "risk": {"level": "high", "action": "force_mfa_callback"},
        "latency_ms": 55.0,
    }

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = mock_response_payload

    with patch.object(adapter.session, "post", return_value=mock_resp) as mock_post:
        resp = adapter.analyze(sample_request)

        assert mock_post.called
        call_kwargs = mock_post.call_args[1]
        assert "files" in call_kwargs
        assert "audio_file" in call_kwargs["files"]
        filename, data_bytes, mime = call_kwargs["files"]["audio_file"]
        assert mime == "audio/wav"
        assert len(data_bytes) > 0
        assert call_kwargs["data"]["request_id"] == sample_request.request_id

        assert isinstance(resp, AnalyzeResponse)
        assert resp.model1.decision == "spoof"
        assert resp.risk.level == "high"


def test_http_adapter_env_var_api_key(monkeypatch, sample_request):
    """Verify HTTPAdapter resolves API key from environment variable."""
    monkeypatch.setenv("TEST_API_KEY_VAR", "env-secret-999")

    adapter = HTTPAdapter(
        url="https://api.test/analyze",
        api_key_env="TEST_API_KEY_VAR",
        payload_format="json",
    )

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "request_id": sample_request.request_id,
        "model1": {"p_spoof": 0.1, "decision": "bonafide"},
        "model2": {"similarity": 0.9, "decision": "match"},
        "model3": {"flags": [], "manipulation_score": 0.1, "transcript": ""},
        "risk": {"level": "low", "action": "allow"},
        "latency_ms": 10.0,
    }

    with patch.object(adapter.session, "post", return_value=mock_resp) as mock_post:
        adapter.analyze(sample_request)
        headers = mock_post.call_args[1]["headers"]
        assert headers["Authorization"] == "Bearer env-secret-999"


def test_http_adapter_server_error(sample_request):
    """Verify HTTPAdapter raises ConnectionError on 500 error."""
    adapter = HTTPAdapter(url="https://api.test/analyze", retries=0)

    mock_resp = MagicMock()
    mock_resp.status_code = 500
    mock_resp.text = "Internal Server Error"

    with patch.object(adapter.session, "post", return_value=mock_resp):
        with pytest.raises(ConnectionError) as exc_info:
            adapter.analyze(sample_request)
        assert "HTTP 500" in str(exc_info.value)


# ============================================================================
# 3. Contract Compliance Inspector Tests
# ============================================================================

def test_inspect_contract_compliance_valid():
    """Verify that a compliant dictionary returns is_valid=True with 0 issues."""
    valid_payload = {
        "request_id": "test-uuid-1",
        "model1": {"p_spoof": 0.25, "decision": "bonafide"},
        "model2": {"similarity": 0.85, "decision": "match"},
        "model3": {"flags": [], "manipulation_score": 0.1, "transcript": "ok"},
        "risk": {"level": "low", "action": "allow"},
        "latency_ms": 12.5,
    }
    result = inspect_contract_compliance(valid_payload)
    assert result.is_valid is True
    assert len(result.missing_fields) == 0
    assert len(result.unexpected_fields) == 0
    assert len(result.validation_errors) == 0


def test_inspect_contract_compliance_missing_fields():
    """Verify that missing fields (e.g. model1.decision and risk) are detected."""
    incomplete_payload = {
        "request_id": "test-uuid-2",
        "model1": {"p_spoof": 0.25},  # missing decision
        "model2": {"similarity": 0.85, "decision": "match"},
        "model3": {"flags": [], "manipulation_score": 0.1, "transcript": ""},
        # missing risk completely
        "latency_ms": 12.5,
    }
    result = inspect_contract_compliance(incomplete_payload)
    assert result.is_valid is False
    assert "model1.decision" in result.missing_fields
    assert "risk" in result.missing_fields


def test_inspect_contract_compliance_unexpected_fields():
    """Verify that unexpected extra fields are detected."""
    extra_payload = {
        "request_id": "test-uuid-3",
        "model1": {"p_spoof": 0.25, "decision": "bonafide", "extra_model1_metric": 123},
        "model2": {"similarity": 0.85, "decision": "match"},
        "model3": {"flags": [], "manipulation_score": 0.1, "transcript": ""},
        "risk": {"level": "low", "action": "allow"},
        "latency_ms": 12.5,
        "unexpected_top_level_key": "debug",
    }
    result = inspect_contract_compliance(extra_payload)
    assert result.is_valid is False
    assert "unexpected_top_level_key" in result.unexpected_fields
    assert "model1.extra_model1_metric" in result.unexpected_fields


# ============================================================================
# 4. CLI check-connection Command Tests
# ============================================================================

def test_check_connection_cli_mock_adapter():
    """Verify check-connection command succeeds with MockAdapter."""
    res = runner.invoke(app, ["check-connection", "--adapter", "mock"])
    assert res.exit_code == 0
    assert "Target Adapter Connection" in res.output
    assert "CONTRACT COMPLIANCE PASSED" in res.output
    assert "Contract Status: MATCH" in res.output


def test_check_connection_cli_local_adapter():
    """Verify check-connection command succeeds with LocalAdapter."""
    res = runner.invoke(app, ["check-connection", "--adapter", "local"])
    assert res.exit_code == 0
    assert "CONTRACT COMPLIANCE PASSED" in res.output


def test_check_connection_cli_detects_contract_violation():
    """Verify check-connection lists missing or unexpected fields and exits with code 1."""
    # Define a flawed mock function that omits risk and adds an extra field
    def flawed_analyze(req):
        return {
            "request_id": req.request_id,
            "model1": {"p_spoof": 0.1, "decision": "bonafide"},
            "model2": {"similarity": 0.8, "decision": "match"},
            "model3": {"flags": [], "manipulation_score": 0.1, "transcript": ""},
            # missing risk
            "unexpected_custom_data": 42,
            "latency_ms": 10.0,
        }

    with patch("adapters.mock.mock_analyze", side_effect=flawed_analyze):
        res = runner.invoke(app, ["check-connection", "--adapter", "local"])
        assert res.exit_code == 1
        assert "CONTRACT COMPLIANCE FAILED" in res.output
        assert "Missing Field" in res.output
        assert "risk" in res.output
        assert "Unexpected Field" in res.output
        assert "unexpected_custom_data" in res.output
