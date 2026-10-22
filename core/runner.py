"""Runner module: Adapter factory, request execution, structured logging, and contract validation."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

from adapters.base import TargetAdapter
from adapters.mock import MockAdapter
from core.config import AppConfig, load_config
from core.models import AnalyzeRequest, AnalyzeResponse, parse_and_validate_response

logger = logging.getLogger("vishing_harness")


def setup_logging(log_dir: str = "logs") -> None:
    """Configure structured logger to write request/response traces safely."""
    path = Path(log_dir)
    path.mkdir(parents=True, exist_ok=True)
    log_file = path / "harness.jsonl"

    handler = logging.FileHandler(log_file, encoding="utf-8")
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('{"time":"%(asctime)s", "level":"%(levelname)s", "message":%(message)s}')
    handler.setFormatter(formatter)

    # Avoid adding multiple duplicate handlers
    if not any(isinstance(h, logging.FileHandler) for h in logger.handlers):
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)


def sanitize_dict_for_logging(data: Dict[str, Any]) -> Dict[str, Any]:
    """Ensure no API keys, tokens, or large audio byte blobs are dumped raw in logs."""
    sanitized: Dict[str, Any] = {}
    sensitive_keys = {"api_key", "authorization", "secret", "token", "password"}
    for k, v in data.items():
        if any(s in k.lower() for s in sensitive_keys):
            sanitized[k] = "***REDACTED***"
        elif k == "audio" and isinstance(v, dict) and "data_b64" in v:
            sanitized[k] = {
                "format": v.get("format"),
                "sample_rate": v.get("sample_rate"),
                "data_b64_len": len(v.get("data_b64", "")),
            }
        elif isinstance(v, dict):
            sanitized[k] = sanitize_dict_for_logging(v)
        else:
            sanitized[k] = v
    return sanitized


def get_adapter(config: Optional[AppConfig] = None) -> TargetAdapter:
    """Factory to instantiate the appropriate TargetAdapter according to configuration."""
    cfg = config or load_config()
    choice = cfg.adapter.lower().strip()

    if choice == "mock":
        return MockAdapter(
            spoof_threshold=cfg.thresholds.spoof_threshold,
            similarity_threshold=cfg.thresholds.similarity_threshold,
        )
    elif choice == "local":
        from adapters.local import LocalAdapter
        return LocalAdapter(
            local_module=cfg.local.local_module,
            local_function=cfg.local.local_function,
            module_path=cfg.local.module_path,
        )
    elif choice == "http":
        from adapters.http import HTTPAdapter
        return HTTPAdapter(cfg.http)
    else:
        raise ValueError(f"Unknown adapter type '{choice}'. Valid options: mock, local, http.")


def run_analysis(
    request: AnalyzeRequest,
    adapter: Optional[TargetAdapter] = None,
    config: Optional[AppConfig] = None,
) -> AnalyzeResponse:
    """Execute an analysis request through the configured adapter, enforcing contract validation and logging."""
    cfg = config or load_config()
    target = adapter or get_adapter(cfg)
    setup_logging()

    # Log sanitized request
    req_dump = sanitize_dict_for_logging(request.model_dump())
    logger.info(json.dumps({"event": "request", "request_id": request.request_id, "payload": req_dump}))

    # Execute adapter
    raw_response = target.analyze(request)

    # Enforce contract validation
    validated_response = parse_and_validate_response(raw_response)

    # Log sanitized response
    resp_dump = sanitize_dict_for_logging(validated_response.model_dump())
    logger.info(json.dumps({"event": "response", "request_id": validated_response.request_id, "payload": resp_dump}))

    return validated_response
