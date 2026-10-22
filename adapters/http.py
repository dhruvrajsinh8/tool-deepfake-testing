"""HTTPAdapter: Remote REST API target adapter."""

from __future__ import annotations

import base64
import json
import logging
import os
import time
from typing import Any, Dict, Optional, Union

import requests
from requests.adapters import HTTPAdapter as RequestsHTTPAdapter
from urllib3.util import Retry

from adapters.base import TargetAdapter
from core.config import HttpConfig
from core.models import AnalyzeRequest, AnalyzeResponse, parse_and_validate_response

logger = logging.getLogger(__name__)


class HTTPAdapter(TargetAdapter):
    """Target adapter that communicates with a remote vishing detection API over HTTP/HTTPS.

    Supports:
    - URL configuration
    - Optional API key resolved from environment variable or explicit string
    - Configurable timeout and retries with exponential backoff
    - JSON (base64 audio) and multipart (raw binary audio upload) payload modes
    """

    def __init__(
        self,
        config: Optional[Union[HttpConfig, Dict[str, Any]]] = None,
        url: Optional[str] = None,
        api_key_env: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout: float = 10.0,
        retries: int = 3,
        payload_format: str = "json",
    ):
        if config is not None:
            if isinstance(config, dict):
                cfg = HttpConfig(**config)
            else:
                cfg = config
            self.url = url or cfg.url
            self.api_key_env = api_key_env if api_key_env is not None else cfg.api_key_env
            self.timeout = timeout if timeout != 10.0 else cfg.timeout
            self.retries = retries if retries != 3 else cfg.retries
            self.payload_format = payload_format if payload_format != "json" else cfg.payload_format
        else:
            self.url = url or "http://localhost:8000/analyze"
            self.api_key_env = api_key_env
            self.timeout = timeout
            self.retries = retries
            self.payload_format = payload_format

        self.api_key = api_key

        # Build requests session with retry strategy
        self.session = requests.Session()
        if self.retries > 0:
            retry_strategy = Retry(
                total=self.retries,
                backoff_factor=0.3,
                status_forcelist=[429, 500, 502, 503, 504],
                allowed_methods=["POST"],
                raise_on_status=False,
            )
            adapter_mount = RequestsHTTPAdapter(max_retries=retry_strategy)
            self.session.mount("http://", adapter_mount)
            self.session.mount("https://", adapter_mount)

    def _resolve_api_key(self) -> Optional[str]:
        if self.api_key:
            return self.api_key
        if self.api_key_env:
            return os.getenv(self.api_key_env)
        return None

    def _get_headers(self) -> Dict[str, str]:
        headers: Dict[str, str] = {}
        api_key = self._resolve_api_key()
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
            headers["X-API-Key"] = api_key
        return headers

    def analyze_raw(self, request: AnalyzeRequest) -> Dict[str, Any]:
        """Send the request to the HTTP endpoint and return raw response JSON dictionary."""
        headers = self._get_headers()
        t0 = time.perf_counter()

        if self.payload_format.lower() == "multipart":
            raw_audio = base64.b64decode(request.audio.data_b64)
            mime = "audio/wav" if request.audio.format == "wav" else "audio/mpeg"
            filename = request.source_filename or f"audio.{request.audio.format}"

            # Provide both 'audio_file' and 'audio' keys for compatibility
            files = {
                "audio_file": (filename, raw_audio, mime),
            }
            data = {
                "request_id": request.request_id,
                "format": request.audio.format,
                "sample_rate": str(request.audio.sample_rate),
                "claimed_identity": json.dumps(request.claimed_identity.model_dump()),
                "context": json.dumps(request.context.model_dump()),
                "source_filename": filename,
            }

            try:
                resp = self.session.post(
                    self.url,
                    data=data,
                    files=files,
                    headers=headers,
                    timeout=self.timeout,
                )
            except requests.exceptions.RequestException as exc:
                raise ConnectionError(f"HTTP request failed to {self.url}: {exc}") from exc
        else:
            # JSON base64 mode
            headers["Content-Type"] = "application/json"
            payload = request.model_dump()
            try:
                resp = self.session.post(
                    self.url,
                    json=payload,
                    headers=headers,
                    timeout=self.timeout,
                )
            except requests.exceptions.RequestException as exc:
                raise ConnectionError(f"HTTP request failed to {self.url}: {exc}") from exc

        elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 2)

        if resp.status_code >= 400:
            raise ConnectionError(
                f"HTTP {resp.status_code} error from {self.url}: {resp.text}"
            )

        try:
            result = resp.json()
        except Exception as exc:
            raise ValueError(f"HTTP response from {self.url} is not valid JSON: {resp.text}") from exc

        if not isinstance(result, dict):
            raise ValueError(f"Expected JSON object from {self.url}, got {type(result).__name__}")

        # Inject measured latency if not provided by server
        if "latency_ms" not in result:
            result["latency_ms"] = elapsed_ms

        return result

    def analyze(self, request: AnalyzeRequest) -> AnalyzeResponse:
        """Submit an AnalyzeRequest over HTTP and validate against the AnalyzeResponse contract."""
        raw_result = self.analyze_raw(request)
        return parse_and_validate_response(raw_result)
