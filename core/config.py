"""Configuration loading and validation."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal, Optional
from pydantic import BaseModel, Field
import yaml


class ThresholdsConfig(BaseModel):
    spoof_threshold: float = 0.5
    similarity_threshold: float = 0.75


class HttpConfig(BaseModel):
    url: str = "http://localhost:8000/analyze"
    api_key_env: Optional[str] = "VISHING_API_KEY"
    timeout: float = 10.0
    retries: int = 3
    payload_format: Literal["json", "multipart"] = "json"


class LocalConfig(BaseModel):
    local_module: Optional[str] = "adapters.mock"
    local_function: Optional[str] = "mock_analyze"
    module_path: Optional[str] = "adapters.mock:mock_analyze"


class PathsConfig(BaseModel):
    reports_dir: str = "reports"
    samples_dir: str = "samples"
    scenarios_dir: str = "scenarios"
    enrolled_dir: str = "data/enrolled"


class AppConfig(BaseModel):
    adapter: Literal["mock", "local", "http"] = "mock"
    send_format: Literal["original", "wav16k", "mp3"] = "original"
    thresholds: ThresholdsConfig = Field(default_factory=ThresholdsConfig)
    http: HttpConfig = Field(default_factory=HttpConfig)
    local: LocalConfig = Field(default_factory=LocalConfig)
    paths: PathsConfig = Field(default_factory=PathsConfig)


def load_config(config_path: Optional[str | Path] = None) -> AppConfig:
    """Load configuration from YAML file or return default configuration."""
    if config_path is None:
        # Search current working directory or relative to project root
        candidates = [
            Path("config.yaml"),
            Path(__file__).resolve().parent.parent / "config.yaml",
        ]
        for candidate in candidates:
            if candidate.is_file():
                config_path = candidate
                break

    if config_path is not None and Path(config_path).is_file():
        with open(config_path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
            return AppConfig.model_validate(raw)

    return AppConfig()
