"""Abstract base class for all target adapters."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any
from core.models import AnalyzeRequest, AnalyzeResponse


class TargetAdapter(ABC):
    """Abstract interface for communicating with the vishing detection system under test."""

    @abstractmethod
    def analyze(self, request: AnalyzeRequest) -> AnalyzeResponse:
        """Submit an AnalyzeRequest to the system under test and return a validated AnalyzeResponse."""
        pass

    def analyze_raw(self, request: AnalyzeRequest) -> Any:
        """Submit an AnalyzeRequest and return the raw output (dict or object) before contract parsing."""
        resp = self.analyze(request)
        if hasattr(resp, "model_dump"):
            return resp.model_dump()
        return resp
