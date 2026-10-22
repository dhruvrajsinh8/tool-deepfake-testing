"""LocalAdapter: In-process target adapter executing a Python callable."""

from __future__ import annotations

import importlib
import logging
from typing import Any, Callable, Optional
from adapters.base import TargetAdapter
from core.models import AnalyzeRequest, AnalyzeResponse, parse_and_validate_response

logger = logging.getLogger(__name__)


class LocalAdapter(TargetAdapter):
    """Adapter that executes an in-process callable dynamically imported

    from `local_module` and `local_function` configured in config.yaml.
    """

    def __init__(
        self,
        local_module: Optional[str] = None,
        local_function: Optional[str] = None,
        module_path: Optional[str] = None,
    ):
        mod_name = local_module
        func_name = local_function

        if not mod_name or not func_name:
            if module_path:
                if ":" in module_path:
                    mod_name, func_name = module_path.split(":", 1)
                elif "." in module_path:
                    mod_name, func_name = module_path.rsplit(".", 1)

        if not mod_name or not func_name:
            raise ValueError(
                "LocalAdapter requires both 'local_module' and 'local_function' "
                "(or a valid 'module_path' like 'package.module:function')."
            )

        self.module_name = mod_name
        self.function_name = func_name

        try:
            mod = importlib.import_module(self.module_name)
        except Exception as exc:
            raise ImportError(
                f"Failed to import local module '{self.module_name}': {exc}"
            ) from exc

        if not hasattr(mod, self.function_name):
            raise AttributeError(
                f"Module '{self.module_name}' has no function/attribute named '{self.function_name}'."
            )

        fn = getattr(mod, self.function_name)
        if not callable(fn):
            raise TypeError(
                f"Attribute '{self.function_name}' in module '{self.module_name}' is not callable."
            )

        self.target_fn: Callable[..., Any] = fn

    def analyze_raw(self, request: AnalyzeRequest) -> Any:
        """Call the local function and return raw result (dict or model) without contract enforcement."""
        result = self.target_fn(request)
        if hasattr(result, "model_dump"):
            return result.model_dump()
        return result

    def analyze(self, request: AnalyzeRequest) -> AnalyzeResponse:
        """Execute the local function and enforce response contract validation."""
        raw_result = self.target_fn(request)
        return parse_and_validate_response(raw_result)
