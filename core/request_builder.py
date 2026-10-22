"""Builds valid AnalyzeRequest payloads from audio files and optional metadata."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Union
import uuid
import yaml

from core.audio import load_and_process_audio
from core.config import load_config
from core.models import AnalyzeRequest, AudioPayload, CallContext, ClaimedIdentity


def load_context_from_file(context_path: Union[str, Path]) -> Dict[str, Any]:
    """Load call context dictionary from a YAML or JSON file."""
    path = Path(context_path)
    if not path.is_file():
        raise FileNotFoundError(f"Context file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return data


def build_request(
    audio_path: Union[str, Path],
    speaker_id: Optional[str] = None,
    context_data: Optional[Dict[str, Any]] = None,
    context_file: Optional[Union[str, Path]] = None,
    request_id: Optional[str] = None,
    send_format: Optional[str] = None,
    target_sr: int = 16000,
) -> AnalyzeRequest:
    """Load and process the audio file, assemble context and speaker ID, and return an AnalyzeRequest."""
    path = Path(audio_path)
    
    if send_format is None:
        try:
            cfg = load_config()
            format_to_use = cfg.send_format
        except Exception:
            format_to_use = "original"
    else:
        format_to_use = send_format

    audio_res = load_and_process_audio(path, target_sr=target_sr, send_format=format_to_use)

    # Merge context file if provided
    merged_ctx: Dict[str, Any] = {}
    if context_file:
        file_ctx = load_context_from_file(context_file)
        if isinstance(file_ctx, dict):
            # Check if nested under 'context' key
            if "context" in file_ctx and isinstance(file_ctx["context"], dict):
                merged_ctx.update(file_ctx["context"])
            else:
                merged_ctx.update(file_ctx)
    if context_data:
        merged_ctx.update(context_data)

    call_context = CallContext(**{
        k: v for k, v in merged_ctx.items()
        if k in CallContext.model_fields
    })

    claimed_identity = ClaimedIdentity(speaker_id=speaker_id)
    req_id = request_id or str(uuid.uuid4())

    return AnalyzeRequest(
        request_id=req_id,
        audio=AudioPayload(
            format=audio_res.format,
            sample_rate=audio_res.sample_rate,
            data_b64=audio_res.b64_audio,
        ),
        claimed_identity=claimed_identity,
        context=call_context,
        source_filename=path.name,
    )
