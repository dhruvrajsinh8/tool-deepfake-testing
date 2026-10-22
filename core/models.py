"""Pydantic schemas defining the Request and Response contract.

System under test has 4 parts:
1. Model 1 (spoof detector): p_spoof (0.0 - 1.0), decision
2. Model 2 (speaker verification): cosine similarity (-1.0 - 1.0), match/mismatch
3. Model 3 (call behavior/intent): transcript, flags[], manipulation_score (0.0 - 1.0)
4. Risk engine: risk level (low/medium/high), action (allow/warn/force_mfa_callback)
"""

from __future__ import annotations

import base64
from dataclasses import dataclass, field
import uuid
from typing import Any, Dict, List, Literal, Optional, Tuple, Type
from pydantic import BaseModel, Field, ValidationError, field_validator


class AudioPayload(BaseModel):
    """Audio data payload formatted for analysis."""
    format: Literal["wav", "mp3"] = Field(default="wav", description="Audio format: 'wav' or 'mp3'")
    sample_rate: int = Field(default=16000, description="Sample rate in Hz (standard is 16kHz)")
    data_b64: str = Field(..., description="Base64-encoded raw audio bytes")

    @field_validator("data_b64")
    @classmethod
    def validate_b64(cls, v: str) -> str:
        if not v or not isinstance(v, str):
            raise ValueError("data_b64 must be a non-empty base64 string")
        try:
            # Validate basic decode
            base64.b64decode(v, validate=True)
        except Exception as e:
            raise ValueError(f"Invalid base64 audio payload: {e}") from e
        return v


class ClaimedIdentity(BaseModel):
    """Claimed speaker identity for speaker verification (Model 2)."""
    speaker_id: Optional[str] = Field(default=None, description="Enrolled speaker identifier")


class CallContext(BaseModel):
    """Contextual metadata about the call and transaction."""
    amount: float = Field(default=0.0, description="Transaction or wire amount in USD")
    new_beneficiary: bool = Field(default=False, description="Whether transaction is to a new beneficiary")
    asks_otp: bool = Field(default=False, description="Whether caller asks for OTP or sensitive codes")
    urgency: str = Field(default="low", description="Urgency level, e.g., low, medium, high")
    call_origin: str = Field(default="unknown", description="Origin of call: inbound, outbound, voip, etc.")


class AnalyzeRequest(BaseModel):
    """Full request contract sent to TargetAdapter."""
    request_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique request identifier")
    audio: AudioPayload
    claimed_identity: ClaimedIdentity = Field(default_factory=ClaimedIdentity)
    context: CallContext = Field(default_factory=CallContext)
    # Optional metadata that helps mock or local adapters identify source file without breaking contract
    source_filename: Optional[str] = Field(default=None, description="Original filename for mock/traceability")


class Model1Spoof(BaseModel):
    """Model 1 output: Voice deepfake / spoof detection."""
    p_spoof: float = Field(..., ge=0.0, le=1.0, description="Probability of synthetic/spoofed voice (0.0 - 1.0)")
    decision: Literal["spoof", "bonafide", "fake", "real"] = Field(..., description="Binary classification decision")


class Model2Speaker(BaseModel):
    """Model 2 output: Speaker verification."""
    similarity: float = Field(..., ge=-1.0, le=1.0, description="Cosine similarity score (-1.0 to 1.0)")
    decision: Literal["match", "mismatch", "not_enrolled", "unknown"] = Field(
        ..., description="Verification decision"
    )


class Model3Behavior(BaseModel):
    """Model 3 output: Call behavior, intent, and transcription."""
    flags: List[str] = Field(default_factory=list, description="Behavioral threat flags detected (e.g. OTP_REQUEST)")
    manipulation_score: float = Field(..., ge=0.0, le=1.0, description="Manipulation/social engineering score (0.0 - 1.0)")
    transcript: str = Field(default="", description="Whisper ASR transcript of the voice segment")


class RiskDecision(BaseModel):
    """Rule-based risk engine decision."""
    level: Literal["low", "medium", "high"] = Field(..., description="Overall risk level")
    action: Literal["allow", "warn", "force_mfa_callback"] = Field(..., description="Recommended mitigation action")


class AnalyzeResponse(BaseModel):
    """Full response contract expected from TargetAdapter."""
    request_id: str = Field(..., description="Request identifier matching the request")
    model1: Model1Spoof
    model2: Model2Speaker
    model3: Model3Behavior
    risk: RiskDecision
    latency_ms: float = Field(..., ge=0.0, description="End-to-end processing latency in milliseconds")


class ContractViolationError(Exception):
    """Raised when adapter response violates the pydantic contract."""
    def __init__(self, message: str, raw_payload: Any = None):
        super().__init__(message)
        self.raw_payload = raw_payload


def parse_and_validate_response(data: Any) -> AnalyzeResponse:
    """Validate raw dictionary or object against AnalyzeResponse contract with detailed error reporting."""
    if isinstance(data, AnalyzeResponse):
        return data
    try:
        return AnalyzeResponse.model_validate(data)
    except Exception as e:
        error_msg = f"Contract violation in analysis response: {e}"
        raise ContractViolationError(error_msg, raw_payload=data) from e


@dataclass
class ContractInspectionResult:
    """Detailed contract verification report."""
    is_valid: bool
    missing_fields: List[str] = field(default_factory=list)
    unexpected_fields: List[str] = field(default_factory=list)
    validation_errors: List[str] = field(default_factory=list)
    response: Optional[AnalyzeResponse] = None
    raw_payload: Any = None


def _inspect_model_fields(
    data: Any,
    model_cls: Type[BaseModel],
    prefix: str = "",
) -> Tuple[List[str], List[str]]:
    """Recursively identify missing and unexpected fields in a dict against a BaseModel schema."""
    missing: List[str] = []
    unexpected: List[str] = []

    if not isinstance(data, dict):
        field_label = prefix[:-1] if prefix else "root"
        return [f"{field_label} (expected dict, got {type(data).__name__})"], []

    fields = model_cls.model_fields
    for field_name, field_info in fields.items():
        full_field = f"{prefix}{field_name}"
        if field_name not in data:
            if field_info.is_required():
                missing.append(full_field)
        else:
            val = data[field_name]
            ann = field_info.annotation
            if isinstance(ann, type) and issubclass(ann, BaseModel):
                sub_missing, sub_unexpected = _inspect_model_fields(val, ann, prefix=f"{full_field}.")
                missing.extend(sub_missing)
                unexpected.extend(sub_unexpected)

    for k in data.keys():
        if k not in fields:
            unexpected.append(f"{prefix}{k}")

    return missing, unexpected


def inspect_contract_compliance(raw_payload: Any) -> ContractInspectionResult:
    """Inspect raw adapter output against the AnalyzeResponse contract.

    Reports:
    - missing_fields: required schema fields not present in payload
    - unexpected_fields: extra fields returned by adapter that are not in schema
    - validation_errors: value constraint or type violations
    """
    if hasattr(raw_payload, "model_dump"):
        data = raw_payload.model_dump()
    elif isinstance(raw_payload, dict):
        data = raw_payload
    else:
        return ContractInspectionResult(
            is_valid=False,
            validation_errors=[f"Expected dictionary or BaseModel, got {type(raw_payload).__name__}"],
            raw_payload=raw_payload,
        )

    missing, unexpected = _inspect_model_fields(data, AnalyzeResponse)
    validation_errors: List[str] = []
    response: Optional[AnalyzeResponse] = None

    try:
        response = AnalyzeResponse.model_validate(data)
    except ValidationError as val_err:
        for err in val_err.errors():
            loc = ".".join(str(item) for item in err.get("loc", []))
            msg = err.get("msg", "Validation error")
            validation_errors.append(f"{loc}: {msg}")
    except Exception as exc:
        validation_errors.append(str(exc))

    is_valid = (len(missing) == 0 and len(unexpected) == 0 and len(validation_errors) == 0)

    return ContractInspectionResult(
        is_valid=is_valid,
        missing_fields=missing,
        unexpected_fields=unexpected,
        validation_errors=validation_errors,
        response=response,
        raw_payload=raw_payload,
    )

