"""MockAdapter: Deterministic fake scoring for testing without live models."""

from __future__ import annotations

import hashlib
import time
from typing import List

from adapters.base import TargetAdapter
from core.models import (
    AnalyzeRequest,
    AnalyzeResponse,
    Model1Spoof,
    Model2Speaker,
    Model3Behavior,
    RiskDecision,
)


class MockAdapter(TargetAdapter):
    """Deterministic mock adapter implementing all 4 components of the system under test."""

    def __init__(self, spoof_threshold: float = 0.5, similarity_threshold: float = 0.75):
        self.spoof_threshold = spoof_threshold
        self.similarity_threshold = similarity_threshold

    def analyze(self, request: AnalyzeRequest) -> AnalyzeResponse:
        t0 = time.perf_counter()

        # Deterministic scoring key from filename or request_id
        src = (request.source_filename or request.request_id or "").lower()

        # 1. Model 1: Spoof Detector
        if any(token in src for token in ["fake", "spoof", "deepfake", "clone", "synth"]):
            p_spoof = 0.94
        elif any(token in src for token in ["real", "bonafide", "genuine", "human"]):
            p_spoof = 0.06
        else:
            # Deterministic hash-based score
            h = int(hashlib.sha256(src.encode("utf-8")).hexdigest()[:8], 16)
            p_spoof = round(0.1 + (h % 800) / 1000.0, 3)

        # Acoustic degradation impact for robustness evaluation (mp3 re-encoding)
        if "32k" in src:
            delta = 0.055 if p_spoof < 0.5 else -0.045
            p_spoof = round(min(0.99, max(0.01, p_spoof + delta)), 3)
        elif "64k" in src:
            delta = 0.030 if p_spoof < 0.5 else -0.025
            p_spoof = round(min(0.99, max(0.01, p_spoof + delta)), 3)
        elif "128k" in src:
            delta = 0.012 if p_spoof < 0.5 else -0.010
            p_spoof = round(min(0.99, max(0.01, p_spoof + delta)), 3)

        m1_decision = "spoof" if p_spoof >= self.spoof_threshold else "bonafide"

        # 2. Model 2: Speaker Verification
        speaker_id = request.claimed_identity.speaker_id
        if not speaker_id:
            similarity = 0.0
            m2_decision = "unknown"
        elif any(token in src for token in ["mismatch", "impersonat", "attacker", "intruder"]):
            similarity = 0.28
            m2_decision = "mismatch"
        elif any(token in src for token in ["match", "real", "bonafide", "enrolled", "verified"]):
            similarity = 0.89
            m2_decision = "match"
        else:
            h_spk = int(hashlib.sha256(f"{src}:{speaker_id}".encode("utf-8")).hexdigest()[:8], 16)
            similarity = round(0.3 + (h_spk % 650) / 1000.0, 3)
            m2_decision = "match" if similarity >= self.similarity_threshold else "mismatch"

        # 3. Model 3: Call Behavior & Intent
        ctx = request.context
        flags: List[str] = []
        if ctx.asks_otp:
            flags.append("OTP_REQUEST")
        if ctx.urgency.lower() in ["high", "critical", "urgent"]:
            flags.append("HIGH_URGENCY")
        if ctx.new_beneficiary:
            flags.append("NEW_BENEFICIARY")
        if ctx.amount > 10000.0:
            flags.append("HIGH_VALUE_TRANSACTION")
        if any(token in src for token in ["vishing", "phishing", "scam", "ceo"]):
            if "AUTHORITY_PRESSURE" not in flags:
                flags.append("AUTHORITY_PRESSURE")

        # Manipulation score derived from contextual behavioral flags
        base_manipulation = 0.12
        if flags:
            manipulation_score = min(0.98, base_manipulation + 0.22 * len(flags))
        else:
            manipulation_score = base_manipulation
        manipulation_score = round(manipulation_score, 3)

        # Synthesize realistic transcript
        if "OTP_REQUEST" in flags and "HIGH_URGENCY" in flags:
            transcript = (
                "This is security operations. We detected suspicious activity on your account. "
                "I need you to urgently read out the 6-digit one-time passcode sent to your mobile."
            )
        elif "NEW_BENEFICIARY" in flags or "HIGH_VALUE_TRANSACTION" in flags:
            transcript = (
                f"Please authorize the immediate payment of ${ctx.amount:,.2f} to the new beneficiary account. "
                "This was approved by executive leadership."
            )
        elif "spoof" in src or "fake" in src:
            transcript = "Good afternoon, I am calling to update our wire transfer instructions immediately."
        else:
            transcript = "Hello, calling to follow up on the quarterly account statement we discussed yesterday."

        # 4. Rule-based Risk Engine
        # Rules:
        # High Risk: High spoof probability OR severe behavioral flags + low speaker similarity
        # Medium Risk: Moderate spoof, or single high-risk behavior
        # Low Risk: Clean voice, enrolled speaker match, no suspicious flags
        if p_spoof >= 0.70 or (p_spoof >= self.spoof_threshold and manipulation_score >= 0.40):
            risk_level = "high"
            risk_action = "force_mfa_callback"
        elif manipulation_score >= 0.60 and m2_decision == "mismatch":
            risk_level = "high"
            risk_action = "force_mfa_callback"
        elif p_spoof >= 0.40 or manipulation_score >= 0.45 or (speaker_id and m2_decision == "mismatch"):
            risk_level = "medium"
            risk_action = "warn"
        else:
            risk_level = "low"
            risk_action = "allow"

        elapsed_ms = round((time.perf_counter() - t0) * 1000.0 + 15.0, 2)

        return AnalyzeResponse(
            request_id=request.request_id,
            model1=Model1Spoof(p_spoof=p_spoof, decision=m1_decision),
            model2=Model2Speaker(similarity=similarity, decision=m2_decision),
            model3=Model3Behavior(flags=flags, manipulation_score=manipulation_score, transcript=transcript),
            risk=RiskDecision(level=risk_level, action=risk_action),
            latency_ms=elapsed_ms,
        )


def mock_analyze(request: AnalyzeRequest) -> AnalyzeResponse:
    """Callable endpoint for local adapter testing."""
    adapter = MockAdapter()
    return adapter.analyze(request)
