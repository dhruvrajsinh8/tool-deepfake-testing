"""Command-line interface for vishing-test-harness using Typer."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box

from core.audio import (
    check_ffmpeg_exists,
    generate_synthetic_tone,
    reencode_wav_to_mp3,
    save_wav,
)
from core.config import load_config
from core.request_builder import build_request
from core.runner import get_adapter, run_analysis
from core.models import (
    ContractViolationError,
    inspect_contract_compliance,
)

app = typer.Typer(
    name="vishing-test-harness",
    help="Voice deepfake / vishing detection system test harness.",
    add_completion=False,
)
console = Console()


def get_risk_style(level: str) -> str:
    level_lower = level.lower()
    if level_lower == "high":
        return "bold red"
    elif level_lower == "medium":
        return "bold yellow"
    return "bold green"


@app.command(name="single", help="Analyze a single audio file through the detection system.")
def single(
    wav: Path = typer.Argument(..., help="Path to audio file (wav/flac/mp3).", exists=True, readable=True),
    speaker: Optional[str] = typer.Option(None, "--speaker", "-s", help="Enrolled speaker ID for verification."),
    context: Optional[Path] = typer.Option(None, "--context", "-c", help="Path to YAML/JSON context file.", exists=True),
    config_path: Optional[Path] = typer.Option(None, "--config", help="Path to custom config.yaml"),
    send_format: Optional[str] = typer.Option(None, "--send-format", help="Override send_format: original | wav16k | mp3"),
) -> None:
    """Analyze a single audio sample with the configured adapter and display component outputs."""
    cfg = load_config(config_path)
    if send_format:
        cfg.send_format = send_format

    console.print(f"[bold cyan]vishing-test-harness[/bold cyan] [dim]• Adapter:[/dim] [yellow]{cfg.adapter}[/yellow] [dim]• Format:[/dim] [yellow]{cfg.send_format}[/yellow]")
    console.print(f"[dim]Analyzing {wav.name}...[/dim]")

    try:
        request = build_request(
            audio_path=wav,
            speaker_id=speaker,
            context_file=context,
            send_format=cfg.send_format,
        )
        response = run_analysis(request, config=cfg)
    except ContractViolationError as cve:
        console.print(f"[bold red]Contract Violation Error:[/bold red] {cve}")
        sys.exit(1)
    except Exception as exc:
        console.print(f"[bold red]Analysis failed:[/bold red] {exc}")
        sys.exit(1)

    # 1. Overview Table
    summary_table = Table(box=box.ROUNDED, show_header=False, padding=(0, 1))
    summary_table.add_column("Key", style="bold cyan", width=18)
    summary_table.add_column("Value")

    summary_table.add_row("Request ID", response.request_id)
    summary_table.add_row("File", wav.name)
    summary_table.add_row("Format Sent", request.audio.format)
    summary_table.add_row("Sample Rate", f"{request.audio.sample_rate} Hz")
    summary_table.add_row("Claimed Speaker", speaker or "[dim]None[/dim]")
    summary_table.add_row("Latency", f"{response.latency_ms:.1f} ms")

    console.print(Panel(summary_table, title="[bold]Request Metadata[/bold]", border_style="cyan"))

    # 2. Components Breakdown Table
    comp_table = Table(box=box.DOUBLE_EDGE, expand=True)
    comp_table.add_column("Component", style="bold white", width=22)
    comp_table.add_column("Key Metrics", width=25)
    comp_table.add_column("Decision / Flags", width=30)
    comp_table.add_column("Details", justify="left")

    # Model 1
    m1_color = "red" if response.model1.decision in ["spoof", "fake"] else "green"
    comp_table.add_row(
        "Model 1 (Spoof Detector)",
        f"p_spoof: [{m1_color}]{response.model1.p_spoof:.3f}[/{m1_color}]",
        f"[{m1_color} bold]{response.model1.decision.upper()}[/{m1_color} bold]",
        f"Threshold: {cfg.thresholds.spoof_threshold}",
    )

    # Model 2
    m2_color = "green" if response.model2.decision == "match" else ("yellow" if response.model2.decision == "unknown" else "red")
    comp_table.add_row(
        "Model 2 (Speaker Match)",
        f"similarity: [{m2_color}]{response.model2.similarity:.3f}[/{m2_color}]",
        f"[{m2_color} bold]{response.model2.decision.upper()}[/{m2_color} bold]",
        f"Target: {speaker or 'N/A'}",
    )

    # Model 3
    flags_str = ", ".join(response.model3.flags) if response.model3.flags else "[dim]None[/dim]"
    m3_color = "red" if response.model3.manipulation_score >= 0.5 else "green"
    comp_table.add_row(
        "Model 3 (Call Behavior)",
        f"manipulation: [{m3_color}]{response.model3.manipulation_score:.3f}[/{m3_color}]",
        f"Flags: {flags_str}",
        f'Transcript: "{response.model3.transcript}"',
    )

    console.print(Panel(comp_table, title="[bold]Model Component Outputs[/bold]", border_style="blue"))

    # 3. Risk Engine Badge
    risk_style = get_risk_style(response.risk.level)
    risk_panel_text = (
        f"Risk Level: [{risk_style}]{response.risk.level.upper()}[/{risk_style}]    "
        f"Action: [{risk_style}]{response.risk.action.upper()}[/{risk_style}]"
    )
    console.print(Panel(risk_panel_text, title="[bold]Risk Engine Decision[/bold]", border_style=risk_style))


@app.command(name="robustness", help="Evaluate audio robustness under MP3 re-encoding at 128k, 64k, and 32k bitrates.")
def robustness(
    wav: Path = typer.Argument(..., help="Path to base audio file (wav/flac/mp3).", exists=True, readable=True),
    speaker: Optional[str] = typer.Option(None, "--speaker", "-s", help="Enrolled speaker ID for verification."),
    context: Optional[Path] = typer.Option(None, "--context", "-c", help="Path to YAML/JSON context file.", exists=True),
    config_path: Optional[Path] = typer.Option(None, "--config", help="Path to custom config.yaml"),
) -> None:
    """Evaluate acoustic robustness by re-encoding audio to MP3 at 128k, 64k, and 32k bitrates

    using ffmpeg and reporting p_spoof change versus the original audio.
    """
    # 1. Verify ffmpeg exists
    try:
        check_ffmpeg_exists(raise_error=True)
    except Exception as exc:
        console.print(f"[bold red]FFmpeg Check Failed:[/bold red] {exc}")
        sys.exit(1)

    cfg = load_config(config_path)
    console.print(f"[bold cyan]vishing-test-harness[/bold cyan] [dim]• Robustness Evaluation[/dim]")
    console.print(f"[dim]Testing base file: {wav.name} with adapter '{cfg.adapter}'[/dim]\n")

    # 2. Baseline evaluation (original audio)
    try:
        req_base = build_request(
            audio_path=wav,
            speaker_id=speaker,
            context_file=context,
            send_format="original",
        )
        resp_base = run_analysis(req_base, config=cfg)
    except Exception as exc:
        console.print(f"[bold red]Failed to analyze baseline audio:[/bold red] {exc}")
        sys.exit(1)

    base_p_spoof = resp_base.model1.p_spoof
    base_decision = resp_base.model1.decision

    # 3. Re-encode to MP3 at 128k, 64k, 32k
    bitrates = ["128k", "64k", "32k"]
    results = []

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir_path = Path(temp_dir)

        for br in bitrates:
            temp_mp3 = temp_dir_path / f"{wav.stem}_{br}.mp3"
            try:
                reencode_wav_to_mp3(wav, temp_mp3, bitrate=br)
                req_mp3 = build_request(
                    audio_path=temp_mp3,
                    speaker_id=speaker,
                    context_file=context,
                    send_format="mp3",
                )
                # Keep traceable filename stem for adapters
                req_mp3.source_filename = f"{wav.stem}_{br}.mp3"
                resp_mp3 = run_analysis(req_mp3, config=cfg)
                p_spoof = resp_mp3.model1.p_spoof
                delta = p_spoof - base_p_spoof

                results.append({
                    "bitrate": br,
                    "p_spoof": p_spoof,
                    "delta": delta,
                    "decision": resp_mp3.model1.decision,
                    "latency_ms": resp_mp3.latency_ms,
                    "risk": resp_mp3.risk.level,
                })
            except Exception as exc:
                console.print(f"[bold red]Failed testing MP3 at {br}:[/bold red] {exc}")
                sys.exit(1)

    # 4. Present Results Table
    table = Table(box=box.ROUNDED, expand=True)
    table.add_column("Audio Condition", style="bold white", width=22)
    table.add_column("Bitrate", justify="center", width=12)
    table.add_column("p_spoof", justify="right", width=12)
    table.add_column("Δ p_spoof", justify="right", width=14)
    table.add_column("Decision", justify="center", width=14)
    table.add_column("Risk Level", justify="center", width=12)
    table.add_column("Latency", justify="right", width=12)

    # Baseline Row
    base_dec_color = "red" if base_decision in ["spoof", "fake"] else "green"
    table.add_row(
        "Original (Baseline)",
        "Original",
        f"{base_p_spoof:.3f}",
        "[dim]-- baseline --[/dim]",
        f"[{base_dec_color} bold]{base_decision.upper()}[/{base_dec_color} bold]",
        f"[{get_risk_style(resp_base.risk.level)}]{resp_base.risk.level.upper()}[/{get_risk_style(resp_base.risk.level)}]",
        f"{resp_base.latency_ms:.1f} ms",
    )

    # Degradation Rows
    max_abs_delta = 0.0
    flips = 0

    for r in results:
        delta_val = r["delta"]
        if abs(delta_val) > max_abs_delta:
            max_abs_delta = abs(delta_val)

        if r["decision"] != base_decision:
            flips += 1

        delta_sign = "+" if delta_val > 0 else ""
        if abs(delta_val) < 0.03:
            delta_style = "green"
        elif abs(delta_val) < 0.08:
            delta_style = "yellow"
        else:
            delta_style = "red"

        delta_str = f"[{delta_style}]{delta_sign}{delta_val:.3f}[/{delta_style}]"
        r_dec_color = "red" if r["decision"] in ["spoof", "fake"] else "green"

        table.add_row(
            f"MP3 Re-encoded",
            r["bitrate"],
            f"{r['p_spoof']:.3f}",
            delta_str,
            f"[{r_dec_color} bold]{r['decision'].upper()}[/{r_dec_color} bold]",
            f"[{get_risk_style(r['risk'])}]{r['risk'].upper()}[/{get_risk_style(r['risk'])}]",
            f"{r['latency_ms']:.1f} ms",
        )

    console.print(Panel(table, title="[bold]MP3 Robustness Evaluation Report[/bold]", border_style="cyan"))

    # Summary Panel
    if flips > 0:
        summary_style = "bold yellow"
        summary_msg = (
            f"[bold yellow]WARNING:[/bold yellow] Audio compression caused {flips} decision flip(s)! "
            f"Maximum p_spoof perturbation was Δ = {max_abs_delta:.3f}."
        )
    else:
        summary_style = "bold green"
        summary_msg = (
            f"[bold green]STABLE:[/bold green] Decision remained consistent across all MP3 bitrates "
            f"(128k, 64k, 32k). Maximum p_spoof perturbation was Δ = {max_abs_delta:.3f}."
        )

    console.print(Panel(summary_msg, title="[bold]Robustness Summary[/bold]", border_style=summary_style))


@app.command(name="check-connection", help="Send a test file and verify that the adapter response strictly complies with the contract.")
def check_connection(
    file: Optional[Path] = typer.Option(None, "--file", "-f", help="Audio file to test. Defaults to sample audio."),
    adapter: Optional[str] = typer.Option(None, "--adapter", "-a", help="Adapter to test (mock, local, http). Defaults to config."),
    config_path: Optional[Path] = typer.Option(None, "--config", help="Path to custom config.yaml"),
) -> None:
    """Send one test file and print whether the response matches the contract,

    listing any missing or unexpected fields.
    """
    cfg = load_config(config_path)
    if adapter:
        cfg.adapter = adapter

    console.print(f"[bold cyan]vishing-test-harness[/bold cyan] [dim]• Connection & Contract Check[/dim]")
    console.print(f"[dim]Testing adapter:[/dim] [yellow]{cfg.adapter}[/yellow]")

    # 1. Resolve test audio file
    test_audio_path: Path
    temp_dir_holder = None

    if file is not None and file.is_file():
        test_audio_path = file
    else:
        # Search candidate samples in samples_dir
        sample_candidates = [
            Path(cfg.paths.samples_dir) / "real_sample.wav",
            Path(cfg.paths.samples_dir) / "fake_sample.wav",
            Path("samples/real_sample.wav"),
        ]
        found_sample = None
        for candidate in sample_candidates:
            if candidate.is_file():
                found_sample = candidate
                break

        if found_sample:
            test_audio_path = found_sample
        else:
            # Generate a temporary clean tone
            temp_dir_holder = tempfile.TemporaryDirectory()
            temp_path = Path(temp_dir_holder.name) / "test_tone.wav"
            tone = generate_synthetic_tone(duration_sec=2.0, sample_rate=16000)
            test_audio_path = save_wav(temp_path, tone, sample_rate=16000)

    console.print(f"[dim]Using test audio file: {test_audio_path.name}[/dim]")

    # 2. Instantiate TargetAdapter
    try:
        target_adapter = get_adapter(cfg)
    except Exception as exc:
        console.print(f"[bold red]Adapter Initialization Failed:[/bold red] {exc}")
        if temp_dir_holder:
            temp_dir_holder.cleanup()
        sys.exit(1)

    # 3. Build test request
    try:
        request = build_request(
            audio_path=test_audio_path,
            speaker_id="check_user_01",
            send_format=cfg.send_format,
        )
    except Exception as exc:
        console.print(f"[bold red]Failed to build request:[/bold red] {exc}")
        if temp_dir_holder:
            temp_dir_holder.cleanup()
        sys.exit(1)

    # 4. Execute request and get raw response
    try:
        raw_response = target_adapter.analyze_raw(request)
    except Exception as exc:
        console.print(f"[bold red]Connection / Execution Failed:[/bold red] {exc}")
        if temp_dir_holder:
            temp_dir_holder.cleanup()
        sys.exit(1)

    # 5. Inspect response for contract compliance
    inspection = inspect_contract_compliance(raw_response)

    # 6. Display Connection & Contract Results
    meta_table = Table(box=box.ROUNDED, show_header=False, padding=(0, 1))
    meta_table.add_column("Key", style="bold cyan", width=20)
    meta_table.add_column("Value")

    meta_table.add_row("Adapter Type", cfg.adapter)
    if cfg.adapter == "http":
        meta_table.add_row("Target URL", cfg.http.url)
        meta_table.add_row("Payload Mode", cfg.http.payload_format)
    elif cfg.adapter == "local":
        meta_table.add_row("Module", cfg.local.local_module or "N/A")
        meta_table.add_row("Function", cfg.local.local_function or "N/A")

    if isinstance(raw_response, dict) and "latency_ms" in raw_response:
        meta_table.add_row("Reported Latency", f"{raw_response['latency_ms']:.1f} ms")

    console.print(Panel(meta_table, title="[bold]Target Adapter Connection[/bold]", border_style="cyan"))

    # Contract verification display
    if inspection.is_valid:
        pass_text = (
            "[bold green]✔ CONTRACT COMPLIANCE PASSED[/bold green]\n\n"
            "• All required fields are present\n"
            "• No unexpected or extra fields detected\n"
            "• All field types and constraints valid\n\n"
            f"[dim]Validated against Pydantic schema: AnalyzeResponse[/dim]"
        )
        console.print(Panel(pass_text, title="[bold green]Contract Status: MATCH[/bold green]", border_style="green"))

        # Print quick summary of model results
        if inspection.response:
            resp = inspection.response
            m_table = Table(box=box.SIMPLE)
            m_table.add_column("Model 1 (Spoof)", style="cyan")
            m_table.add_column("Model 2 (Speaker)", style="cyan")
            m_table.add_column("Model 3 (Behavior)", style="cyan")
            m_table.add_column("Risk Level", style="cyan")
            m_table.add_column("Action", style="cyan")

            m_table.add_row(
                f"{resp.model1.decision} ({resp.model1.p_spoof:.2f})",
                f"{resp.model2.decision} ({resp.model2.similarity:.2f})",
                f"score: {resp.model3.manipulation_score:.2f}",
                resp.risk.level.upper(),
                resp.risk.action.upper(),
            )
            console.print(m_table)

        if temp_dir_holder:
            temp_dir_holder.cleanup()
        return

    # If contract compliance failed, list all details
    fail_text = "[bold red]✘ CONTRACT COMPLIANCE FAILED[/bold red]\nThe response received from the adapter does not match the specification contract."
    console.print(Panel(fail_text, title="[bold red]Contract Status: VIOLATION[/bold red]", border_style="red"))

    issues_table = Table(box=box.ROUNDED, expand=True)
    issues_table.add_column("Violation Type", style="bold red", width=22)
    issues_table.add_column("Field / Description", justify="left")

    for f in inspection.missing_fields:
        issues_table.add_row("[bold red]Missing Field[/bold red]", f"[red]{f}[/red]")

    for f in inspection.unexpected_fields:
        issues_table.add_row("[bold yellow]Unexpected Field[/bold yellow]", f"[yellow]{f}[/yellow]")

    for err in inspection.validation_errors:
        issues_table.add_row("[bold red]Value / Constraint Error[/bold red]", err)

    console.print(issues_table)

    if temp_dir_holder:
        temp_dir_holder.cleanup()

    sys.exit(1)


@app.command(name="batch", help="Batch evaluation over a labeled dataset (Scheduled for Milestone 3).")
def batch(
    folder: Path = typer.Argument(..., help="Folder containing audio files."),
    labels: Path = typer.Option(..., "--labels", "-l", help="CSV file with ground truth labels."),
) -> None:
    console.print("[yellow]Batch evaluation command will be enabled in Milestone 3.[/yellow]")


@app.command(name="scenario", help="Execute YAML behavioral test scenarios (Scheduled for Milestone 4).")
def scenario(
    scenarios_dir: Path = typer.Argument(..., help="Path to scenario YAML files."),
) -> None:
    console.print("[yellow]Scenario evaluation command will be enabled in Milestone 4.[/yellow]")


@app.command(name="load", help="Run latency & load testing (Scheduled for Milestone 7).")
def load_test(
    n: int = typer.Option(50, "--n", help="Total number of requests."),
    concurrency: int = typer.Option(5, "--concurrency", help="Concurrent workers."),
) -> None:
    console.print("[yellow]Load testing command will be enabled in Milestone 7.[/yellow]")


if __name__ == "__main__":
    app()
