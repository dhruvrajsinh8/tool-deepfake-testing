# Vishing & Voice Deepfake Detection Test Harness

A testing, benchmarking, and red-teaming harness for voice deepfake and vishing (voice phishing) detection systems.

The harness evaluates voice-based security systems across 4 components:
1. **Model 1 (Spoof Detector)**: Voice synthesis/cloning detection (`p_spoof`).
2. **Model 2 (Speaker Verification)**: Cosine similarity against enrolled identity (`similarity`).
3. **Model 3 (Call Behavior & Intent)**: NLP-based social engineering flag detection (`manipulation_score`, `flags`).
4. **Risk Decision Engine**: Rule-based transaction mitigation (`allow`, `warn`, `force_mfa_callback`).

---

## Features

- **Multi-Format Audio Handling**: Ingests `wav`, `flac`, and `mp3` audio files.
- **Configurable Audio Negotiation**: `send_format: original | wav16k | mp3`.
- **Acoustic Robustness Testing**: Re-encodes audio to MP3 at 128k, 64k, and 32k bitrates using `ffmpeg` to test spoof model resilience against lossy compression.
- **Pluggable Target Adapters**:
  - `MockAdapter`: Built-in deterministic mock with acoustic degradation emulation.
  - `LocalAdapter`: Dynamic in-process execution of Python callables (`local_module` / `local_function`).
  - `HTTPAdapter`: Remote REST API connector supporting JSON base64 and multipart modes, retries with backoff, and API key auth.
- **Contract Verification (`check-connection`)**: Inspects adapter outputs for full Pydantic schema compliance, highlighting missing or unexpected fields.

---

## Prerequisites

- **Python 3.10+**
- **FFmpeg**: Required for MP3 encoding/decoding and robustness evaluation.
  ```bash
  # macOS
  brew install ffmpeg

  # Ubuntu/Debian
  sudo apt-get install ffmpeg
  ```

---

## Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd <repo-name>

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

## Quickstart

Run commands using the `./harness` wrapper:

```bash
# 1. Analyze a single audio file
./harness single samples/real_sample.wav

# 2. Analyze with context metadata
./harness single samples/fake_sample.wav --context samples/context_urgent_otp.yaml

# 3. Transmit as compressed MP3
./harness single samples/real_sample.wav --send-format mp3

# 4. Run MP3 acoustic degradation robustness test (128k, 64k, 32k)
./harness robustness samples/real_sample.wav

# 5. Check adapter connectivity & schema contract compliance
./harness check-connection
./harness check-connection --adapter local
./harness check-connection --adapter http
```

---

## Running Tests

Run the full pytest suite (25 tests covering contracts, audio round-trips, adapters, and CLI):

```bash
pytest -v
```

---

## Configuration

Edit `config.yaml` to configure target adapter, thresholds, and endpoints:

```yaml
adapter: mock  # Options: mock, local, http
send_format: original  # Options: original | wav16k | mp3

thresholds:
  spoof_threshold: 0.5
  similarity_threshold: 0.75

http:
  url: "http://localhost:8000/analyze"
  api_key_env: "VISHING_API_KEY"
  timeout: 10.0
  retries: 3
  payload_format: "json"  # Options: json or multipart

local:
  local_module: "adapters.mock"
  local_function: "mock_analyze"
```
