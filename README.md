# NotionHealth AI v2.0

AI-Powered Personal Health & Wellness Management System with Notion MCP and TRIBEv2 Brain Prediction

NotionHealth AI connects [Claude Desktop](https://claude.ai/download) to your personal Notion workspace via the Model Context Protocol (MCP), giving you 16 AI-powered health-tracking tools — and optionally wiring in **Meta's TRIBEv2** brain-predictive foundation model for real fMRI-quality brain-activity predictions from health narratives.

> **TRIBEv2** (`facebook/tribev2`) is a gated model hosted by Facebook Research on HuggingFace. It requires requesting access to Meta's LLaMA 3.2 3B model before it will download. See [Gated Model Access](#gated-model-access-huggingface) below.

---

## What It Does

| Capability | Without TRIBEv2 | With TRIBEv2 |
| --- | --- | --- |
| Log weight, sleep, mood, exercise | Yes | Yes |
| Medication & appointment management | Yes | Yes |
| Health goals & symptom analysis | Yes | Yes |
| AI health insights (Claude) | Yes | Yes |
| Brain response prediction | Simulation-based | Real fMRI predictions (20,484 cortical vertices) |
| Brain-health correlation | Research patterns | Actual model inference |
| Optimal daily schedule | Yes | Yes |
| Cognitive health report | Yes | Yes |

---

## Quick Start

### Prerequisites

| Requirement | Where | Notes |
| --- | --- | --- |
| Python 3.10+ | [python.org](https://python.org) | See [Apple Silicon note](#apple-silicon-m1m2m3m4) |
| Notion account | [notion.so](https://notion.so) | Free |
| Claude Desktop | [claude.ai/download](https://claude.ai/download) | Free |
| Notion API key | [notion.so/my-integrations](https://www.notion.so/my-integrations) | Free |
| Anthropic API key | [console.anthropic.com](https://console.anthropic.com) | Pay-as-you-go |
| HuggingFace token | [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) | Only for real TRIBEv2 weights |

### 1. Clone and install

```bash
git clone https://github.com/rotsl/notion-health-ai.git
cd notion-health-ai

# Standard install (no TRIBEv2)
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -e .
```

### 2. Configure `.env`

```bash
cp .env.example .env
```

Fill in at minimum:

```text
NOTION_API_KEY=secret_your_token_here
ANTHROPIC_API_KEY=sk-ant-your_key_here
```

### 3. Create Notion databases

Share a Notion page with your integration (Page → Share → Connections → your integration), then:

```bash
python scripts/setup_notion.py
```

The script will:

- Prompt for any missing API keys and save them to `.env`
- Archive existing databases from a previous run (so you start fresh)
- Create 6 databases with the full schema
- Seed realistic sample data (7 health metrics, 2 meds, 3 appointments, 2 goals, 2 symptoms, 1 brain entry)
- Build a rich dashboard page with embedded database views and command reference
- Write all database IDs back into `.env` automatically

### 4. Connect Claude Desktop

Edit `~/Library/Application Support/Claude/claude_desktop_config.json` (macOS) or `%APPDATA%\Claude\claude_desktop_config.json` (Windows):

```json
{
  "mcpServers": {
    "notion-health-ai": {
      "command": "/absolute/path/to/notion-health-ai/venv/bin/notion-health-mcp"
    }
  }
}
```

Find the absolute path:

```bash
echo "$(pwd)/venv/bin/notion-health-mcp"
```

Fully quit and reopen Claude Desktop (Cmd+Q on Mac). Ask Claude: *"What health tools do you have available?"* — you should see all 16 tools.

---

## Apple Silicon (M1/M2/M3/M4)

The standard `venv` created by Rosetta Python (x86_64) can only install torch up to 2.2.2, which is incompatible with TRIBEv2 (requires torch ≥ 2.5.1). **You must use a native arm64 Python.**

Check if Homebrew Python 3.11 is available:

```bash
/opt/homebrew/bin/python3.11 --version
# If not: brew install python@3.11
```

Create a native arm64 venv:

```bash
/opt/homebrew/bin/python3.11 -m venv venv_arm64
source venv_arm64/bin/activate
pip install -e .
```

Then point Claude Desktop at the arm64 venv:

```json
{
  "mcpServers": {
    "notion-health-ai": {
      "command": "/absolute/path/to/notion-health-ai/venv_arm64/bin/notion-health-mcp"
    }
  }
}
```

The arm64 venv works for **all features** including TRIBEv2. The x86_64 venv works for everything **except** TRIBEv2 model loading.

---

## TRIBEv2 — Real Brain Predictions

TRIBEv2 (`facebook/tribev2`) is Meta's deep multimodal brain encoding model trained on fMRI data. It predicts brain activity across ~20,484 cortical vertices (fsaverage5 surface mesh) from text, audio, or video stimuli.

- Paper: [TRIBE v2: A Predictive Foundation Model](https://ai.meta.com/blog/tribe-v2-brain-predictive-foundation-model/)
- Model card: [huggingface.co/facebook/tribev2](https://huggingface.co/facebook/tribev2)
- Source: [github.com/facebookresearch/tribev2](https://github.com/facebookresearch/tribev2)
- License: CC-BY-NC-4.0 (non-commercial use only)

### Gated Model Access (HuggingFace)

TRIBEv2's text prediction pipeline uses **Meta-Llama/Llama-3.2-3B** as an internal text encoder. This is a **gated model** — you must request access before it will download:

1. Create a HuggingFace account at [huggingface.co](https://huggingface.co)
2. Go to [huggingface.co/meta-llama/Llama-3.2-3B](https://huggingface.co/meta-llama/Llama-3.2-3B) and click **Request access** (usually approved within minutes)
3. Create a token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) with at least **Read** scope
4. Add to your `.env`:

```text
HUGGING_FACE_TOKEN=hf_your_token_here
```

Without this token the system falls back to simulation-based predictions — fully functional for all 16 MCP tools.

### TRIBEv2 Installation (Apple Silicon / GPU)

See [GPU Setup](#gpu-setup) for full platform instructions. Quick summary for Apple Silicon:

```bash
source venv_arm64/bin/activate

# Install torch 2.5.1 + numpy 2.2.6 (exact versions required by TRIBEv2)
pip install "torch==2.5.1" "numpy==2.2.6" torchvision torchaudio

# Install whisperx (Whisper-based audio transcription for the text pipeline)
pip install whisperx

# Clone and install TRIBEv2 package
git clone https://github.com/facebookresearch/tribev2 /tmp/tribev2
pip install -e /tmp/tribev2
```

**First run** downloads:

- TRIBEv2 weights: ~676 MB (`best.ckpt`) → cached in `~/.cache/huggingface/`
- LLaMA 3.2 3B: ~1.65 GB → cached in `~/.cache/huggingface/`
- spaCy model + Whisper model → cached automatically

Subsequent runs use the cache and are much faster.

### Brain Regions Predicted

The system maps HCP MMP1.0 parcellation ROIs (181 labels) to 9 functional brain regions:

| Region | Key HCP ROIs | Activated by |
| --- | --- | --- |
| Visual Cortex | V1, V2, V3, V4, LO1/2 | Reading, nature, creative work |
| Auditory Cortex | A1, LBelt, MBelt, STGa | Music, social, spoken learning |
| Prefrontal Cortex | 46, 9a, 8Ad, FEF | Deep work, decision-making |
| Motor Cortex | 4, 6a, 6d, MI | Exercise, sport |
| Hippocampus | PHA1-3, RSC, POS1/2 | Sleep, learning, memory |
| Amygdala | pOFC, OFC, Pir | Social, music, stress |
| Insula | FOP1-5, PoI1/2 | Meditation, interoception |
| Anterior Cingulate | 24dd, a24, 33pr | Mindfulness, attention |
| Default Mode Network | d23ab, 7m, PCV | Rest, creativity, self-reflection |

---

## GPU Setup

### Apple Silicon (MPS) — Recommended for Mac users

The brain model (fMRI encoder) runs on MPS. The internal feature extractors (Whisper, LLaMA) run on CPU since they don't yet support MPS.

```bash
# Requires: /opt/homebrew/bin/python3.11 (native arm64)
/opt/homebrew/bin/python3.11 -m venv venv_arm64
source venv_arm64/bin/activate
pip install "torch==2.5.1" "numpy==2.2.6" torchvision
```

Verify MPS:

```python
import torch
print(torch.backends.mps.is_available())  # True
```

### NVIDIA GPU (CUDA) — Linux/Windows

```bash
python3 -m venv venv
source venv/bin/activate
pip install "torch==2.5.1" --index-url https://download.pytorch.org/whl/cu121
pip install "numpy==2.2.6" torchvision torchaudio whisperx
git clone https://github.com/facebookresearch/tribev2 /tmp/tribev2
pip install -e /tmp/tribev2
pip install -e .
```

CUDA 12.1+ required. Verify:

```python
import torch
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))
```

### CPU Only — Any Platform

Works everywhere, but TRIBEv2 inference is slow (~5–10 min per prediction).

```bash
pip install "torch==2.5.1" "numpy==2.2.6"
git clone https://github.com/facebookresearch/tribev2 /tmp/tribev2
pip install -e /tmp/tribev2
pip install -e .
```

### No GPU — Simulation Mode (Default)

If TRIBEv2 is not installed or the HuggingFace token is missing, the system uses neuroscience-researched simulation patterns. All 16 MCP tools work identically — only the source of brain predictions differs.

---

## Available MCP Tools (16 total)

### Health Management (12 tools)

| Tool | Description |
| --- | --- |
| `log_health_metric` | Log weight, sleep, mood, exercise, steps, water, vitals |
| `get_health_summary` | Aggregated summary for any date range |
| `add_medication` | Add a medication with dosage and schedule |
| `get_medication_schedule` | Today's or any day's medication list |
| `add_appointment` | Schedule a medical appointment |
| `get_upcoming_appointments` | Upcoming appointments for next N days |
| `set_health_goal` | Create a health goal with target and deadline |
| `get_goal_progress` | Check progress on all active goals |
| `update_goal_progress` | Update current value toward a goal |
| `log_symptom` | Log a symptom with severity, body part, triggers |
| `analyze_symptoms` | AI-powered symptom pattern analysis |
| `get_ai_insights` | Claude-generated health insights and recommendations |

### TRIBEv2 Brain Prediction (4 tools)

| Tool | Description |
| --- | --- |
| `predict_brain_response` | Predict brain activation for an activity (exercise, meditation, etc.) |
| `analyze_brain_health_correlation` | Correlate your health metrics with brain activity networks |
| `get_optimal_schedule` | Circadian-aligned daily schedule from brain science |
| `get_cognitive_health_report` | Full report: health trends + brain analysis + correlations + schedule |

---

## Supported Activities for Brain Prediction

| Activity | Key Brain Regions | Primary Benefits |
| --- | --- | --- |
| `exercise` | Motor Cortex, Hippocampus, Prefrontal Cortex | Neuroplasticity, BDNF release, memory |
| `meditation` | Prefrontal Cortex, Anterior Cingulate, Insula | Stress reduction, emotional regulation |
| `sleep` | Hippocampus, Default Mode Network | Memory consolidation, glymphatic clearance |
| `reading` | Visual Cortex, Hippocampus, Temporal Lobe | Cognitive reserve, language |
| `music` | Auditory Cortex, Amygdala, Temporal Lobe | Mood, memory, motor coordination |
| `social` | Temporal Lobe, Amygdala, Insula | Oxytocin, emotional wellbeing |
| `work` | Prefrontal Cortex, Anterior Cingulate | Executive function, focus |
| `relaxation` | Default Mode Network, Posterior Cingulate | Recovery, stress relief |
| `nature` | Visual Cortex, Amygdala | Cortisol reduction, visual restoration |
| `learning` | Hippocampus, Prefrontal Cortex | Neuroplasticity, cognitive reserve |
| `creative` | Default Mode Network, Prefrontal Cortex | Problem-solving, dopamine |
| `mindfulness` | Insula, Anterior Cingulate | Interoception, present-moment awareness |

---

## Notion Dashboard

Running `python scripts/setup_notion.py` creates:

- **6 databases**: Health Metrics, Medications, Appointments, Health Goals, Symptom Log, Brain Analysis
- **Dashboard page** with embedded database views, brain science timing tables, and AI command reference

The Brain Analysis database is auto-populated when you ask Claude to run `get_cognitive_health_report` or `predict_brain_response`. Each session records:

| Field | Description |
| --- | --- |
| Cognitive Load | Estimated mental effort (0–10) |
| Stress Level | Stress indicator (0–10) |
| Emotional Valence | Positive/negative balance (−1 to +1) |
| Default Mode Network | DMN activation (0–1) |
| Executive Control | PFC/ACC network (0–1) |
| Salience Network | Insula/ACC alertness (0–1) |
| Insights | AI-generated observations |
| Recommendations | Personalised action items |
| TRIBEv2 Model Used | Whether real weights or simulation |

---

## Example Claude Conversations

```text
"Log my weight as 172, mood 7, slept 7.5 hours"
"What's my sleep trend for the past two weeks?"
"Add Vitamin D3, 2000 IU, once daily starting today"
"Schedule a checkup with Dr. Smith on April 15 at 10am"
"Set a goal to lose 10 pounds by June 1st, I'm at 172"
"Predict how my brain responds to 30 minutes of meditation"
"Analyse my brain-health correlations for this week"
"Generate an optimal daily schedule based on brain science"
"Give me a full cognitive health report for the past month"
```

---

## Web GUI

The system ships with a browser-based GUI that gives any user access to the full system — health logging, AI insights, and live TRIBEv2 brain visualizations — without installing a CLI.

### Start locally

```bash
source venv/bin/activate   # or venv_arm64/bin/activate
make serve-api
# Open: http://localhost:8000
```

### Configure in the browser

Click **⚙ Settings** and enter:

- Your Notion integration token
- AI provider (Claude / GPT-4o / Gemini) and its API key

Keys are held in `sessionStorage` only — never saved to disk or sent anywhere except your own server as HTTP headers.

### Token cost estimation

Before sending any request, the GUI shows a live token estimate and USD cost for your chosen AI provider:

- **Text** — counts as you type, updates in real time
- **Audio / Video** — estimated from file size (MB → transcription tokens)
- **Multimodal** — combined estimate
- **AI Insights** — per-DB row estimate (health records × days + meds + symptoms + system prompt)

After a response, actual token usage (input + output) and cost are shown beneath the results.

Provider rates used (as of 2026):

| Provider | Input | Output |
| --- | --- | --- |
| Claude Sonnet | $3 / MTok | $15 / MTok |
| GPT-4o | $5 / MTok | $15 / MTok |
| Gemini 2.0 Flash | $0.075 / MTok | $0.30 / MTok |

See [howto-use-web-gui.md](howto-use-web-gui.md) for the full guide including audio/video brain prediction and the visualizations gallery.

---

## Docker

```bash
cp .env.example .env   # fill in HUGGING_FACE_TOKEN if you have one
docker compose up
# Open: http://localhost:8000
```

Users provide their Notion + AI keys via the Settings panel in the browser — the container only needs `HUGGING_FACE_TOKEN` for real TRIBEv2 predictions (otherwise simulation mode is used).

```bash
make docker-up          # build + start (foreground)
make docker-up-detached # background
make docker-down        # stop
make docker-logs        # tail logs
```

---

## GitHub Codespaces

Run the full system in the cloud with zero local setup:

1. Open this repo on GitHub → **Code** → **Codespaces** → **Create codespace on main**
2. Wait for the container to finish building (~2 min)
3. In the terminal: `make serve-api`
4. Port 8000 is auto-forwarded — click **Open in Browser**

To enable real TRIBEv2 predictions in Codespaces, add `HUGGING_FACE_TOKEN` as a [Codespaces secret](https://github.com/settings/codespaces) — it will be injected automatically.

---

## Brain Visualizations (CLI)

```bash
# Interactive 3D HTML + ROI heatmap (default)
notion-health brain visualize --activity meditation --duration 20

# All types
notion-health brain visualize --activity exercise --duration 30 \
    --type interactive --type static --type heatmap

# Heatmap only, don't open browser
notion-health brain visualize --activity sleep --duration 480 \
    --type heatmap --no-open
```

Generated files land in `visualizations/`. The interactive HTML opens in your default browser automatically (pass `--no-open` to suppress).

---

## CLI Usage

```bash
source venv/bin/activate  # or venv_arm64/bin/activate on Apple Silicon

notion-health --help
notion-health log metric --weight 165 --mood 8 --sleep 7.5
notion-health view summary --days 7
notion-health view medications
notion-health view appointments --days 30
notion-health insights --category overall --period week
notion-health brain predict --activity exercise --duration 30
notion-health brain visualize --activity exercise --duration 30
notion-health brain analyze --days 7
notion-health brain schedule
notion-health interactive
```

---

## Testing

```bash
source venv_arm64/bin/activate   # arm64 venv required for TRIBEv2 tests
pytest tests/ -v
pytest tests/ -v --cov=notion_health_ai
```

All 44 tests pass on `venv_arm64`. The x86_64 `venv` also passes all 44 tests (TRIBEv2 tests use the simulation fallback).

---

## Project Structure

```text
notion-health-ai/
├── src/notion_health_ai/
│   ├── server.py            # MCP server — 16 tools
│   ├── health_manager.py    # Core health logic + TRIBEv2 bridge
│   ├── tribe_integration.py # TRIBEv2 wrapper + HCP ROI mapping
│   ├── ai_insights.py       # Claude-powered health insights
│   ├── models.py            # Pydantic data models
│   ├── notion_client.py     # Notion API wrapper
│   ├── cli.py               # Command-line interface
│   └── utils.py             # Utilities
├── config/
│   ├── database_schema.json # Notion database property definitions
│   └── gui_template.json    # Dashboard layout template
├── scripts/
│   └── setup_notion.py      # One-time Notion workspace setup
├── tests/                   # pytest test suite (44 tests)
├── venv/                    # x86_64 venv (all features except TRIBEv2)
├── venv_arm64/              # arm64 venv (all features including TRIBEv2)
├── cache/tribev2/           # TRIBEv2 feature cache (gitignored)
├── .env                     # Your API keys (gitignored)
├── .env.example             # Template
├── requirements.txt
├── pyproject.toml
└── howto-use-notion-health-ai.md
```

---

## Troubleshooting

### Tools Don't Appear in Claude Desktop

- Fully quit with Cmd+Q (not just close the window) then reopen
- Confirm the path in `claude_desktop_config.json` is absolute — no `~`, no relative paths
- Test the binary directly: `/your/path/venv_arm64/bin/notion-health-mcp`
- Check logs: `~/Library/Logs/Claude/mcp-server-notion-health-ai.log`

### No Module Named notion_health_ai

Run `pip install -e .` from the project root with venv active.

### torch Version Error / TRIBEv2 Won't Install

- On Apple Silicon, use `/opt/homebrew/bin/python3.11` for the venv — not the system Python or any x86_64 Python
- Confirm: `python -c "import platform; print(platform.machine())"` should print `arm64`

### Gated Repo Error from HuggingFace

- Request access to `meta-llama/Llama-3.2-3B` at [huggingface.co/meta-llama/Llama-3.2-3B](https://huggingface.co/meta-llama/Llama-3.2-3B)
- Add your HuggingFace token to `.env` as `HUGGING_FACE_TOKEN=hf_xxx`

### TRIBEv2 Word Alignment Error

The text must be at least ~15 words — short inputs fail the Whisper alignment step. The system catches this and falls back to simulation automatically.

### Brain Analysis Database Missing

Re-run `python scripts/setup_notion.py` — it creates any missing databases. Or manually add the Brain Analysis database using the schema in `config/database_schema.json`.

### AI Insights Show "No insights generated"

This happens when:

1. **No health data** is logged in Notion for the selected period — log at least a few days of data first
2. **AI API key** is missing or invalid — check Settings in the Web GUI or `ANTHROPIC_API_KEY` in `.env`
3. **Notion databases not connected** — re-run `python scripts/setup_notion.py`

Without an AI key, the engine runs in simulation mode and still returns rule-based insight cards from your actual data.

### Notion 400 Error During Setup

Your Notion integration must be shared with the parent page (Page → Share → Connections).

---

## Features

- Health metrics tracking (weight, sleep, mood, exercise, vitals)
- Medication management with schedules
- Appointment scheduling and reminders
- Health goal setting and progress tracking
- Symptom logging with AI analysis
- Claude-powered health insights
- TRIBEv2 real fMRI brain predictions (with HuggingFace token + arm64/GPU)
- Brain-health correlation analysis
- Optimal daily schedule generation
- Cognitive health reporting
- Full Notion dashboard with 6 embedded databases
- 16 MCP tools for Claude Desktop
- 44-test pytest suite

---

## Privacy

- All health data lives in **your own Notion workspace** — nothing is stored by this project
- The Anthropic API receives health summaries only when you explicitly request AI insights
- API keys are stored in your local `.env` only — never committed to version control
- TRIBEv2 inference runs entirely locally (model weights on your device); no data is sent to Meta

---

## License

MIT License

TRIBEv2 model weights: [CC-BY-NC-4.0](https://creativecommons.org/licenses/by-nc/4.0/) — non-commercial use only.

---

## Acknowledgments

- [Meta AI / Facebook Research](https://ai.meta.com) for TRIBEv2 (`facebook/tribev2`)
- [Notion](https://notion.so) for the API and MCP support
- [Anthropic](https://anthropic.com) for Claude and the Claude API
</content>
