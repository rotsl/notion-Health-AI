# How to Use NotionHealth AI

NotionHealth AI lets you track your health — weight, sleep, mood, exercise, medications, appointments, goals, and symptoms — by talking to Claude naturally. Everything saves directly to your Notion workspace in real time.

Optionally powered by **Meta AI's TRIBEv2** brain-predictive foundation model, it also predicts real fMRI brain activation patterns from your health narratives and activity descriptions.

---

## What You Need

| Requirement | Where | Cost |
| --- | --- | --- |
| [Notion](https://notion.so) account | notion.so | Free |
| [Claude Desktop](https://claude.ai/download) | claude.ai/download | Free |
| Notion API key | [notion.so/my-integrations](https://www.notion.so/my-integrations) | Free |
| Anthropic API key | [console.anthropic.com](https://console.anthropic.com) | Pay-as-you-go |
| Python 3.10+ | [python.org](https://python.org) | Free |
| HuggingFace token *(TRIBEv2 only)* | [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) | Free — needs LLaMA 3.2 access |

> **Without a HuggingFace token:** The system uses simulation-based brain analysis — all 16 MCP tools work exactly the same. The token is only needed to run real TRIBEv2 model weights.

---

## One-Time Setup (~15 minutes)

### Step 1 — Install the project

#### Standard install (no TRIBEv2)

```bash
git clone https://github.com/rotsl/notion-health-ai.git
cd notion-health-ai
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
```

#### Apple Silicon (M1/M2/M3/M4) — for TRIBEv2 support

The standard `venv` on Apple Silicon uses Rosetta (x86_64) Python and can only install torch 2.2.2, which is too old for TRIBEv2. You need a **native arm64** Python:

```bash
# Check if Homebrew Python 3.11 is installed
/opt/homebrew/bin/python3.11 --version
# If not: brew install python@3.11

# Create native arm64 venv
/opt/homebrew/bin/python3.11 -m venv venv_arm64
source venv_arm64/bin/activate
pip install -r requirements.txt
pip install -e .
```

The arm64 venv supports **all features** including TRIBEv2. The standard venv works for everything except TRIBEv2 model loading.

### Step 2 — Get your Notion API key

1. Go to [notion.so/my-integrations](https://www.notion.so/my-integrations)
2. Click **+ New integration**, name it `NotionHealth AI`, choose your workspace, click **Submit**
3. Copy the token (starts with `secret_` or `ntn_`)

### Step 3 — Get your Anthropic API key

1. Go to [console.anthropic.com](https://console.anthropic.com) → API Keys
2. Click **+ Create Key**, copy it (starts with `sk-ant-`)

### Step 4 — Configure your environment

```bash
cp .env.example .env
```

Open `.env` and fill in:

```text
NOTION_API_KEY=your_notion_token_here
ANTHROPIC_API_KEY=your_anthropic_key_here
```

Leave the `NOTION_*_DATABASE_ID` fields blank — the setup script fills them automatically.

#### To enable TRIBEv2 (optional)

1. Go to [huggingface.co/meta-llama/Llama-3.2-3B](https://huggingface.co/meta-llama/Llama-3.2-3B) and click **Request access** (usually approved within minutes — this is a gated Facebook Research / Meta model)
2. Create a token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
3. Add to `.env`:

```text
HUGGING_FACE_TOKEN=hf_your_token_here
```

### Step 5 — Create your Notion databases and dashboard

First, share a Notion page with your integration:

1. Open Notion and navigate to the page you want to use
2. Click **Share** (top right) → **Add people, emails, groups, or integrations**
3. Find your integration, click **Invite**

Then run:

```bash
python scripts/setup_notion.py
```

The script will:

- Prompt for any missing API keys and save them to `.env`
- Archive databases from any previous run (so you always start fresh)
- Ask you to select a parent page
- Create **6 databases**: Health Metrics, Medications, Appointments, Health Goals, Symptom Log, Brain Analysis
- Seed realistic sample data so the dashboard looks populated from day one
- Build a **dashboard page** with embedded database views, brain science timing tables, and AI command reference
- Write all database IDs back into your `.env` automatically

### Step 6 — Connect to Claude Desktop

Open `~/Library/Application Support/Claude/claude_desktop_config.json` (macOS) or `%APPDATA%\Claude\claude_desktop_config.json` (Windows) and add:

```json
{
  "mcpServers": {
    "notion-health-ai": {
      "command": "/absolute/path/to/notion-health-ai/venv/bin/notion-health-mcp"
    }
  }
}
```

**On Apple Silicon with TRIBEv2**, use the arm64 venv instead:

```json
{
  "mcpServers": {
    "notion-health-ai": {
      "command": "/absolute/path/to/notion-health-ai/venv_arm64/bin/notion-health-mcp"
    }
  }
}
```

To find your absolute path:

```bash
# Standard venv
echo "$(pwd)/venv/bin/notion-health-mcp"

# arm64 venv
echo "$(pwd)/venv_arm64/bin/notion-health-mcp"
```

**Fully quit and reopen Claude Desktop** (Cmd+Q on Mac — not just close the window).

### Step 7 — Install TRIBEv2 (optional, Apple Silicon/GPU only)

Skip this step if you don't need real brain predictions.

```bash
source venv_arm64/bin/activate

# Install exact versions required by TRIBEv2
pip install "torch==2.5.1" "numpy==2.2.6" torchvision torchaudio

# Install whisperx (used for TTS→Whisper transcription in the text pipeline)
pip install whisperx

# Install FFmpeg (required by whisperx audio decoder)
brew install ffmpeg  # macOS; or: sudo apt install ffmpeg

# Install TRIBEv2 package
git clone https://github.com/facebookresearch/tribev2 /tmp/tribev2
pip install -e /tmp/tribev2

# Reinstall the project to pick up all deps
pip install -e .
```

**First run** downloads (all cached after first use):
- TRIBEv2 weights (`best.ckpt`): ~676 MB
- LLaMA 3.2 3B (internal text encoder): ~1.65 GB
- Whisper model + spaCy `en_core_web_lg`: ~400 MB

### Step 8 — Verify the connection

Open a new Claude Desktop conversation and ask:

> "What health tools do you have available?"

You should see 16 tools. If nothing appears, see Troubleshooting below.

---

## The Notion Dashboard

After running `setup_notion.py`, open Notion and find your dashboard page. It contains:

### Health Metrics
Daily entries — weight (lbs), sleep hours & quality (1–10), exercise type & minutes, steps, mood (1–10), energy (1–10), water glasses, blood pressure, heart rate.

### Medications
Medication name, dosage, frequency (dropdown), start/end dates, prescribing doctor, purpose. Active checkbox for current medications.

### Appointments
Doctor name, date, time, facility, appointment type (checkup/dental/mental health/etc.), status (scheduled → confirmed → completed).

### Health Goals
Goal type, title, target value, current value, unit, target date, status (not started / in progress / on track / behind / achieved).

### Symptom Log
Symptom name, severity (1–10), date/time, body part, duration (minutes), triggers, notes.

### Brain Analysis
Auto-populated by Claude when you use TRIBEv2 tools. Shows per-session cognitive metrics:

| Column | What it means |
| --- | --- |
| Cognitive Load | Estimated mental effort (0–10) |
| Stress Level | Stress indicator (0–10) |
| Emotional Valence | Positive/negative mood balance (−1 to +1) |
| Default Mode Network | Self-referential thinking activation |
| Executive Control | Prefrontal/ACC decision-making network |
| Salience Network | Insula/ACC alertness level |
| Insights | AI-generated observations |
| Recommendations | Personalised action steps |
| TRIBEv2 Model Used | Checkbox — real weights or simulation |

---

## Daily Use — Just Talk to Claude

### Logging your health

```text
Log my weight as 172 pounds, mood 7, and 7.5 hours of sleep
```

```text
I ran for 40 minutes today, steps were 9500, energy felt like an 8
```

```text
Log today: weight 168, slept 8 hours quality 9, drank 8 glasses of water
```

```text
My blood pressure was 118/76 and resting heart rate 62 this morning
```

### Checking your stats

```text
How did I sleep this week?
```

```text
What's my weight trend for the past month?
```

```text
Give me a health summary for the last 7 days
```

### Medications

```text
Add vitamin D3, 2000 IU, once daily starting today
```

```text
What medications do I need to take today?
```

### Appointments

```text
Schedule a checkup with Dr. Smith on April 15 at 10am at City Medical Center
```

```text
What appointments do I have coming up?
```

### Health goals

```text
Set a goal to reach 165 pounds by June 1st. I'm currently at 172.
```

```text
How am I doing on my weight goal?
```

### Symptoms

```text
Log a headache, severity 6 out of 10, started about an hour ago
```

```text
Analyze my symptoms from the past month — are there any patterns?
```

### AI insights

```text
Give me AI health insights for this week
```

```text
Are there correlations between my mood and how much I exercise?
```

> **Note:** AI insights require health data to be logged in Notion for the selected period. If no entries exist yet, seed some data first (use the Health Log tab in the Web GUI, or log via Claude Desktop). Without an Anthropic API key, insights run in **simulation mode** — the cards are still generated from your actual data using heuristics rather than an LLM.

---

## Brain Analysis with TRIBEv2

TRIBEv2 (`facebook/tribev2`) is Meta AI's brain-predictive foundation model, trained on fMRI data from real brain scans. Given text describing an activity or narrative, it predicts activation across ~20,484 cortical surface vertices (fsaverage5 mesh).

The system maps these predictions to 9 functional brain regions using the HCP MMP1.0 parcellation (181 ROIs).

### What you can ask Claude

```text
Predict my brain response to 30 minutes of meditation
```

```text
Analyse my brain-health correlations for the past week
```

```text
What's the optimal daily schedule based on brain science?
```

```text
Give me a full cognitive health report for the past month
```

### The 4 TRIBEv2 tools

| Tool | What it does |
| --- | --- |
| `predict_brain_response` | Predict which brain regions activate for a specific activity and duration |
| `analyze_brain_health_correlation` | Find correlations between your sleep/exercise/mood and brain network activations |
| `get_optimal_schedule` | Circadian-aligned daily schedule from brain science |
| `get_cognitive_health_report` | Full report: health trends + real brain predictions + correlations + personalised schedule |

### Supported activities

| Activity | Key brain benefits |
| --- | --- |
| Exercise | Motor cortex + hippocampus; BDNF release; neuroplasticity |
| Meditation | Prefrontal cortex + anterior cingulate; amygdala down-regulation |
| Sleep | Hippocampal memory consolidation; glymphatic brain cleaning |
| Reading | Hippocampus + visual cortex; cognitive reserve building |
| Music | Auditory cortex + temporal lobe; mood enhancement |
| Social | Temporal lobe + insula; oxytocin; emotional regulation |
| Nature | Visual cortex restoration; amygdala calming; cortisol reduction |
| Creative | Default mode network; problem-solving; dopamine |
| Learning | Hippocampus + prefrontal cortex; neuroplasticity |
| Mindfulness | Insula + anterior cingulate; present-moment awareness |
| Work | Prefrontal cortex + anterior cingulate; executive function |
| Relaxation | Default mode network; recovery; stress relief |

### Optimal activity timing

| Time | Activity | Brain science reason |
| --- | --- | --- |
| 6:00–6:30 AM | Meditation | Theta wave state enhances prefrontal training |
| 6:30–7:30 AM | Exercise | Peak cortisol optimises BDNF release |
| 9:00 AM–12:00 PM | Deep work / Learning | Peak alertness and working memory |
| 12:00–1:00 PM | Light walk in nature | Visual cortex restoration, amygdala calming |
| 2:00–4:00 PM | Creative work | Moderate arousal optimal for divergent thinking |
| 4:00–6:00 PM | Social activities | Natural social energy peak |
| 8:30–9:00 PM | Wind-down / Reading | Melatonin production initiation |
| 10:00 PM–6:00 AM | Sleep (8 hours) | Memory consolidation, glymphatic clearance |

### About the gated model

TRIBEv2's text prediction pipeline uses **LLaMA 3.2 3B** (Meta) as an internal text encoder. This is a gated model on HuggingFace — you must request access at [huggingface.co/meta-llama/Llama-3.2-3B](https://huggingface.co/meta-llama/Llama-3.2-3B) before it will download. Access is typically approved within minutes.

TRIBEv2 weights are licensed **CC-BY-NC-4.0** (non-commercial use only).

---

## What Gets Stored in Notion

| Database | What it tracks |
| --- | --- |
| **Health Metrics** | Daily weight, sleep hours & quality, exercise type & minutes, steps, mood, energy, water glasses, blood pressure, heart rate |
| **Medications** | Name, dosage, frequency, start/end dates, prescribing doctor, purpose |
| **Appointments** | Doctor, date, time, facility, type, reason, status |
| **Health Goals** | Goal type, target value, current progress, deadline, status |
| **Symptom Log** | Symptom name, severity, body part, duration, triggers, notes |
| **Brain Analysis** | TRIBEv2 session: cognitive load, stress, emotional valence, network activations, insights, recommendations |

---

## Rating Scales

### Mood & Energy (1–10)

| Score | Meaning |
| --- | --- |
| 1–2 | Very low / exhausted |
| 3–4 | Below average |
| 5 | Average |
| 6–7 | Good |
| 8–9 | Great |
| 10 | Excellent / peak |

### Symptom Severity (1–10)

| Score | Meaning |
| --- | --- |
| 1–2 | Barely noticeable |
| 3–4 | Mild |
| 5–6 | Moderate |
| 7–8 | Severe |
| 9–10 | Unbearable |

### Sleep Quality (1–10)

How rested you felt, not just hours slept.

---

## Command Line (Optional)

```bash
source venv/bin/activate         # or venv_arm64/bin/activate

notion-health log metric --weight 165 --mood 8 --sleep 7.5
notion-health view summary --days 7
notion-health view medications
notion-health view appointments --days 30
notion-health insights --category overall --period week
notion-health brain predict --activity exercise --duration 30
notion-health brain analyze --days 7
notion-health brain schedule
notion-health interactive
```

---

## Troubleshooting

### Tools don't appear in Claude Desktop

- Fully quit Claude Desktop (Cmd+Q on Mac) and reopen — don't just close the window
- Confirm the path in `claude_desktop_config.json` is absolute — no `~`, no relative paths
- Test the binary directly in terminal: `/your/path/venv_arm64/bin/notion-health-mcp`
- Check logs: `~/Library/Logs/Claude/mcp-server-notion-health-ai.log`

### "Database not found" or tool calls failing

- All `NOTION_*_DATABASE_ID` entries in `.env` must be filled in
- The Notion integration must be added to the parent page (Page → `...` → Connections)

### "No module named src" or import errors

- Run `pip install -e .` from the project root with the venv active
- The MCP binary path must point to the venv that has the package installed

### API errors / "No insights generated"

- Check `ANTHROPIC_API_KEY` in `.env` is valid and has credits
- Confirm health data is logged in the Health Metrics database for the selected period — insights cannot be generated from an empty database
- If using the Web GUI: check the browser console or server log (`make serve-api`) for the exact error returned by the AI provider
- Insights fall back gracefully to simulation mode — all other tools still work

### TRIBEv2: "Gated repo" or 401 error

- You must request access to `meta-llama/Llama-3.2-3B` at [huggingface.co/meta-llama/Llama-3.2-3B](https://huggingface.co/meta-llama/Llama-3.2-3B)
- Add `HUGGING_FACE_TOKEN=hf_xxx` to `.env` and ensure it has at least Read scope

### TRIBEv2: torch version error / "TRIBEv2 not available"

- On Apple Silicon you **must** use `/opt/homebrew/bin/python3.11` for the venv
- Confirm arch: `python -c "import platform; print(platform.machine())"` → must be `arm64`
- Install exact versions: `pip install "torch==2.5.1" "numpy==2.2.6"`

### TRIBEv2: word alignment error / prediction falls back to simulation

- The health narrative text must be at least ~15 words for Whisper alignment to succeed
- The system catches this and falls back to simulation automatically — no action needed

### TRIBEv2: slow first run

- First run downloads ~2.7 GB total (TRIBEv2 weights + LLaMA 3.2 3B + Whisper). All cached after that.
- Subsequent predictions for the same text are instant (cached feature embeddings)

### Brain Analysis database missing

- Re-run `python scripts/setup_notion.py` — it creates any missing databases
- Or manually create it in Notion using the schema in `config/database_schema.json`

---

## Web GUI

You don't need to use Claude Desktop. The system also ships a browser-based GUI that gives any user access to the full system without installing a CLI.

### Start the web server

```bash
source venv/bin/activate   # or venv_arm64/bin/activate on Apple Silicon
make serve-api
# Open: http://localhost:8000
```

Or with Docker (no Python setup needed):

```bash
cp .env.example .env   # add HUGGING_FACE_TOKEN if you have one
docker compose up
# Open: http://localhost:8000
```

Or open in GitHub Codespaces — zero local setup, runs in the cloud. See [howto-use-web-gui.md](howto-use-web-gui.md).

### What the web GUI can do

- **Brain Predict tab** — enter text, upload audio, or upload video → get live fMRI activation predictions with interactive 3D brain visualizations, heatmaps, and animated GIFs
- **Health Log tab** — fill in the health form (weight, sleep, mood, steps, vitals) → saves to your Notion database
- **AI Insights tab** — AI-generated health summary cards (choose Claude, GPT-4o, or Gemini)
- **Gallery tab** — browse all previously generated visualizations

Users enter their own Notion token and AI API key in the **⚙ Settings** panel — keys stay in browser sessionStorage and are never saved server-side.

---

## Privacy

- All health data lives in **your own Notion workspace** — nothing is stored by this project
- The Anthropic API receives anonymised health summaries only when you request AI insights
- API keys are stored only in your local `.env` (CLI) or browser sessionStorage (web GUI) — never committed to version control
- TRIBEv2 inference runs entirely on your own device (locally) — no data is sent to Meta
- LLaMA 3.2 3B and TRIBEv2 weights are downloaded once to `~/.cache/huggingface/` and run locally

---

*NotionHealth AI v2.0 — MIT License*
*[github.com/rotsl/notion-health-ai](https://github.com/rotsl/notion-health-ai)*
</content>
