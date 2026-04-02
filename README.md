<!--
SPDX-License-Identifier: MIT
Copyright (c) 2026 Rohan R
Author: Rohan R
-->

# NotionHealth AI

[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/fastapi-local%20api-009688)](https://fastapi.tiangolo.com/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Notion](https://img.shields.io/badge/notion-health%20data-black)](https://www.notion.so/)

NotionHealth AI is a local-first health tracker for Notion. You run it on your own machine, connect it to Claude through MCP if you want the assistant workflow, and use the same local setup for the browser UI.

The project covers day-to-day health logging, summaries, goals, symptoms, appointments, AI-written insights, and optional TRIBEv2 brain analysis. If TRIBEv2 is not installed, the rest of the app still works and the brain features fall back to simulation mode.

https://github.com/user-attachments/assets/0add8628-bd76-4b43-afd9-b8e483e4967e

> ⚠️ Consult a qualified health professional for medical advice, diagnosis, or treatment.

## What you need

| Item | Notes |
| --- | --- |
| Python 3.10+ | Python 3.11 is the safest default |
| Notion account | Needed for the databases |
| Notion API key | Required |
| Anthropic API key | Required for AI-backed insights |
| Claude Desktop | Optional, only for the MCP workflow |
| Hugging Face token | Optional, only for the real TRIBEv2 path |

## Install

```bash
git clone https://github.com/rotsl/notion-health-ai.git
cd notion-health-ai
python3 -m venv venv
source venv/bin/activate
pip install -e .
cp .env.example .env
```

On Apple Silicon, use a native arm64 Python if you want TRIBEv2:

```bash
/opt/homebrew/bin/python3.11 -m venv venv_arm64
source venv_arm64/bin/activate
pip install -e .
```

Set at least these values in `.env`:

```text
NOTION_API_KEY=your_notion_token_here
ANTHROPIC_API_KEY=your_anthropic_key_here
```

Add this if you want the real TRIBEv2 model path:

```text
HUGGING_FACE_TOKEN=hf_your_token_here
```

## Create the Notion databases

Share a Notion page with your integration, then run:

```bash
python scripts/setup_notion.py
```

The script creates the databases, seeds sample data, builds the dashboard, and writes the generated database IDs back into `.env`.

## Run the local app

```bash
source venv/bin/activate
make serve-api
```

Open `http://127.0.0.1:8000`.

If you prefer the direct command:

```bash
uvicorn notion_health_ai.api:app --reload --host 127.0.0.1 --port 8000
```

To stop the local app:

```bash
Ctrl+C
```

Useful local checks:

```bash
curl http://127.0.0.1:8000/api/status
pytest tests/ -q
```

If port `8000` is still busy after stopping the app, find the process first:

```bash
lsof -nP -iTCP:8000 -sTCP:LISTEN
```

Then stop it cleanly:

```bash
kill <PID>
```

Only if the process is stuck, force it:

```bash
kill -9 <PID>
```

## Use the browser UI

The browser UI runs from the same local FastAPI server. Open the Settings panel and provide:

| Field | Required | Notes |
| --- | --- | --- |
| Notion token | Yes | Your Notion integration token |
| AI provider | Yes | Anthropic, OpenAI, or Gemini |
| AI API key | Yes | Key for the provider you chose |
| Hugging Face token | No | Only needed if the local server does not already have one |

The browser stores those values in session storage only.

The main tabs:

- `Brain Predict`: text, audio, video, and multimodal TRIBEv2 analysis
- `Health Log`: weight, sleep, mood, energy, steps, water, exercise, blood pressure, heart rate, and notes
- `AI Insights`: generated summaries from your Notion health data
- `Visualizations`: generated brain outputs saved under `visualizations/`

The Gallery now has a `Clear Saved Outputs` button. It deletes generated visualization files from the local `visualizations/` directory and keeps the placeholder file.

## TRIBEv2 behavior

TRIBEv2 is optional. If you want the real model path:

1. Request access to `meta-llama/Llama-3.2-3B` on Hugging Face
2. Create a read token
3. Put that token in `HUGGING_FACE_TOKEN`
4. Install the heavier local dependencies in the same environment you use to run the app

Without those pieces, brain features fall back to simulation mode.

For text-based brain analysis in the browser UI, the app asks for a simulation duration before it runs. If your connected Notion data already contains useful timing data, the app can offer those values directly. Otherwise it offers a small set of realistic presets and bounds any custom value to a sensible range.

Visualization timelines now follow the selected simulation duration, and generated titles are written to describe the actual simulation instead of a generic label.

## Use with Claude Desktop

If you want the MCP workflow, add the local server command to Claude Desktop:

```json
{
  "mcpServers": {
    "notion-health-ai": {
      "command": "/absolute/path/to/notion-health-ai/venv/bin/notion-health-mcp"
    }
  }
}
```

If you are using the arm64 environment, point Claude at `venv_arm64/bin/notion-health-mcp` instead.

Then fully quit and reopen Claude Desktop.

Example prompts for Claude:

```text
Log my weight as 172 pounds, mood 7, and 7.5 hours of sleep
```

```text
I ran for 40 minutes today and took 9,500 steps
```

```text
Create a goal to sleep 8 hours a night by the end of the month
```

```text
Log a headache at severity 6 with light sensitivity
```

```text
Predict my brain response to 30 minutes of meditation
```

```text
Give me a cognitive health report for this week
```

## Troubleshooting

| Problem | What to check |
| --- | --- |
| Claude shows no tools | Confirm the MCP command path and fully restart Claude Desktop |
| Notion calls fail | Make sure the integration has access to the right page and databases |
| Database IDs are missing | Run `python scripts/setup_notion.py` again |
| Brain predictions stay in simulation | Check `HUGGING_FACE_TOKEN` and TRIBEv2 installation |
| Duration options look wrong | Rewrite the activity text more clearly, or use a connected Notion timing value |
| Gallery is empty | Generate a visualization first |
| Apple Silicon torch issues | Use `venv_arm64` with native Python 3.11 |

## Project files

- [CONTRIBUTING.md](CONTRIBUTING.md)
- [SECURITY.md](SECURITY.md)
- [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md)
- [CHANGELOG.md](CHANGELOG.md)

## License

The code in this repo is MIT licensed. See [LICENSE](LICENSE).

TRIBEv2 is separate from this repo. If you enable the real model path, you also need to follow the upstream Meta and Hugging Face terms for the TRIBEv2 model and the gated `meta-llama/Llama-3.2-3B` dependency.

At the time this README was updated, the upstream TRIBEv2 pages were not perfectly consistent about the exact model license label, so check the current upstream pages before redistribution or commercial use:

- TRIBEv2 model page: <https://huggingface.co/facebook/tribev2>
- TRIBEv2 source repo: <https://github.com/facebookresearch/tribev2>
- Llama 3.2 model page: <https://huggingface.co/meta-llama/Llama-3.2-3B>
