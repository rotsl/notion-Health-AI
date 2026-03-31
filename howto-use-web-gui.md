# How to Use the NotionHealth AI Web GUI

The web GUI is a single-page app served by the FastAPI backend. It lets any user run the full system — health logging, AI insights, and live TRIBEv2 brain visualizations — from a browser, without installing a CLI.

---

## 1. Start the Server

### Locally (Python venv)

```bash
# From the repo root
source venv/bin/activate          # or venv_arm64/bin/activate on Apple Silicon
uvicorn notion_health_ai.api:app --reload --port 8000
# Open: http://localhost:8000
```

Or with Make:

```bash
make serve-api
```

### Docker

```bash
cp .env.example .env              # fill in keys
docker compose up                 # http://localhost:8000
```

### GitHub Codespaces

1. Open this repo on GitHub → **Code** → **Codespaces** → **Create codespace on main**
2. The container installs deps automatically
3. Run `make serve-api` in the terminal
4. Port 8000 is auto-forwarded; click **Open in Browser**

---

## 2. Configure API Keys (Settings Panel)

Click the **⚙ Settings** button in the top-right corner. Three fields appear:

| Field | Required? | Where to get it |
| --- | --- | --- |
| **Notion Token** | Yes | [notion.so/my-integrations](https://www.notion.so/my-integrations) — create an integration, copy the secret |
| **AI Provider** | Yes | Pick Claude, GPT-4o, or Gemini from the dropdown |
| **AI API Key** | Yes | [console.anthropic.com](https://console.anthropic.com), [platform.openai.com](https://platform.openai.com), or [aistudio.google.com](https://aistudio.google.com) |
| **HuggingFace Token** | No | Only needed if the server doesn't have `HUGGING_FACE_TOKEN` set — for personal self-hosted deploys |

Keys are stored in **browser sessionStorage only** — they are never logged, saved to disk, or sent to any third party. They are transmitted as HTTP headers on each request to your own server.

Click **Save & Close**. The three status dots in the header turn green when connections succeed.

---

## 3. Brain Prediction Tab

### Text Input

1. Select the **Brain Predict** tab
2. Choose **Text** mode
3. Type a description of an activity, e.g.:
   > *I spent 30 minutes meditating this morning, focusing on slow breathing and clearing my mind*
4. Check the visualization types you want: **Interactive 3D**, **Static PNG**, **Animated GIF**, **ROI Heatmap**
5. Click **Predict Brain Response**

**What happens:**

- The text is sent to `POST /api/brain/text`
- TRIBEv2 encodes the text using LLaMA 3.2 3B and predicts fMRI activation across 20 484 cortical vertices
- The result panel shows: model badge (Real / Simulation), wellness score, activation table per brain region, and the selected visualizations

### Audio Input

1. Select **Audio** mode in the Brain Predict tab
2. Drag-and-drop or click to upload an `.mp3`, `.wav`, or `.m4a` file
3. Click **Predict Brain Response**

The audio is sent as `multipart/form-data` to `POST /api/brain/audio`. TRIBEv2 uses Wav2Vec-BERT to encode the audio.

### Video Input

1. Select **Video** mode
2. Upload an `.mp4` or `.mov` file (keep it short — under 60 s for faster results)
3. Click **Predict Brain Response**

Sent to `POST /api/brain/video`. TRIBEv2 uses V-JEPA2 for video encoding.

### Multimodal

Upload any combination of text + audio + video in the **Multimodal** mode. All modalities are fused by TRIBEv2 before prediction.

### Simulation Mode

If the server was started without a `HUGGING_FACE_TOKEN` (or the TRIBEv2 weights have not been downloaded), predictions run in **simulation mode** — all 16 brain region tools still return values, but they are based on cognitive science heuristics rather than the real neural network. The result badge shows **[Simulation]** in this case.

---

## 4. Viewing Live Visualizations

### Interactive 3D Brain (HTML)

When the **Interactive 3D** type is selected, the result panel embeds the Plotly brain mesh in an `<iframe>`. You can:

- Rotate, zoom, and pan with mouse/touch
- Hover over vertices to see activation values
- Use the Plotly toolbar (camera icon) to save a PNG

### ROI Heatmap (PNG)

The heatmap shows activation for 9 functional regions (Visual, Auditory, Motor, etc.) as a colour-coded bar chart. It appears inline in the result panel and links to the full-size PNG.

### Static PNG / GIF

If selected, links to these files appear below the heatmap. Click to open in a new tab.

### Visualizations Gallery Tab

All generated files are browseable in the **Gallery** tab — sorted by date, with type badges (Interactive / Heatmap / Static / GIF). Click any card to open the file.

---

## 5. Health Log Tab

Fill in the form fields (date, weight, sleep hours, mood 1–10, energy 1–10, steps, water glasses, exercise, blood pressure, heart rate, notes) and click **Log Health Data**.

The entry is written to your Notion **Health Metrics** database. The **Load Summary** button pulls a configurable number of recent days and displays an AI-generated summary card.

---

## 6. AI Insights Tab

Select a time period (7 / 14 / 30 / 90 days) and a category (Overall, Exercise, Sleep, Mood, Nutrition, Brain Health). Click **Generate Insights**.

The AI provider you chose in Settings analyses your Notion data and returns insight cards (patterns, recommendations, alerts).

### Requirements for AI Insights

- **Health data must exist** in your Notion Health Metrics database for the selected time period. If no records are logged yet, the system has nothing to analyze.
- **An AI API key** must be configured in Settings. Without a key, the engine runs in **simulation mode** — it still returns insight cards based on whatever data is available, but they are rule-based rather than AI-generated.

### "No insights generated"

This message appears when:

1. **No health data** is logged in Notion for the selected period — log at least a few days of data first (use the Health Log tab or Claude Desktop).
2. **AI API call failed** — check that your API key is valid and the provider is reachable. The server log (`make serve-api`) will show the exact error.
3. **Notion databases not set up** — run `python scripts/setup_notion.py` to create the required databases and connect them to the integration.

> **Tip:** Simulation mode insight cards are shown with a `[Simulation]` badge. They are based on the actual data in your Notion database but generated by heuristics rather than an LLM. They are fully informative and useful even without an AI key.

---

## 7. Running for Others (Self-Hosting)

### Environment variables the server needs

```bash
HUGGING_FACE_TOKEN=hf_...   # optional — enables real TRIBEv2 predictions
```

All other keys (Notion, AI) are provided **per request** by the user via the Settings panel — the server never needs them in its environment.

### GitHub Repo Secrets (for Actions / Codespaces)

Set `HUGGING_FACE_TOKEN` as a repository secret in **Settings → Secrets and variables → Actions**. The CI workflow and Codespaces devcontainer will pick it up automatically.

### Public deployment (e.g. Railway, Fly.io, Render)

```bash
# Set env var on the platform
HUGGING_FACE_TOKEN=hf_...

# Deploy
railway up   # or fly deploy, etc.
```

Users visit the URL, enter their own Notion + AI keys in Settings, and the full system works.

---

## 8. Token Cost Estimation

Before every request the GUI shows a live estimate of how many tokens will be consumed and what it will cost with your chosen provider.

### Where estimates appear

| Location | When shown |
| --- | --- |
| Below the text textarea | Updates in real time as you type |
| Below audio/video dropzone | Updates after file is selected, based on file size |
| Next to Predict button | Summary badge showing total input tokens |
| AI Insights tab | Per-DB breakdown (health records + meds + symptoms + system prompt) |
| Results panel (after response) | **Actual** tokens used (input + output) and exact cost |

### How estimates are calculated

- **Text** — `len(text) / 3.8` characters per token (standard English approximation)
- **Audio** — `file_MB × 180 + 200` tokens (≈ 1 MB MP3 = 1 min speech = ~180 transcription tokens)
- **Video** — `(file_MB / 10) × 180 + 500` tokens (audio track transcription + visual context overhead)
- **AI Insights** — `days × 120` (health rows) + `3 × 80` (meds) + `(days/3) × 100` (symptoms) + `600` (system prompt)

For text input with Claude selected, clicking **Predict** will call `/api/estimate-tokens` for an exact count using the Anthropic token-counting API (requires your AI key to be set).

### Provider rates (as of 2026)

| Provider | Input cost | Output cost |
| --- | --- | --- |
| Claude Sonnet | $3.00 / MTok | $15.00 / MTok |
| GPT-4o | $5.00 / MTok | $15.00 / MTok |
| Gemini 2.0 Flash | $0.075 / MTok | $0.30 / MTok |

Estimates are shown in USD. The badge colour indicates relative size:

- **Green** — under 500 tokens (~$0.002 with Claude)
- **Yellow** — 500–3 000 tokens
- **Red** — over 3 000 tokens

---

## 9. Troubleshooting

| Problem | Fix |
| --- | --- |
| Status dot red after saving keys | Check that the Notion integration has access to your databases (share each DB with the integration in Notion) |
| "TRIBEv2 not loaded" badge | Server needs `HUGGING_FACE_TOKEN`; predictions fall back to simulation |
| Audio/video upload fails | File too large (>100 MB) or unsupported codec; convert to mp3/mp4 first |
| Interactive 3D brain blank | Allow iframes in your browser; or open the linked HTML file directly |
| `RuntimeError: Ratio of unmatched words` | Text stimulus too short — use at least 15–20 words describing the activity |
| "No insights generated" | No health data logged for the selected period, or the AI API key is missing/invalid — see Section 6 |
