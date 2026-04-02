# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Rohan R
# Author: Rohan R

"""
NotionHealth AI — FastAPI HTTP server.

Exposes all 21 health + brain tools as REST endpoints so the GitHub Pages
web GUI (and any other HTTP client) can call them.

API keys are passed per-request in HTTP headers:
  X-Notion-Key   : Notion integration token
  X-AI-Provider  : anthropic | openai | gemini  (default: anthropic)
  X-AI-Key       : AI API key for the selected provider

HuggingFace token for TRIBEv2 is read from the environment (HUGGING_FACE_TOKEN)
because the model weights are loaded once at server startup.

Start with:
    uvicorn notion_health_ai.api:app --reload --port 8000
or:
    make serve-api
"""

from __future__ import annotations

import importlib.util
import os
import re
import shutil
import tempfile
from collections import Counter
from contextlib import asynccontextmanager
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, File, Header, HTTPException, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from loguru import logger
from pydantic import BaseModel, Field

# ── Load .env early so HUGGING_FACE_TOKEN is available before anything else ───

_HERE = Path(__file__).resolve()
PROJECT_ROOT = _HERE.parent.parent.parent

try:
    from dotenv import load_dotenv as _load_dotenv

    _load_dotenv(PROJECT_ROOT / ".env", override=False)
except ImportError:
    pass  # python-dotenv not installed — rely on env vars being set externally

# ── Project root & paths ──────────────────────────────────────────────────────
VIZ_DIR = PROJECT_ROOT / "visualizations"
VIZ_DIR.mkdir(parents=True, exist_ok=True)
STATIC_DIR = _HERE.parent / "static"
STATIC_DIR.mkdir(parents=True, exist_ok=True)

# ── Lazy-loaded global TRIBEv2 analyzer (model loaded once) ──────────────────

_tribe_analyzer = None

TEXT_DURATION_PROFILES: Dict[str, Dict[str, Any]] = {
    "exercise": {
        "label": "Exercise or movement",
        "min": 10,
        "max": 180,
        "presets": [20, 30, 45, 60, 90],
        "keywords": [
            "run",
            "running",
            "walk",
            "walking",
            "workout",
            "exercise",
            "gym",
            "cardio",
            "cycle",
            "cycling",
            "swim",
            "swimming",
            "lift",
            "lifting",
            "yoga",
        ],
    },
    "meditation": {
        "label": "Meditation or mindfulness",
        "min": 5,
        "max": 90,
        "presets": [10, 15, 20, 30, 45],
        "keywords": ["meditation", "mindfulness", "breathing", "breathwork", "journal"],
    },
    "sleep": {
        "label": "Sleep or rest",
        "min": 240,
        "max": 720,
        "presets": [360, 420, 480, 540, 600],
        "keywords": ["sleep", "slept", "nap", "rest", "bed", "bedtime", "asleep"],
    },
    "focus": {
        "label": "Focused work or learning",
        "min": 15,
        "max": 240,
        "presets": [25, 45, 60, 90, 120],
        "keywords": [
            "study",
            "reading",
            "read",
            "work",
            "coding",
            "code",
            "project",
            "learn",
            "learning",
            "lecture",
        ],
    },
    "relaxation": {
        "label": "Relaxation or leisure",
        "min": 10,
        "max": 180,
        "presets": [15, 30, 45, 60, 90],
        "keywords": [
            "music",
            "social",
            "friends",
            "family",
            "relax",
            "relaxing",
            "nature",
            "creative",
            "painting",
            "drawing",
        ],
    },
}
DEFAULT_TEXT_DURATION_PROFILE = "focus"


def _get_tribe_analyzer():
    """Return the singleton TribeHealthAnalyzer, initializing on first call."""
    global _tribe_analyzer
    if _tribe_analyzer is None:
        from notion_health_ai.tribe_integration import TribeHealthAnalyzer

        _tribe_analyzer = TribeHealthAnalyzer()
    return _tribe_analyzer


def _has_server_hf_token() -> bool:
    return bool(
        (os.getenv("HUGGING_FACE_TOKEN") or "").strip() or (os.getenv("HF_TOKEN") or "").strip()
    )


def _apply_hf_token(hf_token: Optional[str]) -> bool:
    """Apply request-scoped HF token to process env so TRIBEv2 can download/use weights."""
    token = (hf_token or "").strip()
    if not token:
        return _has_server_hf_token()

    # Set all three env var names used by huggingface_hub across versions.
    os.environ["HUGGING_FACE_TOKEN"] = token
    os.environ["HF_TOKEN"] = token
    os.environ["HUGGINGFACE_HUB_TOKEN"] = token
    return True


def _hf_login(token: str) -> None:
    """Call huggingface_hub.login() so the credential cache is populated before any download."""
    try:
        from huggingface_hub import login as _login

        _login(token=token, add_to_git_credential=False)
        logger.info("HuggingFace login successful")
    except Exception as e:
        logger.warning(f"HuggingFace login warning (non-fatal): {e}")


async def _ensure_tribe_ready(hf_token: Optional[str] = None):
    """Ensure analyzer is initialized and opportunistically load real model when token exists."""
    tribe = _get_tribe_analyzer()
    has_token = _apply_hf_token(hf_token)

    if not tribe.is_initialized:
        await tribe.initialize()

    # If previously initialized in simulation mode, try loading real weights once token is present.
    if has_token and not tribe.model_wrapper.is_loaded:
        # Authenticate with HuggingFace *before* load_model() attempts any download.
        token = (os.environ.get("HUGGING_FACE_TOKEN") or os.environ.get("HF_TOKEN") or "").strip()
        if token:
            _hf_login(token)
        cache_dir = str(PROJECT_ROOT / "cache" / "tribev2")
        loaded = await tribe.model_wrapper.load_model(cache_dir=cache_dir)
        if loaded:
            logger.info("TRIBEv2 model loaded after receiving HF token")

    return tribe


# ── Helper: build per-request managers ───────────────────────────────────────


def _build_health_manager(notion_key: str):
    from notion_health_ai.health_manager import HealthManager

    return HealthManager(notion_api_key=notion_key)


async def _build_tribe_health_context(
    notion_key: Optional[str],
    context_days: int,
) -> Optional[str]:
    if not notion_key or context_days <= 0:
        return None
    try:
        mgr = _build_health_manager(notion_key)
        return await mgr.build_tribe_text_context(context_days=context_days)
    except Exception as exc:
        logger.warning(f"Failed to build Notion health context for TRIBEv2 text path: {exc}")
        return None


def _extract_duration_minutes(text: str) -> Optional[int]:
    """Extract an explicit duration from free text, if present."""
    match = re.search(
        r"(?P<value>\d+(?:\.\d+)?)\s*(?P<unit>hours?|hrs?|hr|minutes?|mins?|min)\b",
        text,
        flags=re.IGNORECASE,
    )
    if not match:
        return None

    value = float(match.group("value"))
    unit = match.group("unit").lower()
    if unit.startswith("h"):
        return int(round(value * 60))
    return int(round(value))


def _infer_text_duration_profile(text: str) -> Dict[str, Any]:
    """Infer a sensible duration range from the user's narrative."""
    text_l = text.lower()
    selected_key = DEFAULT_TEXT_DURATION_PROFILE
    for key, profile in TEXT_DURATION_PROFILES.items():
        if any(keyword in text_l for keyword in profile["keywords"]):
            selected_key = key
            break

    profile = dict(TEXT_DURATION_PROFILES[selected_key])
    explicit_minutes = _extract_duration_minutes(text)
    suggested_minutes = profile["presets"][1]
    if explicit_minutes and profile["min"] <= explicit_minutes <= profile["max"]:
        suggested_minutes = explicit_minutes

    profile.update(
        {
            "key": selected_key,
            "suggested_minutes": suggested_minutes,
            "explicit_minutes": explicit_minutes,
            "prompt": (
                f"Choose a realistic {profile['label'].lower()} duration between "
                f"{profile['min']} and {profile['max']} minutes."
            ),
        }
    )
    return profile


async def _get_dataset_duration_options(
    notion_key: Optional[str],
    context_days: int,
) -> List[Dict[str, Any]]:
    """Collect duration options from connected Notion health data."""
    if not notion_key or context_days <= 0:
        return []

    mgr = _build_health_manager(notion_key)
    start_date = date.today() - timedelta(days=context_days)
    observations: List[Dict[str, Any]] = []

    try:
        if mgr.health_database_id:
            metrics = await mgr.get_health_metrics(start_date, date.today(), limit=100)
            metrics = sorted(metrics, key=lambda metric: metric.metric_date, reverse=True)
            for metric in metrics:
                if metric.exercise_minutes and 10 <= metric.exercise_minutes <= 180:
                    observations.append(
                        {
                            "minutes": int(metric.exercise_minutes),
                            "label": (
                                f"{int(metric.exercise_minutes)} min exercise"
                                f" ({metric.metric_date.isoformat()})"
                            ),
                            "source": "health_metric",
                            "date": metric.metric_date.isoformat(),
                        }
                    )
                if metric.sleep_hours:
                    sleep_minutes = int(round(metric.sleep_hours * 60))
                    if 240 <= sleep_minutes <= 720:
                        observations.append(
                            {
                                "minutes": sleep_minutes,
                                "label": (
                                    f"{sleep_minutes} min sleep "
                                    f"({metric.metric_date.isoformat()})"
                                ),
                                "source": "sleep_metric",
                                "date": metric.metric_date.isoformat(),
                            }
                        )
    except Exception as exc:
        logger.warning(f"Failed to load health-metric durations for TRIBEv2: {exc}")

    try:
        if mgr.symptoms_database_id:
            symptoms = await mgr.get_symptom_history(start_date, date.today())
            symptoms = sorted(symptoms, key=lambda symptom: symptom.symptom_date, reverse=True)
            for symptom in symptoms:
                if symptom.duration_minutes and 5 <= symptom.duration_minutes <= 240:
                    observations.append(
                        {
                            "minutes": int(symptom.duration_minutes),
                            "label": (
                                f"{int(symptom.duration_minutes)} min "
                                f"{symptom.symptom_name.lower()}"
                                f" ({symptom.symptom_date.isoformat()})"
                            ),
                            "source": "symptom_log",
                            "date": symptom.symptom_date.isoformat(),
                        }
                    )
    except Exception as exc:
        logger.warning(f"Failed to load symptom durations for TRIBEv2: {exc}")

    counts = Counter(item["minutes"] for item in observations)
    deduped: List[Dict[str, Any]] = []
    seen: set[int] = set()
    for item in observations:
        minutes = item["minutes"]
        if minutes in seen:
            continue
        seen.add(minutes)
        count = counts[minutes]
        if count > 1:
            item["label"] = f"{item['label']} • seen {count}x"
        deduped.append(item)

    return deduped[:8]


async def _build_duration_options(
    text: str,
    notion_key: Optional[str],
    context_days: int,
) -> Dict[str, Any]:
    """Build the duration choices shown to the frontend before prediction."""
    profile = _infer_text_duration_profile(text)
    dataset_options = await _get_dataset_duration_options(notion_key, context_days)
    return {
        "success": True,
        "activity_profile": {
            "activity_type": profile["key"],
            "label": profile["label"],
            "min_minutes": profile["min"],
            "max_minutes": profile["max"],
            "preset_minutes": profile["presets"],
            "suggested_minutes": profile["suggested_minutes"],
            "explicit_minutes": profile["explicit_minutes"],
            "prompt": profile["prompt"],
        },
        "dataset_options": dataset_options,
        "dataset_prompt": (
            "Choose one of your recent durations from connected Notion data."
            if dataset_options
            else None
        ),
    }


def _slugify_output_label(value: str) -> str:
    """Create readable filename stems for generated visualization assets."""
    slug = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
    return slug[:80] or "brain_simulation"


async def _resolve_duration_minutes(
    text: str,
    notion_key: Optional[str],
    context_days: int,
    duration_minutes: Optional[int],
    duration_source: str,
) -> Dict[str, Any]:
    """Validate and resolve the duration used for TRIBEv2 text simulation."""
    duration_options = await _build_duration_options(text, notion_key, context_days)
    profile = duration_options["activity_profile"]
    dataset_options = duration_options["dataset_options"]

    source = duration_source if duration_source in {"activity", "dataset"} else "activity"

    if source == "dataset":
        if not notion_key:
            raise HTTPException(
                status_code=400,
                detail="Connected Notion data is required to use dataset-based duration options.",
            )
        if not dataset_options:
            raise HTTPException(
                status_code=400,
                detail="No recent duration values were found in your connected Notion data.",
            )

        allowed_minutes = {item["minutes"] for item in dataset_options}
        selected_minutes = duration_minutes or dataset_options[0]["minutes"]
        if selected_minutes not in allowed_minutes:
            raise HTTPException(
                status_code=422,
                detail=(
                    "Dataset-backed duration must match one of the available connected-data values."
                ),
            )

        return {
            "minutes": selected_minutes,
            "source": "dataset",
            "activity_profile": profile,
            "dataset_options": dataset_options,
        }

    selected_minutes = duration_minutes or profile["suggested_minutes"]
    min_minutes = profile["min_minutes"]
    max_minutes = profile["max_minutes"]
    if not min_minutes <= selected_minutes <= max_minutes:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Please choose a realistic duration between {min_minutes} and "
                f"{max_minutes} minutes for this kind of activity."
            ),
        )

    return {
        "minutes": selected_minutes,
        "source": "activity",
        "activity_profile": profile,
        "dataset_options": dataset_options,
    }


def _is_missing_config(value: Optional[str]) -> bool:
    if value is None:
        return True
    return str(value).strip().lower() in {"", "none", "null"}


def _ensure_required_databases(mgr: Any, required: List[str]) -> None:
    env_key_map = {
        "health_database_id": "NOTION_HEALTH_DATABASE_ID",
        "medication_database_id": "NOTION_MEDICATION_DATABASE_ID",
        "appointment_database_id": "NOTION_APPOINTMENT_DATABASE_ID",
        "goals_database_id": "NOTION_GOALS_DATABASE_ID",
        "symptoms_database_id": "NOTION_SYMPTOMS_DATABASE_ID",
        "brain_database_id": "NOTION_BRAIN_DATABASE_ID",
    }
    missing = [env_key_map[k] for k in required if _is_missing_config(getattr(mgr, k, None))]
    if missing:
        raise HTTPException(
            status_code=400,
            detail=(
                "Missing Notion database configuration: "
                + ", ".join(missing)
                + ". Run 'python scripts/setup_notion.py'"
                + " or set these env vars in .env and restart the API."
            ),
        )


def _build_ai_engine(provider: str, ai_key: str):
    from notion_health_ai.ai_insights import AIInsightsEngine

    if provider == "openai":
        os.environ["OPENAI_API_KEY"] = ai_key
        return AIInsightsEngine(
            openai_api_key=ai_key,
            model_provider="openai",
            model_name="gpt-4o",
        )
    elif provider == "gemini":
        os.environ["GEMINI_API_KEY"] = ai_key
        return AIInsightsEngine(
            gemini_api_key=ai_key,
            model_provider="gemini",
            model_name="gemini-2.0-flash",
        )
    else:  # anthropic (default)
        os.environ["ANTHROPIC_API_KEY"] = ai_key
        return AIInsightsEngine(
            anthropic_api_key=ai_key,
            model_provider="anthropic",
        )


# ── FastAPI lifespan — kick off TRIBEv2 loading once token is in env ─────────


@asynccontextmanager
async def _lifespan(application: FastAPI):
    """Keep startup lightweight and defer TRIBEv2 initialization until needed."""
    if _has_server_hf_token():
        logger.info("HUGGING_FACE_TOKEN detected — TRIBEv2 load will be attempted on demand")
    else:
        logger.info("No HUGGING_FACE_TOKEN — TRIBEv2 will run in simulation mode")
    yield


# ── FastAPI app ───────────────────────────────────────────────────────────────

app = FastAPI(
    title="NotionHealth AI API",
    description="REST API for NotionHealth AI — health tracking + TRIBEv2 brain predictions",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=_lifespan,
)


@app.exception_handler(HTTPException)
async def http_exception_handler(_, exc: HTTPException):
    """Ensure API errors are always JSON with a consistent shape."""
    return JSONResponse(
        status_code=exc.status_code,
        content={"success": False, "error": exc.detail},
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(_, exc: RequestValidationError):
    """Return request validation failures as JSON payloads."""
    return JSONResponse(
        status_code=422,
        content={"success": False, "error": "Request validation failed", "details": exc.errors()},
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(_, exc: Exception):
    """Avoid plaintext 500 responses that break frontend JSON parsing."""
    logger.exception(f"Unhandled API exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"success": False, "error": str(exc) or "Internal Server Error"},
    )


app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:8000", "http://localhost:8000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve generated visualizations at /visualizations/<filename>
app.mount("/visualizations", StaticFiles(directory=str(VIZ_DIR)), name="visualizations")

# Serve web GUI static assets at /static/<filename>
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/", response_class=FileResponse, include_in_schema=False)
async def index():
    """Serve the web GUI."""
    html = STATIC_DIR / "index.html"
    if not html.exists():
        from fastapi.responses import HTMLResponse

        return HTMLResponse("<h2>Web GUI not found. Run: notion-health-api</h2>", status_code=404)
    return FileResponse(str(html))


# ── Dependency helpers ────────────────────────────────────────────────────────


def _require_notion_key(x_notion_key: Optional[str] = Header(None)) -> str:
    if not x_notion_key:
        raise HTTPException(status_code=401, detail="Missing X-Notion-Key header")
    return x_notion_key


def _get_ai_provider(x_ai_provider: Optional[str] = Header(None)) -> str:
    return (x_ai_provider or "anthropic").lower()


def _get_ai_key(x_ai_key: Optional[str] = Header(None)) -> Optional[str]:
    return x_ai_key


# ── Request / Response models ─────────────────────────────────────────────────


class HealthMetricRequest(BaseModel):
    date: Optional[str] = None
    weight: Optional[float] = None
    sleep_hours: Optional[float] = None
    sleep_quality: Optional[int] = None
    exercise_type: Optional[str] = None
    exercise_minutes: Optional[int] = None
    steps: Optional[int] = None
    mood: Optional[int] = None
    energy: Optional[int] = None
    water_glasses: Optional[int] = None
    blood_pressure_systolic: Optional[int] = None
    blood_pressure_diastolic: Optional[int] = None
    heart_rate: Optional[int] = None
    notes: Optional[str] = None


class MedicationRequest(BaseModel):
    name: str
    dosage: str
    frequency: str
    start_date: str
    times: Optional[List[str]] = None
    prescribing_doctor: Optional[str] = None
    purpose: Optional[str] = None
    notes: Optional[str] = None


class AppointmentRequest(BaseModel):
    date: str
    time: str
    doctor_name: str
    appointment_type: str
    facility: Optional[str] = None
    reason: Optional[str] = None
    notes: Optional[str] = None


class GoalRequest(BaseModel):
    goal_type: str
    title: str
    target_value: float
    unit: str
    target_date: str
    description: Optional[str] = None


class SymptomRequest(BaseModel):
    symptom_name: str
    severity: int
    date: Optional[str] = None
    time: Optional[str] = None
    body_part: Optional[str] = None
    duration_minutes: Optional[int] = None
    triggers: Optional[List[str]] = None
    notes: Optional[str] = None


class BrainTextRequest(BaseModel):
    text: str
    viz_types: Optional[List[str]] = None
    context_days: int = Field(7, ge=0, le=30)
    duration_minutes: Optional[int] = Field(None, ge=1, le=720)
    duration_source: str = "activity"


class BrainMultimodalRequest(BaseModel):
    text: Optional[str] = None
    viz_types: Optional[List[str]] = None
    context_days: int = Field(7, ge=0, le=30)
    duration_minutes: Optional[int] = Field(None, ge=1, le=720)
    duration_source: str = "activity"


class VizRequest(BaseModel):
    viz_type: str = "interactive"
    title: Optional[str] = None
    auto_open: bool = False  # server-side, don't auto-open browser


class InsightsRequest(BaseModel):
    category: str = "overall"
    period: str = "week"
    brain_context: Optional[Dict[str, Any]] = None


class GoalProgressUpdate(BaseModel):
    goal_id: str
    current_value: float


# ── Status endpoint ───────────────────────────────────────────────────────────


@app.get("/api/status")
async def status(
    x_hf_token: Optional[str] = Header(None),
):
    """Return server status and available features."""
    _apply_hf_token(x_hf_token)
    tribe = _tribe_analyzer
    return {
        "status": "ok",
        "version": "2.0.0",
        "features": {
            "tribev2_available": importlib.util.find_spec("tribev2") is not None,
            "torch_available": importlib.util.find_spec("torch") is not None,
            "model_loaded": tribe.model_wrapper.is_loaded if tribe else False,
            "hf_token_set": _has_server_hf_token(),
        },
        "viz_dir": str(VIZ_DIR),
        "ai_providers": ["anthropic", "openai", "gemini"],
    }


class TokenEstRequest(BaseModel):
    text: str
    provider: str = "anthropic"


@app.post("/api/estimate-tokens")
async def estimate_tokens(
    body: TokenEstRequest,
    x_ai_key: Optional[str] = Header(None),
):
    """Return accurate token count for text. Falls back to char/3.8 estimate if no API key."""
    provider = body.provider.lower()
    text = body.text
    if provider == "anthropic" and x_ai_key:
        try:
            import anthropic as _ant

            client = _ant.Anthropic(api_key=x_ai_key)
            resp = client.messages.count_tokens(
                model="claude-sonnet-4-6",
                messages=[{"role": "user", "content": text}],
            )
            return {"provider": provider, "input_tokens": resp.input_tokens, "estimated": False}
        except Exception:
            pass
    # Client-side fallback estimate
    n = max(1, round(len(text) / 3.8))
    return {"provider": provider, "input_tokens": n, "estimated": True}


@app.post("/api/init")
async def initialize_tribe(
    x_hf_token: Optional[str] = Header(None),
):
    """Initialize TRIBEv2 model (loads weights if available). Call once after startup."""
    tribe = await _ensure_tribe_ready(x_hf_token)
    return {
        "success": True,
        "model_loaded": tribe.model_wrapper.is_loaded,
    }


# ── Health Metrics ────────────────────────────────────────────────────────────


@app.post("/api/health/log")
async def log_health_metric(
    body: HealthMetricRequest,
    x_notion_key: Optional[str] = Header(None),
):
    if not x_notion_key:
        raise HTTPException(401, "Missing X-Notion-Key header")
    mgr = _build_health_manager(x_notion_key)
    from datetime import datetime

    dt = None
    if body.date:
        dt = datetime.strptime(body.date, "%Y-%m-%d").date()
    result = await mgr.log_health_metric(
        metric_date=dt,
        weight=body.weight,
        sleep_hours=body.sleep_hours,
        sleep_quality=body.sleep_quality,
        exercise_type=body.exercise_type,
        exercise_minutes=body.exercise_minutes,
        steps=body.steps,
        mood=body.mood,
        energy=body.energy,
        water_glasses=body.water_glasses,
        blood_pressure_systolic=body.blood_pressure_systolic,
        blood_pressure_diastolic=body.blood_pressure_diastolic,
        heart_rate=body.heart_rate,
        notes=body.notes,
    )
    return result.model_dump()


@app.get("/api/health/summary")
async def get_health_summary(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    period: str = "week",
    x_notion_key: Optional[str] = Header(None),
):
    if not x_notion_key:
        raise HTTPException(401, "Missing X-Notion-Key header")
    from datetime import date, datetime, timedelta

    mgr = _build_health_manager(x_notion_key)
    _ensure_required_databases(mgr, ["health_database_id"])
    today = date.today()
    if start_date and end_date:
        start = datetime.strptime(start_date, "%Y-%m-%d").date()
        end = datetime.strptime(end_date, "%Y-%m-%d").date()
    else:
        if period == "today":
            start = today
        elif period == "week":
            start = today - timedelta(days=7)
        elif period == "month":
            start = today - timedelta(days=30)
        else:
            start = today - timedelta(days=90)
        end = today

    if start > end:
        raise HTTPException(400, "start_date must be before or equal to end_date")

    summary = await mgr.get_health_summary(start, end)
    metrics = await mgr.get_health_metrics(start, end, limit=500)

    summary_payload = summary.model_dump()
    # Backward-compatible aliases expected by the current web GUI.
    summary_payload.update(
        {
            "average_weight": summary.avg_weight,
            "average_sleep_hours": summary.avg_sleep_hours,
            "average_mood": summary.avg_mood,
            "average_energy": summary.avg_energy,
            "entry_count": len(metrics),
        }
    )

    return {"success": True, "summary": summary_payload}


@app.get("/api/health/metrics")
async def get_health_metrics(
    days: int = 7,
    x_notion_key: Optional[str] = Header(None),
):
    if not x_notion_key:
        raise HTTPException(401, "Missing X-Notion-Key header")
    from datetime import date, timedelta

    mgr = _build_health_manager(x_notion_key)
    end = date.today()
    start = end - timedelta(days=days)
    metrics = await mgr.get_health_metrics(start, end)
    return {"success": True, "metrics": [m.model_dump() for m in metrics]}


# ── Medications ───────────────────────────────────────────────────────────────


@app.post("/api/medications/add")
async def add_medication(body: MedicationRequest, x_notion_key: Optional[str] = Header(None)):
    if not x_notion_key:
        raise HTTPException(401, "Missing X-Notion-Key header")
    mgr = _build_health_manager(x_notion_key)
    result = await mgr.add_medication(
        name=body.name,
        dosage=body.dosage,
        frequency=body.frequency,
        start_date=body.start_date,
        prescribing_doctor=body.prescribing_doctor,
        purpose=body.purpose,
        notes=body.notes,
    )
    return result.model_dump()


@app.get("/api/medications")
async def get_medications(x_notion_key: Optional[str] = Header(None)):
    if not x_notion_key:
        raise HTTPException(401, "Missing X-Notion-Key header")
    mgr = _build_health_manager(x_notion_key)
    from datetime import date

    result = await mgr.get_medication_schedule(date.today())
    return result.model_dump() if hasattr(result, "model_dump") else result


# ── Appointments ──────────────────────────────────────────────────────────────


@app.post("/api/appointments/add")
async def add_appointment(body: AppointmentRequest, x_notion_key: Optional[str] = Header(None)):
    if not x_notion_key:
        raise HTTPException(401, "Missing X-Notion-Key header")
    mgr = _build_health_manager(x_notion_key)
    result = await mgr.add_appointment(
        appointment_date=body.date,
        appointment_time=body.time,
        doctor_name=body.doctor_name,
        appointment_type=body.appointment_type,
        facility=body.facility,
        reason=body.reason,
        notes=body.notes,
    )
    return result.model_dump()


@app.get("/api/appointments")
async def get_appointments(days: int = 30, x_notion_key: Optional[str] = Header(None)):
    if not x_notion_key:
        raise HTTPException(401, "Missing X-Notion-Key header")
    mgr = _build_health_manager(x_notion_key)
    result = await mgr.get_upcoming_appointments(days=days)
    return result.model_dump() if hasattr(result, "model_dump") else result


# ── Goals ─────────────────────────────────────────────────────────────────────


@app.post("/api/goals/set")
async def set_goal(body: GoalRequest, x_notion_key: Optional[str] = Header(None)):
    if not x_notion_key:
        raise HTTPException(401, "Missing X-Notion-Key header")
    mgr = _build_health_manager(x_notion_key)
    result = await mgr.set_health_goal(
        goal_type=body.goal_type,
        title=body.title,
        target_value=body.target_value,
        unit=body.unit,
        target_date=body.target_date,
    )
    return result.model_dump()


@app.get("/api/goals")
async def get_goals(x_notion_key: Optional[str] = Header(None)):
    if not x_notion_key:
        raise HTTPException(401, "Missing X-Notion-Key header")
    mgr = _build_health_manager(x_notion_key)
    result = await mgr.get_active_goals()
    return {"success": True, "goals": [g.model_dump() for g in result]}


# ── Symptoms ──────────────────────────────────────────────────────────────────


@app.post("/api/symptoms/log")
async def log_symptom(body: SymptomRequest, x_notion_key: Optional[str] = Header(None)):
    if not x_notion_key:
        raise HTTPException(401, "Missing X-Notion-Key header")
    mgr = _build_health_manager(x_notion_key)
    result = await mgr.log_symptom(
        symptom_name=body.symptom_name,
        severity=body.severity,
        body_part=body.body_part,
        duration_minutes=body.duration_minutes,
        notes=body.notes,
    )
    return result.model_dump()


@app.get("/api/symptoms/analyze")
async def analyze_symptoms(days: int = 30, x_notion_key: Optional[str] = Header(None)):
    if not x_notion_key:
        raise HTTPException(401, "Missing X-Notion-Key header")
    from datetime import date, timedelta

    mgr = _build_health_manager(x_notion_key)
    end = date.today()
    start = end - timedelta(days=days)
    symptoms = await mgr.get_symptom_history(start, end)
    return {"success": True, "count": len(symptoms), "symptoms": [s.model_dump() for s in symptoms]}


# ── AI Insights ───────────────────────────────────────────────────────────────


@app.post("/api/ai/insights")
async def get_insights(
    body: InsightsRequest,
    x_notion_key: Optional[str] = Header(None),
    x_ai_provider: Optional[str] = Header(None),
    x_ai_key: Optional[str] = Header(None),
):
    if not x_notion_key:
        raise HTTPException(401, "Missing X-Notion-Key header")
    if not x_ai_key:
        raise HTTPException(401, "Missing X-AI-Key header")

    from datetime import date, timedelta

    provider = (x_ai_provider or "anthropic").lower()
    mgr = _build_health_manager(x_notion_key)
    _ensure_required_databases(mgr, ["health_database_id"])
    engine = _build_ai_engine(provider, x_ai_key or "")
    today = date.today()
    start = today - timedelta(
        days=7 if body.period == "week" else 30 if body.period == "month" else 90
    )
    metrics = await mgr.get_health_metrics(start, today)
    summary = await mgr.get_health_summary(start, today)
    insights = await engine.generate_health_insights(
        metrics,
        summary,
        body.category,
        brain_context=body.brain_context,
    )
    token_usage = getattr(engine, "_last_token_usage", None)

    if not insights:
        return {
            "success": False,
            "error": (
                "No insights generated."
                " Verify Notion database data and AI provider configuration."
            ),
            "insights": [],
            "token_usage": token_usage,
        }

    return {
        "success": True,
        "insights": [i.model_dump() for i in insights],
        "token_usage": token_usage,
    }


# ── Brain — Text ──────────────────────────────────────────────────────────────


@app.get("/api/brain/duration-options")
async def brain_duration_options(
    text: str = "",
    context_days: int = 7,
    x_notion_key: Optional[str] = Header(None),
):
    """Return realistic duration choices for TRIBEv2 text analysis."""
    return await _build_duration_options(
        text=text,
        notion_key=x_notion_key,
        context_days=context_days,
    )


@app.post("/api/brain/text")
async def brain_from_text(
    body: BrainTextRequest,
    x_hf_token: Optional[str] = Header(None),
    x_notion_key: Optional[str] = Header(None),
):
    """Predict brain activation from text using LLaMA 3.2 (TRIBEv2 text pipeline)."""
    tribe = await _ensure_tribe_ready(x_hf_token)
    duration = await _resolve_duration_minutes(
        text=body.text,
        notion_key=x_notion_key,
        context_days=body.context_days,
        duration_minutes=body.duration_minutes,
        duration_source=body.duration_source,
    )
    health_context = await _build_tribe_health_context(x_notion_key, body.context_days)
    text_stimulus = (
        f"Simulate a {duration['minutes']} minute {duration['activity_profile']['label'].lower()} "
        f"session.\n\nNarrative: {body.text}"
    )
    if health_context:
        text_stimulus = f"{text_stimulus}\n\nConnected health context: {health_context}"

    # Use the existing text-based prediction with visualization
    viz_types = body.viz_types or ["interactive", "heatmap"]
    roi_dict = await tribe.model_wrapper.predict_response(text_stimulus=text_stimulus)
    brain_responses = tribe._roi_dict_to_brain_responses(roi_dict)

    import numpy as np

    mean_act = float(
        np.mean(
            [
                v.get("activation_level", 0.5) if isinstance(v, dict) else v
                for v in brain_responses.values()
            ]
        )
    )
    duration_bonus = min(1.0, duration["minutes"] / 60) * 0.5
    wellness_score = round(5.0 + mean_act * 4.0 + duration_bonus, 2)

    label = f"{duration['activity_profile']['label']} text simulation"
    title = f"TRIBEv2 Text Simulation: {label} ({duration['minutes']} min)"
    stem = _slugify_output_label(
        f"text_{duration['activity_profile']['activity_type']}_{duration['minutes']}min"
    )
    flat_roi = {k: v.get("activation_level", 0.5) for k, v in brain_responses.items()}

    viz_paths: Dict[str, Optional[str]] = {}
    if tribe.model_wrapper._last_raw_predictions is not None:
        viz_paths = await tribe._generate_visualizations(
            tribe.model_wrapper._last_raw_predictions,
            title=title,
            stem=stem,
            viz_types=viz_types,
            modality="text (LLaMA 3.2 via gTTS → Whisper)",
            roi_activations=flat_roi,
            duration_minutes=duration["minutes"],
        )
    else:
        viz_paths = await tribe._generate_visualizations(
            None,
            title=title,
            stem=stem,
            viz_types=["heatmap"],
            modality="text",
            roi_activations=flat_roi,
            duration_minutes=duration["minutes"],
        )

    # Convert absolute paths to relative URLs served by /visualizations/
    viz_urls = _paths_to_urls(viz_paths)

    return {
        "success": True,
        "modality": "text",
        "brain_regions": brain_responses,
        "wellness_score": wellness_score,
        "model_used": tribe.model_wrapper.is_loaded,
        "visualizations": viz_urls,
        "health_context_used": bool(health_context),
        "duration_minutes": duration["minutes"],
        "duration_source": duration["source"],
        "duration_guidance": duration["activity_profile"]["prompt"],
    }


# ── Brain — Audio upload ──────────────────────────────────────────────────────


@app.post("/api/brain/audio")
async def brain_from_audio(
    file: UploadFile = File(...),
    viz_types: str = "interactive,heatmap",
    x_hf_token: Optional[str] = Header(None),
):
    """Upload an audio file (.wav/.mp3/.flac/.ogg) and predict brain activation."""
    suffix = Path(file.filename).suffix if file.filename else ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        tribe = await _ensure_tribe_ready(x_hf_token)

        vtypes = [v.strip() for v in viz_types.split(",") if v.strip()]
        result = await tribe.predict_brain_from_audio(tmp_path, viz_types=vtypes)
        result["visualizations"] = _paths_to_urls(result.get("visualizations", {}))
        return result
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


# ── Brain — Video upload ──────────────────────────────────────────────────────


@app.post("/api/brain/video")
async def brain_from_video(
    file: UploadFile = File(...),
    viz_types: str = "interactive,heatmap",
    x_hf_token: Optional[str] = Header(None),
):
    """Upload a video file (.mp4/.avi/.mov/.mkv/.webm) and predict brain activation."""
    suffix = Path(file.filename).suffix if file.filename else ".mp4"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        tribe = await _ensure_tribe_ready(x_hf_token)

        vtypes = [v.strip() for v in viz_types.split(",") if v.strip()]
        result = await tribe.predict_brain_from_video(tmp_path, viz_types=vtypes)
        result["visualizations"] = _paths_to_urls(result.get("visualizations", {}))
        return result
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


# ── Brain — Multimodal ────────────────────────────────────────────────────────


@app.post("/api/brain/multimodal")
async def brain_multimodal(
    text: Optional[str] = None,
    audio_file: Optional[UploadFile] = File(None),
    video_file: Optional[UploadFile] = File(None),
    viz_types: str = "interactive,heatmap",
    duration_minutes: Optional[int] = None,
    duration_source: str = "activity",
    x_hf_token: Optional[str] = Header(None),
    x_notion_key: Optional[str] = Header(None),
    context_days: int = 7,
):
    """Predict brain activation from multiple modalities simultaneously."""
    audio_tmp = video_tmp = None
    try:
        if audio_file:
            suf = Path(audio_file.filename).suffix if audio_file.filename else ".wav"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suf) as f:
                shutil.copyfileobj(audio_file.file, f)
                audio_tmp = f.name

        if video_file:
            suf = Path(video_file.filename).suffix if video_file.filename else ".mp4"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suf) as f:
                shutil.copyfileobj(video_file.file, f)
                video_tmp = f.name

        tribe = await _ensure_tribe_ready(x_hf_token)
        health_context = await _build_tribe_health_context(x_notion_key, context_days)
        text_stimulus = text
        duration_result = None
        if text:
            duration_result = await _resolve_duration_minutes(
                text=text,
                notion_key=x_notion_key,
                context_days=context_days,
                duration_minutes=duration_minutes,
                duration_source=duration_source,
            )
            text_stimulus = (
                f"Simulate a {duration_result['minutes']} minute "
                f"{duration_result['activity_profile']['label'].lower()} session.\n\n"
                f"Narrative: {text}"
            )
            if health_context:
                text_stimulus = f"{text_stimulus}\n\nConnected health context: {health_context}"

        vtypes = [v.strip() for v in viz_types.split(",") if v.strip()]
        result = await tribe.predict_brain_multimodal(
            video_path=video_tmp,
            audio_path=audio_tmp,
            text_stimulus=text_stimulus,
            viz_types=vtypes,
            duration_minutes=duration_result["minutes"] if duration_result else None,
            output_label=(
                f"{duration_result['activity_profile']['label']} multimodal simulation"
                if duration_result
                else None
            ),
        )
        result["visualizations"] = _paths_to_urls(result.get("visualizations", {}))
        result["health_context_used"] = bool(health_context)
        if duration_result:
            result["duration_minutes"] = duration_result["minutes"]
            result["duration_source"] = duration_result["source"]
            result["duration_guidance"] = duration_result["activity_profile"]["prompt"]
        return result
    finally:
        for p in [audio_tmp, video_tmp]:
            if p:
                try:
                    os.unlink(p)
                except OSError:
                    pass


# ── Brain — Re-visualize stored prediction ────────────────────────────────────


@app.post("/api/brain/visualize")
async def regenerate_visualization(body: VizRequest):
    """Re-generate a visualization from the most recent stored TRIBEv2 prediction."""
    tribe = _get_tribe_analyzer()
    result = await tribe.generate_brain_visualization(
        viz_type=body.viz_type,
        title=body.title or "Brain Activation",
        auto_open=False,
    )
    if result.get("success"):
        result["url"] = _path_to_url(result.get("path", ""))
    return result


# ── Optimal schedule & cognitive report ──────────────────────────────────────


@app.get("/api/brain/schedule")
async def get_optimal_schedule():
    tribe = _get_tribe_analyzer()
    return await tribe.generate_optimal_schedule()


@app.get("/api/brain/report")
async def get_cognitive_report(days: int = 30, x_notion_key: Optional[str] = Header(None)):
    if not x_notion_key:
        raise HTTPException(401, "Missing X-Notion-Key header")
    mgr = _build_health_manager(x_notion_key)
    result = await mgr.get_cognitive_health_report(days)
    return result.model_dump()


# ── Visualization file listing ────────────────────────────────────────────────


@app.get("/api/visualizations")
async def list_visualizations():
    """List all generated visualization files with their URLs."""
    files = []
    for p in sorted(VIZ_DIR.iterdir(), key=lambda f: f.stat().st_mtime, reverse=True):
        if p.suffix in {".html", ".png", ".gif", ".mp4"}:
            files.append(
                {
                    "name": p.name,
                    "url": f"/visualizations/{p.name}",
                    "type": p.suffix.lstrip("."),
                    "size_kb": p.stat().st_size // 1024,
                }
            )
    return {"files": files}


@app.post("/api/visualizations/clear")
async def clear_visualizations():
    """Delete generated visualization files while keeping placeholders intact."""
    deleted = []
    for p in VIZ_DIR.iterdir():
        if p.name == ".gitkeep":
            continue
        if p.is_file() and p.suffix in {".html", ".png", ".gif", ".mp4"}:
            p.unlink(missing_ok=True)
            deleted.append(p.name)
    return {"success": True, "deleted_count": len(deleted), "deleted_files": deleted}


# ── Helpers ───────────────────────────────────────────────────────────────────


def _path_to_url(path: Optional[str]) -> Optional[str]:
    if not path:
        return None
    p = Path(path)
    if p.is_relative_to(VIZ_DIR):
        return f"/visualizations/{p.name}"
    if p.parent == VIZ_DIR or p.name:
        return f"/visualizations/{p.name}"
    return None


def _paths_to_urls(paths: Dict[str, Optional[str]]) -> Dict[str, Optional[str]]:
    return {k: _path_to_url(v) for k, v in (paths or {}).items()}


# ── Entry point ───────────────────────────────────────────────────────────────


def main():
    import uvicorn
    from dotenv import load_dotenv

    _project_root = Path(__file__).resolve().parent.parent.parent
    load_dotenv(_project_root / ".env")

    port = int(os.getenv("API_PORT", "8000"))
    host = os.getenv("API_HOST", "0.0.0.0")
    logger.info(f"Starting NotionHealth AI API on {host}:{port}")
    uvicorn.run("notion_health_ai.api:app", host=host, port=port, reload=False)


if __name__ == "__main__":
    main()
