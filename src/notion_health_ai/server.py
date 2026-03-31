"""
NotionHealth AI MCP Server - Model Context Protocol implementation.

This module implements the MCP server for NotionHealth AI,
exposing health management tools for AI assistants.
Includes Tribe v2 brain prediction integration.
"""

import asyncio
import json
import os
from typing import Optional, List, Dict, Any
from datetime import datetime, date, timedelta
from loguru import logger

# MCP imports
try:
    from mcp.server import Server
    from mcp.server.stdio import stdio_server
    from mcp.types import (
        Tool,
        TextContent,
        ImageContent,
        EmbeddedResource,
        Resource,
        Prompt,
        PromptArgument,
    )

    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False
    logger.warning("MCP package not available, using mock server")

from notion_health_ai.health_manager import HealthManager
from notion_health_ai.ai_insights import AIInsightsEngine
from notion_health_ai.tribe_integration import TribeHealthAnalyzer
from notion_health_ai.models import MCPCallResult


class NotionHealthMCPServer:
    """
    MCP Server for NotionHealth AI.

    Exposes health management tools via the Model Context Protocol
    for integration with AI assistants like Claude.
    Includes Tribe v2 brain response prediction capabilities.
    """

    def __init__(self):
        """Initialize the MCP server."""
        self.server_name = os.getenv("MCP_SERVER_NAME", "notion-health-ai")
        self.server_version = os.getenv("MCP_SERVER_VERSION", "2.0.0")

        # Initialize health manager
        self.health_manager = HealthManager()

        # Initialize AI insights engine
        self.ai_engine = AIInsightsEngine()

        # Initialize Tribe v2 analyzer
        self.tribe_analyzer = TribeHealthAnalyzer()

        # Create MCP server instance
        if MCP_AVAILABLE:
            self.server = Server(self.server_name)
            self._setup_handlers()

        logger.info(
            f"NotionHealth MCP Server initialized: {self.server_name} v{self.server_version}"
        )

    def _setup_handlers(self):
        """Setup MCP server handlers."""
        if not MCP_AVAILABLE:
            return

        # Register tools
        @self.server.list_tools()
        async def list_tools() -> List[Tool]:
            """List available health management tools."""
            return [
                Tool(
                    name="log_health_metric",
                    description="Log a health metric entry (weight, sleep, mood, exercise, etc.)",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "date": {
                                "type": "string",
                                "description": "Date in YYYY-MM-DD format (optional, defaults to today)",
                            },
                            "weight": {
                                "type": "number",
                                "description": "Weight in pounds",
                            },
                            "sleep_hours": {
                                "type": "number",
                                "description": "Hours of sleep",
                            },
                            "sleep_quality": {
                                "type": "integer",
                                "minimum": 1,
                                "maximum": 10,
                                "description": "Sleep quality rating 1-10",
                            },
                            "exercise_type": {
                                "type": "string",
                                "enum": [
                                    "running",
                                    "walking",
                                    "cycling",
                                    "swimming",
                                    "strength_training",
                                    "yoga",
                                    "hiit",
                                    "cardio",
                                    "other",
                                ],
                                "description": "Type of exercise",
                            },
                            "exercise_minutes": {
                                "type": "integer",
                                "description": "Exercise duration in minutes",
                            },
                            "steps": {
                                "type": "integer",
                                "description": "Daily step count",
                            },
                            "mood": {
                                "type": "integer",
                                "minimum": 1,
                                "maximum": 10,
                                "description": "Mood rating 1-10",
                            },
                            "energy": {
                                "type": "integer",
                                "minimum": 1,
                                "maximum": 10,
                                "description": "Energy level 1-10",
                            },
                            "water_glasses": {
                                "type": "integer",
                                "description": "Glasses of water consumed",
                            },
                            "blood_pressure_systolic": {
                                "type": "integer",
                                "description": "Systolic blood pressure (mmHg)",
                            },
                            "blood_pressure_diastolic": {
                                "type": "integer",
                                "description": "Diastolic blood pressure (mmHg)",
                            },
                            "heart_rate": {
                                "type": "integer",
                                "description": "Resting heart rate (bpm)",
                            },
                            "notes": {
                                "type": "string",
                                "description": "Additional notes",
                            },
                        },
                    },
                ),
                Tool(
                    name="get_health_summary",
                    description="Get a summary of health data for a time period",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "start_date": {
                                "type": "string",
                                "description": "Start date in YYYY-MM-DD format",
                            },
                            "end_date": {
                                "type": "string",
                                "description": "End date in YYYY-MM-DD format",
                            },
                            "period": {
                                "type": "string",
                                "enum": ["today", "week", "month", "custom"],
                                "description": "Predefined time period",
                            },
                        },
                    },
                ),
                Tool(
                    name="add_medication",
                    description="Add a medication to track",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "name": {
                                "type": "string",
                                "description": "Medication name",
                            },
                            "dosage": {
                                "type": "string",
                                "description": "Dosage (e.g., '100mg')",
                            },
                            "frequency": {
                                "type": "string",
                                "enum": [
                                    "once_daily",
                                    "twice_daily",
                                    "three_times_daily",
                                    "weekly",
                                    "as_needed",
                                ],
                                "description": "How often to take",
                            },
                            "start_date": {
                                "type": "string",
                                "description": "Start date in YYYY-MM-DD format",
                            },
                            "times": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Times to take (HH:MM format)",
                            },
                            "prescribing_doctor": {
                                "type": "string",
                                "description": "Doctor who prescribed",
                            },
                            "purpose": {
                                "type": "string",
                                "description": "What the medication is for",
                            },
                            "notes": {
                                "type": "string",
                                "description": "Additional notes",
                            },
                        },
                        "required": ["name", "dosage", "frequency", "start_date"],
                    },
                ),
                Tool(
                    name="get_medication_schedule",
                    description="Get medication schedule for today or a specific date",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "date": {
                                "type": "string",
                                "description": "Date in YYYY-MM-DD format (optional, defaults to today)",
                            },
                        },
                    },
                ),
                Tool(
                    name="add_appointment",
                    description="Add a medical appointment",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "date": {
                                "type": "string",
                                "description": "Appointment date in YYYY-MM-DD format",
                            },
                            "time": {
                                "type": "string",
                                "description": "Appointment time in HH:MM format",
                            },
                            "doctor_name": {
                                "type": "string",
                                "description": "Doctor or provider name",
                            },
                            "appointment_type": {
                                "type": "string",
                                "enum": [
                                    "checkup",
                                    "follow_up",
                                    "specialist",
                                    "dental",
                                    "vision",
                                    "mental_health",
                                    "lab_work",
                                    "other",
                                ],
                                "description": "Type of appointment",
                            },
                            "facility": {
                                "type": "string",
                                "description": "Clinic or hospital name",
                            },
                            "reason": {
                                "type": "string",
                                "description": "Reason for visit",
                            },
                            "notes": {
                                "type": "string",
                                "description": "Additional notes",
                            },
                        },
                        "required": ["date", "time", "doctor_name", "appointment_type"],
                    },
                ),
                Tool(
                    name="get_upcoming_appointments",
                    description="Get upcoming appointments",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "days": {
                                "type": "integer",
                                "description": "Number of days to look ahead (default: 30)",
                            },
                        },
                    },
                ),
                Tool(
                    name="set_health_goal",
                    description="Set a health goal",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "goal_type": {
                                "type": "string",
                                "enum": [
                                    "weight",
                                    "exercise",
                                    "sleep",
                                    "steps",
                                    "water_intake",
                                    "meditation",
                                    "custom",
                                ],
                                "description": "Type of goal",
                            },
                            "title": {
                                "type": "string",
                                "description": "Goal title",
                            },
                            "target_value": {
                                "type": "number",
                                "description": "Target value to achieve",
                            },
                            "unit": {
                                "type": "string",
                                "description": "Unit of measurement",
                            },
                            "target_date": {
                                "type": "string",
                                "description": "Deadline in YYYY-MM-DD format",
                            },
                            "description": {
                                "type": "string",
                                "description": "Detailed description",
                            },
                        },
                        "required": ["goal_type", "title", "target_value", "unit", "target_date"],
                    },
                ),
                Tool(
                    name="get_goal_progress",
                    description="Check progress on health goals",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "goal_type": {
                                "type": "string",
                                "description": "Filter by goal type (optional)",
                            },
                        },
                    },
                ),
                Tool(
                    name="update_goal_progress",
                    description="Update progress on a health goal",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "goal_id": {
                                "type": "string",
                                "description": "Goal identifier",
                            },
                            "current_value": {
                                "type": "number",
                                "description": "Current progress value",
                            },
                        },
                        "required": ["goal_id", "current_value"],
                    },
                ),
                Tool(
                    name="log_symptom",
                    description="Log a symptom entry",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "symptom_name": {
                                "type": "string",
                                "description": "Name of the symptom",
                            },
                            "severity": {
                                "type": "integer",
                                "minimum": 1,
                                "maximum": 10,
                                "description": "Severity rating 1-10",
                            },
                            "date": {
                                "type": "string",
                                "description": "Date in YYYY-MM-DD format (optional)",
                            },
                            "time": {
                                "type": "string",
                                "description": "Time in HH:MM format (optional)",
                            },
                            "body_part": {
                                "type": "string",
                                "description": "Affected body part",
                            },
                            "duration_minutes": {
                                "type": "integer",
                                "description": "Duration in minutes",
                            },
                            "triggers": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Identified triggers",
                            },
                            "notes": {
                                "type": "string",
                                "description": "Additional notes",
                            },
                        },
                        "required": ["symptom_name", "severity"],
                    },
                ),
                Tool(
                    name="analyze_symptoms",
                    description="Analyze symptom patterns",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "start_date": {
                                "type": "string",
                                "description": "Start date for analysis",
                            },
                            "end_date": {
                                "type": "string",
                                "description": "End date for analysis",
                            },
                        },
                    },
                ),
                Tool(
                    name="get_ai_insights",
                    description="Get AI-powered health insights and recommendations",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "category": {
                                "type": "string",
                                "enum": ["weight", "sleep", "exercise", "mood", "overall", "brain"],
                                "description": "Category to analyze (optional, defaults to overall)",
                            },
                            "period": {
                                "type": "string",
                                "enum": ["week", "month", "all"],
                                "description": "Time period to analyze",
                            },
                        },
                    },
                ),
                # Tribe v2 Brain Prediction Tools
                Tool(
                    name="predict_brain_response",
                    description="Predict brain response to a wellness activity using Tribe v2 model",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "activity": {
                                "type": "string",
                                "enum": [
                                    "exercise",
                                    "meditation",
                                    "sleep",
                                    "reading",
                                    "music",
                                    "social",
                                    "work",
                                    "relaxation",
                                    "nature",
                                    "learning",
                                    "creative",
                                    "mindfulness",
                                ],
                                "description": "Activity to analyze",
                            },
                            "duration_minutes": {
                                "type": "integer",
                                "description": "Duration of the activity in minutes (default: 30)",
                            },
                        },
                        "required": ["activity"],
                    },
                ),
                Tool(
                    name="analyze_brain_health_correlation",
                    description="Analyze correlations between health metrics and brain activity using Tribe v2",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "days": {
                                "type": "integer",
                                "description": "Number of days to analyze (default: 7)",
                            },
                        },
                    },
                ),
                Tool(
                    name="get_optimal_schedule",
                    description="Generate an optimal daily schedule based on Tribe v2 brain science",
                    inputSchema={
                        "type": "object",
                        "properties": {},
                    },
                ),
                Tool(
                    name="get_cognitive_health_report",
                    description=(
                        "Generate a comprehensive cognitive health report combining health data, "
                        "TRIBEv2 brain-activity analysis, brain-health correlations, and an "
                        "optimal daily schedule. Best for weekly or monthly reviews."
                    ),
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "days": {
                                "type": "integer",
                                "description": "Days of data to include (default: 30)",
                            },
                        },
                    },
                ),
                # ── Multimodal TRIBEv2 tools ────────────────────────────────────
                Tool(
                    name="predict_brain_from_video",
                    description=(
                        "Predict brain activation from a video file using TRIBEv2's full multimodal pipeline: "
                        "V-JEPA2 (visual frames) + Wav2Vec-BERT (audio) + LLaMA 3.2 (WhisperX transcript). "
                        "Generates an interactive 3D brain visualization automatically opened in the browser. "
                        "Requires an absolute file path to a .mp4, .avi, .mov, .mkv, or .webm file."
                    ),
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "video_path": {
                                "type": "string",
                                "description": "Absolute path to the video file",
                            },
                            "viz_types": {
                                "type": "array",
                                "items": {
                                    "type": "string",
                                    "enum": ["interactive", "static", "gif", "mp4", "heatmap"],
                                },
                                "description": "Visualizations to generate (default: [interactive, heatmap])",
                            },
                        },
                        "required": ["video_path"],
                    },
                ),
                Tool(
                    name="predict_brain_from_audio",
                    description=(
                        "Predict brain activation from an audio file using TRIBEv2's Wav2Vec-BERT + "
                        "LLaMA 3.2 (via WhisperX transcription) feature extractors. "
                        "Generates an interactive 3D brain visualization in the browser. "
                        "Supports .wav, .mp3, .flac, .ogg files."
                    ),
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "audio_path": {
                                "type": "string",
                                "description": "Absolute path to the audio file",
                            },
                            "viz_types": {
                                "type": "array",
                                "items": {
                                    "type": "string",
                                    "enum": ["interactive", "static", "gif", "mp4", "heatmap"],
                                },
                                "description": "Visualizations to generate (default: [interactive, heatmap])",
                            },
                        },
                        "required": ["audio_path"],
                    },
                ),
                Tool(
                    name="predict_brain_multimodal",
                    description=(
                        "Predict brain activation from multiple modalities simultaneously using TRIBEv2. "
                        "Combines V-JEPA2 (video), Wav2Vec-BERT (audio), and LLaMA 3.2 (text) by averaging "
                        "cortical predictions across all provided inputs. "
                        "Provide at least one of: video_path, audio_path, or text_stimulus."
                    ),
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "video_path": {
                                "type": "string",
                                "description": "Absolute path to video file (optional)",
                            },
                            "audio_path": {
                                "type": "string",
                                "description": "Absolute path to audio file (optional)",
                            },
                            "text_stimulus": {
                                "type": "string",
                                "description": "Text narrative / description (optional)",
                            },
                            "viz_types": {
                                "type": "array",
                                "items": {
                                    "type": "string",
                                    "enum": ["interactive", "static", "gif", "mp4", "heatmap"],
                                },
                                "description": "Visualizations to generate (default: [interactive, heatmap])",
                            },
                        },
                    },
                ),
                Tool(
                    name="generate_brain_visualization",
                    description=(
                        "Re-generate a brain visualization from the most recent TRIBEv2 prediction "
                        "stored in memory. Use this to get a different format (e.g. animated GIF or "
                        "MP4) without re-running the model. Requires a prior prediction call."
                    ),
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "viz_type": {
                                "type": "string",
                                "enum": ["interactive", "static", "gif", "mp4", "heatmap"],
                                "description": "Type of visualization to generate",
                            },
                            "title": {
                                "type": "string",
                                "description": "Plot title (optional)",
                            },
                            "auto_open": {
                                "type": "boolean",
                                "description": "Auto-open HTML in browser for interactive type (default: true)",
                            },
                        },
                        "required": ["viz_type"],
                    },
                ),
                Tool(
                    name="open_brain_visualization",
                    description=(
                        "Open a previously generated brain visualization file in the default browser "
                        "or OS viewer. Pass the file path returned by any predict_brain_* or "
                        "generate_brain_visualization tool."
                    ),
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "path": {
                                "type": "string",
                                "description": "Absolute path to the visualization file (.html, .png, .gif, .mp4)",
                            },
                        },
                        "required": ["path"],
                    },
                ),
            ]

        @self.server.call_tool()
        async def call_tool(name: str, arguments: Dict[str, Any]) -> List[TextContent]:
            """Handle tool calls."""
            try:
                result = await self._handle_tool_call(name, arguments)
                return [TextContent(type="text", text=json.dumps(result, default=str))]
            except Exception as e:
                logger.error(f"Tool call error: {e}")
                return [
                    TextContent(
                        type="text",
                        text=json.dumps(
                            {
                                "success": False,
                                "error": str(e),
                            }
                        ),
                    )
                ]

    async def _handle_tool_call(self, name: str, arguments: Dict[str, Any]) -> Dict:
        """
        Handle a tool call.

        Args:
            name: Tool name
            arguments: Tool arguments

        Returns:
            Result dictionary
        """
        logger.info(f"Tool call: {name} with args: {arguments}")

        # Health metrics
        if name == "log_health_metric":
            metric_date = None
            if arguments.get("date"):
                metric_date = datetime.strptime(arguments["date"], "%Y-%m-%d").date()

            result = await self.health_manager.log_health_metric(
                metric_date=metric_date,
                weight=arguments.get("weight"),
                sleep_hours=arguments.get("sleep_hours"),
                sleep_quality=arguments.get("sleep_quality"),
                exercise_type=arguments.get("exercise_type"),
                exercise_minutes=arguments.get("exercise_minutes"),
                steps=arguments.get("steps"),
                mood=arguments.get("mood"),
                energy=arguments.get("energy"),
                water_glasses=arguments.get("water_glasses"),
                blood_pressure_systolic=arguments.get("blood_pressure_systolic"),
                blood_pressure_diastolic=arguments.get("blood_pressure_diastolic"),
                heart_rate=arguments.get("heart_rate"),
                notes=arguments.get("notes"),
            )
            return result.model_dump()

        elif name == "get_health_summary":
            period = arguments.get("period", "week")
            today = date.today()

            if period == "today":
                start_date = today
                end_date = today
            elif period == "week":
                start_date = today - timedelta(days=7)
                end_date = today
            elif period == "month":
                start_date = today - timedelta(days=30)
                end_date = today
            else:
                start_date = datetime.strptime(arguments["start_date"], "%Y-%m-%d").date()
                end_date = datetime.strptime(arguments["end_date"], "%Y-%m-%d").date()

            summary = await self.health_manager.get_health_summary(start_date, end_date)
            return summary.model_dump()

        # Medications
        elif name == "add_medication":
            start_date = datetime.strptime(arguments["start_date"], "%Y-%m-%d").date()
            result = await self.health_manager.add_medication(
                name=arguments["name"],
                dosage=arguments["dosage"],
                frequency=arguments["frequency"],
                start_date=start_date,
                times=arguments.get("times", []),
                prescribing_doctor=arguments.get("prescribing_doctor"),
                purpose=arguments.get("purpose"),
                notes=arguments.get("notes"),
            )
            return result.model_dump()

        elif name == "get_medication_schedule":
            for_date = None
            if arguments.get("date"):
                for_date = datetime.strptime(arguments["date"], "%Y-%m-%d").date()
            schedule = await self.health_manager.get_medication_schedule(for_date)
            return {"success": True, "schedule": schedule}

        # Appointments
        elif name == "add_appointment":
            appointment_date = datetime.strptime(arguments["date"], "%Y-%m-%d").date()
            result = await self.health_manager.add_appointment(
                appointment_date=appointment_date,
                appointment_time=arguments["time"],
                doctor_name=arguments["doctor_name"],
                appointment_type=arguments["appointment_type"],
                facility=arguments.get("facility"),
                reason=arguments.get("reason"),
                notes=arguments.get("notes"),
            )
            return result.model_dump()

        elif name == "get_upcoming_appointments":
            days = arguments.get("days", 30)
            appointments = await self.health_manager.get_upcoming_appointments(days)
            return {
                "success": True,
                "appointments": [a.model_dump() for a in appointments],
            }

        # Goals
        elif name == "set_health_goal":
            target_date = datetime.strptime(arguments["target_date"], "%Y-%m-%d").date()
            result = await self.health_manager.set_health_goal(
                goal_type=arguments["goal_type"],
                title=arguments["title"],
                target_value=arguments["target_value"],
                unit=arguments["unit"],
                target_date=target_date,
                description=arguments.get("description"),
            )
            return result.model_dump()

        elif name == "get_goal_progress":
            goals = await self.health_manager.get_active_goals()
            return {
                "success": True,
                "goals": [g.model_dump() for g in goals],
            }

        elif name == "update_goal_progress":
            result = await self.health_manager.update_goal_progress(
                goal_id=arguments["goal_id"],
                current_value=arguments["current_value"],
            )
            return result.model_dump()

        # Symptoms
        elif name == "log_symptom":
            symptom_date = None
            if arguments.get("date"):
                symptom_date = datetime.strptime(arguments["date"], "%Y-%m-%d").date()

            result = await self.health_manager.log_symptom(
                symptom_name=arguments["symptom_name"],
                severity=arguments["severity"],
                symptom_date=symptom_date,
                symptom_time=arguments.get("time"),
                body_part=arguments.get("body_part"),
                duration_minutes=arguments.get("duration_minutes"),
                triggers=arguments.get("triggers"),
                notes=arguments.get("notes"),
            )
            return result.model_dump()

        elif name == "analyze_symptoms":
            start_date = None
            end_date = None
            if arguments.get("start_date"):
                start_date = datetime.strptime(arguments["start_date"], "%Y-%m-%d").date()
            if arguments.get("end_date"):
                end_date = datetime.strptime(arguments["end_date"], "%Y-%m-%d").date()

            symptoms = await self.health_manager.get_symptom_history(start_date, end_date)
            insight = await self.ai_engine.analyze_symptom_patterns(symptoms)
            return insight.model_dump()

        # AI Insights
        elif name == "get_ai_insights":
            category = arguments.get("category", "overall")
            period = arguments.get("period", "week")

            today = date.today()
            if period == "week":
                start_date = today - timedelta(days=7)
            elif period == "month":
                start_date = today - timedelta(days=30)
            else:
                start_date = today - timedelta(days=90)

            metrics = await self.health_manager.get_health_metrics(start_date, today)
            summary = await self.health_manager.get_health_summary(start_date, today)

            insights = await self.ai_engine.generate_health_insights(metrics, summary, category)
            return {
                "success": True,
                "insights": [i.model_dump() for i in insights],
            }

        # Tribe v2 Brain Prediction Tools
        elif name == "predict_brain_response":
            activity = arguments.get("activity")
            duration = arguments.get("duration_minutes", 30)

            result = await self.tribe_analyzer.predict_brain_response(activity, duration)
            return result

        elif name == "analyze_brain_health_correlation":
            days = arguments.get("days", 7)

            end_date = date.today()
            start_date = end_date - timedelta(days=days)

            metrics = await self.health_manager.get_health_metrics(start_date, end_date)
            summary = await self.health_manager.get_health_summary(start_date, end_date)

            result = await self.tribe_analyzer.analyze_brain_health_correlation(metrics, summary)
            return result

        elif name == "get_optimal_schedule":
            result = await self.tribe_analyzer.generate_optimal_schedule()
            return result

        elif name == "get_cognitive_health_report":
            days = arguments.get("days", 30)
            result = await self.health_manager.get_cognitive_health_report(days)
            return result.model_dump()

        # ── Multimodal TRIBEv2 tools ─────────────────────────────────────────
        elif name == "predict_brain_from_video":
            result = await self.tribe_analyzer.predict_brain_from_video(
                video_path=arguments["video_path"],
                viz_types=arguments.get("viz_types"),
            )
            return result

        elif name == "predict_brain_from_audio":
            result = await self.tribe_analyzer.predict_brain_from_audio(
                audio_path=arguments["audio_path"],
                viz_types=arguments.get("viz_types"),
            )
            return result

        elif name == "predict_brain_multimodal":
            result = await self.tribe_analyzer.predict_brain_multimodal(
                video_path=arguments.get("video_path"),
                audio_path=arguments.get("audio_path"),
                text_stimulus=arguments.get("text_stimulus"),
                viz_types=arguments.get("viz_types"),
            )
            return result

        elif name == "generate_brain_visualization":
            result = await self.tribe_analyzer.generate_brain_visualization(
                viz_type=arguments.get("viz_type", "interactive"),
                title=arguments.get("title", "Brain Activation"),
                auto_open=arguments.get("auto_open", True),
            )
            return result

        elif name == "open_brain_visualization":
            path = arguments.get("path", "")
            try:
                from .brain_viz import open_file

                open_file(path)
                return {"success": True, "opened": path}
            except Exception as e:
                return {"success": False, "error": str(e)}

        else:
            return {"success": False, "error": f"Unknown tool: {name}"}

    async def run(self):
        """Run the MCP server."""
        if not MCP_AVAILABLE:
            logger.error("MCP package not available. Install with: pip install mcp")
            return

        async with stdio_server() as (read_stream, write_stream):
            logger.info("Starting MCP server...")
            await self.server.run(
                read_stream,
                write_stream,
                self.server.create_initialization_options(),
            )

    # Handler methods for CLI
    async def handle_log_health_metric(self, arguments: Dict) -> Dict:
        """Handle log_health_metric tool call."""
        return await self._handle_tool_call("log_health_metric", arguments)

    async def handle_add_medication(self, arguments: Dict) -> Dict:
        """Handle add_medication tool call."""
        return await self._handle_tool_call("add_medication", arguments)

    async def handle_get_health_summary(self, arguments: Dict) -> Dict:
        """Handle get_health_summary tool call."""
        return await self._handle_tool_call("get_health_summary", arguments)

    async def handle_add_appointment(self, arguments: Dict) -> Dict:
        """Handle add_appointment tool call."""
        return await self._handle_tool_call("add_appointment", arguments)

    async def handle_set_health_goal(self, arguments: Dict) -> Dict:
        """Handle set_health_goal tool call."""
        return await self._handle_tool_call("set_health_goal", arguments)

    async def handle_log_symptom(self, arguments: Dict) -> Dict:
        """Handle log_symptom tool call."""
        return await self._handle_tool_call("log_symptom", arguments)

    def list_tools(self) -> List:
        """List all available tools (for testing)."""
        return [
            type("Tool", (), {"name": "log_health_metric"})(),
            type("Tool", (), {"name": "get_health_summary"})(),
            type("Tool", (), {"name": "add_medication"})(),
            type("Tool", (), {"name": "get_medication_schedule"})(),
            type("Tool", (), {"name": "add_appointment"})(),
            type("Tool", (), {"name": "get_upcoming_appointments"})(),
            type("Tool", (), {"name": "set_health_goal"})(),
            type("Tool", (), {"name": "get_goal_progress"})(),
            type("Tool", (), {"name": "update_goal_progress"})(),
            type("Tool", (), {"name": "log_symptom"})(),
            type("Tool", (), {"name": "analyze_symptoms"})(),
            type("Tool", (), {"name": "get_ai_insights"})(),
            type("Tool", (), {"name": "predict_brain_response"})(),
            type("Tool", (), {"name": "analyze_brain_health_correlation"})(),
            type("Tool", (), {"name": "get_optimal_schedule"})(),
            type("Tool", (), {"name": "get_cognitive_health_report"})(),
            type("Tool", (), {"name": "predict_brain_from_video"})(),
            type("Tool", (), {"name": "predict_brain_from_audio"})(),
            type("Tool", (), {"name": "predict_brain_multimodal"})(),
            type("Tool", (), {"name": "generate_brain_visualization"})(),
            type("Tool", (), {"name": "open_brain_visualization"})(),
        ]


def main():
    """Main entry point for the MCP server."""
    # Resolve project root so this works regardless of working directory
    _project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    # Load .env from project root
    try:
        from dotenv import load_dotenv

        load_dotenv(os.path.join(_project_root, ".env"))
    except ImportError:
        pass

    # Configure logging — resolve relative paths against project root
    log_level = os.getenv("LOG_LEVEL", "INFO")
    _log_file = os.getenv("LOG_FILE", "logs/notion-health-ai.log")
    if not os.path.isabs(_log_file):
        _log_file = os.path.join(_project_root, _log_file)
    logger.add(_log_file, rotation="10 MB", retention="7 days", level=log_level)

    # Create and run server
    server = NotionHealthMCPServer()
    asyncio.run(server.run())


if __name__ == "__main__":
    main()
