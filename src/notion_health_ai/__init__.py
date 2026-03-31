"""
NotionHealth AI - AI-Powered Personal Health & Wellness Management System

A comprehensive health management system using Notion MCP (Model Context Protocol)
for seamless integration with AI assistants like Claude.

Version 2.0 adds Tribe v2 brain prediction capabilities.
"""

__version__ = "2.0.0"
__author__ = "NotionHealth AI Team"
__description__ = "AI-Powered Personal Health & Wellness Management System with Notion MCP and Tribe v2 Brain Prediction"

from notion_health_ai.models import (
    HealthMetric,
    Medication,
    Appointment,
    HealthGoal,
    SymptomLog,
    HealthSummary,
    AIInsight,
    MoodLevel,
    EnergyLevel,
    ExerciseType,
    MedicationFrequency,
    AppointmentType,
    AppointmentStatus,
    GoalType,
    GoalStatus,
    SymptomSeverity,
    MCPCallResult,
)
from notion_health_ai.health_manager import HealthManager
from notion_health_ai.ai_insights import AIInsightsEngine
from notion_health_ai.tribe_integration import (
    TribeHealthAnalyzer,
    TribeModelWrapper,
    ActivityType,
    BrainRegion,
    BrainResponse,
    ActivityPrediction,
    BrainHealthCorrelation,
)

__all__ = [
    # Core models
    "HealthMetric",
    "Medication",
    "Appointment",
    "HealthGoal",
    "SymptomLog",
    "HealthSummary",
    "AIInsight",
    "MoodLevel",
    "EnergyLevel",
    "ExerciseType",
    "MedicationFrequency",
    "AppointmentType",
    "AppointmentStatus",
    "GoalType",
    "GoalStatus",
    "SymptomSeverity",
    "MCPCallResult",
    # Core classes
    "HealthManager",
    "AIInsightsEngine",
    # Tribe v2 integration
    "TribeHealthAnalyzer",
    "TribeModelWrapper",
    "ActivityType",
    "BrainRegion",
    "BrainResponse",
    "ActivityPrediction",
    "BrainHealthCorrelation",
]
