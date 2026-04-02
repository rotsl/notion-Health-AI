# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Rohan R
# Author: Rohan R

"""
NotionHealth AI - AI-Powered Personal Health & Wellness Management System.

Keep package import lightweight so the API and CLI can start even when the
optional TRIBEv2 runtime is unavailable.
"""

from importlib import import_module

__version__ = "2.0.0"
__author__ = "Rohan R"
__description__ = (
    "AI-Powered Personal Health & Wellness Management System"
    " with Notion MCP and Tribe v2 Brain Prediction"
)

_EXPORTS = {
    # Core classes
    "HealthManager": ("notion_health_ai.health_manager", "HealthManager"),
    "AIInsightsEngine": ("notion_health_ai.ai_insights", "AIInsightsEngine"),
    # Models
    "AIInsight": ("notion_health_ai.models", "AIInsight"),
    "Appointment": ("notion_health_ai.models", "Appointment"),
    "AppointmentStatus": ("notion_health_ai.models", "AppointmentStatus"),
    "AppointmentType": ("notion_health_ai.models", "AppointmentType"),
    "EnergyLevel": ("notion_health_ai.models", "EnergyLevel"),
    "ExerciseType": ("notion_health_ai.models", "ExerciseType"),
    "GoalStatus": ("notion_health_ai.models", "GoalStatus"),
    "GoalType": ("notion_health_ai.models", "GoalType"),
    "HealthGoal": ("notion_health_ai.models", "HealthGoal"),
    "HealthMetric": ("notion_health_ai.models", "HealthMetric"),
    "HealthSummary": ("notion_health_ai.models", "HealthSummary"),
    "MCPCallResult": ("notion_health_ai.models", "MCPCallResult"),
    "Medication": ("notion_health_ai.models", "Medication"),
    "MedicationFrequency": ("notion_health_ai.models", "MedicationFrequency"),
    "MoodLevel": ("notion_health_ai.models", "MoodLevel"),
    "SymptomLog": ("notion_health_ai.models", "SymptomLog"),
    "SymptomSeverity": ("notion_health_ai.models", "SymptomSeverity"),
    # Tribe integration
    "ActivityPrediction": ("notion_health_ai.tribe_integration", "ActivityPrediction"),
    "ActivityType": ("notion_health_ai.tribe_integration", "ActivityType"),
    "BrainHealthCorrelation": ("notion_health_ai.tribe_integration", "BrainHealthCorrelation"),
    "BrainRegion": ("notion_health_ai.tribe_integration", "BrainRegion"),
    "BrainResponse": ("notion_health_ai.tribe_integration", "BrainResponse"),
    "TribeHealthAnalyzer": ("notion_health_ai.tribe_integration", "TribeHealthAnalyzer"),
    "TribeModelWrapper": ("notion_health_ai.tribe_integration", "TribeModelWrapper"),
}

__all__ = list(_EXPORTS)


def __getattr__(name: str):
    if name not in _EXPORTS:
        raise AttributeError(f"module 'notion_health_ai' has no attribute {name!r}")
    module_name, attr_name = _EXPORTS[name]
    module = import_module(module_name)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value
