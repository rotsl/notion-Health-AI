"""
Pydantic models for health data structures.

This module defines all the data models used throughout NotionHealth AI,
including health metrics, medications, appointments, goals, and symptoms.
"""

from __future__ import annotations
from datetime import datetime, date, time
from enum import Enum
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field, field_validator, ConfigDict


class MoodLevel(int, Enum):
    """Mood level on a scale of 1-10."""

    VERY_LOW = 1
    LOW = 2
    SOMEWHAT_LOW = 3
    BELOW_AVERAGE = 4
    AVERAGE = 5
    ABOVE_AVERAGE = 6
    SOMEWHAT_HIGH = 7
    HIGH = 8
    VERY_HIGH = 9
    EXCELLENT = 10


class EnergyLevel(int, Enum):
    """Energy level on a scale of 1-10."""

    EXHAUSTED = 1
    VERY_LOW = 2
    LOW = 3
    SOMEWHAT_LOW = 4
    MODERATE = 5
    SOMEWHAT_HIGH = 6
    HIGH = 7
    VERY_HIGH = 8
    ENERGETIC = 9
    PEAK = 10


class ExerciseType(str, Enum):
    """Types of exercise activities."""

    RUNNING = "running"
    WALKING = "walking"
    CYCLING = "cycling"
    SWIMMING = "swimming"
    STRENGTH_TRAINING = "strength_training"
    YOGA = "yoga"
    PILATES = "pilates"
    HIIT = "hiit"
    CARDIO = "cardio"
    SPORTS = "sports"
    OTHER = "other"


class MedicationFrequency(str, Enum):
    """Medication dosing frequency."""

    ONCE_DAILY = "once_daily"
    TWICE_DAILY = "twice_daily"
    THREE_TIMES_DAILY = "three_times_daily"
    FOUR_TIMES_DAILY = "four_times_daily"
    EVERY_OTHER_DAY = "every_other_day"
    WEEKLY = "weekly"
    AS_NEEDED = "as_needed"
    CUSTOM = "custom"


class AppointmentType(str, Enum):
    """Types of medical appointments."""

    CHECKUP = "checkup"
    FOLLOW_UP = "follow_up"
    SPECIALIST = "specialist"
    EMERGENCY = "emergency"
    DENTAL = "dental"
    VISION = "vision"
    MENTAL_HEALTH = "mental_health"
    PHYSICAL_THERAPY = "physical_therapy"
    LAB_WORK = "lab_work"
    IMAGING = "imaging"
    VACCINATION = "vaccination"
    OTHER = "other"


class AppointmentStatus(str, Enum):
    """Appointment status."""

    SCHEDULED = "scheduled"
    CONFIRMED = "confirmed"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    RESCHEDULED = "rescheduled"
    NO_SHOW = "no_show"


class GoalType(str, Enum):
    """Types of health goals."""

    WEIGHT = "weight"
    EXERCISE = "exercise"
    SLEEP = "sleep"
    WATER_INTAKE = "water_intake"
    STEPS = "steps"
    MEDITATION = "meditation"
    NUTRITION = "nutrition"
    MEDICATION_ADHERENCE = "medication_adherence"
    MOOD = "mood"
    ENERGY = "energy"
    CUSTOM = "custom"


class GoalStatus(str, Enum):
    """Goal progress status."""

    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    ON_TRACK = "on_track"
    BEHIND = "behind"
    ACHIEVED = "achieved"
    PAUSED = "paused"
    ABANDONED = "abandoned"


class SymptomSeverity(int, Enum):
    """Symptom severity on a scale of 1-10."""

    MINIMAL = 1
    VERY_MILD = 2
    MILD = 3
    MODERATELY_MILD = 4
    MODERATE = 5
    MODERATELY_SEVERE = 6
    SEVERE = 7
    VERY_SEVERE = 8
    EXTREMELY_SEVERE = 9
    UNBEARABLE = 10


class HealthMetric(BaseModel):
    """
    A single health metric record.

    Tracks daily health measurements including weight, sleep,
    exercise, mood, and other vital signs.
    """

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    id: Optional[str] = Field(None, description="Unique identifier (Notion page ID)")
    metric_date: date = Field(..., description="Date of the health measurement")
    weight: Optional[float] = Field(None, ge=50, le=500, description="Weight in pounds")
    weight_unit: str = Field("lbs", description="Weight unit (lbs or kg)")
    body_fat_percent: Optional[float] = Field(None, ge=0, le=100, description="Body fat percentage")
    sleep_hours: Optional[float] = Field(None, ge=0, le=24, description="Hours of sleep")
    sleep_quality: Optional[int] = Field(None, ge=1, le=10, description="Sleep quality rating 1-10")
    exercise_type: Optional[ExerciseType] = Field(None, description="Type of exercise")
    exercise_minutes: Optional[int] = Field(None, ge=0, description="Exercise duration in minutes")
    exercise_calories: Optional[int] = Field(
        None, ge=0, description="Calories burned during exercise"
    )
    steps: Optional[int] = Field(None, ge=0, description="Daily step count")
    mood: Optional[MoodLevel] = Field(None, description="Mood rating 1-10")
    energy: Optional[EnergyLevel] = Field(None, description="Energy level 1-10")
    water_glasses: Optional[int] = Field(None, ge=0, description="Glasses of water consumed")
    blood_pressure_systolic: Optional[int] = Field(
        None, ge=60, le=250, description="Systolic BP (mmHg)"
    )
    blood_pressure_diastolic: Optional[int] = Field(
        None, ge=40, le=150, description="Diastolic BP (mmHg)"
    )
    heart_rate: Optional[int] = Field(None, ge=30, le=220, description="Resting heart rate (bpm)")
    temperature: Optional[float] = Field(None, ge=95, le=110, description="Body temperature (F)")
    notes: Optional[str] = Field(None, max_length=2000, description="Additional notes")
    created_at: datetime = Field(
        default_factory=datetime.now, description="Record creation timestamp"
    )
    updated_at: datetime = Field(default_factory=datetime.now, description="Last update timestamp")

    @field_validator("weight_unit")
    @classmethod
    def validate_weight_unit(cls, v):
        if v not in ("lbs", "kg"):
            raise ValueError("Weight unit must be lbs or kg")
        return v


class Medication(BaseModel):
    """
    Medication record for tracking prescriptions and supplements.

    Stores medication details including dosage, frequency, and schedule.
    """

    model_config = ConfigDict(extra="forbid")

    id: Optional[str] = Field(None, description="Unique identifier (Notion page ID)")
    name: str = Field(..., min_length=1, max_length=200, description="Medication name")
    generic_name: Optional[str] = Field(None, max_length=200, description="Generic/chemical name")
    dosage: str = Field(..., min_length=1, max_length=100, description="Dosage (e.g., '100mg')")
    dosage_unit: Optional[str] = Field(None, max_length=20, description="Dosage unit")
    frequency: MedicationFrequency = Field(..., description="How often to take")
    custom_frequency: Optional[str] = Field(
        None, max_length=100, description="Custom frequency description"
    )
    times: List[str] = Field(default_factory=list, description="Times to take (HH:MM format)")
    start_date: date = Field(..., description="When medication was prescribed/started")
    end_date: Optional[date] = Field(None, description="When medication should end")
    prescribing_doctor: Optional[str] = Field(
        None, max_length=200, description="Doctor who prescribed"
    )
    pharmacy: Optional[str] = Field(None, max_length=200, description="Pharmacy name")
    prescription_number: Optional[str] = Field(
        None, max_length=50, description="Prescription/Rx number"
    )
    refill_date: Optional[date] = Field(None, description="Next refill date")
    total_refills: Optional[int] = Field(None, ge=0, description="Total refills allowed")
    remaining_refills: Optional[int] = Field(None, ge=0, description="Remaining refills")
    purpose: Optional[str] = Field(None, max_length=500, description="What it's for")
    side_effects: Optional[List[str]] = Field(
        default_factory=list, description="Known side effects"
    )
    notes: Optional[str] = Field(None, max_length=2000, description="Additional notes")
    is_active: bool = Field(True, description="Whether medication is currently active")
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)


class Appointment(BaseModel):
    """
    Medical appointment record.

    Tracks doctor visits, specialist appointments, and other medical scheduling.
    """

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    id: Optional[str] = Field(None, description="Unique identifier (Notion page ID)")
    appointment_date: date = Field(..., description="Appointment date")
    appointment_time: time = Field(..., description="Appointment time")
    doctor_name: str = Field(
        ..., min_length=1, max_length=200, description="Doctor or provider name"
    )
    doctor_type: Optional[str] = Field(None, max_length=100, description="Type of doctor/specialty")
    facility: Optional[str] = Field(None, max_length=200, description="Clinic/hospital name")
    facility_address: Optional[str] = Field(None, max_length=500, description="Facility address")
    facility_phone: Optional[str] = Field(None, max_length=20, description="Facility phone number")
    appointment_type: AppointmentType = Field(..., description="Type of appointment")
    status: AppointmentStatus = Field(AppointmentStatus.SCHEDULED, description="Current status")
    reason: Optional[str] = Field(None, max_length=500, description="Reason for visit")
    preparation_notes: Optional[str] = Field(
        None, max_length=2000, description="Preparation instructions"
    )
    questions_to_ask: Optional[List[str]] = Field(
        default_factory=list, description="Questions for doctor"
    )
    follow_up_required: bool = Field(False, description="Whether follow-up is needed")
    follow_up_date: Optional[date] = Field(None, description="Follow-up appointment date")
    notes: Optional[str] = Field(None, max_length=2000, description="Post-appointment notes")
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)


class HealthGoal(BaseModel):
    """
    Health and wellness goal.

    Tracks progress toward health objectives like weight loss,
    fitness milestones, or habit formation.
    """

    model_config = ConfigDict(extra="forbid")

    id: Optional[str] = Field(None, description="Unique identifier (Notion page ID)")
    goal_type: GoalType = Field(..., description="Category of goal")
    custom_type: Optional[str] = Field(None, max_length=100, description="Custom goal type name")
    title: str = Field(..., min_length=1, max_length=200, description="Goal title")
    description: Optional[str] = Field(None, max_length=1000, description="Detailed description")
    target_value: float = Field(..., description="Target value to achieve")
    current_value: float = Field(0, description="Current progress value")
    unit: str = Field(..., min_length=1, max_length=50, description="Unit of measurement")
    start_date: date = Field(default_factory=date.today, description="When goal was set")
    target_date: date = Field(..., description="Deadline for achieving goal")
    status: GoalStatus = Field(GoalStatus.NOT_STARTED, description="Current goal status")
    progress_percent: float = Field(0, ge=0, le=100, description="Percentage complete")
    milestones: List[Dict[str, Any]] = Field(default_factory=list, description="Key milestones")
    rewards: Optional[List[str]] = Field(default_factory=list, description="Reward system")
    notes: Optional[str] = Field(None, max_length=2000, description="Additional notes")
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)

    def calculate_progress(self) -> float:
        """Calculate and update progress percentage."""
        if self.target_value == 0:
            return 0
        progress = (self.current_value / self.target_value) * 100
        return min(max(progress, 0), 100)


class SymptomLog(BaseModel):
    """
    Symptom tracking record.

    Logs symptoms with severity, duration, and contextual information
    for pattern analysis and medical consultations.
    """

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    id: Optional[str] = Field(None, description="Unique identifier (Notion page ID)")
    symptom_date: date = Field(..., description="Date symptom was experienced")
    symptom_time: Optional[time] = Field(None, description="Time symptom started")
    symptom_name: str = Field(..., min_length=1, max_length=200, description="Symptom name")
    body_part: Optional[str] = Field(None, max_length=100, description="Affected body part")
    severity: SymptomSeverity = Field(..., description="Severity rating 1-10")
    duration_minutes: Optional[int] = Field(None, ge=0, description="Duration in minutes")
    duration_description: Optional[str] = Field(
        None, max_length=100, description="Duration description"
    )
    triggers: Optional[List[str]] = Field(default_factory=list, description="Identified triggers")
    remedies_tried: Optional[List[str]] = Field(default_factory=list, description="What helped")
    was_medications_taken: bool = Field(False, description="Whether medication was taken")
    medications_taken: Optional[List[str]] = Field(
        default_factory=list, description="Medications taken"
    )
    impact_on_daily: Optional[int] = Field(
        None, ge=1, le=10, description="Impact on daily life 1-10"
    )
    mood_at_time: Optional[MoodLevel] = Field(None, description="Mood when symptom occurred")
    associated_symptoms: Optional[List[str]] = Field(
        default_factory=list, description="Related symptoms"
    )
    notes: Optional[str] = Field(None, max_length=2000, description="Additional notes")
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)


class HealthSummary(BaseModel):
    """
    Aggregated health summary for a time period.

    Provides statistical summaries and trends for health metrics.
    """

    model_config = ConfigDict(extra="forbid")

    period_start: date = Field(..., description="Start date of summary period")
    period_end: date = Field(..., description="End date of summary period")
    total_days: int = Field(..., description="Number of days in period")

    # Weight statistics
    avg_weight: Optional[float] = Field(None, description="Average weight")
    weight_change: Optional[float] = Field(None, description="Weight change in period")
    min_weight: Optional[float] = Field(None, description="Minimum weight")
    max_weight: Optional[float] = Field(None, description="Maximum weight")

    # Sleep statistics
    avg_sleep_hours: Optional[float] = Field(None, description="Average sleep hours")
    avg_sleep_quality: Optional[float] = Field(None, description="Average sleep quality")
    total_sleep_hours: Optional[float] = Field(None, description="Total sleep hours")

    # Exercise statistics
    total_exercise_minutes: Optional[int] = Field(None, description="Total exercise minutes")
    total_exercise_days: Optional[int] = Field(None, description="Days with exercise")
    total_calories_burned: Optional[int] = Field(None, description="Total calories burned")
    total_steps: Optional[int] = Field(None, description="Total steps")
    avg_daily_steps: Optional[float] = Field(None, description="Average daily steps")

    # Mood and energy
    avg_mood: Optional[float] = Field(None, description="Average mood")
    avg_energy: Optional[float] = Field(None, description="Average energy")

    # Health metrics
    avg_systolic: Optional[float] = Field(None, description="Average systolic BP")
    avg_diastolic: Optional[float] = Field(None, description="Average diastolic BP")
    avg_heart_rate: Optional[float] = Field(None, description="Average heart rate")

    # Streaks
    exercise_streak: Optional[int] = Field(None, description="Current exercise streak")
    sleep_streak: Optional[int] = Field(None, description="Days meeting sleep goal")
    water_streak: Optional[int] = Field(None, description="Days meeting water goal")

    # Medication adherence
    medication_adherence: Optional[float] = Field(None, description="Medication adherence %")


class AIInsight(BaseModel):
    """
    AI-generated health insight.

    Contains actionable recommendations and analysis based on health data.
    """

    model_config = ConfigDict(extra="forbid")

    id: Optional[str] = Field(None, description="Unique identifier")
    generated_at: datetime = Field(
        default_factory=datetime.now, description="When insight was generated"
    )
    category: str = Field(..., description="Category (sleep, weight, exercise, etc.)")
    title: str = Field(..., min_length=1, max_length=200, description="Insight title")
    summary: str = Field(..., min_length=1, max_length=500, description="Brief summary")
    details: str = Field(..., description="Detailed analysis")
    recommendations: List[str] = Field(
        default_factory=list, description="Actionable recommendations"
    )
    data_points_used: int = Field(0, description="Number of data points analyzed")
    confidence_score: float = Field(0.5, ge=0, le=1, description="Confidence level")
    is_actionable: bool = Field(True, description="Whether action can be taken")
    priority: int = Field(1, ge=1, le=5, description="Priority level 1-5 (1 highest)")
    valid_until: Optional[datetime] = Field(None, description="When insight expires")
    tags: List[str] = Field(default_factory=list, description="Tags for categorization")


class BrainAnalysisSession(BaseModel):
    """
    A single TRIBEv2 brain analysis session.

    Records brain activity predictions derived from health data narratives,
    capturing cognitive load, stress, and neural network activations.
    """

    model_config = ConfigDict(extra="forbid")

    id: Optional[str] = Field(None, description="Unique identifier (Notion page ID)")
    session_date: date = Field(default_factory=date.today, description="Date of analysis")
    session_time: Optional[time] = Field(None, description="Time of analysis")
    input_type: str = Field(
        "health_narrative", description="Input used: health_narrative, symptom_text, activity"
    )
    input_summary: str = Field(..., description="Brief description of the health data analysed")

    # TRIBEv2 neural prediction metrics
    mean_activation: Optional[float] = Field(
        None, description="Mean brain activation across ~20k vertices"
    )
    activation_variance: Optional[float] = Field(None, description="Variance in activation")
    cognitive_load_score: Optional[float] = Field(
        None, ge=0, le=10, description="Estimated cognitive load 0-10"
    )
    stress_indicator: Optional[float] = Field(
        None, ge=0, le=10, description="Estimated stress level 0-10"
    )
    emotional_valence: Optional[float] = Field(
        None, ge=-1, le=1, description="Emotional tone: -1 (negative) to +1 (positive)"
    )

    # Brain network activations
    default_mode_activity: Optional[float] = Field(
        None, ge=0, le=1, description="Default mode network 0-1"
    )
    executive_control_activity: Optional[float] = Field(
        None, ge=0, le=1, description="Executive control network 0-1"
    )
    salience_network_activity: Optional[float] = Field(
        None, ge=0, le=1, description="Salience/stress detection network 0-1"
    )

    insights: Optional[str] = Field(None, description="AI-generated cognitive insights")
    recommendations: List[str] = Field(
        default_factory=list, description="Actionable recommendations"
    )
    health_context: Optional[str] = Field(None, description="Health data snapshot used")
    tribe_version: str = Field("v2", description="TRIBEv2 model version")
    model_available: bool = Field(
        False, description="True if TRIBEv2 was directly run, False if AI fallback used"
    )
    created_at: datetime = Field(default_factory=datetime.now)


class CognitiveInsight(BaseModel):
    """
    Aggregated cognitive health insight across multiple TRIBEv2 sessions.

    Captures trends, correlations between brain activity and health data,
    and high-level recommendations for cognitive wellness.
    """

    model_config = ConfigDict(extra="forbid")

    id: Optional[str] = Field(None)
    generated_at: datetime = Field(default_factory=datetime.now)
    analysis_period_days: int = Field(7, description="Days of sessions analysed")
    sessions_analyzed: int = Field(0)
    data_points_used: int = Field(0)

    avg_cognitive_load: Optional[float] = Field(
        None, description="Mean cognitive load across sessions"
    )
    avg_stress_indicator: Optional[float] = Field(
        None, description="Mean stress level across sessions"
    )
    cognitive_load_trend: Optional[str] = Field(None, description="improving / stable / worsening")
    stress_trend: Optional[str] = Field(None, description="improving / stable / worsening")

    # Cross-domain correlations (Pearson r, range -1 to +1)
    sleep_cognition_correlation: Optional[float] = Field(
        None, description="Sleep quality vs cognitive load"
    )
    mood_brain_correlation: Optional[float] = Field(None, description="Mood vs emotional valence")
    exercise_cognition_correlation: Optional[float] = Field(
        None, description="Exercise vs cognitive load"
    )

    key_findings: List[str] = Field(default_factory=list)
    recommendations: List[str] = Field(default_factory=list)


class MCPCallResult(BaseModel):
    """Result from an MCP tool call."""

    model_config = ConfigDict(extra="forbid")

    success: bool = Field(..., description="Whether the operation succeeded")
    message: str = Field(..., description="Result message")
    data: Optional[Dict[str, Any]] = Field(None, description="Result data")
    error: Optional[str] = Field(None, description="Error message if failed")
