"""
Tests for Health Manager module.
"""

import pytest
from datetime import date, timedelta
from unittest.mock import AsyncMock, patch, MagicMock

from notion_health_ai.health_manager import HealthManager
from notion_health_ai.models import (
    HealthMetric,
    Medication,
    Appointment,
    HealthGoal,
    SymptomLog,
    MCPCallResult,
    MoodLevel,
    EnergyLevel,
    ExerciseType,
    MedicationFrequency,
    AppointmentType,
    AppointmentStatus,
    GoalType,
    GoalStatus,
    SymptomSeverity,
)


class TestHealthManager:
    """Test cases for HealthManager class."""

    def test_initialization(self, health_manager):
        """Test HealthManager initialization."""
        assert health_manager.notion_api_key == "test_key"
        assert health_manager.health_database_id == "test_health_db"
        assert health_manager.medication_database_id == "test_med_db"
        assert health_manager.appointment_database_id == "test_appt_db"
        assert health_manager.goals_database_id == "test_goals_db"
        assert health_manager.symptoms_database_id == "test_symptoms_db"

    @pytest.mark.asyncio
    async def test_log_health_metric_success(self, health_manager, mock_notion_client):
        """Test successful health metric logging."""
        health_manager.client = mock_notion_client
        mock_notion_client.create_health_metric = AsyncMock(return_value="test_page_id")

        result = await health_manager.log_health_metric(
            weight=165.5,
            sleep_hours=7.5,
            mood=8,
            energy=7,
        )

        assert result.success is True
        assert "Successfully logged" in result.message
        assert result.data["page_id"] == "test_page_id"

    @pytest.mark.asyncio
    async def test_log_health_metric_with_exercise(self, health_manager, mock_notion_client):
        """Test health metric logging with exercise data."""
        health_manager.client = mock_notion_client
        mock_notion_client.create_health_metric = AsyncMock(return_value="test_page_id")

        result = await health_manager.log_health_metric(
            exercise_type="running",
            exercise_minutes=45,
            steps=10000,
        )

        assert result.success is True

    @pytest.mark.asyncio
    async def test_add_medication_success(self, health_manager, mock_notion_client):
        """Test successful medication addition."""
        health_manager.client = mock_notion_client
        mock_notion_client.create_medication = AsyncMock(return_value="med_page_id")

        result = await health_manager.add_medication(
            name="Aspirin",
            dosage="100mg",
            frequency="once_daily",
            start_date=date.today(),
            purpose="Pain relief",
        )

        assert result.success is True
        assert "Aspirin" in result.message

    @pytest.mark.asyncio
    async def test_add_appointment_success(self, health_manager, mock_notion_client):
        """Test successful appointment addition."""
        health_manager.client = mock_notion_client
        mock_notion_client.create_appointment = AsyncMock(return_value="appt_page_id")

        result = await health_manager.add_appointment(
            appointment_date=date.today() + timedelta(days=7),
            appointment_time="10:00",
            doctor_name="Dr. Smith",
            appointment_type="checkup",
            facility="City Hospital",
        )

        assert result.success is True

    @pytest.mark.asyncio
    async def test_set_health_goal_success(self, health_manager, mock_notion_client):
        """Test successful goal setting."""
        health_manager.client = mock_notion_client
        mock_notion_client.create_goal = AsyncMock(return_value="goal_page_id")

        result = await health_manager.set_health_goal(
            goal_type="weight",
            title="Lose 10 pounds",
            target_value=155,
            unit="lbs",
            target_date=date.today() + timedelta(days=30),
        )

        assert result.success is True

    @pytest.mark.asyncio
    async def test_log_symptom_success(self, health_manager, mock_notion_client):
        """Test successful symptom logging."""
        health_manager.client = mock_notion_client
        mock_notion_client.create_symptom = AsyncMock(return_value="symptom_page_id")

        result = await health_manager.log_symptom(
            symptom_name="Headache",
            severity=6,
            body_part="Head",
            duration_minutes=30,
        )

        assert result.success is True
        assert "Headache" in result.message

    @pytest.mark.asyncio
    async def test_get_health_metrics(self, health_manager, mock_notion_client, mock_health_metrics):
        """Test retrieving health metrics."""
        health_manager.client = mock_notion_client
        mock_notion_client.query_health_metrics = AsyncMock(return_value=mock_health_metrics)

        metrics = await health_manager.get_health_metrics(
            start_date=date.today() - timedelta(days=7),
            end_date=date.today(),
        )

        assert len(metrics) == 2
        assert metrics[0].weight == 165.5

    @pytest.mark.asyncio
    async def test_get_health_summary(self, health_manager, mock_notion_client, mock_health_metrics):
        """Test generating health summary."""
        health_manager.client = mock_notion_client
        mock_notion_client.query_health_metrics = AsyncMock(return_value=mock_health_metrics)

        summary = await health_manager.get_health_summary(
            start_date=date.today() - timedelta(days=7),
            end_date=date.today(),
        )

        assert summary.total_days > 0
        assert summary.avg_weight is not None


class TestHealthModels:
    """Test cases for health data models."""

    def test_health_metric_creation(self):
        """Test HealthMetric model creation."""
        metric = HealthMetric(
            metric_date=date.today(),
            weight=165.5,
            sleep_hours=7.5,
            mood=MoodLevel(8),
            energy=EnergyLevel(7),
        )

        assert metric.weight == 165.5
        assert metric.sleep_hours == 7.5
        assert metric.mood.value == 8

    def test_medication_creation(self):
        """Test Medication model creation."""
        med = Medication(
            name="Test Med",
            dosage="50mg",
            frequency=MedicationFrequency.ONCE_DAILY,
            start_date=date.today(),
            is_active=True,
        )

        assert med.name == "Test Med"
        assert med.frequency == MedicationFrequency.ONCE_DAILY

    def test_appointment_creation(self):
        """Test Appointment model creation."""
        from datetime import time
        
        appt = Appointment(
            appointment_date=date.today(),
            appointment_time=time(10, 30),
            doctor_name="Dr. Test",
            appointment_type=AppointmentType.CHECKUP,
            status=AppointmentStatus.SCHEDULED,
        )

        assert appt.doctor_name == "Dr. Test"
        assert appt.appointment_type == AppointmentType.CHECKUP

    def test_health_goal_creation(self):
        """Test HealthGoal model creation."""
        goal = HealthGoal(
            title="Test Goal",
            goal_type=GoalType.WEIGHT,
            target_value=150,
            current_value=155,
            unit="lbs",
            target_date=date.today() + timedelta(days=30),
            status=GoalStatus.IN_PROGRESS,
        )

        assert goal.title == "Test Goal"
        assert goal.progress_percent == 0  # default; calculate_progress() must be called explicitly

    def test_symptom_log_creation(self):
        """Test SymptomLog model creation."""
        symptom = SymptomLog(
            symptom_date=date.today(),
            symptom_name="Test Symptom",
            severity=SymptomSeverity(7),
            body_part="Test Area",
        )

        assert symptom.symptom_name == "Test Symptom"
        assert symptom.severity.value == 7
