"""
Pytest configuration for NotionHealth AI tests.
"""

import pytest
import os
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

# Add src directory to path for imports
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))


@pytest.fixture(scope="session", autouse=True)
def setup_test_environment():
    """Set up test environment variables."""
    os.environ.setdefault("NOTION_API_KEY", "test_notion_key")
    os.environ.setdefault("ANTHROPIC_API_KEY", "test_anthropic_key")
    os.environ.setdefault("OPENAI_API_KEY", "test_openai_key")
    os.environ.setdefault("NOTION_HEALTH_DATABASE_ID", "test_health_db")
    os.environ.setdefault("NOTION_MEDICATION_DATABASE_ID", "test_med_db")
    os.environ.setdefault("NOTION_APPOINTMENT_DATABASE_ID", "test_appt_db")
    os.environ.setdefault("NOTION_GOALS_DATABASE_ID", "test_goals_db")
    os.environ.setdefault("NOTION_SYMPTOMS_DATABASE_ID", "test_symptoms_db")

    yield

    # Cleanup if needed


@pytest.fixture
def mock_notion_response():
    """Create a mock Notion API response."""
    return {
        "id": "test_page_id",
        "created_time": "2026-03-29T10:00:00Z",
        "last_edited_time": "2026-03-29T10:00:00Z",
        "properties": {
            "Date": {"date": {"start": "2026-03-29"}},
            "Weight": {"number": 165},
            "Mood": {"number": 8},
            "Energy": {"number": 7},
            "Sleep Hours": {"number": 7.5},
        },
    }


@pytest.fixture
def mock_anthropic_response():
    """Create a mock Anthropic API response."""
    return {
        "content": [
            {
                "type": "text",
                "text": (
                    '{"title": "Test Insight", "summary": "Test summary",'
                    ' "recommendations": ["Do this"]}'
                ),
            }
        ]
    }


@pytest.fixture
def mock_health_metrics():
    """Create mock health metrics data."""
    from datetime import date
    from notion_health_ai.models import HealthMetric, MoodLevel, EnergyLevel

    return [
        HealthMetric(
            id="metric_1",
            metric_date=date(2026, 3, 29),
            weight=165.5,
            sleep_hours=7.5,
            sleep_quality=8,
            mood=MoodLevel(8),
            energy=EnergyLevel(7),
            exercise_minutes=30,
            steps=8000,
            water_glasses=6,
        ),
        HealthMetric(
            id="metric_2",
            metric_date=date(2026, 3, 28),
            weight=166.0,
            sleep_hours=6.5,
            sleep_quality=6,
            mood=MoodLevel(6),
            energy=EnergyLevel(5),
            exercise_minutes=45,
            steps=10000,
            water_glasses=8,
        ),
    ]


@pytest.fixture
def mock_health_summary():
    """Create a mock health summary."""
    from datetime import date
    from notion_health_ai.models import HealthSummary

    return HealthSummary(
        period_start=date(2026, 3, 22),
        period_end=date(2026, 3, 29),
        total_days=7,
        avg_weight=165.75,
        weight_change=-0.5,
        avg_sleep_hours=7.0,
        avg_sleep_quality=7.0,
        total_exercise_minutes=75,
        total_exercise_days=2,
        total_steps=18000,
        avg_daily_steps=9000.0,
        avg_mood=7.0,
        avg_energy=6.0,
    )


@pytest.fixture
def mock_httpx_client():
    """Create a mock httpx AsyncClient."""
    mock_client = AsyncMock()
    mock_response = MagicMock()
    mock_response.json.return_value = {"id": "test_id", "results": []}
    mock_response.raise_for_status = MagicMock()
    mock_client.get.return_value = mock_response
    mock_client.post.return_value = mock_response
    mock_client.patch.return_value = mock_response
    return mock_client


@pytest.fixture
def mock_notion_client(mock_httpx_client):
    """Create a mock NotionClient with pre-configured responses."""
    with patch("notion_health_ai.notion_client.httpx.AsyncClient") as mock_async_client:
        mock_async_client.return_value = mock_httpx_client
        from notion_health_ai.notion_client import NotionClient

        client = NotionClient(api_key="test_key")
        yield client


@pytest.fixture
def health_manager():
    """Create a HealthManager instance for testing."""
    from notion_health_ai.health_manager import HealthManager

    return HealthManager(
        notion_api_key="test_key",
        health_database_id="test_health_db",
        medication_database_id="test_med_db",
        appointment_database_id="test_appt_db",
        goals_database_id="test_goals_db",
        symptoms_database_id="test_symptoms_db",
    )


@pytest.fixture
def tribe_analyzer():
    """Create a TribeHealthAnalyzer instance for testing."""
    from notion_health_ai.tribe_integration import TribeHealthAnalyzer

    return TribeHealthAnalyzer(use_model=False)
