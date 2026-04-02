"""
Tests for MCP Server module.
"""

from datetime import date, timedelta
from unittest.mock import AsyncMock, patch

import pytest

from notion_health_ai.models import MCPCallResult
from notion_health_ai.server import NotionHealthMCPServer


class TestNotionHealthMCPServer:
    """Test cases for the MCP server."""

    @pytest.fixture
    def server(self):
        """Create a server instance for testing."""
        with patch.dict(
            "os.environ",
            {
                "NOTION_API_KEY": "test_key",
                "NOTION_HEALTH_DATABASE_ID": "test_health_db",
                "NOTION_MEDICATION_DATABASE_ID": "test_med_db",
                "NOTION_APPOINTMENT_DATABASE_ID": "test_appt_db",
                "NOTION_GOALS_DATABASE_ID": "test_goals_db",
                "NOTION_SYMPTOMS_DATABASE_ID": "test_symptoms_db",
            },
        ):
            return NotionHealthMCPServer()

    def test_server_initialization(self, server):
        """Test server initialization."""
        assert server is not None
        assert server.health_manager is not None

    @pytest.mark.asyncio
    async def test_handle_log_health_metric(self, server):
        """Test handling log_health_metric tool call."""
        mock_result = MCPCallResult(
            success=True,
            message="Successfully logged health metric",
            data={"page_id": "test_id"},
        )
        server.health_manager.log_health_metric = AsyncMock(return_value=mock_result)

        arguments = {
            "weight": 165.5,
            "sleep_hours": 7.5,
            "mood": 8,
            "energy": 7,
        }

        result = await server.handle_log_health_metric(arguments)

        assert result["success"] is True
        assert "Successfully logged" in result["message"]

    @pytest.mark.asyncio
    async def test_handle_add_medication(self, server):
        """Test handling add_medication tool call."""
        mock_result = MCPCallResult(
            success=True,
            message="Successfully added medication: Aspirin",
        )
        server.health_manager.add_medication = AsyncMock(return_value=mock_result)

        arguments = {
            "name": "Aspirin",
            "dosage": "100mg",
            "frequency": "once_daily",
            "start_date": date.today().isoformat(),
        }

        result = await server.handle_add_medication(arguments)

        assert result["success"] is True

    @pytest.mark.asyncio
    async def test_handle_get_health_summary(self, server):
        """Test handling get_health_summary tool call."""
        from notion_health_ai.models import HealthSummary

        mock_summary = HealthSummary(
            period_start=date.today() - timedelta(days=7),
            period_end=date.today(),
            total_days=7,
            avg_weight=165.0,
            avg_sleep_hours=7.0,
            avg_mood=7.5,
            avg_energy=6.5,
        )

        server.health_manager.get_health_summary = AsyncMock(return_value=mock_summary)

        arguments = {
            "start_date": (date.today() - timedelta(days=7)).isoformat(),
            "end_date": date.today().isoformat(),
        }

        result = await server.handle_get_health_summary(arguments)

        assert result["avg_weight"] == 165.0
        assert result["total_days"] == 7

    @pytest.mark.asyncio
    async def test_handle_add_appointment(self, server):
        """Test handling add_appointment tool call."""
        mock_result = MCPCallResult(
            success=True,
            message="Successfully added appointment",
        )
        server.health_manager.add_appointment = AsyncMock(return_value=mock_result)

        arguments = {
            "date": (date.today() + timedelta(days=7)).isoformat(),
            "time": "10:00",
            "doctor_name": "Dr. Smith",
            "appointment_type": "checkup",
        }

        result = await server.handle_add_appointment(arguments)

        assert result["success"] is True

    @pytest.mark.asyncio
    async def test_handle_set_health_goal(self, server):
        """Test handling set_health_goal tool call."""
        mock_result = MCPCallResult(
            success=True,
            message="Successfully set goal",
        )
        server.health_manager.set_health_goal = AsyncMock(return_value=mock_result)

        arguments = {
            "goal_type": "weight",
            "title": "Lose 10 pounds",
            "target_value": 155,
            "unit": "lbs",
            "target_date": (date.today() + timedelta(days=30)).isoformat(),
        }

        result = await server.handle_set_health_goal(arguments)

        assert result["success"] is True

    @pytest.mark.asyncio
    async def test_handle_log_symptom(self, server):
        """Test handling log_symptom tool call."""
        mock_result = MCPCallResult(
            success=True,
            message="Successfully logged symptom: Headache",
        )
        server.health_manager.log_symptom = AsyncMock(return_value=mock_result)

        arguments = {
            "symptom_name": "Headache",
            "severity": 6,
            "body_part": "Head",
        }

        result = await server.handle_log_symptom(arguments)

        assert result["success"] is True

    def test_list_tools(self, server):
        """Test that tools are properly listed."""
        tools = server.list_tools()

        tool_names = [t.name for t in tools]

        assert "log_health_metric" in tool_names
        assert "add_medication" in tool_names
        assert "add_appointment" in tool_names
        assert "set_health_goal" in tool_names
        assert "log_symptom" in tool_names
        assert "get_health_summary" in tool_names
