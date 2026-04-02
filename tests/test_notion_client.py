# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Rohan R
# Author: Rohan R

"""Tests for Notion client parsing behavior."""

from datetime import date
from unittest.mock import AsyncMock

import httpx
import pytest

from notion_health_ai.models import SymptomLog, SymptomSeverity
from notion_health_ai.notion_client import NotionClient


class TestNotionClient:
    """Test parsing of Notion API payloads."""

    def test_parse_health_metric_handles_null_property_objects(self):
        """Null property payloads from Notion should not break metric parsing."""
        client = NotionClient(api_key="test_key")
        page = {
            "id": "page_123",
            "properties": {
                "Date": {"date": {"start": "2026-04-02"}},
                "Weight": {"number": 172.5},
                "Sleep Hours": {"number": 7.0},
                "Sleep Quality": None,
                "Exercise Type": None,
                "Exercise Minutes": {"number": 30},
                "Notes": None,
            },
        }

        metric = client._parse_health_metric(page)

        assert metric is not None
        assert metric.id == "page_123"
        assert metric.weight == 172.5
        assert metric.sleep_hours == 7.0
        assert metric.exercise_minutes == 30
        assert metric.sleep_quality is None
        assert metric.exercise_type is None
        assert metric.notes is None

    @pytest.mark.asyncio
    async def test_create_symptom_falls_back_to_legacy_duration_property(self):
        """Symptom writes should retry with the legacy Duration property when needed."""
        client = NotionClient(api_key="test_key")
        symptom = SymptomLog(
            symptom_date=date(2026, 4, 1),
            symptom_name="Eye Strain",
            severity=SymptomSeverity(2),
            duration_minutes=30,
        )
        first_response = httpx.Response(
            400,
            request=httpx.Request("POST", "https://api.notion.com/v1/pages"),
            text=(
                '{"object":"error","status":400,"code":"validation_error",'
                '"message":"Duration (min) is not a property that exists."}'
            ),
        )
        client._request = AsyncMock(
            side_effect=[
                httpx.HTTPStatusError(
                    "bad request",
                    request=first_response.request,
                    response=first_response,
                ),
                {"id": "symptom_page_id"},
            ]
        )

        page_id = await client.create_symptom("db_123", symptom)

        assert page_id == "symptom_page_id"
        assert client._request.await_count == 2
        retry_payload = client._request.await_args_list[1].args[2]
        assert "Duration" in retry_payload["properties"]
        assert "Duration (min)" not in retry_payload["properties"]
