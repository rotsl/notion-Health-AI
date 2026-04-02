# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Rohan R
# Author: Rohan R

"""Tests for API-layer duration guidance and TRIBEv2 text simulation."""

from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from notion_health_ai.api import (
    BrainTextRequest,
    _infer_text_duration_profile,
    _resolve_duration_minutes,
    app,
    brain_from_text,
)
from notion_health_ai.tribe_integration import TribeHealthAnalyzer


class TestDurationGuidance:
    """Test duration inference and validation for TRIBEv2 text analysis."""

    def test_infer_text_duration_profile_for_sleep_text(self):
        """Sleep-like text should produce sleep-range guidance."""
        profile = _infer_text_duration_profile("I slept for 8 hours and woke up refreshed.")

        assert profile["key"] == "sleep"
        assert profile["suggested_minutes"] == 480
        assert profile["min"] == 240
        assert profile["max"] == 720

    @pytest.mark.asyncio
    async def test_resolve_duration_minutes_rejects_unrealistic_activity_value(self):
        """Activity-derived custom durations should stay inside realistic bounds."""
        with pytest.raises(HTTPException) as exc:
            await _resolve_duration_minutes(
                text="I went for a short run before work.",
                notion_key=None,
                context_days=7,
                duration_minutes=500,
                duration_source="activity",
            )

        assert exc.value.status_code == 422
        assert "realistic duration" in str(exc.value.detail)

    @pytest.mark.asyncio
    async def test_resolve_duration_minutes_accepts_connected_dataset_value(self):
        """Dataset-backed duration must pass when it matches an available connected value."""
        with patch(
            "notion_health_ai.api._get_dataset_duration_options",
            new=AsyncMock(
                return_value=[
                    {
                        "minutes": 45,
                        "label": "45 min exercise (2026-04-02)",
                        "source": "health_metric",
                        "date": "2026-04-02",
                    }
                ]
            ),
        ):
            result = await _resolve_duration_minutes(
                text="I exercised and felt focused afterwards.",
                notion_key="test-key",
                context_days=7,
                duration_minutes=45,
                duration_source="dataset",
            )

        assert result["minutes"] == 45
        assert result["source"] == "dataset"


class TestBrainTextDurationFlow:
    """Test that the text endpoint uses the chosen duration in TRIBEv2 prompts."""

    @pytest.mark.asyncio
    async def test_brain_from_text_includes_duration_in_text_stimulus(self):
        """The API should carry the selected duration into the TRIBEv2 text prompt."""
        analyzer = TribeHealthAnalyzer(use_model=False)
        analyzer.model_wrapper.predict_response = AsyncMock(
            return_value={
                "prefrontal_cortex": {"activation": 0.8, "type": "excitatory"},
                "motor_cortex": {"activation": 0.6, "type": "excitatory"},
            }
        )
        analyzer._generate_visualizations = AsyncMock(return_value={})

        with patch(
            "notion_health_ai.api._ensure_tribe_ready",
            new=AsyncMock(return_value=analyzer),
        ):
            with patch(
                "notion_health_ai.api._build_tribe_health_context",
                new=AsyncMock(return_value=None),
            ):
                result = await brain_from_text(
                    BrainTextRequest(
                        text="I did a focused 45 minute study session and felt sharp.",
                        duration_minutes=45,
                        duration_source="activity",
                    ),
                    None,
                    None,
                )

        text_stimulus = analyzer.model_wrapper.predict_response.await_args.kwargs["text_stimulus"]
        assert "45 minute" in text_stimulus
        assert result["duration_minutes"] == 45
        assert result["duration_source"] == "activity"
        viz_kwargs = analyzer._generate_visualizations.await_args.kwargs
        assert "Focused work or learning text simulation" in viz_kwargs["title"]
        assert viz_kwargs["duration_minutes"] == 45


class TestVisualizationCleanup:
    """Test clearing generated visualization outputs."""

    def test_clear_visualizations_keeps_gitkeep(self, tmp_path):
        """The clear endpoint should remove generated files but keep the placeholder."""
        viz_dir = tmp_path / "visualizations"
        viz_dir.mkdir()
        (viz_dir / ".gitkeep").write_text("")
        (viz_dir / "brain.html").write_text("test")
        (viz_dir / "brain.png").write_text("test")
        (viz_dir / "notes.txt").write_text("leave me")

        with patch("notion_health_ai.api.VIZ_DIR", Path(viz_dir)):
            client = TestClient(app)
            response = client.post("/api/visualizations/clear")

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["deleted_count"] == 2
        assert (viz_dir / ".gitkeep").exists()
        assert not (viz_dir / "brain.html").exists()
        assert not (viz_dir / "brain.png").exists()
        assert (viz_dir / "notes.txt").exists()
