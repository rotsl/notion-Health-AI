"""
Tests for AI Insights and Tribe v2 Integration modules.
"""

import pytest
from datetime import date, timedelta
from unittest.mock import AsyncMock, patch, MagicMock

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


class TestAIInsightsEngine:
    """Test cases for AIInsightsEngine class."""

    @pytest.fixture
    def engine(self):
        """Create an AI insights engine for testing."""
        with patch.dict("os.environ", {
            "ANTHROPIC_API_KEY": "test_key",
            "OPENAI_API_KEY": "test_key",
        }):
            return AIInsightsEngine()

    def test_initialization(self, engine):
        """Test engine initialization."""
        assert engine is not None
        assert engine.anthropic_api_key == "test_key"
        assert engine.openai_api_key == "test_key"

    @pytest.mark.asyncio
    async def test_generate_health_insights(self, engine, mock_health_metrics, mock_health_summary):
        """Test generating health insights."""
        insights = await engine.generate_health_insights(
            mock_health_metrics,
            mock_health_summary,
            category="overall"
        )

        # Should return a list of insights
        assert isinstance(insights, list)

    @pytest.mark.asyncio
    async def test_generate_category_specific_insights(self, engine, mock_health_metrics, mock_health_summary):
        """Test generating category-specific insights."""
        for category in ["weight", "sleep", "exercise", "mood"]:
            insights = await engine.generate_health_insights(
                mock_health_metrics,
                mock_health_summary,
                category=category
            )
            assert isinstance(insights, list)


class TestTribeModelWrapper:
    """Test cases for TribeModelWrapper class."""

    def test_initialization(self):
        """Test wrapper initialization."""
        wrapper = TribeModelWrapper(device="cpu")
        assert wrapper.device == "cpu"
        assert wrapper.is_loaded is False

    @pytest.mark.asyncio
    async def test_simulate_prediction(self):
        """Test simulated prediction."""
        wrapper = TribeModelWrapper(device="cpu")
        
        prediction = wrapper._simulate_prediction("exercise")
        
        assert "prefrontal_cortex" in prediction
        assert "motor_cortex" in prediction
        assert prediction["motor_cortex"]["activation"] > 0.5

    @pytest.mark.asyncio
    async def test_simulate_prediction_meditation(self):
        """Test simulated prediction for meditation."""
        wrapper = TribeModelWrapper(device="cpu")
        
        prediction = wrapper._simulate_prediction("meditation")
        
        assert prediction["amygdala"]["type"] == "inhibitory"
        assert prediction["prefrontal_cortex"]["activation"] > 0.7

    @pytest.mark.asyncio
    async def test_predict_response(self):
        """Test predict_response method."""
        wrapper = TribeModelWrapper(device="cpu")
        
        result = await wrapper.predict_response(text_stimulus="running exercise")
        
        assert isinstance(result, dict)
        assert len(result) > 0


class TestTribeHealthAnalyzer:
    """Test cases for TribeHealthAnalyzer class."""

    @pytest.fixture
    def analyzer(self):
        """Create an analyzer instance for testing."""
        return TribeHealthAnalyzer(use_model=False)

    def test_initialization(self, analyzer):
        """Test analyzer initialization."""
        assert analyzer is not None
        assert analyzer.activity_patterns is not None
        assert len(analyzer.activity_patterns) > 0

    def test_activity_patterns_loaded(self, analyzer):
        """Test that activity patterns are loaded."""
        assert ActivityType.EXERCISE in analyzer.activity_patterns
        assert ActivityType.MEDITATION in analyzer.activity_patterns
        assert ActivityType.SLEEP in analyzer.activity_patterns

    @pytest.mark.asyncio
    async def test_predict_brain_response(self, analyzer):
        """Test brain response prediction."""
        result = await analyzer.predict_brain_response("exercise", duration_minutes=30)

        assert result["success"] is True
        assert "prediction" in result
        assert result["prediction"]["activity"] == "exercise"
        assert "brain_regions" in result["prediction"]
        assert "wellness_score" in result["prediction"]
        assert len(result["prediction"]["benefits"]) > 0

    @pytest.mark.asyncio
    async def test_predict_brain_response_meditation(self, analyzer):
        """Test meditation brain response prediction."""
        result = await analyzer.predict_brain_response("meditation", duration_minutes=20)

        assert result["success"] is True
        assert "amygdala" in result["prediction"]["brain_regions"]
        # Amygdala should be inhibitory for meditation
        assert result["prediction"]["brain_regions"]["amygdala"]["response"] == "inhibitory"

    @pytest.mark.asyncio
    async def test_predict_brain_response_invalid_activity(self, analyzer):
        """Test prediction with invalid activity."""
        result = await analyzer.predict_brain_response("invalid_activity")

        assert result["success"] is False
        assert "error" in result

    @pytest.mark.asyncio
    async def test_analyze_brain_health_correlation(self, analyzer, mock_health_metrics, mock_health_summary):
        """Test brain-health correlation analysis."""
        result = await analyzer.analyze_brain_health_correlation(
            mock_health_metrics,
            mock_health_summary
        )

        assert result["success"] is True
        assert "analysis" in result
        assert "correlations" in result["analysis"]
        assert "recommendations" in result["analysis"]
        assert len(result["analysis"]["correlations"]) > 0

    @pytest.mark.asyncio
    async def test_generate_optimal_schedule(self, analyzer):
        """Test optimal schedule generation."""
        result = await analyzer.generate_optimal_schedule()

        assert result["success"] is True
        assert "schedule" in result
        assert "slots" in result["schedule"]
        assert len(result["schedule"]["slots"]) > 0

        # Check that sleep is prioritized
        sleep_slot = next(
            (s for s in result["schedule"]["slots"] if "Sleep" in s["activity"]),
            None
        )
        assert sleep_slot is not None
        assert sleep_slot["priority"] in ("Critical", "High")


class TestBrainResponseModel:
    """Test cases for brain response data models."""

    def test_brain_response_creation(self):
        """Test BrainResponse model creation."""
        response = BrainResponse(
            region=BrainRegion.PREFRONTAL_CORTEX,
            activation_level=0.85,
            response_type="excitatory",
            confidence=0.90,
            description="Executive function region"
        )

        assert response.region == BrainRegion.PREFRONTAL_CORTEX
        assert response.activation_level == 0.85
        assert response.response_type == "excitatory"

    def test_brain_response_to_dict(self):
        """Test BrainResponse serialization."""
        response = BrainResponse(
            region=BrainRegion.HIPPOCAMPUS,
            activation_level=0.75,
            response_type="excitatory",
            confidence=0.85
        )

        data = response.to_dict()

        assert data["region"] == "hippocampus"
        assert data["activation_level"] == 0.75
        assert data["response_type"] == "excitatory"

    def test_activity_prediction_creation(self):
        """Test ActivityPrediction model creation."""
        prediction = ActivityPrediction(
            activity=ActivityType.EXERCISE,
            duration_minutes=30,
            brain_responses=[
                BrainResponse(
                    region=BrainRegion.MOTOR_CORTEX,
                    activation_level=0.85,
                    response_type="excitatory",
                    confidence=0.85
                )
            ],
            overall_wellness_score=8.0,
            benefits=["Improved mood", "Better sleep"],
            recommendations=["Stay hydrated"],
            optimal_timing="Morning",
            intensity_level="moderate"
        )

        assert prediction.activity == ActivityType.EXERCISE
        assert len(prediction.brain_responses) == 1
        assert len(prediction.benefits) == 2

    def test_brain_health_correlation_creation(self):
        """Test BrainHealthCorrelation model creation."""
        correlation = BrainHealthCorrelation(
            health_metric="Sleep Hours",
            brain_region=BrainRegion.HIPPOCAMPUS,
            correlation_strength=0.75,
            insight="Quality sleep enhances memory",
            confidence=0.85
        )

        assert correlation.health_metric == "Sleep Hours"
        assert correlation.correlation_strength == 0.75


class TestActivityTypes:
    """Test cases for activity type handling."""

    def test_all_activity_types_exist(self):
        """Test that all expected activity types are defined."""
        expected_types = [
            "exercise", "meditation", "sleep", "reading",
            "music", "social", "work", "relaxation",
            "nature", "learning", "creative", "mindfulness"
        ]

        for activity in expected_types:
            assert ActivityType(activity) in ActivityType

    def test_activity_type_values(self):
        """Test activity type enum values."""
        assert ActivityType.EXERCISE.value == "exercise"
        assert ActivityType.MEDITATION.value == "meditation"
        assert ActivityType.SLEEP.value == "sleep"


class TestBrainRegions:
    """Test cases for brain region handling."""

    def test_all_brain_regions_exist(self):
        """Test that all expected brain regions are defined."""
        expected_regions = [
            "visual_cortex", "auditory_cortex", "prefrontal_cortex",
            "motor_cortex", "hippocampus", "amygdala", "insula",
            "anterior_cingulate", "posterior_cingulate", "temporal_lobe",
            "parietal_lobe", "occipital_lobe", "cerebellum",
            "brain_stem", "default_mode_network"
        ]

        for region in expected_regions:
            assert BrainRegion(region) in BrainRegion

    def test_brain_region_values(self):
        """Test brain region enum values."""
        assert BrainRegion.PREFRONTAL_CORTEX.value == "prefrontal_cortex"
        assert BrainRegion.HIPPOCAMPUS.value == "hippocampus"
        assert BrainRegion.AMYGDALA.value == "amygdala"
