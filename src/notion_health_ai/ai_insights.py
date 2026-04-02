"""
AI Insights Engine - Health analysis and recommendations.

This module provides AI-powered health insights using Claude or other
LLM providers to analyze health data and generate actionable recommendations.
"""

import json
import os
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional

from loguru import logger

from notion_health_ai.models import (
    AIInsight,
    HealthMetric,
    HealthSummary,
    SymptomLog,
)


class AIInsightsEngine:
    """
    AI-powered health insights engine.

    Analyzes health data to provide personalized insights,
    recommendations, and trend analysis.
    """

    def __init__(
        self,
        anthropic_api_key: Optional[str] = None,
        openai_api_key: Optional[str] = None,
        gemini_api_key: Optional[str] = None,
        model_provider: str = "anthropic",
        model_name: str = "claude-sonnet-4-6",
    ):
        self.anthropic_api_key = anthropic_api_key or os.getenv("ANTHROPIC_API_KEY")
        self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        self.gemini_api_key = gemini_api_key or os.getenv("GEMINI_API_KEY")
        self.model_provider = model_provider
        self.model_name = model_name

        self._init_client()
        logger.info(f"AIInsightsEngine initialized with {model_provider} / {model_name}")

    def _init_client(self):
        """Initialize the AI client based on provider."""
        if self.model_provider == "anthropic":
            try:
                import anthropic

                self.client = anthropic.Anthropic(api_key=self.anthropic_api_key)
            except ImportError:
                logger.warning("Anthropic package not installed, using mock mode")
                self.client = None
        elif self.model_provider == "openai":
            try:
                import openai

                self.client = openai.OpenAI(api_key=self.openai_api_key)
            except ImportError:
                logger.warning("OpenAI package not installed, using mock mode")
                self.client = None
        elif self.model_provider == "gemini":
            try:
                import google.generativeai as genai

                genai.configure(api_key=self.gemini_api_key)
                self.client = genai.GenerativeModel(self.model_name or "gemini-2.0-flash")
            except ImportError:
                logger.warning("google-generativeai package not installed, using mock mode")
                self.client = None
        else:
            self.client = None

    async def generate_health_insights(
        self,
        metrics: List[HealthMetric],
        summary: HealthSummary,
        category: Optional[str] = None,
        brain_context: Optional[Dict[str, Any]] = None,
    ) -> List[AIInsight]:
        """
        Generate AI insights from health data.

        Args:
            metrics: List of health metrics to analyze
            summary: Health summary statistics
            category: Optional category filter (weight, sleep, exercise, etc.)

        Returns:
            List of AIInsight objects
        """
        insights = []

        # Generate insights for each category
        categories = [category] if category else ["weight", "sleep", "exercise", "mood", "overall"]

        for cat in categories:
            try:
                insight = await self._generate_category_insight(
                    cat,
                    metrics,
                    summary,
                    brain_context=brain_context,
                )
                if insight:
                    insights.append(insight)
            except Exception as e:
                logger.error(f"Failed to generate insight for {cat}: {e}")

        return insights

    async def _generate_category_insight(
        self,
        category: str,
        metrics: List[HealthMetric],
        summary: HealthSummary,
        brain_context: Optional[Dict[str, Any]] = None,
    ) -> Optional[AIInsight]:
        """
        Generate insight for a specific category.

        Args:
            category: Category to analyze
            metrics: Health metrics data
            summary: Summary statistics

        Returns:
            AIInsight object or None
        """
        # Build prompt based on category
        prompt = self._build_analysis_prompt(
            category,
            metrics,
            summary,
            brain_context=brain_context,
        )

        # Get AI response
        if self.client:
            response = await self._call_ai(prompt)
        else:
            response = self._generate_mock_insight(category, summary)

        # Parse response into AIInsight
        return self._parse_insight_response(category, response, len(metrics))

    def _build_analysis_prompt(
        self,
        category: str,
        metrics: List[HealthMetric],
        summary: HealthSummary,
        brain_context: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Build analysis prompt for the AI.

        Args:
            category: Category to analyze
            metrics: Health metrics data
            summary: Summary statistics

        Returns:
            Formatted prompt string
        """
        category_prompts = {
            "weight": f"""
Analyze this weight data and provide actionable health insights:

Weight Statistics (last {summary.total_days} days):
- Average weight: {summary.avg_weight} lbs
- Weight change: {summary.weight_change} lbs
- Min/Max: {summary.min_weight} / {summary.max_weight} lbs

Provide:
1. A clear trend analysis
2. Health implications
3. Actionable recommendations
4. Realistic goal suggestions
""",
            "sleep": f"""
Analyze this sleep data and provide actionable health insights:

Sleep Statistics (last {summary.total_days} days):
- Average sleep: {summary.avg_sleep_hours} hours
- Average sleep quality: {summary.avg_sleep_quality}/10
- Total sleep hours: {summary.total_sleep_hours} hours

Provide:
1. Sleep quality assessment
2. Impact on overall health
3. Sleep improvement recommendations
4. Optimal sleep schedule suggestions
""",
            "exercise": f"""
Analyze this exercise data and provide actionable health insights:

Exercise Statistics (last {summary.total_days} days):
- Total exercise minutes: {summary.total_exercise_minutes} minutes
- Exercise days: {summary.total_exercise_days} days
- Total steps: {summary.total_steps}
- Average daily steps: {summary.avg_daily_steps}

Provide:
1. Activity level assessment
2. Health benefits achieved
3. Exercise recommendations
4. Goal suggestions for improvement
""",
            "mood": f"""
Analyze this mood and energy data and provide actionable health insights:

Mood & Energy Statistics (last {summary.total_days} days):
- Average mood: {summary.avg_mood}/10
- Average energy: {summary.avg_energy}/10

Provide:
1. Mental health assessment
2. Factors affecting mood
3. Energy optimization strategies
4. Wellness recommendations
""",
            "overall": f"""
Provide an overall health assessment based on these statistics:

Period: {summary.period_start} to {summary.period_end} ({summary.total_days} days)

Key Metrics:
- Average weight: {summary.avg_weight} lbs, change: {summary.weight_change} lbs
- Average sleep: {summary.avg_sleep_hours} hours, quality: {summary.avg_sleep_quality}/10
- Exercise days: {summary.total_exercise_days}, total minutes: {summary.total_exercise_minutes}
- Average daily steps: {summary.avg_daily_steps}
- Average mood: {summary.avg_mood}/10
- Average energy: {summary.avg_energy}/10

Provide a comprehensive health assessment with:
1. Overall health score (1-100)
2. Key strengths
3. Areas for improvement
4. Top 3 priority recommendations
""",
        }

        brain_section = ""
        if brain_context:
            brain_section = f"""

Latest brain simulation context:
- Input modality: {brain_context.get("modality") or "unknown"}
- Duration: {brain_context.get("duration_minutes") or "unknown"} minutes
- Wellness score: {brain_context.get("wellness_score") or "unknown"}
- Top brain regions: {", ".join(brain_context.get("top_regions") or []) or "unknown"}
- Notes: {brain_context.get("summary") or "No extra notes"}
"""

        output_contract = """

Return only valid JSON with this exact shape:
{
  "title": "short title",
  "summary": "2 concise sentences max",
  "details": "short plain-language overview paragraph",
  "structured_details": {
    "metric_snapshot": ["bullet", "bullet"],
    "key_strengths": ["bullet", "bullet"],
    "watch_items": ["bullet", "bullet"],
    "brain_context": ["bullet", "bullet"],
    "next_steps": ["bullet", "bullet", "bullet"]
  },
  "recommendations": ["action", "action", "action"]
}

Rules:
- No markdown tables
- No code fences
- No headings inside strings
- Keep every bullet short and concrete
- If brain context is provided, include it in structured_details.brain_context
- Recommendations must be practical, realistic, and health-oriented
"""

        return (
            category_prompts.get(category, category_prompts["overall"])
            + brain_section
            + output_contract
        )

    async def _call_ai(self, prompt: str) -> str:
        """
        Call the AI model with the prompt.

        Args:
            prompt: Analysis prompt

        Returns:
            AI response text
        """
        self._last_token_usage = None
        try:
            if self.model_provider == "anthropic" and self.client:
                response = self.client.messages.create(
                    model=self.model_name,
                    max_tokens=1000,
                    messages=[
                        {
                            "role": "user",
                            "content": prompt,
                        }
                    ],
                )
                if hasattr(response, "usage"):
                    self._last_token_usage = {
                        "input_tokens": response.usage.input_tokens,
                        "output_tokens": response.usage.output_tokens,
                    }
                return response.content[0].text

            elif self.model_provider == "openai" and self.client:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    max_tokens=1000,
                    messages=[
                        {"role": "user", "content": prompt},
                    ],
                )
                if hasattr(response, "usage") and response.usage:
                    self._last_token_usage = {
                        "input_tokens": response.usage.prompt_tokens,
                        "output_tokens": response.usage.completion_tokens,
                    }
                return response.choices[0].message.content

            elif self.model_provider == "gemini" and self.client:
                response = self.client.generate_content(prompt)
                usage = getattr(response, "usage_metadata", None)
                if usage:
                    self._last_token_usage = {
                        "input_tokens": getattr(usage, "prompt_token_count", 0),
                        "output_tokens": getattr(usage, "candidates_token_count", 0),
                    }
                return response.text

            else:
                return self._generate_mock_response()

        except Exception as e:
            logger.error(f"AI call failed: {e}")
            return self._generate_mock_response()

    def _generate_mock_insight(self, category: str, summary: HealthSummary) -> Dict:
        """Generate mock insight when AI is not available."""
        _wt_dir = (
            "decreased" if summary.weight_change and summary.weight_change < 0 else "increased"
        )
        _wt_prog = (
            "healthy progress"
            if summary.weight_change and abs(summary.weight_change) < 2
            else "room for improvement"
        )
        mock_insights = {
            "weight": {
                "title": "Weight Trend Analysis",
                "summary": (
                    f"Your weight has {_wt_dir}"
                    f" by {abs(summary.weight_change or 0):.1f} lbs over the analysis period."
                ),
                "details": (
                    f"Your average weight is {summary.avg_weight or 0:.1f} lbs"
                    f" with a range of {summary.min_weight or 0:.1f}"
                    f" to {summary.max_weight or 0:.1f} lbs."
                    f" This shows {_wt_prog}."
                ),
                "structured_details": {
                    "metric_snapshot": [
                        f"Average weight: {summary.avg_weight or 0:.1f} lbs",
                        f"Weight change: {summary.weight_change or 0:+.1f} lbs",
                    ],
                    "key_strengths": ["Weight tracking is consistent enough to show a trend."],
                    "watch_items": ["Monitor whether the recent direction continues next week."],
                    "brain_context": [],
                    "next_steps": [
                        "Keep logging weight at the same time each day",
                        "Pair weight trends with exercise and sleep review",
                    ],
                },
                "recommendations": [
                    "Continue monitoring weight daily for better trend visibility",
                    "Aim for 0.5-1 lb per week for sustainable weight change",
                    "Combine dietary changes with regular exercise",
                ],
            },
            "sleep": {
                "title": "Sleep Quality Analysis",
                "summary": (
                    f"You're averaging {summary.avg_sleep_hours or 0:.1f} hours of sleep"
                    f" with quality rating of {summary.avg_sleep_quality or 0:.1f}/10."
                ),
                "details": (
                    "Sleep is crucial for recovery, cognitive function, and overall health."
                    " Your current sleep pattern shows opportunities for optimization."
                ),
                "structured_details": {
                    "metric_snapshot": [
                        f"Average sleep: {summary.avg_sleep_hours or 0:.1f} hours",
                        f"Average sleep quality: {summary.avg_sleep_quality or 0:.1f}/10",
                    ],
                    "key_strengths": ["Sleep duration is being tracked regularly."],
                    "watch_items": [
                        "Sleep quality can still improve even if duration is adequate."
                    ],
                    "brain_context": [],
                    "next_steps": [
                        "Set a stable bedtime for the next 7 days",
                        "Track whether quality improves when wake time stays fixed",
                    ],
                },
                "recommendations": [
                    "Aim for 7-9 hours of sleep consistently",
                    "Establish a regular sleep schedule",
                    "Avoid screens 1 hour before bedtime",
                    "Create a cool, dark sleeping environment",
                ],
            },
            "exercise": {
                "title": "Activity Level Analysis",
                "summary": (
                    f"You exercised {summary.total_exercise_days} days"
                    f" with {summary.total_exercise_minutes} total minutes of activity."
                ),
                "details": (
                    f"Your average daily steps are {summary.avg_daily_steps or 0:.0f}."
                    " The WHO recommends 150 minutes of moderate activity per week."
                ),
                "structured_details": {
                    "metric_snapshot": [
                        f"Exercise minutes: {summary.total_exercise_minutes or 0}",
                        f"Exercise days: {summary.total_exercise_days or 0}",
                    ],
                    "key_strengths": ["Baseline movement is present through daily steps."],
                    "watch_items": ["Weekly exercise minutes may still be below target."],
                    "brain_context": [],
                    "next_steps": [
                        "Add one extra planned movement block this week",
                        "Use low-friction sessions on busy days",
                    ],
                },
                "recommendations": [
                    "Aim for at least 150 minutes of moderate exercise weekly",
                    "Try to reach 10,000 steps daily",
                    "Include both cardio and strength training",
                    "Take short walking breaks during work",
                ],
            },
            "mood": {
                "title": "Mental Wellness Analysis",
                "summary": (
                    f"Your average mood is {summary.avg_mood or 0:.1f}/10"
                    f" and energy level is {summary.avg_energy or 0:.1f}/10."
                ),
                "details": (
                    "Mental wellness is just as important as physical health."
                    " Tracking mood patterns can help identify triggers"
                    " and optimize daily routines."
                ),
                "structured_details": {
                    "metric_snapshot": [
                        f"Average mood: {summary.avg_mood or 0:.1f}/10",
                        f"Average energy: {summary.avg_energy or 0:.1f}/10",
                    ],
                    "key_strengths": ["Mood and energy are being tracked together."],
                    "watch_items": ["Look for days when mood and energy diverge."],
                    "brain_context": [],
                    "next_steps": [
                        "Review sleep and exercise on low-energy days",
                        "Keep notes on unusually good or bad days",
                    ],
                },
                "recommendations": [
                    "Practice daily mindfulness or meditation",
                    "Maintain social connections",
                    "Exercise regularly to boost mood",
                    "Consider journaling to track emotional patterns",
                ],
            },
            "overall": {
                "title": "Overall Health Assessment",
                "summary": (
                    "Your health metrics show a balanced picture with areas of strength"
                    " and opportunities for improvement."
                ),
                "details": (
                    f"Based on {summary.total_days} days of data, you're making progress"
                    " in multiple health dimensions. Focus on consistency and gradual improvement."
                ),
                "structured_details": {
                    "metric_snapshot": [
                        f"Sleep: {summary.avg_sleep_hours or 0:.1f} hrs avg",
                        f"Exercise: {summary.total_exercise_minutes or 0} mins total",
                        f"Mood: {summary.avg_mood or 0:.1f}/10 avg",
                    ],
                    "key_strengths": [
                        "There is enough data here to review trends instead of one-off days."
                    ],
                    "watch_items": ["Focus first on the metric with the clearest downside trend."],
                    "brain_context": [],
                    "next_steps": [
                        "Keep logging consistently for the next 7 days",
                        "Choose one realistic habit change instead of many",
                    ],
                },
                "recommendations": [
                    "Maintain consistent sleep schedule",
                    "Continue regular exercise routine",
                    "Monitor weight trends weekly",
                    "Practice stress management techniques",
                ],
            },
        }

        return mock_insights.get(category, mock_insights["overall"])

    def _generate_mock_response(self) -> str:
        """Generate mock AI response."""
        return json.dumps(
            {
                "title": "Health Analysis",
                "summary": "Your health data shows positive trends.",
                "details": "Based on the available data, you're maintaining good health practices.",
                "recommendations": [
                    "Continue current healthy habits",
                    "Stay consistent with tracking",
                    "Focus on areas needing improvement",
                ],
            }
        )

    def _parse_insight_response(
        self,
        category: str,
        response: Dict,
        data_points: int,
    ) -> AIInsight:
        """
        Parse AI response into AIInsight object.

        Args:
            category: Insight category
            response: AI response dict or string
            data_points: Number of data points analyzed

        Returns:
            AIInsight object
        """
        if isinstance(response, str):
            try:
                response = json.loads(response)
            except json.JSONDecodeError:
                response = {
                    "title": f"{category.title()} Analysis",
                    "summary": response[:200],
                    "details": response,
                    "recommendations": [],
                }

        return AIInsight(
            category=category,
            title=response.get("title", f"{category.title()} Insight"),
            summary=response.get("summary", "Health insight generated."),
            details=response.get("details", ""),
            structured_details=(
                response.get("structured_details")
                if isinstance(response.get("structured_details"), dict)
                else None
            ),
            recommendations=response.get("recommendations", []),
            data_points_used=data_points,
            confidence_score=0.85,
            priority=self._calculate_priority(category),
            valid_until=datetime.now() + timedelta(hours=24),
            tags=[category, "ai-generated", "health-insight"],
        )

    def _calculate_priority(self, category: str) -> int:
        """Calculate insight priority based on category."""
        priorities = {
            "overall": 1,
            "sleep": 2,
            "mood": 2,
            "exercise": 3,
            "weight": 3,
        }
        return priorities.get(category, 3)

    async def analyze_symptom_patterns(
        self,
        symptoms: List[SymptomLog],
    ) -> AIInsight:
        """
        Analyze symptom patterns using AI.

        Args:
            symptoms: List of symptom logs to analyze

        Returns:
            AIInsight with symptom analysis
        """
        if not symptoms:
            return AIInsight(
                category="symptoms",
                title="Symptom Analysis",
                summary="No symptoms logged in the analysis period.",
                details=(
                    "Continue tracking any symptoms you experience for better pattern recognition."
                ),
                recommendations=["Log symptoms as they occur for better tracking"],
                data_points_used=0,
            )

        # Group symptoms by name
        symptom_counts: Dict[str, int] = {}
        for symptom in symptoms:
            name = symptom.symptom_name
            symptom_counts[name] = symptom_counts.get(name, 0) + 1

        # Find most common symptoms
        sorted_symptoms = sorted(symptom_counts.items(), key=lambda x: x[1], reverse=True)
        top_symptoms = sorted_symptoms[:5]

        # Build analysis
        top_names = ", ".join([s[0] for s in top_symptoms[:3]])
        summary = f"Analyzed {len(symptoms)} symptom entries. Most frequent: {top_names}."
        details = "Symptom frequency analysis:\n"
        for name, count in top_symptoms:
            details += f"- {name}: {count} occurrences\n"

        # Generate recommendations
        recommendations = [
            "Track potential triggers (food, stress, sleep)",
            "Note time of day symptoms occur",
            "Discuss recurring symptoms with your doctor",
            "Consider keeping a detailed symptom diary",
        ]

        return AIInsight(
            category="symptoms",
            title="Symptom Pattern Analysis",
            summary=summary,
            details=details,
            recommendations=recommendations,
            data_points_used=len(symptoms),
            confidence_score=0.75,
            priority=2,
            valid_until=datetime.now() + timedelta(hours=24),
            tags=["symptoms", "pattern-analysis", "ai-generated"],
        )

    async def generate_correlation_insights(
        self,
        metrics: List[HealthMetric],
    ) -> List[AIInsight]:
        """
        Find correlations between different health metrics.

        Args:
            metrics: Health metrics to analyze for correlations

        Returns:
            List of AIInsight objects about correlations
        """
        insights = []

        # Analyze sleep-mood correlation
        sleep_mood_data = [
            (m.sleep_hours, m.mood.value) for m in metrics if m.sleep_hours and m.mood
        ]
        if len(sleep_mood_data) >= 5:
            correlation = self._calculate_correlation(
                [d[0] for d in sleep_mood_data], [d[1] for d in sleep_mood_data]
            )
            if abs(correlation) > 0.3:
                insights.append(
                    AIInsight(
                        category="correlation",
                        title="Sleep-Mood Connection",
                        summary=(
                            f"Found a {'positive' if correlation > 0 else 'negative'}"
                            f" correlation ({correlation:.2f}) between sleep and mood."
                        ),
                        details=(
                            "Your mood tends to be better when you get more sleep."
                            " This is consistent with research showing sleep quality"
                            " significantly impacts emotional well-being."
                        ),
                        recommendations=[
                            "Prioritize sleep on nights before important days",
                            "Track your mood after different sleep durations",
                        ],
                        data_points_used=len(sleep_mood_data),
                        confidence_score=0.7,
                        priority=2,
                        tags=["sleep", "mood", "correlation"],
                    )
                )

        # Analyze exercise-energy correlation
        exercise_energy_data = [
            (m.exercise_minutes, m.energy.value) for m in metrics if m.exercise_minutes and m.energy
        ]
        if len(exercise_energy_data) >= 5:
            correlation = self._calculate_correlation(
                [d[0] for d in exercise_energy_data], [d[1] for d in exercise_energy_data]
            )
            if abs(correlation) > 0.3:
                insights.append(
                    AIInsight(
                        category="correlation",
                        title="Exercise-Energy Connection",
                        summary=(
                            f"Found a {'positive' if correlation > 0 else 'negative'}"
                            f" correlation ({correlation:.2f}) between exercise and energy."
                        ),
                        details=(
                            "Your energy levels are connected to your exercise patterns."
                            " Regular physical activity can boost energy levels throughout the day."
                        ),
                        recommendations=[
                            "Try morning exercise for sustained energy",
                            "Light exercise can help when feeling low energy",
                        ],
                        data_points_used=len(exercise_energy_data),
                        confidence_score=0.7,
                        priority=2,
                        tags=["exercise", "energy", "correlation"],
                    )
                )

        return insights

    def _calculate_correlation(self, x: List[float], y: List[float]) -> float:
        """
        Calculate Pearson correlation coefficient.

        Args:
            x: First variable values
            y: Second variable values

        Returns:
            Correlation coefficient (-1 to 1)
        """
        n = len(x)
        if n != len(y) or n == 0:
            return 0.0

        mean_x = sum(x) / n
        mean_y = sum(y) / n

        numerator = sum((x[i] - mean_x) * (y[i] - mean_y) for i in range(n))
        denominator_x = sum((x[i] - mean_x) ** 2 for i in range(n)) ** 0.5
        denominator_y = sum((y[i] - mean_y) ** 2 for i in range(n)) ** 0.5

        if denominator_x == 0 or denominator_y == 0:
            return 0.0

        return numerator / (denominator_x * denominator_y)

    async def generate_prediction(
        self,
        metrics: List[HealthMetric],
        metric_type: str,
        days_ahead: int = 7,
    ) -> Dict[str, Any]:
        """
        Generate predictions for future health metrics.

        Args:
            metrics: Historical health metrics
            metric_type: Type of metric to predict (weight, sleep, etc.)
            days_ahead: Number of days to predict ahead

        Returns:
            Dictionary with prediction data
        """
        # Simple linear trend prediction
        values = []
        dates = []

        for m in metrics:
            value = getattr(m, metric_type, None)
            if value is not None:
                values.append(float(value))
                dates.append(m.date)

        if len(values) < 3:
            return {
                "success": False,
                "message": "Not enough data for prediction",
            }

        # Calculate simple linear regression
        n = len(values)
        x_mean = sum(range(n)) / n
        y_mean = sum(values) / n

        numerator = sum((i - x_mean) * (values[i] - y_mean) for i in range(n))
        denominator = sum((i - x_mean) ** 2 for i in range(n))

        if denominator == 0:
            slope = 0
        else:
            slope = numerator / denominator

        intercept = y_mean - slope * x_mean

        # Predict future values
        future_values = []
        for i in range(days_ahead):
            future_x = n + i
            predicted = slope * future_x + intercept
            future_values.append(
                {
                    "day": i + 1,
                    "date": (date.today() + timedelta(days=i + 1)).isoformat(),
                    "predicted_value": round(predicted, 2),
                }
            )

        return {
            "success": True,
            "metric_type": metric_type,
            "trend": "increasing" if slope > 0 else "decreasing" if slope < 0 else "stable",
            "slope": round(slope, 4),
            "current_value": values[-1],
            "predictions": future_values,
        }
