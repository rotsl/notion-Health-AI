"""
Utility functions for NotionHealth AI.

This module provides helper functions used throughout the application.
"""

import os
from datetime import datetime, date, timedelta
from typing import Optional, List, Dict, Any
import json
import re


def load_env_file(env_path: str = ".env") -> Dict[str, str]:
    """
    Load environment variables from a .env file.

    Args:
        env_path: Path to the .env file

    Returns:
        Dictionary of environment variables
    """
    env_vars = {}

    if not os.path.exists(env_path):
        return env_vars

    with open(env_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            if "=" in line:
                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip().strip("\"'")
                env_vars[key] = value

    return env_vars


def format_date(dt: date) -> str:
    """Format a date as YYYY-MM-DD."""
    return dt.strftime("%Y-%m-%d")


def format_time(t) -> str:
    """Format a time as HH:MM."""
    if hasattr(t, "strftime"):
        return t.strftime("%H:%M")
    return str(t)


def parse_date(date_str: str) -> Optional[date]:
    """
    Parse a date string in various formats.

    Args:
        date_str: Date string to parse

    Returns:
        date object or None
    """
    formats = [
        "%Y-%m-%d",
        "%Y/%m/%d",
        "%d-%m-%Y",
        "%d/%m/%Y",
        "%m-%d-%Y",
        "%m/%d/%Y",
    ]

    for fmt in formats:
        try:
            return datetime.strptime(date_str, fmt).date()
        except ValueError:
            continue

    # Try relative dates
    date_str = date_str.lower().strip()
    today = date.today()

    if date_str == "today":
        return today
    elif date_str == "yesterday":
        return today - timedelta(days=1)
    elif date_str == "tomorrow":
        return today + timedelta(days=1)
    elif date_str.endswith(" days ago"):
        try:
            days = int(date_str.split()[0])
            return today - timedelta(days=days)
        except (ValueError, IndexError):
            pass

    return None


def parse_time(time_str: str) -> Optional[int]:
    """
    Parse a time string and return minutes since midnight.

    Args:
        time_str: Time string (HH:MM or similar)

    Returns:
        Minutes since midnight or None
    """
    time_str = time_str.strip().lower()

    # Handle HH:MM format
    if ":" in time_str:
        try:
            hours, minutes = time_str.split(":")
            return int(hours) * 60 + int(minutes)
        except (ValueError, IndexError):
            pass

    # Handle AM/PM format
    match = re.match(r"(\d{1,2})(?::(\d{2}))?\s*(am|pm)?", time_str)
    if match:
        hours = int(match.group(1))
        minutes = int(match.group(2)) if match.group(2) else 0
        am_pm = match.group(3)

        if am_pm == "pm" and hours < 12:
            hours += 12
        elif am_pm == "am" and hours == 12:
            hours = 0

        return hours * 60 + minutes

    return None


def calculate_bmi(weight_lbs: float, height_inches: float) -> float:
    """
    Calculate BMI from weight in pounds and height in inches.

    Args:
        weight_lbs: Weight in pounds
        height_inches: Height in inches

    Returns:
        BMI value
    """
    if height_inches <= 0:
        return 0

    # BMI = (weight in pounds * 703) / (height in inches)^2
    bmi = (weight_lbs * 703) / (height_inches**2)
    return round(bmi, 1)


def get_bmi_category(bmi: float) -> str:
    """
    Get BMI category label.

    Args:
        bmi: BMI value

    Returns:
        Category string
    """
    if bmi < 18.5:
        return "Underweight"
    elif bmi < 25:
        return "Normal weight"
    elif bmi < 30:
        return "Overweight"
    else:
        return "Obese"


def lbs_to_kg(lbs: float) -> float:
    """Convert pounds to kilograms."""
    return round(lbs * 0.453592, 2)


def kg_to_lbs(kg: float) -> float:
    """Convert kilograms to pounds."""
    return round(kg * 2.20462, 2)


def calculate_average(values: List[float]) -> Optional[float]:
    """
    Calculate average of a list of values.

    Args:
        values: List of numeric values

    Returns:
        Average value or None
    """
    if not values:
        return None
    return sum(values) / len(values)


def calculate_trend(values: List[float]) -> Optional[str]:
    """
    Determine trend direction from a list of values.

    Args:
        values: List of numeric values in chronological order

    Returns:
        Trend string: "increasing", "decreasing", or "stable"
    """
    if len(values) < 2:
        return None

    # Simple comparison of first and last third
    n = len(values)
    first_third = values[: n // 3]
    last_third = values[2 * n // 3 :]

    avg_first = sum(first_third) / len(first_third)
    avg_last = sum(last_third) / len(last_third)

    diff = avg_last - avg_first
    threshold = avg_first * 0.02  # 2% threshold

    if diff > threshold:
        return "increasing"
    elif diff < -threshold:
        return "decreasing"
    else:
        return "stable"


def format_duration(minutes: int) -> str:
    """
    Format duration in minutes to human-readable string.

    Args:
        minutes: Duration in minutes

    Returns:
        Formatted string (e.g., "1h 30m")
    """
    if minutes < 60:
        return f"{minutes}m"

    hours = minutes // 60
    remaining_minutes = minutes % 60

    if remaining_minutes == 0:
        return f"{hours}h"

    return f"{hours}h {remaining_minutes}m"


def get_day_of_week(dt: date) -> str:
    """Get day of week name."""
    return dt.strftime("%A")


def is_weekend(dt: date) -> bool:
    """Check if date is a weekend."""
    return dt.weekday() >= 5


def get_week_number(dt: date) -> int:
    """Get ISO week number."""
    return dt.isocalendar()[1]


def generate_date_range(start: date, end: date) -> List[date]:
    """
    Generate a list of dates between start and end (inclusive).

    Args:
        start: Start date
        end: End date

    Returns:
        List of dates
    """
    dates = []
    current = start
    while current <= end:
        dates.append(current)
        current += timedelta(days=1)
    return dates


def truncate_text(text: str, max_length: int = 100) -> str:
    """
    Truncate text to max_length characters.

    Args:
        text: Text to truncate
        max_length: Maximum length

    Returns:
        Truncated text with ellipsis if needed
    """
    if len(text) <= max_length:
        return text
    return text[: max_length - 3] + "..."


def safe_float(value: Any, default: float = 0.0) -> float:
    """Safely convert value to float."""
    try:
        return float(value)
    except (ValueError, TypeError):
        return default


def safe_int(value: Any, default: int = 0) -> int:
    """Safely convert value to int."""
    try:
        return int(value)
    except (ValueError, TypeError):
        return default


def format_health_score(score: float) -> str:
    """
    Format a health score with emoji indicator.

    Args:
        score: Score from 0-100

    Returns:
        Formatted string with emoji
    """
    if score >= 90:
        return f"🌟 {score:.0f}/100 (Excellent)"
    elif score >= 75:
        return f"😊 {score:.0f}/100 (Good)"
    elif score >= 60:
        return f"😐 {score:.0f}/100 (Fair)"
    elif score >= 40:
        return f"😟 {score:.0f}/100 (Needs Work)"
    else:
        return f"😰 {score:.0f}/100 (Poor)"


def get_mood_emoji(mood: int) -> str:
    """Get emoji for mood level."""
    if mood >= 9:
        return "😄"
    elif mood >= 7:
        return "🙂"
    elif mood >= 5:
        return "😐"
    elif mood >= 3:
        return "😟"
    else:
        return "😢"


def get_energy_emoji(energy: int) -> str:
    """Get emoji for energy level."""
    if energy >= 9:
        return "⚡"
    elif energy >= 7:
        return "💪"
    elif energy >= 5:
        return "👌"
    elif energy >= 3:
        return "😴"
    else:
        return "😫"


def get_exercise_emoji(exercise_type: str) -> str:
    """Get emoji for exercise type."""
    emojis = {
        "running": "🏃",
        "walking": "🚶",
        "cycling": "🚴",
        "swimming": "🏊",
        "strength_training": "🏋️",
        "yoga": "🧘",
        "hiit": "🔥",
        "cardio": "❤️",
        "sports": "⚽",
        "other": "🎯",
    }
    return emojis.get(exercise_type.lower(), "💪")


def create_health_report(
    summary: Dict[str, Any],
    insights: List[Dict[str, Any]],
    format_type: str = "text",
) -> str:
    """
    Create a formatted health report.

    Args:
        summary: Health summary dictionary
        insights: List of insight dictionaries
        format_type: Output format ("text", "markdown", "json")

    Returns:
        Formatted report string
    """
    if format_type == "json":
        return json.dumps(
            {
                "summary": summary,
                "insights": insights,
            },
            indent=2,
            default=str,
        )

    lines = []
    lines.append("=" * 50)
    lines.append("NOTIONHEALTH AI - HEALTH REPORT")
    lines.append("=" * 50)
    lines.append("")

    # Summary section
    lines.append("📊 SUMMARY")
    lines.append("-" * 30)
    for key, value in summary.items():
        if value is not None:
            lines.append(f"  {key}: {value}")
    lines.append("")

    # Insights section
    if insights:
        lines.append("💡 INSIGHTS")
        lines.append("-" * 30)
        for insight in insights:
            lines.append(f"  • {insight.get('title', 'Insight')}")
            lines.append(f"    {insight.get('summary', '')}")
            lines.append("")

    lines.append("=" * 50)

    if format_type == "markdown":
        return "\n".join(lines).replace("=", "#").replace("-", "*")

    return "\n".join(lines)
