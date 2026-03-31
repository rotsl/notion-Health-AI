"""
CLI Interface for NotionHealth AI.

Provides command-line interface for interacting with the health management system.
"""

import asyncio
import os
import sys
from datetime import datetime, date, timedelta
from typing import Optional
import click
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, BarColumn
from questionary import select, text, confirm

from notion_health_ai.health_manager import HealthManager
from notion_health_ai.ai_insights import AIInsightsEngine
from notion_health_ai.tribe_integration import TribeHealthAnalyzer
from notion_health_ai.utils import (
    format_date,
    parse_date,
    get_mood_emoji,
    get_energy_emoji,
    get_exercise_emoji,
    format_health_score,
)

console = Console()


# Initialize managers
health_manager = None
ai_engine = None
tribe_analyzer = None


def get_health_manager() -> HealthManager:
    """Get or create health manager instance."""
    global health_manager
    if health_manager is None:
        health_manager = HealthManager()
    return health_manager


def get_ai_engine() -> AIInsightsEngine:
    """Get or create AI insights engine instance."""
    global ai_engine
    if ai_engine is None:
        ai_engine = AIInsightsEngine()
    return ai_engine


def get_tribe_analyzer() -> TribeHealthAnalyzer:
    """Get or create Tribe v2 analyzer instance."""
    global tribe_analyzer
    if tribe_analyzer is None:
        tribe_analyzer = TribeHealthAnalyzer()
    return tribe_analyzer


@click.group()
@click.version_option(version="2.0.0", prog_name="notion-health")
def main():
    """
    NotionHealth AI - AI-Powered Personal Health Management with Tribe v2 Brain Prediction

    Track health metrics, medications, appointments, and get AI insights
    powered by Meta's Tribe v2 brain predictive foundation model,
    all stored in your Notion workspace.
    """
    pass


# ===========================================
# Log Commands
# ===========================================


@main.group()
def log():
    """Log health data entries."""
    pass


@log.command()
@click.option("--weight", type=float, help="Weight in pounds")
@click.option("--sleep", type=float, help="Hours of sleep")
@click.option("--sleep-quality", type=int, help="Sleep quality (1-10)")
@click.option("--exercise", type=int, help="Exercise minutes")
@click.option("--exercise-type", type=str, help="Exercise type")
@click.option("--steps", type=int, help="Daily steps")
@click.option("--mood", type=int, help="Mood rating (1-10)")
@click.option("--energy", type=int, help="Energy level (1-10)")
@click.option("--water", type=int, help="Glasses of water")
@click.option("--notes", type=str, help="Additional notes")
def metric(**kwargs):
    """Log a health metric entry."""
    manager = get_health_manager()

    # Check if any values provided
    if not any(v is not None for k, v in kwargs.items() if k != "notes"):
        # Interactive mode
        console.print(Panel("📝 Log Health Metric", style="blue"))

        kwargs["weight"] = text("Weight (lbs, leave empty to skip):").ask()
        kwargs["weight"] = float(kwargs["weight"]) if kwargs["weight"] else None

        kwargs["sleep"] = text("Sleep hours (leave empty to skip):").ask()
        kwargs["sleep"] = float(kwargs["sleep"]) if kwargs["sleep"] else None

        kwargs["mood"] = text("Mood (1-10, leave empty to skip):").ask()
        kwargs["mood"] = int(kwargs["mood"]) if kwargs["mood"] else None

        kwargs["energy"] = text("Energy (1-10, leave empty to skip):").ask()
        kwargs["energy"] = int(kwargs["energy"]) if kwargs["energy"] else None

        kwargs["exercise"] = text("Exercise minutes (leave empty to skip):").ask()
        kwargs["exercise"] = int(kwargs["exercise"]) if kwargs["exercise"] else None

        kwargs["notes"] = text("Notes (optional):").ask() or None

    # Log the metric
    result = asyncio.run(
        manager.log_health_metric(
            weight=kwargs.get("weight"),
            sleep_hours=kwargs.get("sleep"),
            sleep_quality=kwargs.get("sleep_quality"),
            exercise_minutes=kwargs.get("exercise"),
            exercise_type=kwargs.get("exercise_type"),
            steps=kwargs.get("steps"),
            mood=kwargs.get("mood"),
            energy=kwargs.get("energy"),
            water_glasses=kwargs.get("water"),
            notes=kwargs.get("notes"),
        )
    )

    if result.success:
        console.print(f"✅ {result.message}", style="green")
    else:
        console.print(f"❌ {result.message}", style="red")
        if result.error:
            console.print(f"   Error: {result.error}", style="red")


@log.command()
@click.option("--name", required=True, help="Medication name")
@click.option("--dosage", required=True, help="Dosage (e.g., '100mg')")
@click.option(
    "--frequency",
    type=click.Choice(["once_daily", "twice_daily", "three_times_daily", "weekly", "as_needed"]),
    required=True,
    help="How often to take",
)
@click.option("--start-date", type=str, default="today", help="Start date (YYYY-MM-DD or 'today')")
@click.option("--doctor", help="Prescribing doctor")
@click.option("--purpose", help="What it's for")
def medication(**kwargs):
    """Add a medication to track."""
    manager = get_health_manager()

    start_date = parse_date(kwargs["start_date"]) or date.today()

    result = asyncio.run(
        manager.add_medication(
            name=kwargs["name"],
            dosage=kwargs["dosage"],
            frequency=kwargs["frequency"],
            start_date=start_date,
            prescribing_doctor=kwargs.get("doctor"),
            purpose=kwargs.get("purpose"),
        )
    )

    if result.success:
        console.print(f"✅ {result.message}", style="green")
    else:
        console.print(f"❌ {result.message}", style="red")


@log.command()
@click.option("--symptom", required=True, help="Symptom name")
@click.option("--severity", type=int, required=True, help="Severity (1-10)")
@click.option("--body-part", help="Affected body part")
@click.option("--duration", type=int, help="Duration in minutes")
@click.option("--notes", help="Additional notes")
def symptom(**kwargs):
    """Log a symptom entry."""
    manager = get_health_manager()

    result = asyncio.run(
        manager.log_symptom(
            symptom_name=kwargs["symptom"],
            severity=kwargs["severity"],
            body_part=kwargs.get("body_part"),
            duration_minutes=kwargs.get("duration"),
            notes=kwargs.get("notes"),
        )
    )

    if result.success:
        console.print(f"✅ {result.message}", style="green")
    else:
        console.print(f"❌ {result.message}", style="red")


# ===========================================
# View Commands
# ===========================================


@main.group()
def view():
    """View health data."""
    pass


@view.command()
@click.option("--days", type=int, default=7, help="Number of days to show")
def summary(days: int):
    """View health summary."""
    manager = get_health_manager()

    end_date = date.today()
    start_date = end_date - timedelta(days=days)

    summary_data = asyncio.run(manager.get_health_summary(start_date, end_date))

    # Create summary table
    table = Table(title=f"📊 Health Summary ({days} days)")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    table.add_column("Trend", style="yellow")

    # Add rows
    if summary_data.avg_weight:
        trend = "↓" if summary_data.weight_change and summary_data.weight_change < 0 else "↑"
        table.add_row("Weight", f"{summary_data.avg_weight:.1f} lbs", trend)

    if summary_data.avg_sleep_hours:
        table.add_row("Sleep", f"{summary_data.avg_sleep_hours:.1f} hrs", "")

    if summary_data.total_exercise_minutes:
        table.add_row("Exercise", f"{summary_data.total_exercise_minutes} min", "")

    if summary_data.avg_mood:
        table.add_row(
            "Mood", f"{summary_data.avg_mood:.1f}/10", get_mood_emoji(int(summary_data.avg_mood))
        )

    if summary_data.avg_energy:
        table.add_row(
            "Energy",
            f"{summary_data.avg_energy:.1f}/10",
            get_energy_emoji(int(summary_data.avg_energy)),
        )

    console.print(table)


@view.command()
def medications():
    """View medication schedule."""
    manager = get_health_manager()

    schedule = asyncio.run(manager.get_medication_schedule())

    table = Table(title="💊 Medication Schedule (Today)")
    table.add_column("Time", style="cyan")
    table.add_column("Medication", style="green")
    table.add_column("Dosage", style="yellow")
    table.add_column("Purpose", style="blue")

    for med in schedule:
        table.add_row(
            med["time"],
            med["medication"],
            med["dosage"],
            med.get("purpose", ""),
        )

    console.print(table)


@view.command()
@click.option("--days", type=int, default=30, help="Days to look ahead")
def appointments(days: int):
    """View upcoming appointments."""
    manager = get_health_manager()

    appointments_list = asyncio.run(manager.get_upcoming_appointments(days))

    table = Table(title=f"📅 Upcoming Appointments ({days} days)")
    table.add_column("Date", style="cyan")
    table.add_column("Time", style="green")
    table.add_column("Doctor", style="yellow")
    table.add_column("Type", style="blue")
    table.add_column("Facility", style="magenta")

    for appt in appointments_list:
        table.add_row(
            format_date(appt.date),
            appt.time.strftime("%H:%M") if appt.time else "",
            appt.doctor_name,
            appt.appointment_type.value,
            appt.facility or "",
        )

    console.print(table)


@view.command()
def goals():
    """View health goals progress."""
    manager = get_health_manager()

    goals_list = asyncio.run(manager.get_active_goals())

    table = Table(title="🎯 Health Goals")
    table.add_column("Goal", style="cyan")
    table.add_column("Type", style="green")
    table.add_column("Progress", style="yellow")
    table.add_column("Target", style="blue")
    table.add_column("Deadline", style="magenta")

    for goal in goals_list:
        progress = f"{goal.current_value}/{goal.target_value} {goal.unit}"
        percent = (goal.current_value / goal.target_value * 100) if goal.target_value > 0 else 0

        table.add_row(
            goal.title,
            goal.goal_type.value,
            f"{progress} ({percent:.0f}%)",
            f"{goal.target_value} {goal.unit}",
            format_date(goal.target_date),
        )

    console.print(table)


# ===========================================
# Add Commands
# ===========================================


@main.group()
def add():
    """Add new entries."""
    pass


@add.command()
@click.option("--date", required=True, help="Appointment date (YYYY-MM-DD)")
@click.option("--time", required=True, help="Appointment time (HH:MM)")
@click.option("--doctor", required=True, help="Doctor name")
@click.option(
    "--type",
    "appointment_type",
    type=click.Choice(
        [
            "checkup",
            "follow_up",
            "specialist",
            "dental",
            "vision",
            "mental_health",
            "lab_work",
            "other",
        ]
    ),
    required=True,
    help="Type of appointment",
)
@click.option("--facility", help="Facility name")
@click.option("--reason", help="Reason for visit")
def appointment(**kwargs):
    """Add a medical appointment."""
    manager = get_health_manager()

    appointment_date = parse_date(kwargs["date"])
    if not appointment_date:
        console.print("❌ Invalid date format. Use YYYY-MM-DD.", style="red")
        return

    result = asyncio.run(
        manager.add_appointment(
            appointment_date=appointment_date,
            appointment_time=kwargs["time"],
            doctor_name=kwargs["doctor"],
            appointment_type=kwargs["appointment_type"],
            facility=kwargs.get("facility"),
            reason=kwargs.get("reason"),
        )
    )

    if result.success:
        console.print(f"✅ {result.message}", style="green")
    else:
        console.print(f"❌ {result.message}", style="red")


@add.command()
@click.option(
    "--type",
    "goal_type",
    type=click.Choice(
        ["weight", "exercise", "sleep", "steps", "water_intake", "meditation", "custom"]
    ),
    required=True,
    help="Type of goal",
)
@click.option("--title", required=True, help="Goal title")
@click.option("--target", type=float, required=True, help="Target value")
@click.option("--unit", required=True, help="Unit of measurement")
@click.option("--deadline", required=True, help="Target date (YYYY-MM-DD)")
@click.option("--description", help="Goal description")
def goal(**kwargs):
    """Set a health goal."""
    manager = get_health_manager()

    target_date = parse_date(kwargs["deadline"])
    if not target_date:
        console.print("❌ Invalid date format. Use YYYY-MM-DD.", style="red")
        return

    result = asyncio.run(
        manager.set_health_goal(
            goal_type=kwargs["goal_type"],
            title=kwargs["title"],
            target_value=kwargs["target"],
            unit=kwargs["unit"],
            target_date=target_date,
            description=kwargs.get("description"),
        )
    )

    if result.success:
        console.print(f"✅ {result.message}", style="green")
    else:
        console.print(f"❌ {result.message}", style="red")


# ===========================================
# Insights Command
# ===========================================


@main.command()
@click.option(
    "--category",
    type=click.Choice(["weight", "sleep", "exercise", "mood", "overall", "brain"]),
    default="overall",
    help="Category to analyze",
)
@click.option(
    "--period", type=click.Choice(["week", "month", "all"]), default="week", help="Time period"
)
def insights(category: str, period: str):
    """Get AI-powered health insights with Tribe v2 brain prediction."""
    manager = get_health_manager()
    engine = get_ai_engine()

    # Calculate date range
    today = date.today()
    if period == "week":
        start_date = today - timedelta(days=7)
    elif period == "month":
        start_date = today - timedelta(days=30)
    else:
        start_date = today - timedelta(days=90)

    # Get data
    metrics = asyncio.run(manager.get_health_metrics(start_date, today))
    summary = asyncio.run(manager.get_health_summary(start_date, today))

    # Generate insights
    insights_list = asyncio.run(engine.generate_health_insights(metrics, summary, category))

    # Display insights
    console.print(Panel(f"🧠 AI Health Insights - {category.title()}", style="blue"))

    for insight in insights_list:
        console.print(f"\n[bold cyan]{insight.title}[/bold cyan]")
        console.print(f"\n{insight.summary}")
        console.print(f"\n{insight.details}")

        if insight.recommendations:
            console.print("\n[yellow]Recommendations:[/yellow]")
            for i, rec in enumerate(insight.recommendations, 1):
                console.print(f"  {i}. {rec}")

        console.print(
            f"\n[dim]Confidence: {insight.confidence_score * 100:.0f}% | Priority: {insight.priority}[/dim]"
        )
        console.print("-" * 50)


# ===========================================
# Tribe v2 Brain Prediction Commands
# ===========================================


@main.group()
def brain():
    """Tribe v2 brain response prediction commands."""
    pass


@brain.command()
@click.option(
    "--activity",
    type=click.Choice(
        ["exercise", "meditation", "sleep", "reading", "music", "social", "work", "relaxation"]
    ),
    required=True,
    help="Activity to analyze",
)
@click.option("--duration", type=int, default=30, help="Duration in minutes")
def predict(activity: str, duration: int):
    """Predict brain response to a wellness activity using Tribe v2."""
    analyzer = get_tribe_analyzer()

    console.print(Panel(f"🧠 Tribe v2 Brain Response Prediction", style="blue"))
    console.print(
        f"\nAnalyzing brain response to [cyan]{activity}[/cyan] for [yellow]{duration}[/yellow] minutes...\n"
    )

    result = asyncio.run(analyzer.predict_brain_response(activity, duration))

    if result.get("success"):
        prediction = result.get("prediction", {})

        table = Table(title="Predicted Brain Response")
        table.add_column("Region", style="cyan")
        table.add_column("Response", style="green")
        table.add_column("Intensity", style="yellow")

        for region, data in prediction.get("brain_regions", {}).items():
            table.add_row(
                region.replace("_", " ").title(),
                data.get("response", "N/A"),
                f"{data.get('intensity', 0):.1%}",
            )

        console.print(table)

        console.print(
            f"\n[bold]Overall Wellness Score:[/bold] {prediction.get('wellness_score', 0):.1f}/10"
        )
        console.print(f"[bold]Recommendation:[/bold] {prediction.get('recommendation', 'N/A')}")

        if prediction.get("benefits"):
            console.print("\n[green]Expected Benefits:[/green]")
            for benefit in prediction.get("benefits", []):
                console.print(f"  ✨ {benefit}")
    else:
        console.print(f"❌ Prediction failed: {result.get('error', 'Unknown error')}", style="red")


@brain.command()
@click.option("--days", type=int, default=7, help="Number of days to analyze")
def analyze(days: int):
    """Analyze your health data with Tribe v2 brain correlation."""
    manager = get_health_manager()
    analyzer = get_tribe_analyzer()

    end_date = date.today()
    start_date = end_date - timedelta(days=days)

    # Get health data
    metrics = asyncio.run(manager.get_health_metrics(start_date, end_date))
    summary = asyncio.run(manager.get_health_summary(start_date, end_date))

    console.print(Panel(f"🧠 Tribe v2 Brain-Health Correlation Analysis", style="blue"))
    console.print(f"\nAnalyzing {len(metrics)} health entries over {days} days...\n")

    # Run analysis
    result = asyncio.run(analyzer.analyze_brain_health_correlation(metrics, summary))

    if result.get("success"):
        analysis = result.get("analysis", {})

        # Display correlations
        table = Table(title="Brain-Health Correlations")
        table.add_column("Health Metric", style="cyan")
        table.add_column("Brain Region", style="green")
        table.add_column("Correlation", style="yellow")
        table.add_column("Insight", style="blue")

        for corr in analysis.get("correlations", []):
            table.add_row(
                corr.get("metric", "N/A"),
                corr.get("brain_region", "N/A"),
                f"{corr.get('strength', 0):.2f}",
                corr.get("insight", "N/A"),
            )

        console.print(table)

        # Display recommendations
        if analysis.get("recommendations"):
            console.print("\n[bold green]Personalized Recommendations:[/bold green]")
            for rec in analysis.get("recommendations", []):
                console.print(f"  🎯 {rec}")

        # Display optimal timing
        if analysis.get("optimal_timing"):
            console.print(f"\n[bold yellow]Optimal Activity Timing:[/bold yellow]")
            for activity, timing in analysis.get("optimal_timing", {}).items():
                console.print(f"  ⏰ {activity}: {timing}")
    else:
        console.print(f"❌ Analysis failed: {result.get('error', 'Unknown error')}", style="red")


@brain.command()
def schedule():
    """Generate optimal daily schedule based on brain prediction."""
    analyzer = get_tribe_analyzer()

    console.print(Panel(f"🧠 Optimal Daily Schedule (Tribe v2 AI)", style="blue"))
    console.print("\nGenerating schedule based on brain response patterns...\n")

    result = asyncio.run(analyzer.generate_optimal_schedule())

    if result.get("success"):
        schedule = result.get("schedule", {})

        table = Table(title="Optimal Daily Schedule")
        table.add_column("Time", style="cyan")
        table.add_column("Activity", style="green")
        table.add_column("Brain Benefit", style="yellow")
        table.add_column("Priority", style="blue")

        for slot in schedule.get("slots", []):
            table.add_row(
                slot.get("time", "N/A"),
                slot.get("activity", "N/A"),
                slot.get("brain_benefit", "N/A"),
                slot.get("priority", "N/A"),
            )

        console.print(table)

        console.print(f"\n[bold]Schedule Score:[/bold] {schedule.get('score', 0):.1f}/10")
        console.print(f"[bold]Notes:[/bold] {schedule.get('notes', 'N/A')}")
    else:
        console.print(
            f"❌ Schedule generation failed: {result.get('error', 'Unknown error')}", style="red"
        )


_ACTIVITY_PROMPTS = {
    "exercise": "I went for a {d} minute run and full body workout session, feeling my heart rate rise and muscles engage throughout the effort",
    "meditation": "I spent {d} minutes in quiet meditation, focusing gently on my breath and letting thoughts pass without holding onto them",
    "sleep": "I enjoyed a deep and restful sleep lasting {d} minutes, cycling through slow wave and REM stages for full recovery",
    "reading": "I sat down and read a book for {d} minutes, fully absorbed in the text and following the narrative closely",
    "music": "I listened to music for {d} minutes, letting the melodies and rhythms wash over me while I relaxed and enjoyed the sound",
    "social": "I spent {d} minutes in warm and engaging conversation with close friends, laughing and sharing stories together",
    "work": "I worked with deep concentration on challenging tasks for {d} minutes, organising my thoughts and solving complex problems",
    "relaxation": "I spent {d} minutes relaxing completely, letting go of tension and allowing my body and mind to unwind fully",
    "nature": "I walked through a quiet park and enjoyed the natural surroundings for {d} minutes, breathing fresh air and observing the trees and sky",
    "learning": "I studied new and complex material for {d} minutes, taking notes and making connections between concepts to consolidate understanding",
    "creative": "I engaged in open-ended creative work for {d} minutes, brainstorming freely and exploring novel ideas without judgment",
    "mindfulness": "I practiced gentle mindfulness and body awareness for {d} minutes, noticing physical sensations and staying present in each moment",
}


@brain.command()
@click.option(
    "--activity",
    type=click.Choice(list(_ACTIVITY_PROMPTS.keys())),
    default="exercise",
    show_default=True,
    help="Activity to predict brain response for",
)
@click.option("--duration", type=int, default=30, show_default=True, help="Duration in minutes")
@click.option(
    "--type",
    "viz_types",
    multiple=True,
    type=click.Choice(["interactive", "static", "gif", "heatmap"]),
    default=["interactive", "heatmap"],
    show_default=True,
    help="Visualization type(s) to generate (repeat flag for multiple)",
)
@click.option("--no-open", is_flag=True, help="Do not auto-open the browser after generating HTML")
def visualize(activity: str, duration: int, viz_types: tuple, no_open: bool):
    """Generate brain activation visualizations from TRIBEv2 predictions.

    Runs a TRIBEv2 prediction for the given activity and produces interactive
    3D HTML and/or static PNG files saved to the visualizations/ directory.
    """
    from notion_health_ai.brain_viz import BrainVisualizer

    text_stimulus = _ACTIVITY_PROMPTS[activity].format(d=duration)
    title = f"Brain Activation: {activity.title()} ({duration} min)"

    console.print(Panel(f"🧠 Brain Visualization — {activity.title()}", style="blue"))
    console.print(f"\nStimulus: [italic]{text_stimulus}[/italic]\n")

    analyzer = get_tribe_analyzer()

    # Ensure model is initialised and attempt to load TRIBEv2
    with console.status("[cyan]Loading TRIBEv2 model...[/cyan]"):
        asyncio.run(analyzer.initialize())

    model_loaded = analyzer.model_wrapper.is_loaded
    if model_loaded:
        console.print("✅ TRIBEv2 model loaded — generating real fMRI predictions")
    else:
        console.print("⚠️  TRIBEv2 not loaded — using simulation (heatmap only)", style="yellow")

    # Run prediction
    with console.status("[cyan]Running prediction...[/cyan]"):
        roi_dict = asyncio.run(analyzer.model_wrapper.predict_response(text_stimulus=text_stimulus))

    roi_activations = {k: float(v.get("activation", 0.5)) for k, v in roi_dict.items()}
    raw_preds = analyzer.model_wrapper._last_raw_predictions

    # Generate visualizations
    viz = BrainVisualizer()
    paths: dict = {}

    with console.status("[cyan]Rendering visualizations...[/cyan]"):
        if raw_preds is not None:
            paths = viz.generate_all(
                raw_preds,
                roi_activations,
                title=title,
                viz_types=list(viz_types),
                auto_open_interactive=not no_open,
                modality=analyzer.model_wrapper._last_modality,
            )
        else:
            # Simulation fallback: only heatmap is meaningful without raw vertices
            heatmap_path = viz.plot_roi_heatmap(roi_activations, title=title)
            paths = {"heatmap": str(heatmap_path)}
            if "interactive" in viz_types or "static" in viz_types:
                console.print(
                    "[yellow]Note: 3D surface viz requires TRIBEv2 model weights. "
                    "Add HUGGING_FACE_TOKEN to .env to enable.[/yellow]"
                )

    # Report results
    console.print("\n[bold green]Generated files:[/bold green]")
    for vtype, path in paths.items():
        if path:
            console.print(f"  ✅ [cyan]{vtype:12}[/cyan] → {path}")
        else:
            console.print(f"  ❌ [red]{vtype:12}[/red] → failed")

    if not no_open and paths.get("interactive"):
        console.print("\n[dim]Interactive HTML opened in browser.[/dim]")


# ===========================================
# Report Command
# ===========================================


@main.command()
@click.option(
    "--period", type=click.Choice(["week", "month", "custom"]), default="week", help="Report period"
)
@click.option("--start-date", help="Start date for custom period (YYYY-MM-DD)")
@click.option("--end-date", help="End date for custom period (YYYY-MM-DD)")
@click.option("--output", type=click.Path(), help="Output file path")
def report(period: str, start_date: Optional[str], end_date: Optional[str], output: Optional[str]):
    """Generate a health report."""
    manager = get_health_manager()
    engine = get_ai_engine()

    # Calculate date range
    today = date.today()
    if period == "week":
        start = today - timedelta(days=7)
        end = today
    elif period == "month":
        start = today - timedelta(days=30)
        end = today
    else:
        start = parse_date(start_date) if start_date else today - timedelta(days=7)
        end = parse_date(end_date) if end_date else today

    # Get data
    summary = asyncio.run(manager.get_health_summary(start, end))
    metrics = asyncio.run(manager.get_health_metrics(start, end))
    insights_list = asyncio.run(engine.generate_health_insights(metrics, summary))

    # Build report
    lines = []
    lines.append("=" * 60)
    lines.append("NOTIONHEALTH AI - HEALTH REPORT (v2.0 with Tribe v2)")
    lines.append("=" * 60)
    lines.append(f"\nPeriod: {format_date(start)} to {format_date(end)}")
    lines.append(f"Days Analyzed: {summary.total_days}")
    lines.append("")

    # Summary section
    lines.append("📊 SUMMARY")
    lines.append("-" * 40)

    if summary.avg_weight:
        lines.append(f"  Average Weight: {summary.avg_weight:.1f} lbs")
        if summary.weight_change:
            lines.append(f"  Weight Change: {summary.weight_change:+.1f} lbs")

    if summary.avg_sleep_hours:
        lines.append(f"  Average Sleep: {summary.avg_sleep_hours:.1f} hours")

    if summary.total_exercise_minutes:
        lines.append(f"  Total Exercise: {summary.total_exercise_minutes} minutes")

    if summary.avg_mood:
        lines.append(f"  Average Mood: {summary.avg_mood:.1f}/10")

    if summary.avg_energy:
        lines.append(f"  Average Energy: {summary.avg_energy:.1f}/10")

    lines.append("")

    # Insights section
    lines.append("💡 AI INSIGHTS")
    lines.append("-" * 40)

    for insight in insights_list:
        lines.append(f"\n  [{insight.category.upper()}] {insight.title}")
        lines.append(f"  {insight.summary}")

        if insight.recommendations:
            lines.append("  Recommendations:")
            for rec in insight.recommendations:
                lines.append(f"    • {rec}")

    lines.append("")
    lines.append("=" * 60)
    lines.append("Generated by NotionHealth AI with Tribe v2 Brain Prediction")

    report_text = "\n".join(lines)

    # Output
    if output:
        with open(output, "w") as f:
            f.write(report_text)
        console.print(f"✅ Report saved to {output}", style="green")
    else:
        console.print(report_text)


# ===========================================
# Interactive Mode
# ===========================================


@main.command()
def interactive():
    """Start interactive mode."""
    console.print(
        Panel(
            "Welcome to NotionHealth AI v2.0!\n"
            "Track your health, medications, and get AI insights\n"
            "powered by Tribe v2 brain prediction.\n\n"
            "Features:\n"
            "  📊 Health metrics tracking\n"
            "  💊 Medication management\n"
            "  🧠 Tribe v2 brain response prediction\n"
            "  📅 Appointment scheduling\n"
            "  🎯 Goal tracking",
            title="🏥 NotionHealth AI",
            style="blue",
        )
    )

    while True:
        action = select(
            "What would you like to do?",
            choices=[
                "📝 Log health metric",
                "💊 View medication schedule",
                "📅 View appointments",
                "🎯 View goals",
                "🧠 Get AI insights",
                "🧬 Brain prediction (Tribe v2)",
                "📊 Generate report",
                "❌ Exit",
            ],
        ).ask()

        if action is None or action == "❌ Exit":
            console.print("Goodbye! Stay healthy! 💚", style="green")
            break

        elif action == "📝 Log health metric":
            # Trigger log metric command
            ctx = click.Context(metric)
            metric.invoke(ctx)

        elif action == "💊 View medication schedule":
            ctx = click.Context(medications)
            medications.invoke(ctx)

        elif action == "📅 View appointments":
            ctx = click.Context(appointments)
            appointments.invoke(ctx)

        elif action == "🎯 View goals":
            ctx = click.Context(goals)
            goals.invoke(ctx)

        elif action == "🧠 Get AI insights":
            ctx = click.Context(insights)
            insights.invoke(ctx)

        elif action == "🧬 Brain prediction (Tribe v2)":
            # Sub-menu for brain predictions
            brain_action = select(
                "Brain prediction options:",
                choices=[
                    "🎯 Predict activity response",
                    "📈 Analyze brain-health correlation",
                    "📅 Generate optimal schedule",
                    "⬅️  Back to main menu",
                ],
            ).ask()

            if brain_action == "🎯 Predict activity response":
                ctx = click.Context(predict)
                predict.invoke(ctx)
            elif brain_action == "📈 Analyze brain-health correlation":
                ctx = click.Context(analyze)
                analyze.invoke(ctx)
            elif brain_action == "📅 Generate optimal schedule":
                ctx = click.Context(schedule)
                schedule.invoke(ctx)

        elif action == "📊 Generate report":
            ctx = click.Context(report)
            report.invoke(ctx)


if __name__ == "__main__":
    main()
