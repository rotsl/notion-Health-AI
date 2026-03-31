"""
Health Manager - Core health data management module.

This module provides the main interface for managing health data,
including logging metrics, tracking medications, and managing appointments.
"""

import os
from datetime import date, timedelta
from typing import Optional, List, Dict, Any
from loguru import logger

from notion_health_ai.models import (
    HealthMetric,
    Medication,
    Appointment,
    HealthGoal,
    SymptomLog,
    HealthSummary,
    BrainAnalysisSession,
    CognitiveInsight,
    MCPCallResult,
    MoodLevel,
    EnergyLevel,
    ExerciseType,
    MedicationFrequency,
    AppointmentType,
    GoalType,
    SymptomSeverity,
)
from notion_health_ai.notion_client import NotionClient
from notion_health_ai.tribe_integration import TribeHealthAnalyzer


class HealthManager:
    """
    Main health data management class.

    Provides methods for logging, retrieving, and analyzing health data
    stored in Notion databases via the Notion API.
    """

    def __init__(
        self,
        notion_api_key: Optional[str] = None,
        health_database_id: Optional[str] = None,
        medication_database_id: Optional[str] = None,
        appointment_database_id: Optional[str] = None,
        goals_database_id: Optional[str] = None,
        symptoms_database_id: Optional[str] = None,
        brain_database_id: Optional[str] = None,
    ):
        """
        Initialize the Health Manager.

        Args:
            notion_api_key: Notion integration API key
            health_database_id: Health metrics database ID
            medication_database_id: Medications database ID
            appointment_database_id: Appointments database ID
            goals_database_id: Health goals database ID
            symptoms_database_id: Symptoms database ID
        """

        def _clean_cfg(value: Optional[str]) -> Optional[str]:
            if value is None:
                return None
            cleaned = str(value).strip()
            if cleaned.lower() in {"", "none", "null"}:
                return None
            return cleaned

        # Load from environment if not provided
        self.notion_api_key = _clean_cfg(notion_api_key or os.getenv("NOTION_API_KEY"))
        self.health_database_id = _clean_cfg(
            health_database_id or os.getenv("NOTION_HEALTH_DATABASE_ID")
        )
        self.medication_database_id = _clean_cfg(
            medication_database_id or os.getenv("NOTION_MEDICATION_DATABASE_ID")
        )
        self.appointment_database_id = _clean_cfg(
            appointment_database_id or os.getenv("NOTION_APPOINTMENT_DATABASE_ID")
        )
        self.goals_database_id = _clean_cfg(
            goals_database_id or os.getenv("NOTION_GOALS_DATABASE_ID")
        )
        self.symptoms_database_id = _clean_cfg(
            symptoms_database_id or os.getenv("NOTION_SYMPTOMS_DATABASE_ID")
        )
        self.brain_database_id = _clean_cfg(
            brain_database_id or os.getenv("NOTION_BRAIN_DATABASE_ID")
        )

        # Initialize Notion client
        self.client = NotionClient(self.notion_api_key)

        # Lazy-initialised TRIBEv2 analyser
        self._tribe: Optional[TribeHealthAnalyzer] = None

        logger.info("HealthManager initialized")

    # ===========================================
    # Health Metrics Methods
    # ===========================================

    async def log_health_metric(
        self,
        metric_date: Optional[date] = None,
        weight: Optional[float] = None,
        sleep_hours: Optional[float] = None,
        sleep_quality: Optional[int] = None,
        exercise_type: Optional[str] = None,
        exercise_minutes: Optional[int] = None,
        steps: Optional[int] = None,
        mood: Optional[int] = None,
        energy: Optional[int] = None,
        water_glasses: Optional[int] = None,
        blood_pressure_systolic: Optional[int] = None,
        blood_pressure_diastolic: Optional[int] = None,
        heart_rate: Optional[int] = None,
        notes: Optional[str] = None,
    ) -> MCPCallResult:
        """
        Log a health metric entry.

        Args:
            metric_date: Date of the measurement (defaults to today)
            weight: Weight in pounds
            sleep_hours: Hours of sleep
            sleep_quality: Sleep quality rating 1-10
            exercise_type: Type of exercise performed
            exercise_minutes: Duration of exercise in minutes
            steps: Daily step count
            mood: Mood rating 1-10
            energy: Energy level 1-10
            water_glasses: Glasses of water consumed
            blood_pressure_systolic: Systolic BP
            blood_pressure_diastolic: Diastolic BP
            heart_rate: Resting heart rate
            notes: Additional notes

        Returns:
            MCPCallResult with success status and data
        """
        try:
            if not self.health_database_id:
                raise ValueError(
                    "NOTION_HEALTH_DATABASE_ID is not configured. "
                    "Run 'python scripts/setup_notion.py' or set it in .env "
                    "and restart the API."
                )

            metric_date = metric_date or date.today()

            # Create health metric object
            metric = HealthMetric(
                metric_date=metric_date,
                weight=weight,
                sleep_hours=sleep_hours,
                sleep_quality=sleep_quality,
                exercise_type=ExerciseType(exercise_type) if exercise_type else None,
                exercise_minutes=exercise_minutes,
                steps=steps,
                mood=MoodLevel(mood) if mood else None,
                energy=EnergyLevel(energy) if energy else None,
                water_glasses=water_glasses,
                blood_pressure_systolic=blood_pressure_systolic,
                blood_pressure_diastolic=blood_pressure_diastolic,
                heart_rate=heart_rate,
                notes=notes,
            )

            # Save to Notion
            page_id = await self.client.create_health_metric(self.health_database_id, metric)

            logger.info(f"Logged health metric for {metric_date}: weight={weight}, mood={mood}")

            return MCPCallResult(
                success=True,
                message=f"Successfully logged health metric for {metric_date}",
                data={"page_id": page_id, "metric": metric.model_dump()},
            )

        except Exception as e:
            logger.error(f"Failed to log health metric: {e}")
            return MCPCallResult(
                success=False,
                message="Failed to log health metric",
                error=str(e),
            )

    async def get_health_metrics(
        self,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
        limit: int = 30,
    ) -> List[HealthMetric]:
        """
        Retrieve health metrics for a date range.

        Args:
            start_date: Start of date range (defaults to 30 days ago)
            end_date: End of date range (defaults to today)
            limit: Maximum number of records to return

        Returns:
            List of HealthMetric objects
        """
        if not self.health_database_id:
            raise ValueError(
                "NOTION_HEALTH_DATABASE_ID is not configured. "
                "Run 'python scripts/setup_notion.py' or set it in .env "
                "and restart the API."
            )

        start_date = start_date or (date.today() - timedelta(days=30))
        end_date = end_date or date.today()

        metrics = await self.client.query_health_metrics(
            self.health_database_id, start_date, end_date, limit
        )

        logger.info(f"Retrieved {len(metrics)} health metrics from {start_date} to {end_date}")
        return metrics

    async def get_today_summary(self) -> Dict[str, Any]:
        """
        Get today's health summary.

        Returns:
            Dictionary with today's health metrics
        """
        today = date.today()
        metrics = await self.get_health_metrics(today, today, limit=1)

        if metrics:
            metric = metrics[0]
            return {
                "date": today.isoformat(),
                "weight": metric.weight,
                "sleep_hours": metric.sleep_hours,
                "sleep_quality": metric.sleep_quality,
                "exercise_minutes": metric.exercise_minutes,
                "steps": metric.steps,
                "mood": metric.mood.value if metric.mood else None,
                "energy": metric.energy.value if metric.energy else None,
                "notes": metric.notes,
            }

        return {"date": today.isoformat(), "message": "No data logged for today"}

    # ===========================================
    # Medication Methods
    # ===========================================

    async def add_medication(
        self,
        name: str,
        dosage: str,
        frequency: str,
        start_date: date,
        times: Optional[List[str]] = None,
        end_date: Optional[date] = None,
        prescribing_doctor: Optional[str] = None,
        purpose: Optional[str] = None,
        notes: Optional[str] = None,
    ) -> MCPCallResult:
        """
        Add a new medication to track.

        Args:
            name: Medication name
            dosage: Dosage amount
            frequency: How often to take (daily, twice_daily, etc.)
            start_date: When medication started
            times: List of times to take (HH:MM format)
            end_date: When medication should end
            prescribing_doctor: Doctor who prescribed
            purpose: What it's for
            notes: Additional notes

        Returns:
            MCPCallResult with success status
        """
        try:
            medication = Medication(
                name=name,
                dosage=dosage,
                frequency=MedicationFrequency(frequency),
                start_date=start_date,
                times=times or [],
                end_date=end_date,
                prescribing_doctor=prescribing_doctor,
                purpose=purpose,
                notes=notes,
            )

            page_id = await self.client.create_medication(self.medication_database_id, medication)

            logger.info(f"Added medication: {name} {dosage}")

            return MCPCallResult(
                success=True,
                message=f"Successfully added medication: {name}",
                data={"page_id": page_id, "medication": medication.model_dump()},
            )

        except Exception as e:
            logger.error(f"Failed to add medication: {e}")
            return MCPCallResult(
                success=False,
                message="Failed to add medication",
                error=str(e),
            )

    async def get_active_medications(self) -> List[Medication]:
        """
        Get all active medications.

        Returns:
            List of active Medication objects
        """
        medications = await self.client.query_medications(
            self.medication_database_id, active_only=True
        )
        return medications

    async def get_medication_schedule(
        self, for_date: Optional[date] = None
    ) -> List[Dict[str, Any]]:
        """
        Get medication schedule for a specific date.

        Args:
            for_date: Date to get schedule for (defaults to today)

        Returns:
            List of medication schedule entries
        """
        for_date = for_date or date.today()
        medications = await self.get_active_medications()

        schedule = []
        for med in medications:
            for time_slot in med.times:
                schedule.append(
                    {
                        "medication": med.name,
                        "dosage": med.dosage,
                        "time": time_slot,
                        "purpose": med.purpose,
                        "taken": False,  # Would be tracked separately
                    }
                )

        # Sort by time
        schedule.sort(key=lambda x: x["time"])
        return schedule

    # ===========================================
    # Appointment Methods
    # ===========================================

    async def add_appointment(
        self,
        appointment_date: date,
        appointment_time: str,
        doctor_name: str,
        appointment_type: str,
        facility: Optional[str] = None,
        reason: Optional[str] = None,
        notes: Optional[str] = None,
    ) -> MCPCallResult:
        """
        Add a new medical appointment.

        Args:
            appointment_date: Date of appointment
            appointment_time: Time of appointment (HH:MM format)
            doctor_name: Name of doctor/provider
            appointment_type: Type of appointment
            facility: Clinic/hospital name
            reason: Reason for visit
            notes: Additional notes

        Returns:
            MCPCallResult with success status
        """
        try:
            from datetime import datetime as dt

            time_obj = dt.strptime(appointment_time, "%H:%M").time()

            appointment = Appointment(
                appointment_date=appointment_date,
                appointment_time=time_obj,
                doctor_name=doctor_name,
                appointment_type=AppointmentType(appointment_type),
                facility=facility,
                reason=reason,
                notes=notes,
            )

            page_id = await self.client.create_appointment(
                self.appointment_database_id, appointment
            )

            logger.info(f"Added appointment with {doctor_name} on {appointment_date}")

            return MCPCallResult(
                success=True,
                message=f"Successfully added appointment for {appointment_date}",
                data={"page_id": page_id, "appointment": appointment.model_dump()},
            )

        except Exception as e:
            logger.error(f"Failed to add appointment: {e}")
            return MCPCallResult(
                success=False,
                message="Failed to add appointment",
                error=str(e),
            )

    async def get_upcoming_appointments(self, days: int = 30) -> List[Appointment]:
        """
        Get upcoming appointments.

        Args:
            days: Number of days to look ahead

        Returns:
            List of upcoming Appointment objects
        """
        start_date = date.today()
        end_date = start_date + timedelta(days=days)

        appointments = await self.client.query_appointments(
            self.appointment_database_id, start_date, end_date
        )
        return appointments

    # ===========================================
    # Goals Methods
    # ===========================================

    async def set_health_goal(
        self,
        goal_type: str,
        title: str,
        target_value: float,
        unit: str,
        target_date: date,
        description: Optional[str] = None,
    ) -> MCPCallResult:
        """
        Set a new health goal.

        Args:
            goal_type: Type of goal (weight, exercise, sleep, etc.)
            title: Goal title
            target_value: Target value to achieve
            unit: Unit of measurement
            target_date: Deadline for goal
            description: Detailed description

        Returns:
            MCPCallResult with success status
        """
        try:
            goal = HealthGoal(
                goal_type=GoalType(goal_type),
                title=title,
                target_value=target_value,
                unit=unit,
                target_date=target_date,
                description=description,
            )

            page_id = await self.client.create_goal(self.goals_database_id, goal)

            logger.info(f"Set health goal: {title}")

            return MCPCallResult(
                success=True,
                message=f"Successfully set goal: {title}",
                data={"page_id": page_id, "goal": goal.model_dump()},
            )

        except Exception as e:
            logger.error(f"Failed to set health goal: {e}")
            return MCPCallResult(
                success=False,
                message="Failed to set health goal",
                error=str(e),
            )

    async def get_active_goals(self) -> List[HealthGoal]:
        """
        Get all active health goals.

        Returns:
            List of active HealthGoal objects
        """
        goals = await self.client.query_goals(self.goals_database_id, active_only=True)
        return goals

    async def update_goal_progress(self, goal_id: str, current_value: float) -> MCPCallResult:
        """
        Update progress on a health goal.

        Args:
            goal_id: Goal identifier
            current_value: Current progress value

        Returns:
            MCPCallResult with success status
        """
        try:
            await self.client.update_goal_progress(self.goals_database_id, goal_id, current_value)

            return MCPCallResult(
                success=True,
                message=f"Updated goal progress to {current_value}",
            )

        except Exception as e:
            logger.error(f"Failed to update goal progress: {e}")
            return MCPCallResult(
                success=False,
                message="Failed to update goal progress",
                error=str(e),
            )

    # ===========================================
    # Symptom Methods
    # ===========================================

    async def log_symptom(
        self,
        symptom_name: str,
        severity: int,
        symptom_date: Optional[date] = None,
        symptom_time: Optional[str] = None,
        body_part: Optional[str] = None,
        duration_minutes: Optional[int] = None,
        triggers: Optional[List[str]] = None,
        notes: Optional[str] = None,
    ) -> MCPCallResult:
        """
        Log a symptom entry.

        Args:
            symptom_name: Name of the symptom
            severity: Severity rating 1-10
            symptom_date: Date symptom occurred (defaults to today)
            symptom_time: Time symptom started (HH:MM)
            body_part: Affected body part
            duration_minutes: Duration in minutes
            triggers: List of identified triggers
            notes: Additional notes

        Returns:
            MCPCallResult with success status
        """
        try:
            symptom_date = symptom_date or date.today()
            time_obj = None
            if symptom_time:
                from datetime import datetime as dt

                time_obj = dt.strptime(symptom_time, "%H:%M").time()

            symptom = SymptomLog(
                symptom_date=symptom_date,
                symptom_time=time_obj,
                symptom_name=symptom_name,
                severity=SymptomSeverity(severity),
                body_part=body_part,
                duration_minutes=duration_minutes,
                triggers=triggers or [],
                notes=notes,
            )

            page_id = await self.client.create_symptom(self.symptoms_database_id, symptom)

            logger.info(f"Logged symptom: {symptom_name} (severity {severity})")

            return MCPCallResult(
                success=True,
                message=f"Successfully logged symptom: {symptom_name}",
                data={"page_id": page_id, "symptom": symptom.model_dump()},
            )

        except Exception as e:
            logger.error(f"Failed to log symptom: {e}")
            return MCPCallResult(
                success=False,
                message="Failed to log symptom",
                error=str(e),
            )

    async def get_symptom_history(
        self, start_date: Optional[date] = None, end_date: Optional[date] = None
    ) -> List[SymptomLog]:
        """
        Get symptom history for a date range.

        Args:
            start_date: Start of date range
            end_date: End of date range

        Returns:
            List of SymptomLog objects
        """
        start_date = start_date or (date.today() - timedelta(days=30))
        end_date = end_date or date.today()

        symptoms = await self.client.query_symptoms(self.symptoms_database_id, start_date, end_date)
        return symptoms

    # ===========================================
    # Summary Methods
    # ===========================================

    async def get_health_summary(
        self, start_date: Optional[date] = None, end_date: Optional[date] = None
    ) -> HealthSummary:
        """
        Generate a health summary for a date range.

        Args:
            start_date: Start of summary period
            end_date: End of summary period

        Returns:
            HealthSummary object with statistics
        """
        start_date = start_date or (date.today() - timedelta(days=7))
        end_date = end_date or date.today()

        metrics = await self.get_health_metrics(start_date, end_date, limit=100)

        # Calculate statistics
        weights = [m.weight for m in metrics if m.weight is not None]
        sleep_hours = [m.sleep_hours for m in metrics if m.sleep_hours is not None]
        sleep_quality = [m.sleep_quality for m in metrics if m.sleep_quality is not None]
        exercise_minutes = [m.exercise_minutes for m in metrics if m.exercise_minutes is not None]
        steps = [m.steps for m in metrics if m.steps is not None]
        moods = [m.mood.value for m in metrics if m.mood is not None]
        energies = [m.energy.value for m in metrics if m.energy is not None]
        systolic = [
            m.blood_pressure_systolic for m in metrics if m.blood_pressure_systolic is not None
        ]
        diastolic = [
            m.blood_pressure_diastolic for m in metrics if m.blood_pressure_diastolic is not None
        ]
        heart_rates = [m.heart_rate for m in metrics if m.heart_rate is not None]

        import statistics

        summary = HealthSummary(
            period_start=start_date,
            period_end=end_date,
            total_days=(end_date - start_date).days + 1,
            avg_weight=statistics.mean(weights) if weights else None,
            weight_change=(weights[-1] - weights[0]) if len(weights) >= 2 else None,
            min_weight=min(weights) if weights else None,
            max_weight=max(weights) if weights else None,
            avg_sleep_hours=statistics.mean(sleep_hours) if sleep_hours else None,
            avg_sleep_quality=statistics.mean(sleep_quality) if sleep_quality else None,
            total_sleep_hours=sum(sleep_hours) if sleep_hours else None,
            total_exercise_minutes=sum(exercise_minutes) if exercise_minutes else None,
            total_exercise_days=len(exercise_minutes),
            total_steps=sum(steps) if steps else None,
            avg_daily_steps=statistics.mean(steps) if steps else None,
            avg_mood=statistics.mean(moods) if moods else None,
            avg_energy=statistics.mean(energies) if energies else None,
            avg_systolic=statistics.mean(systolic) if systolic else None,
            avg_diastolic=statistics.mean(diastolic) if diastolic else None,
            avg_heart_rate=statistics.mean(heart_rates) if heart_rates else None,
        )

        return summary

    # ===========================================
    # TRIBEv2 Brain Analysis Methods
    # ===========================================

    async def _get_tribe(self) -> TribeHealthAnalyzer:
        """Return (and lazily initialise) the TribeHealthAnalyzer."""
        if self._tribe is None:
            self._tribe = TribeHealthAnalyzer()
            await self._tribe.initialize()
        return self._tribe

    async def run_brain_analysis(
        self,
        activity: str = "exercise",
        duration_minutes: int = 30,
        context_days: int = 7,
    ) -> MCPCallResult:
        """
        Run a TRIBEv2 brain activity prediction and save the session to Notion.

        Args:
            activity: Activity type to analyse (exercise, meditation, sleep, etc.)
            duration_minutes: Duration of the activity in minutes
            context_days: How many days of health data to include as context

        Returns:
            MCPCallResult with prediction data and Notion page ID
        """
        try:
            tribe = await self._get_tribe()

            # Predict brain response for the activity
            prediction = await tribe.predict_brain_response(activity, duration_minutes)
            if not prediction.get("success"):
                return MCPCallResult(
                    success=False,
                    message="Brain prediction failed",
                    error=prediction.get("error", "Unknown error"),
                )

            pred_data = prediction["prediction"]

            # Build a BrainAnalysisSession from the prediction
            top_regions = sorted(
                pred_data.get("brain_regions", {}).items(),
                key=lambda kv: kv[1].get("intensity", 0),
                reverse=True,
            )[:3]
            region_summary = "; ".join(
                f"{r} ({v['response']} {v['intensity']:.2f})" for r, v in top_regions
            )

            session = BrainAnalysisSession(
                input_type="activity",
                input_summary=(
                    f"{activity} for {duration_minutes} min"
                    f" \u2014 top regions: {region_summary}"
                ),
                cognitive_load_score=pred_data.get("wellness_score"),
                stress_indicator=None,
                emotional_valence=None,
                default_mode_activity=pred_data.get("brain_regions", {})
                .get("default_mode_network", {})
                .get("intensity"),
                executive_control_activity=pred_data.get("brain_regions", {})
                .get("prefrontal_cortex", {})
                .get("intensity"),
                salience_network_activity=pred_data.get("brain_regions", {})
                .get("anterior_cingulate", {})
                .get("intensity"),
                insights=f"Optimal timing: {pred_data.get('optimal_timing', 'N/A')}. "
                f"Intensity: {pred_data.get('intensity', 'moderate')}.",
                recommendations=pred_data.get("benefits", [])[:3],
                health_context=f"Activity: {activity}, Duration: {duration_minutes} min",
                model_available=tribe.model_wrapper.is_loaded,
            )

            # Save to Notion if database is configured
            page_id = None
            if self.brain_database_id:
                page_id = await self.client.create_brain_analysis(self.brain_database_id, session)

            logger.info(f"Brain analysis complete: {activity} {duration_minutes}min")
            return MCPCallResult(
                success=True,
                message=f"Brain analysis complete for {activity} ({duration_minutes} min)",
                data={
                    "page_id": page_id,
                    "prediction": pred_data,
                    "session": session.model_dump(),
                },
            )

        except Exception as e:
            logger.error(f"Brain analysis failed: {e}")
            return MCPCallResult(
                success=False,
                message="Brain analysis failed",
                error=str(e),
            )

    async def get_brain_insights(
        self,
        context_days: int = 7,
    ) -> MCPCallResult:
        """
        Correlate recent health metrics with brain activity patterns using TRIBEv2.

        Args:
            context_days: Days of health data to analyse

        Returns:
            MCPCallResult with correlation analysis
        """
        try:
            tribe = await self._get_tribe()
            start_date = date.today() - timedelta(days=context_days)
            metrics = await self.get_health_metrics(start_date, date.today(), limit=100)
            summary = await self.get_health_summary(start_date, date.today())

            analysis = await tribe.analyze_brain_health_correlation(metrics, summary)
            return MCPCallResult(
                success=True,
                message=f"Brain-health correlation analysis for the past {context_days} days",
                data=analysis,
            )

        except Exception as e:
            logger.error(f"Brain insights failed: {e}")
            return MCPCallResult(
                success=False,
                message="Brain insights failed",
                error=str(e),
            )

    async def get_optimal_brain_schedule(self) -> MCPCallResult:
        """
        Return an optimal daily schedule derived from brain science and TRIBEv2 data.

        Returns:
            MCPCallResult with schedule data
        """
        try:
            tribe = await self._get_tribe()
            schedule = await tribe.generate_optimal_schedule()
            return MCPCallResult(
                success=True,
                message="Optimal brain-science daily schedule generated",
                data=schedule,
            )
        except Exception as e:
            logger.error(f"Optimal schedule failed: {e}")
            return MCPCallResult(
                success=False,
                message="Failed to generate optimal schedule",
                error=str(e),
            )

    async def get_cognitive_health_report(
        self,
        days: int = 30,
    ) -> MCPCallResult:
        """
        Generate a comprehensive cognitive health report combining health data,
        TRIBEv2 brain analysis, and personalised recommendations.

        Args:
            days: Days of data to include

        Returns:
            MCPCallResult with full cognitive health report
        """
        try:
            import statistics as _stats

            start_date = date.today() - timedelta(days=days)
            metrics = await self.get_health_metrics(start_date, date.today(), limit=200)
            summary = await self.get_health_summary(start_date, date.today())
            symptoms = await self.get_symptom_history(start_date, date.today())

            tribe = await self._get_tribe()
            correlation_result = await tribe.analyze_brain_health_correlation(metrics, summary)
            schedule_result = await tribe.generate_optimal_schedule()

            # Compute CognitiveInsight aggregate
            cog_loads = [
                m.mood.value for m in metrics if m.mood
            ]  # mood as proxy for cognitive wellness
            stress_proxies = [s.severity.value for s in symptoms]

            avg_cog = _stats.mean(cog_loads) if cog_loads else None
            avg_stress = _stats.mean(stress_proxies) if stress_proxies else None

            # Simple trend: compare first half vs second half
            def _trend(vals: list) -> str:
                if len(vals) < 4:
                    return "stable"
                mid = len(vals) // 2
                return (
                    "improving"
                    if _stats.mean(vals[:mid]) > _stats.mean(vals[mid:])
                    else "worsening"
                )

            cognitive_insight = CognitiveInsight(
                analysis_period_days=days,
                sessions_analyzed=len(metrics),
                data_points_used=len(metrics) + len(symptoms),
                avg_cognitive_load=avg_cog,
                avg_stress_indicator=avg_stress,
                cognitive_load_trend=_trend(cog_loads),
                stress_trend=_trend(stress_proxies),
                key_findings=[
                    c["insight"]
                    for c in correlation_result.get("analysis", {}).get("correlations", [])
                ][:5],
                recommendations=correlation_result.get("analysis", {}).get("recommendations", []),
            )

            report = {
                "period_days": days,
                "health_summary": summary.model_dump(),
                "cognitive_insight": cognitive_insight.model_dump(),
                "brain_health_correlations": correlation_result.get("analysis", {}),
                "optimal_daily_schedule": schedule_result.get("schedule", {}),
                "tribe_model_active": tribe.model_wrapper.is_loaded,
            }

            return MCPCallResult(
                success=True,
                message=f"Cognitive health report for the past {days} days",
                data=report,
            )

        except Exception as e:
            logger.error(f"Cognitive health report failed: {e}")
            return MCPCallResult(
                success=False,
                message="Failed to generate cognitive health report",
                error=str(e),
            )
