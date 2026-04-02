# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Rohan R
# Author: Rohan R

"""
Notion Client - API wrapper for Notion database operations.

This module provides a clean interface for interacting with Notion databases,
handling all API calls and data transformations.
"""

import os
from datetime import date, datetime, timedelta
from typing import Dict, List, Optional

import httpx
from loguru import logger

from notion_health_ai.models import (
    Appointment,
    AppointmentStatus,
    AppointmentType,
    BrainAnalysisSession,
    EnergyLevel,
    ExerciseType,
    GoalStatus,
    GoalType,
    HealthGoal,
    HealthMetric,
    Medication,
    MedicationFrequency,
    MoodLevel,
    SymptomLog,
    SymptomSeverity,
)


class NotionClient:
    """
    Notion API client for health data operations.

    Handles all interactions with Notion databases including
    creating, reading, updating, and querying health records.
    """

    BASE_URL = "https://api.notion.com/v1"
    NOTION_VERSION = "2022-06-28"

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the Notion client.

        Args:
            api_key: Notion integration API key
        """
        self.api_key = api_key or os.getenv("NOTION_API_KEY")
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Notion-Version": self.NOTION_VERSION,
        }
        self.client = httpx.AsyncClient(headers=self.headers, timeout=30.0)
        logger.info("NotionClient initialized")

    async def close(self):
        """Close the HTTP client."""
        await self.client.aclose()

    @staticmethod
    def _parse_time_value(time_str: Optional[str]):
        """Accept both 24-hour and 12-hour Notion time text values."""
        if not time_str:
            return None
        for fmt in ("%H:%M", "%I:%M %p"):
            try:
                return datetime.strptime(time_str.strip(), fmt).time()
            except ValueError:
                continue
        raise ValueError(f"Unsupported time format: {time_str}")

    async def _request(self, method: str, endpoint: str, json_data: Optional[Dict] = None) -> Dict:
        """
        Make an API request to Notion.

        Args:
            method: HTTP method (GET, POST, PATCH)
            endpoint: API endpoint
            json_data: Request body

        Returns:
            Response JSON data
        """
        url = f"{self.BASE_URL}/{endpoint}"

        try:
            if method == "GET":
                response = await self.client.get(url)
            elif method == "POST":
                response = await self.client.post(url, json=json_data)
            elif method == "PATCH":
                response = await self.client.patch(url, json=json_data)
            else:
                raise ValueError(f"Unsupported method: {method}")

            response.raise_for_status()
            return response.json()

        except httpx.HTTPStatusError as e:
            logger.error(f"Notion API error: {e.response.status_code} - {e.response.text}")
            raise
        except Exception as e:
            logger.error(f"Request failed: {e}")
            raise

    # ===========================================
    # Health Metrics Operations
    # ===========================================

    async def create_health_metric(self, database_id: str, metric: HealthMetric) -> str:
        """
        Create a new health metric entry in Notion.

        Args:
            database_id: Health metrics database ID
            metric: HealthMetric object to create

        Returns:
            Created page ID
        """
        properties = {
            "Date": {"date": {"start": metric.metric_date.isoformat()}},
        }

        if metric.weight:
            properties["Weight"] = {"number": metric.weight}
        if metric.sleep_hours:
            properties["Sleep Hours"] = {"number": metric.sleep_hours}
        if metric.sleep_quality:
            properties["Sleep Quality"] = {"number": metric.sleep_quality}
        if metric.exercise_type:
            properties["Exercise Type"] = {"select": {"name": metric.exercise_type.value}}
        if metric.exercise_minutes:
            properties["Exercise Minutes"] = {"number": metric.exercise_minutes}
        if metric.steps:
            properties["Steps"] = {"number": metric.steps}
        if metric.mood:
            properties["Mood"] = {"number": metric.mood.value}
        if metric.energy:
            properties["Energy"] = {"number": metric.energy.value}
        if metric.water_glasses:
            properties["Water Glasses"] = {"number": metric.water_glasses}
        if metric.blood_pressure_systolic:
            properties["BP Systolic"] = {"number": metric.blood_pressure_systolic}
        if metric.blood_pressure_diastolic:
            properties["BP Diastolic"] = {"number": metric.blood_pressure_diastolic}
        if metric.heart_rate:
            properties["Heart Rate"] = {"number": metric.heart_rate}
        if metric.notes:
            properties["Notes"] = {"rich_text": [{"text": {"content": metric.notes}}]}

        data = {
            "parent": {"database_id": database_id},
            "properties": properties,
        }

        response = await self._request("POST", "pages", data)
        logger.info(f"Created health metric page: {response['id']}")
        return response["id"]

    async def query_health_metrics(
        self,
        database_id: str,
        start_date: date,
        end_date: date,
        limit: int = 100,
    ) -> List[HealthMetric]:
        """
        Query health metrics from Notion database.

        Args:
            database_id: Health metrics database ID
            start_date: Start of date range
            end_date: End of date range
            limit: Maximum results

        Returns:
            List of HealthMetric objects
        """
        data = {
            "filter": {
                "and": [
                    {
                        "property": "Date",
                        "date": {"on_or_after": start_date.isoformat()},
                    },
                    {
                        "property": "Date",
                        "date": {"before": (end_date + timedelta(days=1)).isoformat()},
                    },
                ]
            },
            "sorts": [{"property": "Date", "direction": "descending"}],
            "page_size": limit,
        }

        response = await self._request("POST", f"databases/{database_id}/query", data)
        metrics = []

        for page in response.get("results", []):
            metric = self._parse_health_metric(page)
            if metric:
                metrics.append(metric)

        return metrics

    def _parse_health_metric(self, page: Dict) -> Optional[HealthMetric]:
        """Parse a Notion page into a HealthMetric object."""
        try:
            props = page.get("properties", {})

            def get_prop(prop_name: str) -> Dict:
                prop = props.get(prop_name)
                return prop if isinstance(prop, dict) else {}

            def get_number(prop_name: str) -> Optional[float]:
                prop = get_prop(prop_name)
                return prop.get("number")

            def get_date_value(prop_name: str) -> Optional[date]:
                prop = get_prop(prop_name)
                date_value = prop.get("date")
                if not isinstance(date_value, dict):
                    return None
                date_str = date_value.get("start")
                if date_str:
                    return datetime.fromisoformat(date_str.replace("Z", "+00:00")).date()
                return None

            def get_select(prop_name: str) -> Optional[str]:
                prop = get_prop(prop_name)
                select_value = prop.get("select")
                if not isinstance(select_value, dict):
                    return None
                return select_value.get("name")

            def get_text(prop_name: str) -> Optional[str]:
                prop = get_prop(prop_name)
                texts = prop.get("rich_text", [])
                return texts[0].get("text", {}).get("content") if texts else None

            metric_date = get_date_value("Date") or date.today()

            return HealthMetric(
                id=page["id"],
                metric_date=metric_date,
                weight=get_number("Weight"),
                sleep_hours=get_number("Sleep Hours"),
                sleep_quality=(
                    int(get_number("Sleep Quality")) if get_number("Sleep Quality") else None
                ),
                exercise_type=(
                    ExerciseType(get_select("Exercise Type"))
                    if get_select("Exercise Type")
                    else None
                ),
                exercise_minutes=(
                    int(get_number("Exercise Minutes")) if get_number("Exercise Minutes") else None
                ),
                steps=int(get_number("Steps")) if get_number("Steps") else None,
                mood=MoodLevel(int(get_number("Mood"))) if get_number("Mood") else None,
                energy=EnergyLevel(int(get_number("Energy"))) if get_number("Energy") else None,
                water_glasses=(
                    int(get_number("Water Glasses")) if get_number("Water Glasses") else None
                ),
                blood_pressure_systolic=(
                    int(get_number("BP Systolic")) if get_number("BP Systolic") else None
                ),
                blood_pressure_diastolic=(
                    int(get_number("BP Diastolic")) if get_number("BP Diastolic") else None
                ),
                heart_rate=int(get_number("Heart Rate")) if get_number("Heart Rate") else None,
                notes=get_text("Notes"),
            )

        except Exception as e:
            logger.error(f"Failed to parse health metric: {e}")
            return None

    # ===========================================
    # Medication Operations
    # ===========================================

    async def create_medication(self, database_id: str, medication: Medication) -> str:
        """Create a new medication entry in Notion."""
        properties = {
            "Name": {"title": [{"text": {"content": medication.name}}]},
            "Dosage": {"rich_text": [{"text": {"content": medication.dosage}}]},
            "Frequency": {"select": {"name": medication.frequency.value}},
            "Start Date": {"date": {"start": medication.start_date.isoformat()}},
            "Active": {"checkbox": medication.is_active},
        }

        if medication.generic_name:
            properties["Generic Name"] = {
                "rich_text": [{"text": {"content": medication.generic_name}}]
            }
        if medication.end_date:
            properties["End Date"] = {"date": {"start": medication.end_date.isoformat()}}
        if medication.prescribing_doctor:
            properties["Prescribing Doctor"] = {
                "rich_text": [{"text": {"content": medication.prescribing_doctor}}]
            }
        if medication.purpose:
            properties["Purpose"] = {"rich_text": [{"text": {"content": medication.purpose}}]}
        if medication.notes:
            properties["Notes"] = {"rich_text": [{"text": {"content": medication.notes}}]}

        data = {
            "parent": {"database_id": database_id},
            "properties": properties,
        }

        response = await self._request("POST", "pages", data)
        logger.info(f"Created medication page: {response['id']}")
        return response["id"]

    async def query_medications(
        self, database_id: str, active_only: bool = True
    ) -> List[Medication]:
        """Query medications from Notion database."""
        filter_query = {}
        if active_only:
            filter_query = {"property": "Active", "checkbox": {"equals": True}}

        data = {
            "filter": filter_query,
            "sorts": [{"property": "Start Date", "direction": "descending"}],
        }

        response = await self._request("POST", f"databases/{database_id}/query", data)
        medications = []

        for page in response.get("results", []):
            medication = self._parse_medication(page)
            if medication:
                medications.append(medication)

        return medications

    def _parse_medication(self, page: Dict) -> Optional[Medication]:
        """Parse a Notion page into a Medication object."""
        try:
            props = page.get("properties", {})

            def get_title(prop_name: str) -> str:
                prop = props.get(prop_name, {})
                texts = prop.get("title", [])
                return texts[0].get("text", {}).get("content", "") if texts else ""

            def get_text(prop_name: str) -> Optional[str]:
                prop = props.get(prop_name, {})
                texts = prop.get("rich_text", [])
                return texts[0].get("text", {}).get("content") if texts else None

            def get_date_value(prop_name: str) -> Optional[date]:
                prop = props.get(prop_name, {})
                date_str = prop.get("date", {}).get("start")
                if date_str:
                    return datetime.fromisoformat(date_str.replace("Z", "+00:00")).date()
                return date.today()

            def get_select(prop_name: str) -> str:
                prop = props.get(prop_name, {})
                return prop.get("select", {}).get("name", "once_daily")

            def get_checkbox(prop_name: str) -> bool:
                prop = props.get(prop_name, {})
                return prop.get("checkbox", True)

            return Medication(
                id=page["id"],
                name=get_title("Name"),
                dosage=get_text("Dosage") or "",
                frequency=MedicationFrequency(get_select("Frequency")),
                start_date=get_date_value("Start Date"),
                end_date=(
                    get_date_value("End Date") if props.get("End Date", {}).get("date") else None
                ),
                prescribing_doctor=get_text("Prescribing Doctor"),
                purpose=get_text("Purpose"),
                notes=get_text("Notes"),
                is_active=get_checkbox("Active"),
            )

        except Exception as e:
            logger.error(f"Failed to parse medication: {e}")
            return None

    # ===========================================
    # Appointment Operations
    # ===========================================

    async def create_appointment(self, database_id: str, appointment: Appointment) -> str:
        """Create a new appointment entry in Notion."""
        properties = {
            "Date": {"date": {"start": appointment.appointment_date.isoformat()}},
            "Doctor Name": {"title": [{"text": {"content": appointment.doctor_name}}]},
            "Type": {"select": {"name": appointment.appointment_type.value}},
            "Status": {"select": {"name": appointment.status.value}},
        }

        if appointment.appointment_time:
            properties["Time"] = {
                "rich_text": [{"text": {"content": appointment.appointment_time.strftime("%H:%M")}}]
            }
        if appointment.facility:
            properties["Facility"] = {"rich_text": [{"text": {"content": appointment.facility}}]}
        if appointment.reason:
            properties["Reason"] = {"rich_text": [{"text": {"content": appointment.reason}}]}
        if appointment.notes:
            properties["Notes"] = {"rich_text": [{"text": {"content": appointment.notes}}]}

        data = {
            "parent": {"database_id": database_id},
            "properties": properties,
        }

        response = await self._request("POST", "pages", data)
        logger.info(f"Created appointment page: {response['id']}")
        return response["id"]

    async def query_appointments(
        self,
        database_id: str,
        start_date: date,
        end_date: date,
    ) -> List[Appointment]:
        """Query appointments from Notion database."""
        data = {
            "filter": {
                "and": [
                    {
                        "property": "Date",
                        "date": {"on_or_after": start_date.isoformat()},
                    },
                    {
                        "property": "Date",
                        "date": {"before": (end_date + timedelta(days=1)).isoformat()},
                    },
                ]
            },
            "sorts": [{"property": "Date", "direction": "ascending"}],
        }

        response = await self._request("POST", f"databases/{database_id}/query", data)
        appointments = []

        for page in response.get("results", []):
            appointment = self._parse_appointment(page)
            if appointment:
                appointments.append(appointment)

        return appointments

    def _parse_appointment(self, page: Dict) -> Optional[Appointment]:
        """Parse a Notion page into an Appointment object."""
        try:
            props = page.get("properties", {})

            def get_title(prop_name: str) -> str:
                prop = props.get(prop_name, {})
                texts = prop.get("title", [])
                return texts[0].get("text", {}).get("content", "") if texts else ""

            def get_text(prop_name: str) -> Optional[str]:
                prop = props.get(prop_name, {})
                texts = prop.get("rich_text", [])
                return texts[0].get("text", {}).get("content") if texts else None

            def get_date_value(prop_name: str) -> date:
                prop = props.get(prop_name, {})
                date_str = prop.get("date", {}).get("start")
                if date_str:
                    return datetime.fromisoformat(date_str.replace("Z", "+00:00")).date()
                return date.today()

            def get_select(prop_name: str) -> str:
                prop = props.get(prop_name, {})
                return prop.get("select", {}).get("name", "checkup")

            time_str = get_text("Time")
            time_obj = None
            if time_str:
                time_obj = self._parse_time_value(time_str)

            return Appointment(
                id=page["id"],
                appointment_date=get_date_value("Date"),
                appointment_time=time_obj or datetime.now().time(),
                doctor_name=get_title("Doctor Name"),
                appointment_type=AppointmentType(get_select("Type")),
                status=AppointmentStatus(get_select("Status")),
                facility=get_text("Facility"),
                reason=get_text("Reason"),
                notes=get_text("Notes"),
            )

        except Exception as e:
            logger.error(f"Failed to parse appointment: {e}")
            return None

    # ===========================================
    # Goals Operations
    # ===========================================

    async def create_goal(self, database_id: str, goal: HealthGoal) -> str:
        """Create a new health goal entry in Notion."""
        properties = {
            "Title": {"title": [{"text": {"content": goal.title}}]},
            "Type": {"select": {"name": goal.goal_type.value}},
            "Target Value": {"number": goal.target_value},
            "Current Value": {"number": goal.current_value},
            "Unit": {"rich_text": [{"text": {"content": goal.unit}}]},
            "Target Date": {"date": {"start": goal.target_date.isoformat()}},
            "Status": {"select": {"name": goal.status.value}},
            "Progress %": {"number": goal.progress_percent},
        }

        if goal.description:
            properties["Description"] = {"rich_text": [{"text": {"content": goal.description}}]}

        data = {
            "parent": {"database_id": database_id},
            "properties": properties,
        }

        response = await self._request("POST", "pages", data)
        logger.info(f"Created goal page: {response['id']}")
        return response["id"]

    async def query_goals(self, database_id: str, active_only: bool = True) -> List[HealthGoal]:
        """Query health goals from Notion database."""
        filter_query = {}
        if active_only:
            filter_query = {
                "or": [
                    {"property": "Status", "select": {"equals": "in_progress"}},
                    {"property": "Status", "select": {"equals": "not_started"}},
                    {"property": "Status", "select": {"equals": "on_track"}},
                ]
            }

        data = {
            "filter": filter_query,
            "sorts": [{"property": "Target Date", "direction": "ascending"}],
        }

        response = await self._request("POST", f"databases/{database_id}/query", data)
        goals = []

        for page in response.get("results", []):
            goal = self._parse_goal(page)
            if goal:
                goals.append(goal)

        return goals

    async def update_goal_progress(
        self, database_id: str, goal_id: str, current_value: float
    ) -> None:
        """Update goal progress in Notion."""
        data = {
            "properties": {
                "Current Value": {"number": current_value},
            }
        }

        await self._request("PATCH", f"pages/{goal_id}", data)
        logger.info(f"Updated goal progress: {goal_id}")

    def _parse_goal(self, page: Dict) -> Optional[HealthGoal]:
        """Parse a Notion page into a HealthGoal object."""
        try:
            props = page.get("properties", {})

            def get_title(prop_name: str) -> str:
                prop = props.get(prop_name, {})
                texts = prop.get("title", [])
                return texts[0].get("text", {}).get("content", "") if texts else ""

            def get_text(prop_name: str) -> Optional[str]:
                prop = props.get(prop_name, {})
                texts = prop.get("rich_text", [])
                return texts[0].get("text", {}).get("content") if texts else None

            def get_number(prop_name: str) -> float:
                prop = props.get(prop_name, {})
                return prop.get("number", 0)

            def get_date_value(prop_name: str) -> date:
                prop = props.get(prop_name, {})
                date_str = prop.get("date", {}).get("start")
                if date_str:
                    return datetime.fromisoformat(date_str.replace("Z", "+00:00")).date()
                return date.today()

            def get_select(prop_name: str) -> str:
                prop = props.get(prop_name, {})
                return prop.get("select", {}).get("name", "custom")

            return HealthGoal(
                id=page["id"],
                title=get_title("Title"),
                goal_type=GoalType(get_select("Type")),
                target_value=get_number("Target Value"),
                current_value=get_number("Current Value"),
                unit=get_text("Unit") or "",
                target_date=get_date_value("Target Date"),
                status=GoalStatus(get_select("Status")),
                progress_percent=get_number("Progress %"),
                description=get_text("Description"),
            )

        except Exception as e:
            logger.error(f"Failed to parse goal: {e}")
            return None

    # ===========================================
    # Symptom Operations
    # ===========================================

    async def create_symptom(self, database_id: str, symptom: SymptomLog) -> str:
        """Create a new symptom log entry in Notion."""
        properties = {
            "Date": {"date": {"start": symptom.symptom_date.isoformat()}},
            "Symptom": {"title": [{"text": {"content": symptom.symptom_name}}]},
            "Severity": {"number": symptom.severity.value},
        }

        if symptom.symptom_time:
            properties["Time"] = {
                "rich_text": [{"text": {"content": symptom.symptom_time.strftime("%H:%M")}}]
            }
        if symptom.body_part:
            properties["Body Part"] = {"rich_text": [{"text": {"content": symptom.body_part}}]}
        if symptom.duration_minutes:
            properties["Duration (min)"] = {"number": symptom.duration_minutes}
        if symptom.notes:
            properties["Notes"] = {"rich_text": [{"text": {"content": symptom.notes}}]}

        data = {
            "parent": {"database_id": database_id},
            "properties": properties,
        }

        try:
            response = await self._request("POST", "pages", data)
        except httpx.HTTPStatusError as exc:
            response_text = exc.response.text
            if (
                symptom.duration_minutes
                and "Duration (min) is not a property that exists" in response_text
            ):
                properties.pop("Duration (min)", None)
                properties["Duration"] = {"number": symptom.duration_minutes}
                response = await self._request("POST", "pages", data)
            else:
                raise

        logger.info(f"Created symptom page: {response['id']}")
        return response["id"]

    async def query_symptoms(
        self,
        database_id: str,
        start_date: date,
        end_date: date,
    ) -> List[SymptomLog]:
        """Query symptom logs from Notion database."""
        data = {
            "filter": {
                "and": [
                    {
                        "property": "Date",
                        "date": {"on_or_after": start_date.isoformat()},
                    },
                    {
                        "property": "Date",
                        "date": {"before": (end_date + timedelta(days=1)).isoformat()},
                    },
                ]
            },
            "sorts": [{"property": "Date", "direction": "descending"}],
        }

        response = await self._request("POST", f"databases/{database_id}/query", data)
        symptoms = []

        for page in response.get("results", []):
            symptom = self._parse_symptom(page)
            if symptom:
                symptoms.append(symptom)

        return symptoms

    def _parse_symptom(self, page: Dict) -> Optional[SymptomLog]:
        """Parse a Notion page into a SymptomLog object."""
        try:
            props = page.get("properties", {})

            def get_title(prop_name: str) -> str:
                prop = props.get(prop_name, {})
                texts = prop.get("title", [])
                return texts[0].get("text", {}).get("content", "") if texts else ""

            def get_text(prop_name: str) -> Optional[str]:
                prop = props.get(prop_name, {})
                texts = prop.get("rich_text", [])
                return texts[0].get("text", {}).get("content") if texts else None

            def get_number(prop_name: str) -> Optional[int]:
                prop = props.get(prop_name, {})
                val = prop.get("number")
                return int(val) if val else None

            def get_date_value(prop_name: str) -> date:
                prop = props.get(prop_name, {})
                date_str = prop.get("date", {}).get("start")
                if date_str:
                    return datetime.fromisoformat(date_str.replace("Z", "+00:00")).date()
                return date.today()

            time_str = get_text("Time")
            time_obj = None
            if time_str:
                time_obj = self._parse_time_value(time_str)

            severity_val = get_number("Severity") or 5

            return SymptomLog(
                id=page["id"],
                symptom_date=get_date_value("Date"),
                symptom_time=time_obj,
                symptom_name=get_title("Symptom"),
                severity=SymptomSeverity(severity_val),
                body_part=get_text("Body Part"),
                duration_minutes=get_number("Duration (min)") or get_number("Duration"),
                notes=get_text("Notes"),
            )

        except Exception as e:
            logger.error(f"Failed to parse symptom: {e}")
            return None

    # ===========================================
    # Brain Analysis Operations (TRIBEv2)
    # ===========================================

    async def create_brain_analysis(self, database_id: str, session: BrainAnalysisSession) -> str:
        """Create a brain analysis session record in Notion."""
        label = f"Brain Analysis — {session.session_date} ({session.input_type})"
        properties = {
            "Name": {"title": [{"text": {"content": label}}]},
            "Date": {"date": {"start": session.session_date.isoformat()}},
            "Input Type": {"select": {"name": session.input_type}},
            "Input Summary": {"rich_text": [{"text": {"content": session.input_summary[:2000]}}]},
        }

        if session.cognitive_load_score is not None:
            properties["Cognitive Load"] = {"number": session.cognitive_load_score}
        if session.stress_indicator is not None:
            properties["Stress Level"] = {"number": session.stress_indicator}
        if session.emotional_valence is not None:
            properties["Emotional Valence"] = {"number": session.emotional_valence}
        if session.default_mode_activity is not None:
            properties["Default Mode Network"] = {"number": session.default_mode_activity}
        if session.executive_control_activity is not None:
            properties["Executive Control"] = {"number": session.executive_control_activity}
        if session.salience_network_activity is not None:
            properties["Salience Network"] = {"number": session.salience_network_activity}
        if session.insights:
            properties["Insights"] = {"rich_text": [{"text": {"content": session.insights[:2000]}}]}
        if session.recommendations:
            recs = "; ".join(session.recommendations)
            properties["Recommendations"] = {"rich_text": [{"text": {"content": recs[:2000]}}]}
        properties["TRIBEv2 Model Used"] = {"checkbox": session.model_available}

        data = {"parent": {"database_id": database_id}, "properties": properties}
        response = await self._request("POST", "pages", data)
        logger.info(f"Created brain analysis page: {response['id']}")
        return response["id"]

    async def query_brain_analyses(
        self,
        database_id: str,
        start_date: date,
        end_date: date,
        limit: int = 50,
    ) -> List[BrainAnalysisSession]:
        """Query brain analysis sessions from Notion."""
        data = {
            "filter": {
                "and": [
                    {"property": "Date", "date": {"on_or_after": start_date.isoformat()}},
                    {
                        "property": "Date",
                        "date": {"before": (end_date + timedelta(days=1)).isoformat()},
                    },
                ]
            },
            "sorts": [{"property": "Date", "direction": "descending"}],
            "page_size": limit,
        }
        response = await self._request("POST", f"databases/{database_id}/query", data)
        sessions = []
        for page in response.get("results", []):
            s = self._parse_brain_analysis(page)
            if s:
                sessions.append(s)
        return sessions

    def _parse_brain_analysis(self, page: Dict) -> Optional[BrainAnalysisSession]:
        """Parse a Notion page into a BrainAnalysisSession."""
        try:
            props = page.get("properties", {})

            def get_number(k: str) -> Optional[float]:
                return (props.get(k) or {}).get("number")

            def get_text(k: str) -> Optional[str]:
                texts = (props.get(k) or {}).get("rich_text") or []
                return texts[0].get("text", {}).get("content") if texts else None

            def get_select(k: str) -> Optional[str]:
                return ((props.get(k) or {}).get("select") or {}).get("name")

            def get_checkbox(k: str) -> bool:
                return bool((props.get(k) or {}).get("checkbox"))

            def get_date_val(k: str) -> date:
                date_str = ((props.get(k) or {}).get("date") or {}).get("start")
                if date_str:
                    return datetime.fromisoformat(date_str.replace("Z", "+00:00")).date()
                return date.today()

            recs_raw = get_text("Recommendations")
            recommendations = (
                [r.strip() for r in recs_raw.split(";") if r.strip()] if recs_raw else []
            )

            return BrainAnalysisSession(
                id=page["id"],
                session_date=get_date_val("Date"),
                input_type=get_select("Input Type") or "health_narrative",
                input_summary=get_text("Input Summary") or "",
                cognitive_load_score=get_number("Cognitive Load"),
                stress_indicator=get_number("Stress Level"),
                emotional_valence=get_number("Emotional Valence"),
                default_mode_activity=get_number("Default Mode Network"),
                executive_control_activity=get_number("Executive Control"),
                salience_network_activity=get_number("Salience Network"),
                insights=get_text("Insights"),
                recommendations=recommendations,
                model_available=get_checkbox("TRIBEv2 Model Used"),
            )
        except Exception as e:
            logger.error(f"Failed to parse brain analysis: {e}")
            return None
