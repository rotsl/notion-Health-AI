"""
Notion Workspace Setup Script — Creates databases, GUI dashboard, and seeds sample data.

Usage:
    python scripts/setup_notion.py

What it does:
    1. Prompts for any missing API keys and saves them to .env
    2. Archives existing databases (from a previous run) so you start fresh
    3. Creates 6 health databases with the full schema
    4. Seeds realistic sample data (7 health entries, meds, appointments, goals, symptoms, brain)
    5. Builds a rich GUI dashboard page with embedded databases and command reference
"""

import os
import sys
import json
from datetime import date
from typing import Optional, Dict, List, Any

import httpx

# Add src/ to path so relative imports work when running directly
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

try:
    from dotenv import load_dotenv
    HAS_DOTENV = True
except ImportError:
    HAS_DOTENV = False

    def load_dotenv(*args, **kwargs):
        pass


# Path to the project .env file
ENV_PATH = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".env"))

# Mapping from schema database keys → env variable names
DB_ID_KEYS = {
    "health_metrics": "NOTION_HEALTH_DATABASE_ID",
    "medications": "NOTION_MEDICATION_DATABASE_ID",
    "appointments": "NOTION_APPOINTMENT_DATABASE_ID",
    "goals": "NOTION_GOALS_DATABASE_ID",
    "symptoms": "NOTION_SYMPTOMS_DATABASE_ID",
    "brain_analysis": "NOTION_BRAIN_DATABASE_ID",
}


# ──────────────────────────────────────────────────────────────────────────────
# .env helpers (preserve all lines, only update specific keys)
# ──────────────────────────────────────────────────────────────────────────────

def _read_env_values(path: str) -> Dict[str, str]:
    """Return {key: value} for all non-comment key=value lines in an env file."""
    result: Dict[str, str] = {}
    if not os.path.exists(path):
        return result
    with open(path, "r") as f:
        for line in f:
            stripped = line.strip()
            if stripped and not stripped.startswith("#") and "=" in stripped:
                key, value = stripped.split("=", 1)
                result[key.strip()] = value.strip()
    return result


def _write_env_updates(path: str, updates: Dict[str, str]):
    """
    Update specific keys in .env, preserving all other lines exactly as-is.
    Appends keys that don't already exist in the file.
    """
    lines: List[str] = []
    if os.path.exists(path):
        with open(path, "r") as f:
            lines = f.readlines()

    updated: set = set()
    new_lines: List[str] = []
    for line in lines:
        stripped = line.strip()
        if stripped and not stripped.startswith("#") and "=" in stripped:
            key = stripped.split("=", 1)[0].strip()
            if key in updates:
                new_lines.append(f"{key}={updates[key]}\n")
                updated.add(key)
                continue
        new_lines.append(line if line.endswith("\n") else line + "\n")

    # Append keys not already in the file
    for key, value in updates.items():
        if key not in updated:
            new_lines.append(f"{key}={value}\n")

    with open(path, "w") as f:
        f.writelines(new_lines)


# ──────────────────────────────────────────────────────────────────────────────
# API key prompting (runs before NotionSetup is instantiated)
# ──────────────────────────────────────────────────────────────────────────────

def _prompt_api_keys() -> str:
    """
    Prompt for any missing API keys, save them to .env, and return NOTION_API_KEY.
    Keys already present in .env (and not placeholder values) are skipped silently.
    """
    load_dotenv(ENV_PATH, override=False)

    placeholder_prefixes = (
        "your_", "secret_your_", "sk-ant-your", "hf_your",
        "sk-your", "your-",
    )

    def is_placeholder(val: str) -> bool:
        return not val or any(val.startswith(p) for p in placeholder_prefixes)

    print("\n" + "=" * 60)
    print("🔑 API Key Configuration")
    print("=" * 60)

    updates: Dict[str, str] = {}

    # ── NOTION_API_KEY (required) ──────────────────────────────────────
    notion_key = os.getenv("NOTION_API_KEY", "").strip()
    if is_placeholder(notion_key):
        print("\nNotion Integration Token (required)")
        print("  → notion.so/my-integrations → New integration → copy token")
        print("    Token starts with:  ntn_...  or  secret_...")
        notion_key = input("  Enter NOTION_API_KEY: ").strip()
        if not notion_key:
            print("❌ NOTION_API_KEY is required. Exiting.")
            sys.exit(1)
        updates["NOTION_API_KEY"] = notion_key
        os.environ["NOTION_API_KEY"] = notion_key
    else:
        print(f"\n✅ NOTION_API_KEY already set ({notion_key[:8]}...)")

    # ── ANTHROPIC_API_KEY (optional) ───────────────────────────────────
    anthropic_key = os.getenv("ANTHROPIC_API_KEY", "").strip()
    if is_placeholder(anthropic_key):
        print("\nAnthropic API Key  (optional — needed for AI health insights)")
        print("  → console.anthropic.com → API Keys")
        print("  Press Enter to skip (AI insights will be disabled)")
        val = input("  Enter ANTHROPIC_API_KEY: ").strip()
        if val:
            updates["ANTHROPIC_API_KEY"] = val
            os.environ["ANTHROPIC_API_KEY"] = val
        else:
            print("   Skipped — AI insights disabled")
    else:
        print(f"✅ ANTHROPIC_API_KEY already set ({anthropic_key[:8]}...)")

    # ── HUGGING_FACE_TOKEN (optional) ──────────────────────────────────
    hf_token = os.getenv("HUGGING_FACE_TOKEN", "").strip()
    if is_placeholder(hf_token):
        print("\nHuggingFace Token  (optional — needed for REAL TRIBEv2 brain predictions)")
        print("  Without this, fast simulation-based brain analysis is used")
        print("  To use real model:")
        print("    1. Request access at huggingface.co/meta-llama/Llama-3.2-3B  (gated model)")
        print("    2. Get token at huggingface.co/settings/tokens")
        print("  Press Enter to skip (simulation mode)")
        val = input("  Enter HUGGING_FACE_TOKEN: ").strip()
        if val:
            updates["HUGGING_FACE_TOKEN"] = val
            os.environ["HUGGING_FACE_TOKEN"] = val
        else:
            print("   Skipped — TRIBEv2 will use simulation mode")
    else:
        print(f"✅ HUGGING_FACE_TOKEN already set ({hf_token[:8]}...)")

    if updates:
        _write_env_updates(ENV_PATH, updates)
        print(f"\n💾 Saved {len(updates)} key(s) to .env")

    return notion_key


# ──────────────────────────────────────────────────────────────────────────────
# Main setup class
# ──────────────────────────────────────────────────────────────────────────────

class NotionSetup:
    """Set up a Notion workspace for NotionHealth AI."""

    BASE_URL = "https://api.notion.com/v1"
    NOTION_VERSION = "2022-06-28"

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("NOTION_API_KEY")
        if not self.api_key:
            raise ValueError("NOTION_API_KEY not found. Run setup again and enter your token.")

        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Notion-Version": self.NOTION_VERSION,
        }
        self.client = httpx.Client(headers=self.headers, timeout=60.0)
        self.database_ids: Dict[str, str] = {}

    def close(self):
        self.client.close()

    def _request(self, method: str, endpoint: str, json_data: Optional[Dict] = None) -> Dict:
        url = f"{self.BASE_URL}/{endpoint}"
        try:
            if method == "GET":
                response = self.client.get(url)
            elif method == "POST":
                response = self.client.post(url, json=json_data)
            elif method == "PATCH":
                response = self.client.patch(url, json=json_data)
            else:
                raise ValueError(f"Unsupported HTTP method: {method}")
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            print(f"❌ API Error {e.response.status_code}: {e.response.text[:300]}")
            raise

    # ── Delete existing databases ───────────────────────────────────────

    def delete_existing_databases(self):
        """Archive any databases from a previous setup run (IDs stored in .env)."""
        env = _read_env_values(ENV_PATH)
        to_delete = {k: env[v] for k, v in DB_ID_KEYS.items() if env.get(v, "").strip()}

        if not to_delete:
            print("\n   No existing database IDs found — nothing to clean up")
            return

        print(f"\n🗑️  Archiving {len(to_delete)} existing database(s) from previous setup...")
        cleared: Dict[str, str] = {}
        for db_key, db_id in to_delete.items():
            try:
                self._request("PATCH", f"pages/{db_id}", {"archived": True})
                print(f"   ✅ Archived: {db_key} ({db_id[:8]}...)")
            except httpx.HTTPStatusError as e:
                if e.response.status_code == 404:
                    print(f"   ⚠️  Already gone: {db_key}")
                else:
                    print(f"   ⚠️  Could not archive {db_key}: {e.response.status_code}")
            cleared[DB_ID_KEYS[db_key]] = ""

        _write_env_updates(ENV_PATH, cleared)
        print("   Cleared old IDs from .env")

    # ── Page / database creation ────────────────────────────────────────

    def list_pages(self) -> List[Dict]:
        print("\n📄 Fetching accessible pages...")
        response = self._request("POST", "search", {
            "filter": {"property": "object", "value": "page"},
            "page_size": 50,
        })
        pages = response.get("results", [])
        print(f"   Found {len(pages)} accessible page(s)")
        return pages

    def select_parent_page(self) -> str:
        pages = self.list_pages()
        if not pages:
            print("❌ No pages found. Share a Notion page with your integration first:")
            print("   Open Notion → any page → Share → Add integration → Invite")
            sys.exit(1)

        print("\n📁 Select a parent page for the NotionHealth AI workspace:")
        for i, page in enumerate(pages, 1):
            print(f"   {i}. {self._get_page_title(page)}")

        while True:
            try:
                choice = int(input("\nEnter page number: "))
                if 1 <= choice <= len(pages):
                    selected = pages[choice - 1]
                    print(f"✅ Selected: {self._get_page_title(selected)}")
                    return selected["id"]
                print("❌ Invalid choice. Try again.")
            except ValueError:
                print("❌ Please enter a number.")

    def _get_page_title(self, page: Dict) -> str:
        props = page.get("properties", {})
        title_prop = props.get("title", {})
        if title_prop:
            texts = title_prop.get("title", [])
            if texts:
                return texts[0].get("text", {}).get("content", "Untitled")
        return "Untitled"

    def load_schema(self) -> Dict:
        schema_path = os.path.normpath(
            os.path.join(os.path.dirname(__file__), "..", "config", "database_schema.json")
        )
        with open(schema_path, "r") as f:
            return json.load(f)

    def create_database(self, parent_id: str, name: str, properties: Dict) -> str:
        print(f"   Creating: {name}...")
        notion_properties: Dict[str, Any] = {}

        for prop_name, prop_config in properties.items():
            prop_type = prop_config.get("type")
            notion_prop: Dict[str, Any] = {"type": prop_type}

            if prop_type == "title":
                notion_prop["title"] = {}
            elif prop_type == "rich_text":
                notion_prop["rich_text"] = {}
            elif prop_type == "number":
                notion_prop["number"] = prop_config.get("number", {"format": "number"})
            elif prop_type == "date":
                notion_prop["date"] = {}
            elif prop_type == "checkbox":
                notion_prop["checkbox"] = {}
            elif prop_type == "select":
                raw_opts = prop_config.get("select", {}).get("options", [])
                notion_prop["select"] = {
                    "options": [
                        {
                            "name": opt["name"] if isinstance(opt, dict) else str(opt),
                            "color": opt.get("color", "default") if isinstance(opt, dict) else "default",
                        }
                        for opt in raw_opts
                    ]
                }
            elif prop_type == "multi_select":
                raw_opts = prop_config.get("multi_select", {}).get("options", [])
                notion_prop["multi_select"] = {
                    "options": [
                        {
                            "name": opt["name"] if isinstance(opt, dict) else str(opt),
                            "color": opt.get("color", "default") if isinstance(opt, dict) else "default",
                        }
                        for opt in raw_opts
                    ]
                }

            notion_properties[prop_name] = notion_prop

        # Every database needs at least one title property
        if not any(p.get("type") == "title" for p in notion_properties.values()):
            notion_properties["Name"] = {"type": "title", "title": {}}

        response = self._request("POST", "databases", {
            "parent": {"page_id": parent_id},
            "title": [{"type": "text", "text": {"content": name}}],
            "properties": notion_properties,
        })
        db_id = response["id"]
        print(f"   ✅ {name} ({db_id[:8]}...)")
        return db_id

    def create_databases(self, parent_id: str) -> Dict[str, str]:
        print("\n📊 Creating databases...")
        schema = self.load_schema()
        database_ids: Dict[str, str] = {}

        for db_key, db_config in schema.get("databases", {}).items():
            name = db_config.get("name", db_key.replace("_", " ").title())
            properties = db_config.get("properties", {})
            try:
                database_ids[db_key] = self.create_database(parent_id, name, properties)
            except Exception as e:
                print(f"   ❌ Failed to create {name}: {e}")

        self.database_ids = database_ids
        return database_ids

    # ── Sample data seeding ─────────────────────────────────────────────

    def _create_row(self, database_id: str, properties: Dict) -> bool:
        """POST one page (row) into a database."""
        try:
            self._request("POST", "pages", {
                "parent": {"database_id": database_id},
                "properties": properties,
            })
            return True
        except Exception as e:
            print(f"   ⚠️  Row creation failed: {e}")
            return False

    @staticmethod
    def _rt(text: str) -> Dict:
        """Rich text property value."""
        return {"rich_text": [{"text": {"content": text}}]}

    @staticmethod
    def _title(text: str) -> Dict:
        return {"title": [{"text": {"content": text}}]}

    @staticmethod
    def _date(d: str) -> Dict:
        return {"date": {"start": d}}

    @staticmethod
    def _num(n: float) -> Dict:
        return {"number": n}

    @staticmethod
    def _sel(name: str) -> Dict:
        return {"select": {"name": name}}

    @staticmethod
    def _chk(val: bool) -> Dict:
        return {"checkbox": val}

    def seed_sample_data(self, database_ids: Dict[str, str]):
        """Insert realistic sample rows into each database."""
        print("\n🌱 Seeding sample data...")
        rt, title, dt, num, sel, chk = self._rt, self._title, self._date, self._num, self._sel, self._chk

        # ── Health Metrics (7 days of data, weight trending down) ──────
        hm_id = database_ids.get("health_metrics")
        if hm_id:
            rows = [
                ("2026-03-23", 174.0, 7.0, 6, 6, 6, 7200, 6, None, None),
                ("2026-03-24", 173.8, 6.5, 5, 5, 5, 6800, 5, "walking", 20),
                ("2026-03-25", 173.5, 8.0, 8, 8, 8, 9100, 8, "running", 30),
                ("2026-03-26", 173.2, 7.5, 7, 7, 7, 8500, 7, None, None),
                ("2026-03-27", 173.0, 6.0, 5, 6, 6, 7800, 6, "yoga", 45),
                ("2026-03-28", 172.8, 7.0, 7, 7, 7, 8200, 7, "running", 25),
                ("2026-03-29", 172.5, 8.5, 9, 9, 9, 10200, 9, "cycling", 40),
            ]
            for d, weight, sleep, quality, mood, energy, steps, water, ex_type, ex_min in rows:
                props = {
                    "Name": title(f"Health Log {d}"),
                    "Date": dt(d),
                    "Weight": num(weight),
                    "Sleep Hours": num(sleep),
                    "Sleep Quality": num(quality),
                    "Mood": num(mood),
                    "Energy": num(energy),
                    "Steps": num(steps),
                    "Water Glasses": num(water),
                }
                if ex_type:
                    props["Exercise Type"] = sel(ex_type)
                    props["Exercise Minutes"] = num(ex_min)
                self._create_row(hm_id, props)
            print(f"   ✅ Health Metrics: {len(rows)} rows")

        # ── Medications ─────────────────────────────────────────────────
        med_id = database_ids.get("medications")
        if med_id:
            meds = [
                ("Vitamin D3", "2000 IU", "once_daily", "2026-01-01",
                 "Bone health and immune system support", "Dr. Sarah Chen"),
                ("Omega-3 Fish Oil", "1000mg", "once_daily", "2026-02-15",
                 "Cardiovascular health and brain function", "Dr. Sarah Chen"),
            ]
            for name_, dosage, freq, start, purpose, doctor in meds:
                self._create_row(med_id, {
                    "Name": title(name_),
                    "Dosage": rt(dosage),
                    "Frequency": sel(freq),
                    "Start Date": dt(start),
                    "Active": chk(True),
                    "Purpose": rt(purpose),
                    "Prescribing Doctor": rt(doctor),
                })
            print(f"   ✅ Medications: {len(meds)} rows")

        # ── Appointments ────────────────────────────────────────────────
        appt_id = database_ids.get("appointments")
        if appt_id:
            appts = [
                ("Dr. Sarah Chen", "2026-04-15", "10:00 AM", "checkup",
                 "City Medical Center", "Annual physical checkup", "scheduled"),
                ("Dr. James Martinez", "2026-05-02", "2:00 PM", "dental",
                 "Bright Smile Dental", "Routine dental cleaning and X-rays", "scheduled"),
                ("Dr. Priya Sharma", "2026-04-28", "3:30 PM", "mental_health",
                 "Wellness Clinic", "Mental health follow-up and wellness check", "confirmed"),
            ]
            for doc, d, time_, type_, facility, reason, status in appts:
                self._create_row(appt_id, {
                    "Doctor Name": title(doc),
                    "Date": dt(d),
                    "Time": rt(time_),
                    "Type": sel(type_),
                    "Facility": rt(facility),
                    "Reason": rt(reason),
                    "Status": sel(status),
                })
            print(f"   ✅ Appointments: {len(appts)} rows")

        # ── Goals ───────────────────────────────────────────────────────
        goals_id = database_ids.get("goals")
        if goals_id:
            goals = [
                ("Weight Loss — Reach 165 lbs", "weight", 165.0, 172.5,
                 "lbs", "2026-07-01", "in_progress",
                 "Gradual 1–2 lb/week reduction through diet and exercise."),
                ("Daily Exercise — 30 min/day", "exercise", 30.0, 22.0,
                 "min/day avg", "2026-06-01", "on_track",
                 "Currently averaging 22 min/day. Target is 30 min across at least 5 days/week."),
            ]
            for goal_title, type_, target, current, unit, d, status, notes in goals:
                self._create_row(goals_id, {
                    "Title": title(goal_title),
                    "Type": sel(type_),
                    "Target Value": num(target),
                    "Current Value": num(current),
                    "Unit": rt(unit),
                    "Target Date": dt(d),
                    "Status": sel(status),
                    "Notes": rt(notes),
                })
            print(f"   ✅ Goals: {len(goals)} rows")

        # ── Symptoms ────────────────────────────────────────────────────
        sym_id = database_ids.get("symptoms")
        if sym_id:
            symptoms = [
                ("Headache", "2026-03-25", "2:00 PM", 4, "head", 120,
                 "Possible trigger: dehydration. Resolved after drinking water and resting 30 min."),
                ("Lower Back Tension", "2026-03-27", "4:30 PM", 3, "lower back", 60,
                 "After 4+ hours of sitting. Resolved with stretching and short walk."),
            ]
            for sym, d, time_, sev, part, dur, notes in symptoms:
                self._create_row(sym_id, {
                    "Symptom": title(sym),
                    "Date": dt(d),
                    "Time": rt(time_),
                    "Severity": num(sev),
                    "Body Part": rt(part),
                    "Duration": num(dur),
                    "Notes": rt(notes),
                })
            print(f"   ✅ Symptoms: {len(symptoms)} rows")

        # ── Brain Analysis ───────────────────────────────────────────────
        brain_id = database_ids.get("brain_analysis")
        if brain_id:
            self._create_row(brain_id, {
                "Name": title("Running 30 min — TRIBEv2 Prediction"),
                "Date": dt("2026-03-25"),
                "Cognitive Load": num(4.2),
                "Stress Level": num(3.1),
                "Emotional Valence": num(0.72),
                "Default Mode Network": num(0.41),
                "Executive Control": num(0.55),
                "Salience Network": num(0.48),
                "Insights": rt(
                    "Running activates motor cortex (0.615) and hippocampus (0.58). "
                    "BDNF release supports neuroplasticity and memory formation. "
                    "Elevated mood (0.72 valence) and sustained energy predicted post-exercise. "
                    "Low stress response (3.1/10) indicates healthy adaptation to physical load."
                ),
                "Recommendations": rt(
                    "Morning runs between 6:30–7:30 AM optimise BDNF release during peak cortisol. "
                    "Follow with protein-rich breakfast for muscle recovery. "
                    "Pair with 10 min meditation to enhance prefrontal integration of the exercise benefits."
                ),
                "TRIBEv2 Model Used": chk(True),
            })
            print("   ✅ Brain Analysis: 1 row")

    # ── Dashboard creation ──────────────────────────────────────────────

    def create_dashboard_page(self, parent_id: str, database_ids: Dict[str, str]) -> str:
        """Create the main GUI dashboard page with embedded databases and command guide."""
        print("\n🏠 Creating GUI Dashboard...")
        response = self._request("POST", "pages", {
            "parent": {"page_id": parent_id},
            "icon": {"type": "emoji", "emoji": "🏥"},
            "properties": {
                "title": [{"type": "text", "text": {"content": "🏥 NotionHealth AI Dashboard"}}]
            },
        })
        dashboard_id = response["id"]
        print("   Dashboard page created")
        self._add_dashboard_content(dashboard_id, database_ids)
        return dashboard_id

    def _append_blocks(self, page_id: str, blocks: List[Dict]):
        """Append blocks in batches of 100 (Notion API limit per request)."""
        for i in range(0, len(blocks), 100):
            self._request("PATCH", f"blocks/{page_id}/children", {"children": blocks[i:i + 100]})

    # ── Block builder helpers ───────────────────────────────────────────

    @staticmethod
    def _rich(content: str, bold: bool = False, color: str = "default") -> Dict:
        item: Dict[str, Any] = {"type": "text", "text": {"content": content}}
        ann: Dict[str, Any] = {}
        if bold:
            ann["bold"] = True
        if color != "default":
            ann["color"] = color
        if ann:
            item["annotations"] = ann
        return item

    def _h1(self, text: str) -> Dict:
        return {"object": "block", "type": "heading_1",
                "heading_1": {"rich_text": [self._rich(text)]}}

    def _h2(self, text: str) -> Dict:
        return {"object": "block", "type": "heading_2",
                "heading_2": {"rich_text": [self._rich(text)]}}

    def _h3(self, text: str) -> Dict:
        return {"object": "block", "type": "heading_3",
                "heading_3": {"rich_text": [self._rich(text)]}}

    @staticmethod
    def _divider() -> Dict:
        return {"object": "block", "type": "divider", "divider": {}}

    def _para(self, *parts) -> Dict:
        return {"object": "block", "type": "paragraph",
                "paragraph": {"rich_text": list(parts)}}

    def _callout(self, emoji: str, color: str, *parts) -> Dict:
        return {
            "object": "block", "type": "callout",
            "callout": {
                "icon": {"type": "emoji", "emoji": emoji},
                "color": color,
                "rich_text": list(parts),
            },
        }

    @staticmethod
    def _linked_db(db_id: str) -> Dict:
        """Link to an existing database page (clickable reference in Notion)."""
        return {
            "object": "block", "type": "link_to_page",
            "link_to_page": {"type": "database_id", "database_id": db_id},
        }

    def _toggle(self, label: str, children: List[Dict]) -> Dict:
        return {
            "object": "block", "type": "toggle",
            "toggle": {
                "rich_text": [self._rich(label)],
                "children": children,
            },
        }

    # ── Dashboard content ───────────────────────────────────────────────

    def _add_dashboard_content(self, page_id: str, database_ids: Dict[str, str]):
        r = self._rich  # shorthand

        # ── Batch 1: Welcome + Setup Guide + Claude Commands ────────────
        batch1: List[Dict] = []

        # Welcome
        batch1.append(self._h1("👋 Welcome to NotionHealth AI"))
        batch1.append(self._callout(
            "🧠", "blue_background",
            r("v2.0 — Powered by "),
            r("Meta's TRIBEv2", bold=True),
            r(" brain-predictive foundation model + "),
            r("Claude AI", bold=True),
            r(". Talk to Claude Desktop to log health data, get AI insights, "
              "and predict real fMRI brain activation from your health narratives."),
        ))
        batch1.append(self._divider())

        # Setup Guide (flat sections — no nesting to avoid API limits)
        batch1.append(self._h2("⚙️ Setup & Configuration"))
        batch1.append(self._callout(
            "🔑", "gray_background",
            r("Step 1 — Notion Integration Token  (required)\n", bold=True),
            r("Go to notion.so/my-integrations → New integration → copy token (starts with ntn_ or secret_). "
              "Then share this parent page: Page menu → Share → Add integration → Invite."),
        ))
        batch1.append(self._callout(
            "🤖", "gray_background",
            r("Step 2 — Anthropic API Key  (for AI insights)\n", bold=True),
            r("Get from console.anthropic.com → API Keys. "
              "Required for AI health analysis, symptom pattern detection, and correlation reports."),
        ))
        batch1.append(self._callout(
            "🧬", "gray_background",
            r("Step 3 — HuggingFace Token  (optional — for real TRIBEv2 brain predictions)\n", bold=True),
            r("Without this: fast simulation-based brain analysis is used (all 16 tools still work fully). "
              "For real fMRI predictions: "
              "(1) Request access at huggingface.co/meta-llama/Llama-3.2-3B (gated Facebook Research / Meta model — approved in minutes). "
              "(2) Get token at huggingface.co/settings/tokens. "
              "(3) Add HUGGING_FACE_TOKEN=hf_... to your .env file. "
              "First run downloads ~2.7 GB (TRIBEv2 weights + LLaMA 3.2 3B + Whisper). All cached after that."),
        ))
        batch1.append(self._callout(
            "💻", "gray_background",
            r("Step 4 — Claude Desktop MCP Setup\n", bold=True),
            r('Edit ~/Library/Application Support/Claude/claude_desktop_config.json  and add: '
              '{ "mcpServers": { "notion-health-ai": { "command": "/absolute/path/venv_arm64/bin/notion-health-mcp" } } }  '
              'Then Cmd+Q and reopen Claude Desktop. '
              'Run: echo "$(pwd)/venv_arm64/bin/notion-health-mcp" to get your absolute path.'),
        ))
        batch1.append(self._divider())

        # Claude Commands
        batch1.append(self._h2("💬 How to Use with Claude Desktop"))
        batch1.append(self._para(
            r("Open Claude Desktop and type any command below. "
              "Claude uses 16 health tools to read and write your Notion databases in real time.")
        ))
        batch1.append(self._callout(
            "📊", "blue_background",
            r("Health Logging\n", bold=True),
            r('"Log my weight as 172 lbs, mood 8, and 7.5 hours of sleep"\n'
              '"I ran for 30 minutes today, 9500 steps, energy level 8"\n'
              '"Blood pressure 118/76, heart rate 62 this morning"\n'
              '"Log today: weight 170, slept 8 hours quality 9, drank 8 glasses of water"'),
        ))
        batch1.append(self._callout(
            "📈", "green_background",
            r("Check Your Stats\n", bold=True),
            r('"Give me a health summary for the past week"\n'
              '"What is my weight trend for the past month?"\n'
              '"Are there patterns in my symptoms?"\n'
              '"How am I doing on my health goals?"'),
        ))
        batch1.append(self._callout(
            "💊", "yellow_background",
            r("Medications & Appointments\n", bold=True),
            r('"Add vitamin D3, 2000 IU, once daily starting today"\n'
              '"What medications do I need to take today?"\n'
              '"Schedule a checkup with Dr. Smith on April 20 at 10am"\n'
              '"What appointments do I have coming up?"'),
        ))
        batch1.append(self._callout(
            "🧠", "purple_background",
            r("Brain Analysis with TRIBEv2\n", bold=True),
            r('"Predict my brain response to 30 minutes of meditation"\n'
              '"Analyse my brain-health correlations for the past week"\n'
              '"What is the optimal daily schedule based on brain science?"\n'
              '"Give me a full cognitive health report for the past month"'),
        ))
        batch1.append(self._callout(
            "💡", "orange_background",
            r("AI Insights & Analysis\n", bold=True),
            r('"Give me AI health insights for this week"\n'
              '"Are there correlations between my mood and how much I exercise?"\n'
              '"Analyse my symptom patterns from the past month"'),
        ))
        batch1.append(self._divider())

        self._append_blocks(page_id, batch1)

        # ── Batch 2: All databases ───────────────────────────────────────
        batch2: List[Dict] = []

        batch2.append(self._h2("📊 Health Metrics"))
        batch2.append(self._para(
            r("Daily health entries. Click "),
            r("+ New", bold=True),
            r(' to add today\'s data, or tell Claude: "Log my weight and mood for today".'),
        ))
        if "health_metrics" in database_ids:
            batch2.append(self._linked_db(database_ids["health_metrics"]))
        batch2.append(self._divider())

        batch2.append(self._h2("💊 Medications"))
        batch2.append(self._para(
            r('Active medications and dosage schedule. Tell Claude: "Add a new medication" or "What do I take today?"'),
        ))
        if "medications" in database_ids:
            batch2.append(self._linked_db(database_ids["medications"]))
        batch2.append(self._divider())

        batch2.append(self._h2("📅 Upcoming Appointments"))
        batch2.append(self._para(
            r('Scheduled medical visits. Tell Claude: "Schedule an appointment with Dr. X on April 20 at 10am".'),
        ))
        if "appointments" in database_ids:
            batch2.append(self._linked_db(database_ids["appointments"]))
        batch2.append(self._divider())

        batch2.append(self._h2("🎯 Health Goals"))
        batch2.append(self._para(
            r('Wellness targets and progress. Tell Claude: "Set a goal to reach 165 lbs by July 1st".'),
        ))
        if "goals" in database_ids:
            batch2.append(self._linked_db(database_ids["goals"]))
        batch2.append(self._divider())

        batch2.append(self._h2("🩺 Symptom Log"))
        batch2.append(self._para(
            r('Track symptoms over time. Tell Claude: "Log a headache, severity 5, started an hour ago".'),
        ))
        if "symptoms" in database_ids:
            batch2.append(self._linked_db(database_ids["symptoms"]))
        batch2.append(self._divider())

        batch2.append(self._h2("🧠 Brain Analysis (TRIBEv2)"))
        batch2.append(self._callout(
            "🔬", "blue_background",
            r("Powered by Meta's TRIBEv2", bold=True),
            r(" — the world's first brain-predictive foundation model. "
              "Predicts real fMRI activation across 20,484 cortical surface vertices "
              "(fsaverage5 mesh, HCP MMP1.0 parcellation) from health narratives and activity descriptions. "
              "Inference runs entirely on your local device — "),
            r("no data is sent to Meta.", bold=True),
        ))
        if "brain_analysis" in database_ids:
            batch2.append(self._linked_db(database_ids["brain_analysis"]))
        batch2.append(self._divider())

        # Brain science timing (flat paragraphs)
        batch2.append(self._h2("⏰ Optimal Daily Activity Schedule (Brain Science)"))
        timing_rows = [
            ("6:00–6:30 AM", "Morning Meditation",
             "Theta wave state enhances prefrontal cortex training; amygdala calming"),
            ("6:30–7:30 AM", "Exercise",
             "Peak cortisol optimises BDNF release; motor cortex + hippocampus activation"),
            ("9:00 AM–12:00 PM", "Deep Work / Learning",
             "Peak alertness and working memory; hippocampus primed for encoding"),
            ("12:00–1:00 PM", "Light Walk in Nature",
             "Visual cortex restoration; amygdala calming; cortisol reduction"),
            ("2:00–4:00 PM", "Creative Work",
             "Moderate arousal optimal for divergent thinking; default mode network engagement"),
            ("4:00–6:00 PM", "Social Activities",
             "Natural social energy peak; oxytocin release; temporal lobe engagement"),
            ("8:30–9:00 PM", "Wind-Down / Reading",
             "Melatonin production begins; avoid screens; hippocampus memory replay"),
            ("10:00 PM–6:00 AM", "Sleep (8 hours)",
             "Memory consolidation (hippocampus); glymphatic brain cleaning; growth hormone release"),
        ]
        for time_label, activity, reason in timing_rows:
            batch2.append(self._para(
                self._rich(f"{time_label}  ", bold=True),
                self._rich(f"{activity} — "),
                self._rich(reason, color="gray"),
            ))

        self._append_blocks(page_id, batch2)
        print("   ✅ Dashboard content added")

    # ── Save env ────────────────────────────────────────────────────────

    def save_env_file(self, database_ids: Dict[str, str]):
        """Write new database IDs to .env, preserving all other settings."""
        updates = {
            DB_ID_KEYS[k]: v
            for k, v in database_ids.items()
            if k in DB_ID_KEYS
        }
        _write_env_updates(ENV_PATH, updates)
        print(f"\n💾 Saved {len(updates)} database ID(s) to .env")


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("🏥 NotionHealth AI — Workspace Setup")
    print("=" * 60)

    # 1. Prompt for any missing API keys and persist to .env
    notion_key = _prompt_api_keys()

    setup = None
    try:
        setup = NotionSetup(notion_key)

        # 2. Archive existing databases from a previous run
        setup.delete_existing_databases()

        # 3. Choose parent page
        parent_id = setup.select_parent_page()

        # 4. Create fresh databases
        database_ids = setup.create_databases(parent_id)

        # 5. Seed sample data
        setup.seed_sample_data(database_ids)

        # 6. Build the GUI dashboard
        setup.create_dashboard_page(parent_id, database_ids)

        # 7. Persist new database IDs
        setup.save_env_file(database_ids)

        print("\n" + "=" * 60)
        print("✅ Setup Complete!")
        print("=" * 60)
        print()
        print("Your NotionHealth AI workspace is ready.")
        print()
        print("Next steps:")
        print("  1. Open Notion → find  🏥 NotionHealth AI Dashboard")
        print("  2. Explore the sample data seeded in each database")
        print("  3. Connect Claude Desktop (see Setup Guide on the dashboard)")
        print('  4. Ask Claude: "Give me a health summary for this week"')
        print()
        print("📚 Full guide: howto-use-notion-health-ai.md")

    except Exception as e:
        print(f"\n❌ Setup failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        if setup:
            setup.close()


if __name__ == "__main__":
    main()
