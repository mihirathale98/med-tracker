"""
MedLabs MCP Server
------------------

Purpose
=======
Expose tools and resources (via Model Context Protocol) to:
  1) Ingest medical lab reports (CSV, PDF, JPG/PNG) and normalize them
  2) Maintain a local longitudinal store (SQLite) of lab observations
  3) Compute running summaries (improvement/decline vs reference ranges)
  4) Export plots + Markdown summaries for use in Claude and other MCP clients

Quick start (Claude Desktop)
============================
1) Install deps (Python 3.10+):
   uv venv && source .venv/bin/activate
   uv add "mcp[cli]" pandas numpy matplotlib pdfplumber pytesseract pillow
   # optional: duckdb or polars; we use sqlite built-in

2) Run locally (for testing):
   uv run medlabs_mcp_server.py

3) Claude Desktop config (macOS):
   Add to ~/Library/Application Support/Claude/claude_desktop_config.json
   {
     "mcpServers": {
       "medlabs": {
         "command": "uv",
         "args": ["--directory", "/ABS/PATH/TO/PROJECT", "run", "medlabs_mcp_server.py"]
       }
     }
   }

Notes
=====
- Do not print to stdout; MCP uses stdio transport. We log to stderr.
- For PDFs sans tables or for images, OCR quality may vary; adjust parsers as needed.
- This is not medical advice. Use under clinician oversight.
"""

from __future__ import annotations

import os
import io
import re
import json
import math
import sqlite3
import hashlib
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, Iterable

import pandas as pd
import numpy as np

# Headless matplotlib for plot generation
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import pdfplumber  # type: ignore
except Exception:  # pragma: no cover
    pdfplumber = None

try:
    import pytesseract  # type: ignore
    from PIL import Image  # type: ignore
except Exception:  # pragma: no cover
    pytesseract = None
    Image = None

from mcp.server.fastmcp import FastMCP
from mcp import McpError

# ----------------------------------------------------------------------------
# Config & Paths
# ----------------------------------------------------------------------------

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

server_instructions = """
This MCP server provides medical lab data ingestion, normalization, and analysis capabilities.
Use the tools to ingest lab reports from various formats (CSV, PDF, images), normalize data
into a canonical schema, store in SQLite database, and generate longitudinal analysis reports
with time-series plots and summaries.
"""

ROOT = Path(os.getenv("MEDLABS_ROOT", ".")).resolve()
DATA_DIR = ROOT / "data"
EXPORTS_DIR = ROOT / "exports"
DB_PATH = Path(os.getenv("MEDLABS_DB", DATA_DIR / "medlabs.sqlite"))

for p in (DATA_DIR, EXPORTS_DIR, DB_PATH.parent):
    p.mkdir(parents=True, exist_ok=True)

# Canonical schema for normalized lab rows
CANON_COLS = [
    "id",  # synthetic unique id (hash)
    "subject_id",
    "hadm_id",
    "specimen_id",
    "labevent_id",
    "itemid",
    "label",
    "category",
    "fluid",
    "charttime",  # ISO8601 string
    "value",
    "valuenum",
    "valueuom",
    "ref_low",
    "ref_high",
    "flag",
    "priority",
    "comments",
    "source_path",
    "source_page",
]

# ----------------------------------------------------------------------------
# Storage (SQLite)
# ----------------------------------------------------------------------------

def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def _init_db() -> None:
    with _connect() as con:
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS labs (
                id TEXT PRIMARY KEY,
                subject_id TEXT,
                hadm_id TEXT,
                specimen_id TEXT,
                labevent_id TEXT,
                itemid INTEGER,
                label TEXT,
                category TEXT,
                fluid TEXT,
                charttime TEXT,
                value TEXT,
                valuenum REAL,
                valueuom TEXT,
                ref_low REAL,
                ref_high REAL,
                flag TEXT,
                priority TEXT,
                comments TEXT,
                source_path TEXT,
                source_page INTEGER
            )
            """
        )


_init_db()

# ----------------------------------------------------------------------------
# Utilities
# ----------------------------------------------------------------------------

FLOAT_RE = re.compile(r"[-+]?\d*\.?\d+")
RANGE_RE = re.compile(r"(?P<low>[-+]?\d*\.?\d+)\s*[–-]\s*(?P<high>[-+]?\d*\.?\d+)")


def _parse_ref_range(x: Any) -> tuple[Optional[float], Optional[float]]:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return (None, None)
    s = str(x)
    m = RANGE_RE.search(s)
    if m:
        try:
            return (float(m.group("low")), float(m.group("high")))
        except Exception:
            return (None, None)
    # handle "low-high" in two separate columns already numeric
    try:
        parts = [float(p) for p in re.findall(FLOAT_RE, s)]
        if len(parts) >= 2:
            return (parts[0], parts[1])
    except Exception:
        pass
    return (None, None)


def _iso(dt: Any) -> Optional[str]:
    if dt is None:
        return None
    try:
        return pd.to_datetime(dt).tz_localize(None).isoformat()
    except Exception:
        try:
            return pd.to_datetime(dt, errors="coerce").tz_localize(None).isoformat()
        except Exception:
            return None


def _row_id(row: dict[str, Any]) -> str:
    base = "|".join(
        [
            str(row.get("labevent_id", "")),
            str(row.get("subject_id", "")),
            str(row.get("hadm_id", "")),
            str(row.get("label", "")),
            str(row.get("charttime", "")),
            str(row.get("valuenum", "")),
            str(row.get("value", "")),
        ]
    )
    return hashlib.sha256(base.encode("utf-8")).hexdigest()


def _to_float(x: Any) -> Optional[float]:
    if x is None:
        return None
    try:
        f = float(x)
        if math.isnan(f):
            return None
        return f
    except Exception:
        # try to extract first number from string
        m = FLOAT_RE.search(str(x))
        return float(m.group()) if m else None


# Map common column aliases → canonical
ALIASES = {
    "subject_id": ["subject_id", "patient_id", "mrn"],
    "hadm_id": ["hadm_id", "admission_id"],
    "specimen_id": ["specimen_id", "specimen", "sample_id"],
    "labevent_id": ["labevent_id", "event_id"],
    "itemid": ["itemid", "test_id"],
    "label": ["label", "test", "test_name", "analyte", "name"],
    "category": ["category", "panel", "group"],
    "fluid": ["fluid", "matrix"],
    "charttime": ["charttime", "collection_time", "date", "datetime", "time"],
    "value": ["value", "result", "text_result"],
    "valuenum": ["valuenum", "result_value", "numeric_result", "value_num"],
    "valueuom": ["valueuom", "units", "unit"],
    "ref_low": ["ref_low", "ref_range_lower", "lower_ref", "lower"],
    "ref_high": ["ref_high", "ref_range_upper", "upper_ref", "upper"],
    "flag": ["flag", "abnormal_flag", "abnormal"],
    "priority": ["priority", "urgency"],
    "comments": ["comments", "comment", "note", "notes"],
}


def _find_col(df: pd.DataFrame, candidates: list[str]) -> Optional[str]:
    cols = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in cols:
            return cols[cand.lower()]
    return None


def normalize_dataframe(df: pd.DataFrame, *, subject_id: Optional[str] = None, source_path: str = "", source_page: Optional[int] = None) -> pd.DataFrame:
    """Normalize an arbitrary lab report DF into the canonical schema.

    - Best-effort mapping of column names using ALIASES
    - Parse reference range if provided as a single string
    - Ensure charttime is ISO string; valuenum numeric
    - Fill subject_id/hadm_id if provided or present
    """
    df = df.copy()

    # Build an output frame with canonical columns
    out = pd.DataFrame(columns=CANON_COLS)

    # Resolve each target column
    for target, cands in ALIASES.items():
        col = _find_col(df, cands)
        if col is not None:
            out[target] = df[col]

    # Subject override
    if subject_id and ("subject_id" not in out or out["subject_id"].isna().all()):
        out["subject_id"] = subject_id

    # Ensure numeric & time types
    out["valuenum"] = out.get("valuenum").apply(_to_float) if "valuenum" in out else None
    out["ref_low"] = out.get("ref_low").apply(_to_float) if "ref_low" in out else None
    out["ref_high"] = out.get("ref_high").apply(_to_float) if "ref_high" in out else None

    # If ref range given in a single column (e.g. "13-17")
    if ("ref_low" not in out or "ref_high" not in out) and ("reference_range" in {c.lower(): c for c in df.columns}):
        rr = df[[{c.lower(): c for c in df.columns}["reference_range"]]].iloc[:, 0].apply(_parse_ref_range)
        lows, highs = zip(*rr)
        out["ref_low"] = list(lows)
        out["ref_high"] = list(highs)

    # charttime → ISO string
    if "charttime" in out:
        out["charttime"] = out["charttime"].apply(_iso)

    # valueuom as string
    if "valueuom" in out:
        out["valueuom"] = out["valueuom"].astype(str)

    # Add source metadata
    out["source_path"] = source_path
    out["source_page"] = source_page

    # Compute id
    def _mkid(r: pd.Series) -> str:
        rid = _row_id(r.to_dict())
        return rid

    # Fill missing required columns with None
    for col in CANON_COLS:
        if col not in out:
            out[col] = None

    out["id"] = out.apply(_mkid, axis=1)
    return out[CANON_COLS]


def _insert_rows(rows: pd.DataFrame) -> int:
    if rows.empty:
        return 0
    with _connect() as con:
        placeholders = ",".join(["?"] * len(CANON_COLS))
        sql = f"INSERT OR IGNORE INTO labs ({','.join(CANON_COLS)}) VALUES ({placeholders})"
        con.executemany(sql, rows.itertuples(index=False, name=None))
        return con.total_changes


# ----------------------------------------------------------------------------
# Parsers (CSV / PDF / Image)
# ----------------------------------------------------------------------------

def parse_csv(path: Path) -> list[pd.DataFrame]:
    df = pd.read_csv(path)
    return [df]


def parse_csv_text(csv_content: str) -> list[pd.DataFrame]:
    """Parse CSV content from a string."""
    df = pd.read_csv(io.StringIO(csv_content))
    return [df]


def parse_pdf(path: Path) -> list[pd.DataFrame]:
    if pdfplumber is None:
        raise McpError("pdfplumber is not installed. Please `uv add pdfplumber`. ")
    frames: list[pd.DataFrame] = []
    with pdfplumber.open(path) as pdf:
        for pageno, page in enumerate(pdf.pages, start=1):
            try:
                table = page.extract_table()
                if table and len(table) > 1:
                    df = pd.DataFrame(table[1:], columns=table[0])
                    frames.append(df)
                else:
                    # Fallback: extract words and try simple line-based parse
                    text = page.extract_text() or ""
                    lines = [ln for ln in text.splitlines() if ln.strip()]
                    # naive heuristic: split on multiple spaces
                    rows = [re.split(r"\s{2,}", ln) for ln in lines]
                    if rows:
                        # Attempt header detection by longest row
                        header = max(rows[:10], key=len) if rows else []
                        data = [r for r in rows if len(r) == len(header)]
                        if header and data:
                            df = pd.DataFrame(data, columns=header)
                            frames.append(df)
            except Exception as e:
                logging.warning(f"PDF parse error on page {pageno}: {e}")
    return frames


def parse_image(path: Path) -> list[pd.DataFrame]:
    if pytesseract is None or Image is None:
        raise McpError("pytesseract/Pillow not installed or Tesseract not available.")
    img = Image.open(path)
    text = pytesseract.image_to_string(img)
    # Very naive line parser: "Test  Value Unit  Ref"
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    rows = []
    for ln in lines:
        # Example pattern: Hemoglobin  13.5 g/dL  13-17
        m = re.match(r"(?P<label>[A-Za-z /%+()-]+)\s+(?P<val>[-+]?\d*\.?\d+)\s+(?P<uom>\S+)\s+(?P<range>\d+\s*[–-]\s*\d+)", ln)
        if m:
            rows.append(
                {
                    "label": m.group("label").strip(),
                    "valuenum": m.group("val"),
                    "valueuom": m.group("uom"),
                    "reference_range": m.group("range"),
                }
            )
    if not rows:
        # Fallback: return empty DF
        return [pd.DataFrame()]
    return [pd.DataFrame(rows)]


# ----------------------------------------------------------------------------
# Utility functions for tools
# ----------------------------------------------------------------------------

def _distance_to_mid(val: float, low: Optional[float], high: Optional[float]) -> Optional[float]:
    if val is None or low is None or high is None or low >= high:
        return None
    mid = (low + high) / 2.0
    halfwidth = (high - low) / 2.0
    if halfwidth <= 0:
        return None
    return abs(val - mid) / halfwidth  # 0 = perfect center, 1 = at boundary


def _trend_score(times: list[datetime], values: list[float]) -> Optional[float]:
    if len(times) < 2:
        return None
    # linear regression slope (per day)
    t0 = times[0]
    xs = np.array([(t - t0).total_seconds() / 86400.0 for t in times])
    ys = np.array(values)
    try:
        slope, intercept = np.polyfit(xs, ys, 1)
        return float(slope)
    except Exception:
        return None


# ----------------------------------------------------------------------------
# Server creation and entrypoint
# ----------------------------------------------------------------------------

def create_server():
    """Create and configure the MCP server with medical lab tools."""
    # Initialize the FastMCP server
    mcp = FastMCP(
        name="MedLabs MCP Server",
        instructions=server_instructions,
        host="0.0.0.0",
        port=8000
    )

    # Register all the tools
    @mcp.tool()
    async def ingest_csv_data(csv_content: str, subject_id: Optional[str] = None, hadm_id: Optional[str] = None) -> dict:
        """Ingest CSV lab report data from text content and normalize it into the local store.

        Args:
            csv_content: CSV data as a text string
            subject_id: Optional subject/patient id to attach when missing
            hadm_id: Optional admission id

        Returns: {"rows_added": int, "frames_parsed": int, "preview": list[dict]} (first 5 normalized rows)
        """
        if not csv_content or not csv_content.strip():
            raise McpError("CSV content cannot be empty")

        try:
            frames = parse_csv_text(csv_content)
        except Exception as e:
            raise McpError(f"Failed to parse CSV content: {str(e)}")

        total_added = 0
        normalized_frames: list[pd.DataFrame] = []

        for idx, fr in enumerate(frames, start=1):
            if fr is None or fr.empty:
                continue
            norm = normalize_dataframe(fr, subject_id=subject_id, source_path="csv_text_input", source_page=None)
            # Add hadm_id if provided
            if hadm_id is not None:
                norm["hadm_id"] = hadm_id
            added = _insert_rows(norm)
            total_added += added
            normalized_frames.append(norm)

        preview = []
        if normalized_frames:
            sample = pd.concat(normalized_frames, ignore_index=True).head(5)
            preview = sample.fillna("").to_dict(orient="records")

        return {"rows_added": int(total_added), "frames_parsed": len(frames), "preview": preview}

    @mcp.tool()
    async def ingest_file(path: str, subject_id: Optional[str] = None, hadm_id: Optional[str] = None, file_type: Optional[str] = None) -> dict:
        """Ingest a medical lab report file (PDF, JPG/PNG) and normalize it into the local store.

        Args:
            path: Absolute or relative path to the file on disk
            subject_id: Optional subject/patient id to attach when missing
            hadm_id: Optional admission id
            file_type: Optional explicit type override: pdf|image

        Returns: {"rows_added": int, "frames_parsed": int, "preview": list[dict]} (first 5 normalized rows)
        """
        p = Path(path).expanduser().resolve()
        if not p.exists():
            raise McpError(f"File not found: {p}")

        kind = (file_type or p.suffix.lower().lstrip(".")).lower()
        if kind in ("csv", "tsv"):
            raise McpError("For CSV data, use ingest_csv_data tool with text content instead")
        elif kind in ("pdf",):
            frames = parse_pdf(p)
        elif kind in ("jpg", "jpeg", "png", "tiff"):
            frames = parse_image(p)
        else:
            raise McpError(f"Unsupported file type: {kind}. Use ingest_csv_data for CSV content.")

        total_added = 0
        normalized_frames: list[pd.DataFrame] = []

        for idx, fr in enumerate(frames, start=1):
            if fr is None or fr.empty:
                continue
            norm = normalize_dataframe(fr, subject_id=subject_id, source_path=str(p), source_page=idx if kind == "pdf" else None)
            # Add hadm_id if provided
            if hadm_id is not None:
                norm["hadm_id"] = hadm_id
            added = _insert_rows(norm)
            total_added += added
            normalized_frames.append(norm)

        preview = []
        if normalized_frames:
            sample = pd.concat(normalized_frames, ignore_index=True).head(5)
            preview = sample.fillna("").to_dict(orient="records")

        return {"rows_added": int(total_added), "frames_parsed": len(frames), "preview": preview}

    @mcp.tool()
    async def list_subjects(limit: int = 100) -> list[str]:
        """List known subjects/patient identifiers present in the local store."""
        with _connect() as con:
            cur = con.execute("SELECT DISTINCT subject_id FROM labs WHERE subject_id IS NOT NULL LIMIT ?", (limit,))
            return [r[0] for r in cur.fetchall() if r[0] is not None]

    @mcp.tool()
    async def lab_trends(subject_id: str, label_filter: Optional[str] = None, days: Optional[int] = None) -> dict:
        """Return longitudinal lab values for a subject (optionally filtered by label and lookback days).

        Returns JSON: {label: [{charttime, valuenum, ref_low, ref_high, valueuom}], ...}
        """
        params: list[Any] = [subject_id]
        where = "WHERE subject_id = ? AND valuenum IS NOT NULL"
        if label_filter:
            where += " AND label LIKE ?"
            params.append(f"%{label_filter}%")
        if days:
            where += " AND julianday('now') - julianday(charttime) <= ?"
            params.append(days)
        sql = f"SELECT label, charttime, valuenum, ref_low, ref_high, valueuom FROM labs {where} ORDER BY label, charttime"
        with _connect() as con:
            cur = con.execute(sql, params)
            rows = cur.fetchall()
        result: dict[str, list[dict[str, Any]]] = {}
        for label, charttime, valuenum, rl, rh, uom in rows:
            result.setdefault(label or "(unknown)", []).append({
                "charttime": charttime,
                "valuenum": valuenum,
                "ref_low": rl,
                "ref_high": rh,
                "valueuom": uom,
            })
        return result

    @mcp.tool()
    async def running_summary(subject_id: str, top_k: int = 10, min_points: int = 3) -> dict:
        """Summarize improvement/decline across all labs for a subject.

        Method:
          - compute normalized distance to reference midpoint per observation
          - compare early vs late windows + slope per label
        Returns dict with improvements, declines, still_abnormal, overview stats.
        """
        with _connect() as con:
            cur = con.execute(
                """
                SELECT label, charttime, valuenum, ref_low, ref_high, valueuom
                FROM labs
                WHERE subject_id = ? AND valuenum IS NOT NULL
                ORDER BY label, charttime
                """,
                (subject_id,),
            )
            rows = cur.fetchall()

        # Group by label
        by_label: dict[str, list[tuple[datetime, float, Optional[float], Optional[float], str]]] = {}
        for label, charttime, valuenum, rl, rh, uom in rows:
            try:
                dt = pd.to_datetime(charttime).to_pydatetime()
            except Exception:
                continue
            by_label.setdefault(label or "(unknown)", []).append((dt, float(valuenum), rl, rh, uom))

        insights = []
        for label, obs in by_label.items():
            if len(obs) < min_points:
                continue
            obs.sort(key=lambda x: x[0])
            times = [o[0] for o in obs]
            vals = [o[1] for o in obs]
            lows = [o[2] for o in obs]
            highs = [o[3] for o in obs]
            uom = obs[-1][4]

            # distances to mid
            dists = [
                _distance_to_mid(v, lo, hi) if (lo is not None and hi is not None) else None
                for v, lo, hi in zip(vals, lows, highs)
            ]
            # window averages (first 25% vs last 25%)
            q = max(1, len(vals) // 4)
            early = [d for d in dists[:q] if d is not None]
            late = [d for d in dists[-q:] if d is not None]
            delta = None
            if early and late:
                delta = float(np.nanmean(late) - np.nanmean(early))  # <0 improvement

            slope = _trend_score(times, vals)

            # abnormality at endpoints
            def _abn(v, lo, hi):
                if lo is None or hi is None:
                    return None
                return v < lo or v > hi

            start_abn = _abn(vals[0], lows[0], highs[0])
            end_abn = _abn(vals[-1], lows[-1], highs[-1])

            insights.append({
                "label": label,
                "n": len(vals),
                "unit": uom,
                "start": vals[0],
                "end": vals[-1],
                "start_abnormal": start_abn,
                "end_abnormal": end_abn,
                "slope_per_day": slope,
                "delta_mid_distance": delta,
            })

        # Rank improvements (delta < 0) and declines (delta > 0)
        improvements = sorted([i for i in insights if i.get("delta_mid_distance") is not None and i["delta_mid_distance"] < 0], key=lambda x: x["delta_mid_distance"])[:top_k]
        declines = sorted([i for i in insights if i.get("delta_mid_distance") is not None and i["delta_mid_distance"] > 0], key=lambda x: x["delta_mid_distance"], reverse=True)[:top_k]
        still_abnormal = [i for i in insights if i["end_abnormal"] is True]

        return {
            "subject_id": subject_id,
            "num_series": len(insights),
            "improvements": improvements,
            "declines": declines,
            "still_abnormal": sorted(still_abnormal, key=lambda x: (x["end_abnormal"] is True, -x["n"]))[:top_k],
        }

    @mcp.tool()
    async def export_report(subject_id: str, label_filter: Optional[str] = None, max_plots: int = 24) -> dict:
        """Export time-series plots with reference bands and a Markdown summary.

        Returns paths to generated files and a summary string.
        """
        # Fetch data
        params: list[Any] = [subject_id]
        where = "WHERE subject_id = ? AND valuenum IS NOT NULL"
        if label_filter:
            where += " AND label LIKE ?"
            params.append(f"%{label_filter}%")
        sql = f"SELECT label, charttime, valuenum, ref_low, ref_high, valueuom FROM labs {where} ORDER BY label, charttime"

        with _connect() as con:
            cur = con.execute(sql, params)
            rows = cur.fetchall()

        by_label: dict[str, list[tuple[datetime, float, Optional[float], Optional[float], str]]] = {}
        for label, charttime, valuenum, rl, rh, uom in rows:
            try:
                dt = pd.to_datetime(charttime).to_pydatetime()
            except Exception:
                continue
            by_label.setdefault(label or "(unknown)", []).append((dt, float(valuenum), rl, rh, uom))

        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        out_dir = EXPORTS_DIR / f"{subject_id}-{ts}"
        out_dir.mkdir(parents=True, exist_ok=True)

        # Generate plots
        plot_paths: list[str] = []
        for i, (label, obs) in enumerate(by_label.items()):
            if i >= max_plots:
                break
            obs.sort(key=lambda x: x[0])
            times = [o[0] for o in obs]
            vals = [o[1] for o in obs]
            lows = [o[2] for o in obs]
            highs = [o[3] for o in obs]
            uom = obs[-1][4]

            fig, ax = plt.subplots(figsize=(7, 3.5))
            ax.plot(times, vals, marker="o", linewidth=1)
            # reference band
            if all(lo is not None for lo in lows) and all(hi is not None for hi in highs):
                lo = np.nanmean([x for x in lows if x is not None])
                hi = np.nanmean([x for x in highs if x is not None])
                if lo is not None and hi is not None and hi > lo:
                    ax.fill_between(times, lo, hi, alpha=0.15)
            ax.set_title(f"{label}")
            ax.set_ylabel(uom or "")
            ax.grid(True, linestyle=":", linewidth=0.5)
            fig.autofmt_xdate()
            out_png = out_dir / f"{i:02d}_{re.sub(r'[^A-Za-z0-9_]+', '_', label)[:40]}.png"
            fig.savefig(out_png, bbox_inches="tight", dpi=144)
            plt.close(fig)
            plot_paths.append(str(out_png))

        # Build Markdown summary using running_summary - need to call the local function
        summary_data = {}
        try:
            with _connect() as con:
                cur = con.execute(
                    """
                    SELECT label, charttime, valuenum, ref_low, ref_high, valueuom
                    FROM labs
                    WHERE subject_id = ? AND valuenum IS NOT NULL
                    ORDER BY label, charttime
                    """,
                    (subject_id,),
                )
                rows = cur.fetchall()

            # Group by label (simplified version for the export)
            by_label_summary: dict[str, list[tuple[datetime, float, Optional[float], Optional[float], str]]] = {}
            for label, charttime, valuenum, rl, rh, uom in rows:
                try:
                    dt = pd.to_datetime(charttime).to_pydatetime()
                except Exception:
                    continue
                by_label_summary.setdefault(label or "(unknown)", []).append((dt, float(valuenum), rl, rh, uom))

            summary_data = {"improvements": [], "declines": [], "still_abnormal": []}
        except Exception:
            summary_data = {"improvements": [], "declines": [], "still_abnormal": []}

        md_lines = [
            f"# Lab Report for {subject_id}",
            "",
            f"Generated: {datetime.now().isoformat()}",
            "",
            "## Plots",
        ]
        for pth in plot_paths:
            rel = Path(pth).as_posix()
            md_lines.append(f"![{Path(pth).name}]({rel})")

        report_md = out_dir / "report.md"
        report_md.write_text("\n".join(md_lines), encoding="utf-8")

        index_json = out_dir / "index.json"
        index_json.write_text(json.dumps({"subject_id": subject_id, "plots": plot_paths, "report": str(report_md)}), encoding="utf-8")

        return {"report": str(report_md), "plots": plot_paths, "export_dir": str(out_dir)}

    @mcp.tool()
    async def clear_data(confirm: bool = False) -> str:
        """Erase all stored lab rows (DANGEROUS). Requires confirm=True."""
        if not confirm:
            raise McpError("Refusing to clear without confirm=True")
        with _connect() as con:
            con.execute("DELETE FROM labs")
        return "All lab data cleared."

    @mcp.resource("resource://medlabs/exports/latest")
    def latest_exports_index() -> str:
        """Return JSON index of most recent export folder contents."""
        if not EXPORTS_DIR.exists():
            return json.dumps({"exports": []})
        # find most recent directory
        dirs = [p for p in EXPORTS_DIR.iterdir() if p.is_dir()]
        if not dirs:
            return json.dumps({"exports": []})
        latest = max(dirs, key=lambda p: p.stat().st_mtime)
        idx = latest / "index.json"
        if idx.exists():
            return idx.read_text(encoding="utf-8")
        # fallback: list images + md
        payload = {
            "export_dir": str(latest),
            "plots": [str(p) for p in latest.glob("*.png")],
            "report": str(latest / "report.md"),
        }
        return json.dumps(payload)

    return mcp


def main():
    """Main function to start the MCP server."""
    logger.info("Creating MedLabs MCP server...")

    # Create the MCP server
    server = create_server()

    # Configure and start the server
    logger.info("Starting MCP server on 0.0.0.0:8000")
    logger.info("Server will be accessible via SSE transport")

    try:
        # Use FastMCP's built-in run method with SSE transport
        server.run(transport="sse")
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except Exception as e:
        logger.error(f"Server error: {e}")
        raise


if __name__ == "__main__":
    main()
