"""
CYFD Statewide Workforce Training Evaluation Pipeline
======================================================
Author: Kirit Reddy Daida — Training Evaluator II, CYFD
Description:
    Full ETL + evaluation pipeline for New Mexico statewide child welfare
    workforce training programs. Ingests raw training records, computes
    pre/post knowledge gains, county-level KPIs, compliance rates, and
    generates Power BI-ready flat files plus an HTML executive summary.

Usage:
    python evaluation_pipeline.py --input data/raw/ --output data/processed/

Requirements:
    pip install pandas numpy openpyxl jinja2 matplotlib seaborn scipy
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from datetime import datetime, date
from typing import Optional, Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from jinja2 import Environment, BaseLoader

# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
NM_COUNTIES: List[str] = [
    "Bernalillo", "Catron", "Chaves", "Cibola", "Colfax", "Curry",
    "De Baca", "Doña Ana", "Eddy", "Grant", "Guadalupe", "Harding",
    "Hidalgo", "Lea", "Lincoln", "Los Alamos", "Luna", "McKinley",
    "Mora", "Otero", "Quay", "Rio Arriba", "Roosevelt", "Sandoval",
    "San Juan", "San Miguel", "Santa Fe", "Sierra", "Socorro",
    "Taos", "Torrance", "Union", "Valencia",
]

TRAINING_MODULES: List[str] = [
    "Child Safety Fundamentals",
    "Trauma-Informed Care",
    "Mandatory Reporting Requirements",
    "Family Engagement Strategies",
    "Risk & Safety Assessment",
    "Cultural Competency",
    "Documentation & Compliance",
    "Supervisor Leadership Coaching",
]

KNOWLEDGE_GAIN_TARGET: float = 0.15      # 15 % improvement threshold
COMPLETION_RATE_TARGET: float = 0.90     # 90 % completion requirement
COMPLIANCE_WINDOW_DAYS: int = 365        # annual compliance window


# ─────────────────────────────────────────────────────────────────────────────
# Data Generation (synthetic — replace with live DB query in production)
# ─────────────────────────────────────────────────────────────────────────────

def _rng(seed: int = 42) -> np.random.Generator:
    return np.random.default_rng(seed)


def generate_training_records(n: int = 2_500, seed: int = 42) -> pd.DataFrame:
    """Simulate raw training participation records for demo / testing."""
    rng = _rng(seed)
    today = pd.Timestamp.today().normalize()

    counties = rng.choice(NM_COUNTIES, size=n)
    modules = rng.choice(TRAINING_MODULES, size=n)
    roles = rng.choice(
        ["Caseworker", "Supervisor", "Admin", "Specialist"], size=n,
        p=[0.55, 0.20, 0.10, 0.15],
    )

    pre_scores = rng.integers(40, 75, size=n).astype(float)
    gains = rng.normal(loc=18, scale=8, size=n).clip(0, 40)
    post_scores = (pre_scores + gains).clip(0, 100)

    # Simulate some incomplete records
    completed_mask = rng.random(n) < 0.92
    post_scores = np.where(completed_mask, post_scores, np.nan)

    # Training dates: spread over the past 12 months
    days_ago = rng.integers(0, 365, size=n)
    training_dates = pd.to_datetime(
        [today - pd.Timedelta(days=int(d)) for d in days_ago]
    )

    df = pd.DataFrame(
        {
            "employee_id": [f"NM{str(i).zfill(5)}" for i in rng.integers(1000, 9999, size=n)],
            "county": counties,
            "role": roles,
            "module": modules,
            "training_date": training_dates,
            "pre_score": pre_scores.round(1),
            "post_score": post_scores.round(1),
            "completed": completed_mask,
            "delivery_mode": rng.choice(["In-Person", "Virtual", "Hybrid"], size=n, p=[0.30, 0.50, 0.20]),
        }
    )
    return df


def generate_compliance_records(employees_df: pd.DataFrame) -> pd.DataFrame:
    """Build compliance tracking table from employee training history."""
    rng = _rng(99)
    required = pd.DataFrame(
        {
            "employee_id": employees_df["employee_id"].unique(),
            "required_module": np.tile(
                TRAINING_MODULES,
                int(np.ceil(len(employees_df["employee_id"].unique()) / len(TRAINING_MODULES))),
            )[: len(employees_df["employee_id"].unique())],
        }
    )
    required["due_date"] = pd.Timestamp.today() + pd.to_timedelta(
        rng.integers(-60, 120, size=len(required)), unit="D"
    )
    required["compliant"] = rng.random(len(required)) < 0.87
    return required


# ─────────────────────────────────────────────────────────────────────────────
# KPI Computation
# ─────────────────────────────────────────────────────────────────────────────

class KPIEngine:
    """Computes all training evaluation KPIs from raw records."""

    def __init__(self, records: pd.DataFrame, compliance: pd.DataFrame) -> None:
        self.records = records.copy()
        self.compliance = compliance.copy()
        self._preprocess()

    # ── Pre-processing ───────────────────────────────────────────────────────
    def _preprocess(self) -> None:
        df = self.records
        df["knowledge_gain"] = df["post_score"] - df["pre_score"]
        df["pct_gain"] = df["knowledge_gain"] / df["pre_score"].replace(0, np.nan)
        df["month"] = df["training_date"].dt.to_period("M")
        df["quarter"] = df["training_date"].dt.to_period("Q")
        df["met_gain_target"] = df["pct_gain"] >= KNOWLEDGE_GAIN_TARGET
        self.records = df

    # ── Organisation-level KPIs ──────────────────────────────────────────────
    def org_kpis(self) -> Dict[str, float]:
        df = self.records
        completed = df[df["completed"]]
        return {
            "total_trainings": len(df),
            "total_completed": int(df["completed"].sum()),
            "overall_completion_rate": df["completed"].mean() * 100,
            "avg_pre_score": df["pre_score"].mean(),
            "avg_post_score": completed["post_score"].mean(),
            "avg_knowledge_gain_pct": completed["pct_gain"].mean() * 100,
            "pct_meeting_gain_target": completed["met_gain_target"].mean() * 100,
            "counties_covered": df["county"].nunique(),
            "unique_employees": df["employee_id"].nunique(),
        }

    # ── County-level KPIs ────────────────────────────────────────────────────
    def county_kpis(self) -> pd.DataFrame:
        df = self.records
        completed = df[df["completed"]]

        agg = df.groupby("county").agg(
            total_trainings=("employee_id", "count"),
            completion_rate=("completed", "mean"),
            avg_pre=("pre_score", "mean"),
            unique_employees=("employee_id", "nunique"),
        ).round(3)

        gain_agg = completed.groupby("county").agg(
            avg_post=("post_score", "mean"),
            avg_gain_pct=("pct_gain", "mean"),
            pct_met_target=("met_gain_target", "mean"),
        ).round(3)

        result = agg.join(gain_agg, how="left")
        result["completion_rate"] *= 100
        result["avg_gain_pct"] *= 100
        result["pct_met_target"] *= 100
        result["gap_to_target"] = (result["completion_rate"] - COMPLETION_RATE_TARGET * 100).round(2)
        result["risk_flag"] = result["completion_rate"] < (COMPLETION_RATE_TARGET * 100 * 0.85)
        return result.reset_index()

    # ── Module-level KPIs ────────────────────────────────────────────────────
    def module_kpis(self) -> pd.DataFrame:
        df = self.records
        completed = df[df["completed"]]

        agg = df.groupby("module").agg(
            enrollments=("employee_id", "count"),
            completion_rate=("completed", "mean"),
            avg_pre=("pre_score", "mean"),
        ).round(3)

        gain_agg = completed.groupby("module").agg(
            avg_post=("post_score", "mean"),
            avg_gain_pct=("pct_gain", "mean"),
            effect_size=("knowledge_gain", lambda x: x.mean() / x.std() if x.std() > 0 else 0),
        ).round(3)

        result = agg.join(gain_agg, how="left")
        result["completion_rate"] *= 100
        result["avg_gain_pct"] *= 100
        result["underperforming"] = result["avg_gain_pct"] < (KNOWLEDGE_GAIN_TARGET * 100)
        return result.reset_index()

    # ── Time-trend KPIs ──────────────────────────────────────────────────────
    def monthly_trends(self) -> pd.DataFrame:
        df = self.records[self.records["completed"]]
        trend = df.groupby("month").agg(
            trainings=("employee_id", "count"),
            avg_gain_pct=("pct_gain", "mean"),
            completion_rate=("completed", "mean"),
        ).reset_index()
        trend["month"] = trend["month"].astype(str)
        trend["avg_gain_pct"] *= 100
        trend["completion_rate"] *= 100
        return trend

    # ── Statistical significance test (paired t-test) ───────────────────────
    def paired_significance(self) -> Dict[str, float]:
        completed = self.records[self.records["completed"]].dropna(subset=["pre_score", "post_score"])
        t_stat, p_value = stats.ttest_rel(completed["post_score"], completed["pre_score"])
        cohens_d = (
            (completed["post_score"].mean() - completed["pre_score"].mean())
            / completed["knowledge_gain"].std()
        )
        return {
            "t_statistic": round(t_stat, 4),
            "p_value": round(p_value, 6),
            "cohens_d": round(cohens_d, 4),
            "significant": p_value < 0.05,
            "effect_size_label": (
                "Large" if abs(cohens_d) >= 0.8
                else "Medium" if abs(cohens_d) >= 0.5
                else "Small"
            ),
        }

    # ── Compliance KPIs ──────────────────────────────────────────────────────
    def compliance_kpis(self) -> Dict[str, float]:
        comp = self.compliance
        return {
            "overall_compliance_rate": comp["compliant"].mean() * 100,
            "at_risk_employees": int((~comp["compliant"]).sum()),
            "past_due": int((comp["due_date"] < pd.Timestamp.today()) & (~comp["compliant"])).sum()
            if "due_date" in comp.columns else 0,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Export Layer
# ─────────────────────────────────────────────────────────────────────────────

class ExportManager:
    """Writes Power BI-ready flat files, charts, and HTML report."""

    def __init__(self, output_dir: str) -> None:
        self.out = Path(output_dir)
        self.out.mkdir(parents=True, exist_ok=True)
        (self.out / "charts").mkdir(exist_ok=True)

    # ── Excel workbook ───────────────────────────────────────────────────────
    def write_excel(
        self,
        raw: pd.DataFrame,
        county_kpis: pd.DataFrame,
        module_kpis: pd.DataFrame,
        monthly: pd.DataFrame,
    ) -> Path:
        path = self.out / f"training_evaluation_{date.today()}.xlsx"
        with pd.ExcelWriter(path, engine="openpyxl") as writer:
            raw.to_excel(writer, sheet_name="Raw Records", index=False)
            county_kpis.to_excel(writer, sheet_name="County KPIs", index=False)
            module_kpis.to_excel(writer, sheet_name="Module KPIs", index=False)
            monthly.to_excel(writer, sheet_name="Monthly Trends", index=False)
        log.info("Excel workbook written → %s", path)
        return path

    # ── Charts ───────────────────────────────────────────────────────────────
    def write_charts(
        self,
        county_kpis: pd.DataFrame,
        module_kpis: pd.DataFrame,
        monthly: pd.DataFrame,
    ) -> List[Path]:
        sns.set_theme(style="darkgrid", palette="muted")
        paths: List[Path] = []

        # Chart 1 — County Completion Rates
        fig, ax = plt.subplots(figsize=(14, 7))
        sorted_df = county_kpis.sort_values("completion_rate", ascending=True)
        colors = ["#e74c3c" if r else "#2ecc71" for r in sorted_df["risk_flag"]]
        ax.barh(sorted_df["county"], sorted_df["completion_rate"], color=colors)
        ax.axvline(COMPLETION_RATE_TARGET * 100, color="gold", linestyle="--", linewidth=2, label=f"Target {int(COMPLETION_RATE_TARGET*100)}%")
        ax.set_xlabel("Completion Rate (%)")
        ax.set_title("County-Level Training Completion Rates — All 33 NM Counties", fontsize=14, fontweight="bold")
        ax.legend()
        plt.tight_layout()
        p = self.out / "charts" / "county_completion_rates.png"
        fig.savefig(p, dpi=150)
        plt.close(fig)
        paths.append(p)
        log.info("Chart saved → %s", p)

        # Chart 2 — Module Knowledge Gain
        fig, ax = plt.subplots(figsize=(12, 6))
        m_sorted = module_kpis.sort_values("avg_gain_pct", ascending=False)
        bar_colors = ["#e74c3c" if u else "#3498db" for u in m_sorted["underperforming"]]
        ax.bar(range(len(m_sorted)), m_sorted["avg_gain_pct"], color=bar_colors)
        ax.axhline(KNOWLEDGE_GAIN_TARGET * 100, color="gold", linestyle="--", linewidth=2, label="Target 15%")
        ax.set_xticks(range(len(m_sorted)))
        ax.set_xticklabels(m_sorted["module"], rotation=35, ha="right", fontsize=9)
        ax.set_ylabel("Avg Knowledge Gain (%)")
        ax.set_title("Knowledge Gain % by Training Module", fontsize=13, fontweight="bold")
        ax.legend()
        plt.tight_layout()
        p = self.out / "charts" / "module_knowledge_gain.png"
        fig.savefig(p, dpi=150)
        plt.close(fig)
        paths.append(p)

        # Chart 3 — Monthly Trend
        fig, ax1 = plt.subplots(figsize=(12, 5))
        ax2 = ax1.twinx()
        ax1.plot(monthly["month"], monthly["trainings"], marker="o", color="#3498db", label="Trainings")
        ax2.plot(monthly["month"], monthly["avg_gain_pct"], marker="s", color="#e67e22", linestyle="--", label="Avg Gain %")
        ax1.set_ylabel("Total Trainings", color="#3498db")
        ax2.set_ylabel("Avg Knowledge Gain %", color="#e67e22")
        ax1.set_xlabel("Month")
        ax1.set_title("Monthly Training Volume & Knowledge Gain Trend", fontsize=13, fontweight="bold")
        plt.xticks(rotation=45, ha="right", fontsize=8)
        fig.tight_layout()
        p = self.out / "charts" / "monthly_trend.png"
        fig.savefig(p, dpi=150)
        plt.close(fig)
        paths.append(p)

        return paths

    # ── HTML Executive Report ────────────────────────────────────────────────
    def write_html_report(
        self,
        org_kpis: Dict,
        stat_sig: Dict,
        compliance: Dict,
        county_kpis: pd.DataFrame,
    ) -> Path:
        template_str = """<!DOCTYPE html>
<html lang='en'>
<head>
  <meta charset='UTF-8'>
  <title>CYFD Training Evaluation Report — {{ report_date }}</title>
  <style>
    body { font-family: 'Segoe UI', sans-serif; background:#1a1a2e; color:#eee; margin:0; padding:20px; }
    h1 { color:#00d2ff; text-align:center; }
    h2 { color:#a8dadc; border-bottom:1px solid #444; padding-bottom:6px; }
    .grid { display:grid; grid-template-columns:repeat(4,1fr); gap:16px; margin:20px 0; }
    .card { background:#16213e; border-radius:10px; padding:20px; text-align:center; box-shadow:0 2px 8px #0005; }
    .kpi-val { font-size:2rem; font-weight:bold; color:#00d2ff; }
    .kpi-label { font-size:0.85rem; color:#aaa; margin-top:4px; }
    table { width:100%; border-collapse:collapse; margin-top:16px; }
    th { background:#0f3460; color:#a8dadc; padding:10px; text-align:left; }
    td { padding:8px 10px; border-bottom:1px solid #2a2a4a; }
    tr:hover td { background:#1e2a4a; }
    .risk { color:#e74c3c; font-weight:bold; }
    .ok { color:#2ecc71; }
    .badge { display:inline-block; padding:2px 8px; border-radius:12px; font-size:0.75rem; }
    .badge-red { background:#e74c3c22; color:#e74c3c; border:1px solid #e74c3c; }
    .badge-green { background:#2ecc7122; color:#2ecc71; border:1px solid #2ecc71; }
    .footer { text-align:center; color:#555; margin-top:40px; font-size:0.8rem; }
  </style>
</head>
<body>
  <h1>🏛️ CYFD Statewide Training Evaluation Report</h1>
  <p style='text-align:center;color:#aaa;'>Generated: {{ report_date }} &nbsp;|&nbsp; All 33 New Mexico Counties</p>

  <h2>📊 Organization-Level KPIs</h2>
  <div class='grid'>
    <div class='card'><div class='kpi-val'>{{ "%.1f"|format(org.overall_completion_rate) }}%</div><div class='kpi-label'>Overall Completion Rate</div></div>
    <div class='card'><div class='kpi-val'>{{ "%.1f"|format(org.avg_knowledge_gain_pct) }}%</div><div class='kpi-label'>Avg Knowledge Gain</div></div>
    <div class='card'><div class='kpi-val'>{{ org.counties_covered }}</div><div class='kpi-label'>Counties Covered</div></div>
    <div class='card'><div class='kpi-val'>{{ org.unique_employees }}</div><div class='kpi-label'>Unique Employees Trained</div></div>
    <div class='card'><div class='kpi-val'>{{ org.total_trainings }}</div><div class='kpi-label'>Total Training Records</div></div>
    <div class='card'><div class='kpi-val'>{{ "%.1f"|format(org.pct_meeting_gain_target) }}%</div><div class='kpi-label'>Meeting Gain Target (≥15%)</div></div>
    <div class='card'><div class='kpi-val'>{{ "%.1f"|format(stat.cohens_d) }}</div><div class='kpi-label'>Cohen's d Effect Size</div></div>
    <div class='card'><div class='kpi-val'>{{ stat.effect_size_label }}</div><div class='kpi-label'>Effect Size Rating</div></div>
  </div>

  <h2>⚖️ Statistical Significance</h2>
  <p>Paired t-test: <strong>t = {{ stat.t_statistic }}</strong>, p = {{ stat.p_value }}
  &nbsp;→&nbsp;
  {% if stat.significant %}
    <span class='badge badge-green'>✅ Statistically Significant</span>
  {% else %}
    <span class='badge badge-red'>❌ Not Significant</span>
  {% endif %}
  &nbsp; Cohen's d = {{ stat.cohens_d }} ({{ stat.effect_size_label }} effect)</p>

  <h2>📋 County Performance Summary</h2>
  <table>
    <tr>
      <th>County</th><th>Trainings</th><th>Completion %</th>
      <th>Avg Pre</th><th>Avg Post</th><th>Gain %</th><th>Risk</th>
    </tr>
    {% for row in counties %}
    <tr>
      <td>{{ row.county }}</td>
      <td>{{ row.total_trainings }}</td>
      <td class='{% if row.completion_rate >= 90 %}ok{% else %}risk{% endif %}'>{{ "%.1f"|format(row.completion_rate) }}%</td>
      <td>{{ "%.1f"|format(row.avg_pre) }}</td>
      <td>{{ "%.1f"|format(row.avg_post if row.avg_post == row.avg_post else 0) }}</td>
      <td>{{ "%.1f"|format(row.avg_gain_pct if row.avg_gain_pct == row.avg_gain_pct else 0) }}%</td>
      <td>{% if row.risk_flag %}<span class='badge badge-red'>⚠ At Risk</span>{% else %}<span class='badge badge-green'>✅ OK</span>{% endif %}</td>
    </tr>
    {% endfor %}
  </table>

  <div class='footer'>CYFD Statewide Workforce Training Evaluation System · Built by Kirit Reddy Daida</div>
</body>
</html>"""
        env = Environment(loader=BaseLoader())
        tmpl = env.from_string(template_str)
        html = tmpl.render(
            report_date=date.today().strftime("%B %d, %Y"),
            org=type("O", (), org_kpis)(),
            stat=stat_sig,
            compliance=compliance,
            counties=county_kpis.to_dict("records"),
        )
        path = self.out / f"executive_report_{date.today()}.html"
        path.write_text(html, encoding="utf-8")
        log.info("HTML report written → %s", path)
        return path


# ─────────────────────────────────────────────────────────────────────────────
# CLI Entry Point
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="CYFD Statewide Workforce Training Evaluation Pipeline"
    )
    parser.add_argument("--input", default="data/raw/", help="Input directory with raw CSV/Excel files")
    parser.add_argument("--output", default="data/processed/", help="Output directory for processed files")
    parser.add_argument("--records", type=int, default=2500, help="Number of synthetic records (demo mode)")
    parser.add_argument("--demo", action="store_true", help="Run with synthetic demo data")
    return parser.parse_args()


def load_or_generate(args: argparse.Namespace) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if args.demo:
        log.info("Demo mode: generating %d synthetic training records …", args.records)
        records = generate_training_records(args.records)
        compliance = generate_compliance_records(records)
        return records, compliance

    input_path = Path(args.input)
    csv_files = list(input_path.glob("*.csv")) + list(input_path.glob("*.xlsx"))
    if not csv_files:
        log.warning("No input files found in %s. Falling back to demo data.", input_path)
        records = generate_training_records(args.records)
        compliance = generate_compliance_records(records)
        return records, compliance

    dfs = []
    for f in csv_files:
        if f.suffix == ".csv":
            dfs.append(pd.read_csv(f))
        else:
            dfs.append(pd.read_excel(f))
    records = pd.concat(dfs, ignore_index=True)
    compliance_path = input_path / "compliance.csv"
    compliance = pd.read_csv(compliance_path) if compliance_path.exists() else generate_compliance_records(records)
    return records, compliance


def main() -> None:
    args = parse_args()

    # ── Load Data ────────────────────────────────────────────────────────────
    records, compliance = load_or_generate(args)
    log.info("Loaded %d training records across %d employees", len(records), records["employee_id"].nunique())

    # ── Compute KPIs ─────────────────────────────────────────────────────────
    engine = KPIEngine(records, compliance)
    org_kpis = engine.org_kpis()
    county_kpis = engine.county_kpis()
    module_kpis = engine.module_kpis()
    monthly_trends = engine.monthly_trends()
    stat_sig = engine.paired_significance()
    compliance_kpis = engine.compliance_kpis()

    # ── Print Summary ─────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("  CYFD Statewide Training Evaluation — Quick Summary")
    print("="*60)
    for k, v in org_kpis.items():
        print(f"  {k:<35} {v:.2f}" if isinstance(v, float) else f"  {k:<35} {v}")
    print(f"\n  Statistical Significance: p={stat_sig['p_value']} ({stat_sig['effect_size_label']} effect)")
    print(f"  Compliance Rate: {compliance_kpis['overall_compliance_rate']:.1f}%")
    print("="*60 + "\n")

    # ── Export ────────────────────────────────────────────────────────────────
    exporter = ExportManager(args.output)
    exporter.write_excel(records, county_kpis, module_kpis, monthly_trends)
    exporter.write_charts(county_kpis, module_kpis, monthly_trends)
    exporter.write_html_report(org_kpis, stat_sig, compliance_kpis, county_kpis)

    log.info("Pipeline complete. Outputs in: %s", args.output)


if __name__ == "__main__":
    main()
