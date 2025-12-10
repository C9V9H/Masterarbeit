"""
Cell Manufacturing Cost Estimator
"""

from dataclasses import dataclass, replace
from typing import Dict, Tuple, List, Any
from collections import deque  # used for O(1) pops in topological sort
import copy
import math
import os
from urllib.parse import quote
import base64
import json

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, html, dcc, Input, Output, State, dash_table, ctx
import dash


# ============================================================
# Defaults & Presets
# ============================================================

def _derive_ah_v_from_kwh(kwh: float, v: float = 3.7) -> Tuple[float, float]:
    """Helper to derive Ah from kWh and voltage (kept for backwards compatibility)."""
    v = v if v and v > 0 else 3.7
    ah = (kwh * 1000.0) / v
    return ah, v


# ============================================================
# Column names used for cell process steps
# ============================================================

CELL_COLS = {
    # Internal canonical column names used throughout the app/model.
    # Keeping this indirection makes it easy to change names in one place
    # if you ever align with an external CSV or a different DataFrame schema.
    "id": "step_id",
    "step": "step",
    "lead_time_s": "lead_time_s",
    "scrap_rate": "scrap_rate",
    "kw_per_unit": "kw_per_unit",
    "spec_workers_per_machine": "spec_workers_per_machine",
    "supp_workers_per_machine": "supp_workers_per_machine",
    "capex_meur_per_machine": "capex_meur_per_machine",
    "env": "env",
    "footprint_m2": "footprint_m2",
    # DAG connectivity: comma-separated successors & fractions
    "succ": "succ",
    "succ_frac": "succ_frac",
}


# --- Default process steps (used as template for all presets) -----------------

DEFAULT_STEPS = [
    {
        CELL_COLS["id"]: "MIX_C",
        CELL_COLS["step"]: "Cathode Mixing",
        CELL_COLS["lead_time_s"]: 0.017,
        CELL_COLS["scrap_rate"]: 0.01,
        CELL_COLS["kw_per_unit"]: 20.0,
        CELL_COLS["spec_workers_per_machine"]: 0.5,
        CELL_COLS["supp_workers_per_machine"]: 0.0,
        CELL_COLS["capex_meur_per_machine"]: 1.08,
        CELL_COLS["env"]: "none",
        CELL_COLS["footprint_m2"]: 24,
        CELL_COLS["succ"]: "COAT_C",
        CELL_COLS["succ_frac"]: "1",
    },
    {
        CELL_COLS["id"]: "COAT_C",
        CELL_COLS["step"]: "Cathode Coating",
        CELL_COLS["lead_time_s"]: 0.02325,
        CELL_COLS["scrap_rate"]: 0.00,
        CELL_COLS["kw_per_unit"]: 75.0,
        CELL_COLS["spec_workers_per_machine"]: 1.0,
        CELL_COLS["supp_workers_per_machine"]: 0.0,
        CELL_COLS["capex_meur_per_machine"]: 37.8,
        CELL_COLS["env"]: "clean",
        CELL_COLS["footprint_m2"]: 300,
        CELL_COLS["succ"]: "CAL_C",
        CELL_COLS["succ_frac"]: "1",
    },
    {
        CELL_COLS["id"]: "CAL_C",
        CELL_COLS["step"]: "Cathode Calendering",
        CELL_COLS["lead_time_s"]: 0.02325,
        CELL_COLS["scrap_rate"]: 0.00,
        CELL_COLS["kw_per_unit"]: 60.0,
        CELL_COLS["spec_workers_per_machine"]: 0.5,
        CELL_COLS["supp_workers_per_machine"]: 0.0,
        CELL_COLS["capex_meur_per_machine"]: 2.9,
        CELL_COLS["env"]: "none",
        CELL_COLS["footprint_m2"]: 24,
        CELL_COLS["succ"]: "SLIT_C",
        CELL_COLS["succ_frac"]: "1",
    },
    {
        CELL_COLS["id"]: "SLIT_C",
        CELL_COLS["step"]: "Cathode Slitting",
        CELL_COLS["lead_time_s"]: 0.02325,
        CELL_COLS["scrap_rate"]: 0.00,
        CELL_COLS["kw_per_unit"]: 45.0,
        CELL_COLS["spec_workers_per_machine"]: 0.0,
        CELL_COLS["supp_workers_per_machine"]: 0.0,
        CELL_COLS["capex_meur_per_machine"]: 1.15,
        CELL_COLS["env"]: "none",
        CELL_COLS["footprint_m2"]: 24,
        CELL_COLS["succ"]: "VAC",
        CELL_COLS["succ_frac"]: "1",
    },
    {
        CELL_COLS["id"]: "MIX_A",
        CELL_COLS["step"]: "Anode Mixing",
        CELL_COLS["lead_time_s"]: 0.017,
        CELL_COLS["scrap_rate"]: 0.01,
        CELL_COLS["kw_per_unit"]: 20.0,
        CELL_COLS["spec_workers_per_machine"]: 0.5,
        CELL_COLS["supp_workers_per_machine"]: 0.0,
        CELL_COLS["capex_meur_per_machine"]: 1.08,
        CELL_COLS["env"]: "none",
        CELL_COLS["footprint_m2"]: 24,
        CELL_COLS["succ"]: "COAT_A",
        CELL_COLS["succ_frac"]: "1",
    },
    {
        CELL_COLS["id"]: "COAT_A",
        CELL_COLS["step"]: "Anode Coating",
        CELL_COLS["lead_time_s"]: 0.02325,
        CELL_COLS["scrap_rate"]: 0.00,
        CELL_COLS["kw_per_unit"]: 75.0,
        CELL_COLS["spec_workers_per_machine"]: 1.0,
        CELL_COLS["supp_workers_per_machine"]: 0.0,
        CELL_COLS["capex_meur_per_machine"]: 37.8,
        CELL_COLS["env"]: "clean",
        CELL_COLS["footprint_m2"]: 300,
        CELL_COLS["succ"]: "CAL_A",
        CELL_COLS["succ_frac"]: "1",
    },
    {
        CELL_COLS["id"]: "CAL_A",
        CELL_COLS["step"]: "Anode Calendering",
        CELL_COLS["lead_time_s"]: 0.02325,
        CELL_COLS["scrap_rate"]: 0.00,
        CELL_COLS["kw_per_unit"]: 60.0,
        CELL_COLS["spec_workers_per_machine"]: 0.5,
        CELL_COLS["supp_workers_per_machine"]: 0.0,
        CELL_COLS["capex_meur_per_machine"]: 2.9,
        CELL_COLS["env"]: "none",
        CELL_COLS["footprint_m2"]: 24,
        CELL_COLS["succ"]: "SLIT_A",
        CELL_COLS["succ_frac"]: "1",
    },
    {
        CELL_COLS["id"]: "SLIT_A",
        CELL_COLS["step"]: "Anode Slitting",
        CELL_COLS["lead_time_s"]: 0.02325,
        CELL_COLS["scrap_rate"]: 0.00,
        CELL_COLS["kw_per_unit"]: 45.0,
        CELL_COLS["spec_workers_per_machine"]: 0.0,
        CELL_COLS["supp_workers_per_machine"]: 0.0,
        CELL_COLS["capex_meur_per_machine"]: 1.15,
        CELL_COLS["env"]: "none",
        CELL_COLS["footprint_m2"]: 24,
        CELL_COLS["succ"]: "VAC",
        CELL_COLS["succ_frac"]: "1",
    },
    {
        CELL_COLS["id"]: "VAC",
        CELL_COLS["step"]: "Vacuum Drying",
        CELL_COLS["lead_time_s"]: 0.02325,
        CELL_COLS["scrap_rate"]: 0.00,
        CELL_COLS["kw_per_unit"]: 56.0,
        CELL_COLS["spec_workers_per_machine"]: 0.1,
        CELL_COLS["supp_workers_per_machine"]: 0.0,
        CELL_COLS["capex_meur_per_machine"]: 1.2,
        CELL_COLS["env"]: "none",
        CELL_COLS["footprint_m2"]: 11,
        CELL_COLS["succ"]: "CONT",
        CELL_COLS["succ_frac"]: "1",
    },
    {
        CELL_COLS["id"]: "CONT",
        CELL_COLS["step"]: "Contacting",
        CELL_COLS["lead_time_s"]: 3.0,
        CELL_COLS["scrap_rate"]: 0.0010,
        CELL_COLS["kw_per_unit"]: 56.0,
        CELL_COLS["spec_workers_per_machine"]: 0.2,
        CELL_COLS["supp_workers_per_machine"]: 0.0,
        CELL_COLS["capex_meur_per_machine"]: 5.0,
        CELL_COLS["env"]: "none",
        CELL_COLS["footprint_m2"]: 12,
        CELL_COLS["succ"]: "WIND",
        CELL_COLS["succ_frac"]: "1",
    },
    {
        CELL_COLS["id"]: "WIND",
        CELL_COLS["step"]: "Winding",
        CELL_COLS["lead_time_s"]: 1.50,
        CELL_COLS["scrap_rate"]: 0.006,
        CELL_COLS["kw_per_unit"]: 25.0,
        CELL_COLS["spec_workers_per_machine"]: 0.5,
        CELL_COLS["supp_workers_per_machine"]: 0.0,
        CELL_COLS["capex_meur_per_machine"]: 0.80,
        CELL_COLS["env"]: "clean",
        CELL_COLS["footprint_m2"]: 12,
        CELL_COLS["succ"]: "INSRT",
        CELL_COLS["succ_frac"]: "1",
    },
    {
        CELL_COLS["id"]: "INSRT",
        CELL_COLS["step"]: "Insert in Housing",
        CELL_COLS["lead_time_s"]: 0.60,
        CELL_COLS["scrap_rate"]: 0.0015,
        CELL_COLS["kw_per_unit"]: 117.50,
        CELL_COLS["spec_workers_per_machine"]: 1.25,
        CELL_COLS["supp_workers_per_machine"]: 0.0,
        CELL_COLS["capex_meur_per_machine"]: 2.5,
        CELL_COLS["env"]: "clean",
        CELL_COLS["footprint_m2"]: 93,
        CELL_COLS["succ"]: "FILL",
        CELL_COLS["succ_frac"]: "1",
    },
    {
        CELL_COLS["id"]: "FILL",
        CELL_COLS["step"]: "Electrolyte Fill",
        CELL_COLS["lead_time_s"]: 2.0,
        CELL_COLS["scrap_rate"]: 0.001,
        CELL_COLS["kw_per_unit"]: 22.0,
        CELL_COLS["spec_workers_per_machine"]: 0.66,
        CELL_COLS["supp_workers_per_machine"]: 0.0,
        CELL_COLS["capex_meur_per_machine"]: 1.0,
        CELL_COLS["env"]: "dry",
        CELL_COLS["footprint_m2"]: 60,
        CELL_COLS["succ"]: "FORM",
        CELL_COLS["succ_frac"]: "1",
    },
    {
        CELL_COLS["id"]: "FORM",
        CELL_COLS["step"]: "Formation",
        CELL_COLS["lead_time_s"]: 177.1875,
        CELL_COLS["scrap_rate"]: 0.005,
        CELL_COLS["kw_per_unit"]: 10.0,
        CELL_COLS["spec_workers_per_machine"]: 0.005,
        CELL_COLS["supp_workers_per_machine"]: 0.0,
        CELL_COLS["capex_meur_per_machine"]: 0.01,
        CELL_COLS["env"]: "none",
        CELL_COLS["footprint_m2"]: 5,
        CELL_COLS["succ"]: "AGE",
        CELL_COLS["succ_frac"]: "1",
    },
    {
        CELL_COLS["id"]: "AGE",
        CELL_COLS["step"]: "Ageing",
        CELL_COLS["lead_time_s"]: 270.0,
        CELL_COLS["scrap_rate"]: 0.0,
        CELL_COLS["kw_per_unit"]: 10.0,
        CELL_COLS["spec_workers_per_machine"]: 0.002,
        CELL_COLS["supp_workers_per_machine"]: 0.0,
        CELL_COLS["capex_meur_per_machine"]: 0.005,
        CELL_COLS["env"]: "none",
        CELL_COLS["footprint_m2"]: 5,
        CELL_COLS["succ"]: "FIN",
        CELL_COLS["succ_frac"]: "1",
    },
    {
        CELL_COLS["id"]: "FIN",
        CELL_COLS["step"]: "Final EoL Test",
        CELL_COLS["lead_time_s"]: 0.9375,
        CELL_COLS["scrap_rate"]: 0.025,
        CELL_COLS["kw_per_unit"]: 10.0,
        CELL_COLS["spec_workers_per_machine"]: 0.005,
        CELL_COLS["supp_workers_per_machine"]: 0.0,
        CELL_COLS["capex_meur_per_machine"]: 0.005,
        CELL_COLS["env"]: "none",
        CELL_COLS["footprint_m2"]: 5,
        CELL_COLS["succ"]: "",
        CELL_COLS["succ_frac"]: "",
    },
]

# Deterministic ordering for UI; DAG connectivity is defined via succ / succ_frac.
for i, r in enumerate(DEFAULT_STEPS, start=1):
    r["order"] = i
    r.setdefault("machines_override", None)

# Pre-compute a user-facing successor label ("successor_step") from the internal succ field
id_key = CELL_COLS["id"]
step_key = CELL_COLS["step"]
succ_key = CELL_COLS["succ"]

_id_to_step = {r[id_key]: r[step_key] for r in DEFAULT_STEPS}
for r in DEFAULT_STEPS:
    succ_raw = str(r.get(succ_key, "") or "").strip()
    if not succ_raw:
        r["successor_step"] = ""
        continue
    parts = [s.strip() for s in succ_raw.split(",") if s.strip()]
    succ_names = [_id_to_step.get(p, p) for p in parts]
    # For now we assume a single successor per UI; multiple successors will show as comma-separated names
    r["successor_step"] = ", ".join(succ_names)


# --- Default raw materials (used as template for all presets) -----------------

DEFAULT_RAW_MATERIALS: List[Dict[str, Any]] = [
    {
        "name": "NMC",
        "intro_step": "Cathode Mixing",
        "pricing_unit": "kg",
        "g_per_cell": 128.0,
        "eur_per_kg": 25.0,
        "area_per_cell_m2": 0.0,
        "eur_per_m2": 0.0,
    },
    {
        "name": "Graphite+Si",
        "intro_step": "Anode Mixing",
        "pricing_unit": "kg",
        "g_per_cell": 65.38,
        "eur_per_kg": 5.42,
        "area_per_cell_m2": 0.0,
        "eur_per_m2": 0.0,
    },
    {
        "name": "Conductive Carbon",
        "intro_step": "Cathode Mixing",
        "pricing_unit": "kg",
        "g_per_cell": 4.0,
        "eur_per_kg": 3.0,
        "area_per_cell_m2": 0.0,
        "eur_per_m2": 0.0,
    },
    {
        "name": "Binder (PVDF/CMC/SBR)",
        "intro_step": "Cathode Mixing",
        "pricing_unit": "kg",
        "g_per_cell": 6.0,
        "eur_per_kg": 13.59,
        "area_per_cell_m2": 0.0,
        "eur_per_m2": 0.0,
    },
    {
        "name": "Solvent (NMP)",
        "intro_step": "Cathode Mixing",
        "pricing_unit": "kg",
        "g_per_cell": 2.0,
        "eur_per_kg": 2.7,
        "area_per_cell_m2": 0.0,
        "eur_per_m2": 0.0,
    },
    {
        "name": "Aluminum Foil (CC)",
        "intro_step": "Cathode Coating",
        "pricing_unit": "kg",
        "g_per_cell": 22.0,
        "eur_per_kg": 4.87,
        "area_per_cell_m2": 0.0,
        "eur_per_m2": 0.0,
    },
    {
        "name": "Copper Foil (CC)",
        "intro_step": "Anode Coating",
        "pricing_unit": "kg",
        "g_per_cell": 34.0,
        "eur_per_kg": 12.3,
        "area_per_cell_m2": 0.0,
        "eur_per_m2": 0.0,
    },
    {
        "name": "Separator",
        "intro_step": "Winding",
        "pricing_unit": "m2",
        "g_per_cell": 0.0,
        "eur_per_kg": 0.0,
        "area_per_cell_m2": 0.07,
        "eur_per_m2": 0.26,
    },
    {
        "name": "Cell Case",
        "intro_step": "Insert in Housing",
        "pricing_unit": "m2",
        "g_per_cell": 65.5,
        "eur_per_kg": 0.0,
        "area_per_cell_m2": 0.0,
        "eur_per_m2": 0.0,
    },
    {
        "name": "Electrolyte",
        "intro_step": "Electrolyte Fill",
        "pricing_unit": "kg",
        "g_per_cell": 49.37,
        "eur_per_kg": 5.39,
        "area_per_cell_m2": 0.0,
        "eur_per_m2": 0.0,
    },
]


# --- Presets (all defaults live here) ----------------------------------------

nmca_v = _derive_ah_v_from_kwh(0.0927, 3.7)
NMC_CELL_AH, NMC_CELL_V = nmca_v
NMC_CELL_WH = NMC_CELL_AH * NMC_CELL_V

PRESETS: Dict[str, Dict[str, Any]] = {
    "NMC 4680 (Baseline)": {
        "general": {
            "electricity_price_eur_per_kwh": 0.1589,
            "baseline_building_kwh": 101.8,  # baseline building load (kWh/m²/a)
            "specialist_labor_eur_per_h": 44.0,
            "support_labor_eur_per_h": 0.0,
            "building_cost_eur_per_m2": 3360.0,
            "indoor_area_factor": 3.0,
            "outdoor_area_factor": 0.0,
            "outdoor_area_cost": 0.0,
            "clean_area_multiplier": 1.0,
            "clean_capex_eur_per_m2": 2850.0,
            "dry_capex_eur_per_m2": 120.0,
            "clean_opex_eur_per_h_per_m2": 0.0,
            "dry_opex_eur_per_h_per_m2": 0.0,
            "dry_area_multiplier": 1.15,
            "annual_output_gwh": 10.0,
            "cell_ah": NMC_CELL_AH,
            "cell_voltage": NMC_CELL_V,
            "cell_wh": NMC_CELL_WH,
            "working_days": 365.0,
            "shifts_per_day": 3.0,
            "shift_hours": 8.0,
            "avail": 0.855,
        },
        "econ": {
            "project_years": 10,
            "tax_rate": 0.30,
            "depreciation_years_equipment": 7,
            "depreciation_years_building": 33,
            "construction_years": 2,
            "ramp_years": 0,
            "ramp_scrap_rates": [],
            "ramp_output_rates": [],
            "capital_cost_wacc": 0.06,
            "desired_margin": 0.15,
            # Overhead factors (relative to direct labor or assets)
            "indirect_personnel_factor": 0.25,
            "logistics_personnel_factor": 0.15,
            "building_maintenance_factor": 0.02,  # of building CAPEX per year
            "machine_maintenance_factor": 0.03,   # of equipment CAPEX per year
            # Investment factors (multipliers on equipment CAPEX)
            "logistics_investment_factor": 0.05,
            "indirect_investment_factor": 0.10,
        },
        "steps": copy.deepcopy(DEFAULT_STEPS),
        "materials": copy.deepcopy(DEFAULT_RAW_MATERIALS),
        "note": (
            "Baseline NMC 4680-type cell process with ISO environments. "
            "Lead-time-based takt sizing with explicit overhead/maintenance factors."
        ),
    },
    # Example SSB preset (illustrative only)
    "SSB Cell (Example)": {
        "general": {
            "electricity_price_eur_per_kwh": 0.18,
            "baseline_building_kwh": 100.0,
            "specialist_labor_eur_per_h": 48.0,
            "support_labor_eur_per_h": 40.0,
            "building_cost_eur_per_m2": 3600.0,
            "indoor_area_factor": 3.0,
            "outdoor_area_factor": 0.0,
            "outdoor_area_cost": 0.0,
            "clean_area_multiplier": 1.0,
            "clean_capex_eur_per_m2": 2850.0,
            "dry_capex_eur_per_m2": 120.0,
            "clean_opex_eur_per_h_per_m2": 0.0,
            "dry_opex_eur_per_h_per_m2": 0.0,
            "dry_area_multiplier": 1.15,
            "annual_output_gwh": 5.0,
            "cell_ah": 20.0,
            "cell_voltage": 3.8,
            "cell_wh": 20.0 * 3.8,
            "working_days": 330.0,
            "shifts_per_day": 3.0,
            "shift_hours": 8.0,
            "avail": 0.80,
        },
        "econ": {
            "project_years": 12,
            "tax_rate": 0.30,
            "depreciation_years_equipment": 8,
            "depreciation_years_building": 35,
            "construction_years": 3,
            "ramp_years": 0,
            "ramp_scrap_rates": [],
            "ramp_output_rates": [],
            "capital_cost_wacc": 0.07,
            "desired_margin": 0.18,
            "indirect_personnel_factor": 0.30,
            "logistics_personnel_factor": 0.18,
            "building_maintenance_factor": 0.025,
            "machine_maintenance_factor": 0.035,
            "logistics_investment_factor": 0.06,
            "indirect_investment_factor": 0.12,
        },
        "steps": copy.deepcopy(DEFAULT_STEPS),
        "materials": copy.deepcopy(DEFAULT_RAW_MATERIALS),
        "note": "Illustrative solid-state cell preset (parameters not tied to specific product).",
    },
    # Example SIB preset (illustrative only)
    "SIB Cell (Example)": {
        "general": {
            "electricity_price_eur_per_kwh": 0.15,
            "baseline_building_kwh": 100.0,
            "specialist_labor_eur_per_h": 40.0,
            "support_labor_eur_per_h": 32.0,
            "building_cost_eur_per_m2": 3000.0,
            "indoor_area_factor": 3.0,
            "outdoor_area_factor": 0.0,
            "outdoor_area_cost": 0.0,
            "clean_area_multiplier": 1.0,
            "clean_capex_eur_per_m2": 2850.0,
            "dry_capex_eur_per_m2": 120.0,
            "clean_opex_eur_per_h_per_m2": 0.0,
            "dry_opex_eur_per_h_per_m2": 0.0,
            "dry_area_multiplier": 1.15,
            "annual_output_gwh": 8.0,
            "cell_ah": 12.0,
            "cell_voltage": 2.5,
            "cell_wh": 12.0 * 2.5,
            "working_days": 350.0,
            "shifts_per_day": 2.5,
            "shift_hours": 8.0,
            "avail": 0.82,
        },
        "econ": {
            "project_years": 10,
            "tax_rate": 0.28,
            "depreciation_years_equipment": 7,
            "depreciation_years_building": 30,
            "construction_years": 2,
            "ramp_years": 0,
            "ramp_scrap_rates": [],
            "ramp_output_rates": [],
            "capital_cost_wacc": 0.065,
            "desired_margin": 0.16,
            "indirect_personnel_factor": 0.22,
            "logistics_personnel_factor": 0.12,
            "building_maintenance_factor": 0.018,
            "machine_maintenance_factor": 0.028,
            "logistics_investment_factor": 0.05,
            "indirect_investment_factor": 0.09,
        },
        "steps": copy.deepcopy(DEFAULT_STEPS),
        "materials": copy.deepcopy(DEFAULT_RAW_MATERIALS),
        "note": "Illustrative sodium-ion cell preset (parameters not tied to specific product).",
    },
}

# Ensure deterministic ordering on preset step lists and presence of successor_step
for preset in PRESETS.values():
    if "steps" in preset:
        for i, r in enumerate(preset["steps"], start=1):
            r.setdefault("order", i)
            # successor_step is UI-facing; DAG uses "succ"/"succ_frac"
            r.setdefault("successor_step", r.get("successor_step", ""))

DEFAULT_PRESET_KEY = "NMC 4680 (Baseline)"
DEFAULT_GENERAL = PRESETS[DEFAULT_PRESET_KEY]["general"]
DEFAULT_ECON = PRESETS[DEFAULT_PRESET_KEY]["econ"]

# Defaults for the utilization-vs-GWh sweep controls
GWH_SWEEP_MIN_FACTOR = 0.2
GWH_SWEEP_MAX_MULTIPLIER = 5.0
GWH_SWEEP_MIN_FLOOR = 0.1
GWH_SWEEP_DEFAULT_POINTS = 10

def _validate_preset_dict(p: Dict[str, Any]) -> Dict[str, Any]:
    """Lightweight validation/normalization for external presets."""
    if not isinstance(p, dict):
        raise ValueError("Preset must be a JSON object")
    for key in ["general", "econ", "steps", "materials"]:
        if key not in p:
            raise ValueError(f"Preset missing required key '{key}'")
    if not isinstance(p["steps"], list) or not isinstance(p["materials"], list):
        raise ValueError("Preset 'steps' and 'materials' must be lists")
    if "note" not in p:
        p["note"] = ""
    return p


def load_external_presets(asset_dir: str = "assets") -> Dict[str, Dict[str, Any]]:
    """Load *.json presets from the assets directory."""
    presets: Dict[str, Dict[str, Any]] = {}
    if not os.path.isdir(asset_dir):
        return presets
    for fname in os.listdir(asset_dir):
        if not fname.lower().endswith(".json"):
            continue
        fpath = os.path.join(asset_dir, fname)
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                data = json.load(f)
            preset = _validate_preset_dict(data)
            key = preset.get("name") or os.path.splitext(fname)[0]
            presets[key] = preset
        except Exception:
            # Ignore bad files; keep the app running
            continue
    return presets


EXTERNAL_PRESETS = load_external_presets("assets")


def all_presets() -> Dict[str, Dict[str, Any]]:
    """Return combined built-in and external presets."""
    combined = copy.deepcopy(PRESETS)
    for k, v in EXTERNAL_PRESETS.items():
        combined.setdefault(k, copy.deepcopy(v))
    return combined


def current_app_presets(store_data: Dict[str, Any] | None = None) -> Dict[str, Any]:
    """Merge built-in + disk-loaded + in-app imported presets (from store)."""
    base = all_presets()
    if isinstance(store_data, dict):
        for k, v in store_data.items():
            base[k] = v
    return base


def _resolve_cell_params(
    ah_in: Any,
    wh_in: Any,
    v_in: Any,
    default_gen: Dict[str, Any] = DEFAULT_GENERAL,
) -> Tuple[float, float, float]:
    """
    Resolve (Ah, Wh, V) from user input:
    - If two positive values provided -> compute the third.
    - If all three provided -> trust Ah & V, recompute Wh = Ah * V for consistency.
    - If fewer than two -> fall back to preset defaults.
    """

    def _to_pos_or_none(val: Any) -> float | None:
        try:
            f = float(val)
        except (TypeError, ValueError):
            return None
        return f if f > 0 else None

    ah = _to_pos_or_none(ah_in)
    wh = _to_pos_or_none(wh_in)
    v = _to_pos_or_none(v_in)

    provided = {"ah": ah is not None, "wh": wh is not None, "v": v is not None}
    count = sum(provided.values())

    # Start from defaults
    ah_res = ah if ah is not None else float(default_gen["cell_ah"])
    wh_res = wh if wh is not None else float(default_gen["cell_wh"])
    v_res = v if v is not None else float(default_gen["cell_voltage"])

    if count >= 2:
        if ah is None:
            # Wh and V provided
            ah_res = wh_res / v_res if v_res > 0 else float(default_gen["cell_ah"])
        elif wh is None:
            # Ah and V provided
            wh_res = ah_res * v_res
        elif v is None:
            # Ah and Wh provided
            v_res = wh_res / ah_res if ah_res > 0 else float(default_gen["cell_voltage"])
        else:
            # All three provided: keep Ah & V authoritative
            wh_res = ah_res * v_res

    return ah_res, wh_res, v_res


# ============================================================
# DAG helpers: parse successors, build graph, simulate unit flow
# ============================================================

def _parse_successors(successors_str: Any, fractions_str: Any) -> List[Tuple[str, float]]:
    """
    Parse comma-separated successors and fractions into a normalized list.

    Examples:
        "A,B", "0.7,0.3" -> [("A", 0.7), ("B", 0.3)]
        "A,B", ""        -> [("A", 0.5), ("B", 0.5)]
    """
    if not isinstance(successors_str, str) or not successors_str.strip():
        return []

    ids = [s.strip() for s in successors_str.split(",") if s.strip()]
    if not ids:
        return []

    # No fractions -> equal split
    if not isinstance(fractions_str, str) or not fractions_str.strip():
        frac = 1.0 / len(ids)
        return [(sid, frac) for sid in ids]

    fracs_raw: List[float] = []
    for tok in fractions_str.split(","):
        tok = tok.strip()
        if not tok:
            continue
        try:
            fracs_raw.append(float(tok))
        except ValueError:
            fracs_raw.append(float("nan"))

    # Pad / trim to same length as ids
    if len(fracs_raw) < len(ids):
        fracs_raw += [0.0] * (len(ids) - len(fracs_raw))
    fracs_raw = fracs_raw[: len(ids)]

    # Normalize only finite values; if sum<=0, fall back to equal split
    s = sum(f for f in fracs_raw if np.isfinite(f))
    if s <= 0:
        frac = 1.0 / len(ids)
        return [(sid, frac) for sid in ids]

    fracs = [(f / s) if np.isfinite(f) else 0.0 for f in fracs_raw]
    return list(zip(ids, fracs))


def _build_graph(
    df: pd.DataFrame,
    id_col: str,
    succ_col: str,
    frac_col: str,
) -> Dict[str, Any]:
    """
    Build a DAG from the step table.

    Returns a dict with:
      "ids":   [step_ids],
      "succ":  {sid: [(succ_id, frac), ...]},
      "pred":  {sid: [pred_ids]},
      "roots": [root_ids],         # steps without predecessors in the DAG
      "root":  root_id,            # first root (kept for backwards compatibility)
      "sinks": [sink_ids],         # steps without successors
      "topo":  [ids in topological order],
    """
    ids = df[id_col].astype(str).tolist()
    if len(ids) != len(set(ids)):
        raise ValueError("Step IDs must be unique for DAG construction.")

    succ: Dict[str, List[Tuple[str, float]]] = {sid: [] for sid in ids}
    pred: Dict[str, List[str]] = {sid: [] for sid in ids}

    for _, row in df.iterrows():
        sid = str(row[id_col])
        s_list = _parse_successors(row.get(succ_col, ""), row.get(frac_col, ""))
        for tid, frac in s_list:
            if tid not in succ:
                # Ignore dangling edges (helps when user references missing steps).
                continue
            succ[sid].append((tid, frac))
            pred[tid].append(sid)

    roots = [sid for sid in ids if not pred[sid]]
    sinks = [sid for sid in ids if not succ[sid]]

    if not roots:
        raise ValueError("No root step found (every step has at least one predecessor).")

    # Kahn's algorithm for topological order using deque for performance
    in_deg = {sid: len(pred[sid]) for sid in ids}
    queue = deque([sid for sid in ids if in_deg[sid] == 0])
    topo: List[str] = []

    while queue:
        n = queue.popleft()
        topo.append(n)
        for m, _frac in succ[n]:
            in_deg[m] -= 1
            if in_deg[m] == 0:
                queue.append(m)

    if len(topo) != len(ids):
        raise ValueError("Process graph has at least one cycle; only DAGs are supported.")

    return {
        "ids": ids,
        "succ": succ,
        "pred": pred,
        "roots": roots,
        "root": roots[0],  # kept for backwards compatibility; prefer "roots"
        "sinks": sinks,
        "topo": topo,
    }


def _simulate_unit_flow(
    df: pd.DataFrame,
    graph: Dict[str, Any],
    root_ids: List[str] | None,
    id_col: str,
    scrap_col: str,
    root_injection: Dict[str, float] | None = None,
    normalize_roots: bool = True,
) -> Tuple[
    Dict[str, float],
    Dict[str, float],
    Dict[str, float],
    Dict[Tuple[str, str], float],
    float,
]:
    """
    Simulate the flow of unit(s) entering the DAG.

    - If root_ids is None, one unit is injected into *each* root in graph["roots"].
      This is what we use for the main line sizing so that multiple start points
      (parallel process branches) are all active.
    - If a specific list is provided, one unit is injected into each of those
      steps. This is used for material survival from an intro step.

    root_injection: optional map of {root_id: amount}. If None, defaults to 1 per
      active root. If normalize_roots=True, the map is normalized to sum to 1.0
      so upstream demand is not scaled by the number of roots.

    Returns:
      flow_in[step]:     units entering each step per unit configuration
      good_out[step]:    good units leaving each step to successors
      scrap_amt[step]:   units scrapped at each step
      edge_good[(i,j)]:  good units on each edge
      final_good:        total final-good units (aggregated across all sinks)
    """
    ids = graph["ids"]
    succ = graph["succ"]

    rows = {str(r[id_col]): r for _, r in df.iterrows()}

    flow_in: Dict[str, float] = {sid: 0.0 for sid in ids}
    good_out: Dict[str, float] = {sid: 0.0 for sid in ids}
    scrap_amt: Dict[str, float] = {sid: 0.0 for sid in ids}
    edge_good: Dict[Tuple[str, str], float] = {}

    # Resolve which roots to inject into
    if root_ids is None:
        active_roots = list(graph.get("roots", []))
    elif isinstance(root_ids, str):
        active_roots = [root_ids]
    else:
        active_roots = list(root_ids)

    if not active_roots:
        raise ValueError("No root steps supplied for unit-flow simulation.")

    if root_injection is None:
        root_injection = {rid: 1.0 for rid in active_roots}

    root_injection = {
        str(k): float(v) for k, v in (root_injection or {}).items() if k in flow_in
    }
    if not root_injection:
        raise ValueError("Root injection map is empty after validation.")

    if normalize_roots:
        total_injection = sum(root_injection.values())
        if total_injection > 0:
            root_injection = {
                k: v / total_injection for k, v in root_injection.items()
            }

    for rid, amt in root_injection.items():
        if rid not in flow_in:
            raise KeyError(f"Unknown root step id '{rid}' in unit-flow simulation.")
        flow_in[rid] += max(amt, 0.0)

    final_good = 0.0

    for sid in graph["topo"]:
        row = rows[sid]
        scrap_rate = float(row.get(scrap_col, 0.0) or 0.0)
        scrap_rate = min(max(scrap_rate, 0.0), 1.0)

        # Assembly-aware merge: if multiple predecessors feed this step, the
        # assembled inflow is limited by the smallest contributing branch.
        incoming_edges = [edge_good.get((p, sid), 0.0) for p in graph["pred"][sid]]
        if len(incoming_edges) > 1:
            fin = min(incoming_edges)
        else:
            fin = flow_in[sid]
        good = fin * (1.0 - scrap_rate)
        scrap = fin - good

        good_out[sid] = good
        scrap_amt[sid] = scrap

        if succ[sid]:
            remaining = good
            for tid, frac in succ[sid]:
                amt = good * frac
                remaining -= amt
                flow_in[tid] += amt
                edge_good[(sid, tid)] = edge_good.get((sid, tid), 0.0) + amt
            # Any tiny leftover is "final good" that doesn't go to another step
            final_good += max(0.0, remaining)
        else:
            # No successors ⇒ this step's good output is final
            final_good += good

    return flow_in, good_out, scrap_amt, edge_good, final_good


def _apply_successor_ui_to_graph(df: pd.DataFrame) -> pd.DataFrame:
    """
    If a user-facing 'successor_step' column is present, derive the internal
    DAG fields 'succ' and 'succ_frac' from it.

    - Values in 'successor_step' are step names (or comma-separated list of
      names/IDs). The function maps them to step_ids.
    - Empty or 'End' (value == "") means "no successor" (sink step).
    - For multiple successors, fractions are distributed equally for now.
    """
    if "successor_step" not in df.columns:
        return df

    df = df.copy()

    # Build name -> ID mapping once
    name_to_sid: Dict[str, str] = {}
    for _, row in df.iterrows():
        step_name = str(row.get("step", "")).strip()
        sid = str(row.get("step_id", "")).strip()
        if step_name:
            # First occurrence of a name wins (stable mapping)
            name_to_sid.setdefault(step_name, sid)

    succ_values: List[str] = []
    frac_values: List[str] = []

    for _, row in df.iterrows():
        raw = row.get("successor_step", "")
        raw = "" if raw is None else str(raw).strip()

        if not raw:
            # "End" or blank -> sink step
            succ_values.append("")
            frac_values.append("")
            continue

        # Allow comma-separated list of successors in the UI for advanced cases
        parts = [p.strip() for p in raw.split(",") if p.strip()]
        succ_ids: List[str] = []
        for p in parts:
            # Map step name -> step_id; fall back to literal if not found
            succ_ids.append(name_to_sid.get(p, p))

        succ_str = ",".join(succ_ids)
        succ_values.append(succ_str)

        n = len(succ_ids)
        if n == 0:
            frac_values.append("")
        else:
            frac_each = 1.0 / n
            frac_values.append(",".join([f"{frac_each:g}"] * n))

    df["succ"] = succ_values
    df["succ_frac"] = frac_values
    return df


# ============================================================
# Core data structures
# ============================================================

@dataclass(frozen=True)
class GeneralAssumptions:
    electricity_price_eur_per_kwh: float
    baseline_building_kwh: float
    specialist_labor_eur_per_h: float
    support_labor_eur_per_h: float
    building_cost_eur_per_m2: float
    indoor_area_factor: float
    outdoor_area_factor: float
    outdoor_area_cost: float
    clean_area_multiplier: float
    dry_area_multiplier: float
    clean_capex_eur_per_m2: float
    dry_capex_eur_per_m2: float
    clean_opex_eur_per_h_per_m2: float
    dry_opex_eur_per_h_per_m2: float
    annual_output_gwh: float
    cell_ah: float
    cell_wh: float
    cell_voltage: float
    working_days: float
    shifts_per_day: float
    shift_hours: float
    avail: float


@dataclass(frozen=True)
class Economics:
    project_years: int
    tax_rate: float
    depreciation_years_equipment: int
    depreciation_years_building: int
    construction_years: int
    ramp_years: int
    capital_cost_wacc: float
    desired_margin: float
    # Overhead factors
    indirect_personnel_factor: float
    logistics_personnel_factor: float
    building_maintenance_factor: float
    machine_maintenance_factor: float
    # Investment factors
    logistics_investment_factor: float
    indirect_investment_factor: float
    ramp_scrap_rates: List[float] | None = None  # per-ramp-year scrap, revenue-only
    ramp_output_rates: List[float] | None = None


@dataclass
class BatteryCostModel:
    """
    Encapsulates the DAG-based cell manufacturing cost model.

    - Uses lead_time_s + DAG flows to size machines and capacity.
    - Materials survival is computed from actual process scrap from the
      intro step to final good output via re-simulation on the DAG.
    - DAG supports multiple start points (roots) feeding a single conceptual
      final-good node (aggregated over all sinks).
    - Process topology is strictly defined via "successors" from the UI
      (successor_step -> succ/succ_frac).
    """

    general: GeneralAssumptions
    econ: Economics
    steps: pd.DataFrame
    raw_materials: pd.DataFrame

    def compute(self) -> Dict[str, Any]:
        g, e = self.general, self.econ

        # ------------------------------------------------------------------
        # 1) Normalize step table and ensure required columns
        # ------------------------------------------------------------------
        df = self.steps.copy().reset_index(drop=True)

        for col, default in [
            ("step", ""),
            ("step_id", None),
            ("env", "none"),
            ("lead_time_s", 1.0),
            ("scrap_rate", 0.0),
            ("kw_per_unit", 0.0),
            ("spec_workers_per_machine", 0.0),
            ("supp_workers_per_machine", 0.0),
            ("capex_meur_per_machine", 0.0),
            ("footprint_m2", 0.0),
            ("successor_step", ""),  # UI-facing successor label
            ("succ", ""),            # internal DAG successors (step_ids)
            ("succ_frac", ""),       # internal DAG successor fractions
            ("order", None),
            ("machines_override", None),
        ]:
            if col not in df.columns:
                df[col] = default

        df["step"] = df["step"].astype(str).replace({"nan": ""})

        # If no explicit step_id, derive stable IDs from names
        if df["step_id"].isna().all():
            df["step_id"] = [
                f"S{i+1:03d}" if not name else str(name).replace(" ", "_").upper()
                for i, name in enumerate(df["step"])
            ]
        else:
            df["step_id"] = df["step_id"].fillna("").astype(str)

            mask = df["step_id"].eq("")
            df.loc[mask, "step_id"] = [f"S{i + 1:03d}" for i in range(mask.sum())
                                       ]

        # Enforce uniqueness of IDs
        if df["step_id"].duplicated().any():
            seen: Dict[str, int] = {}
            new_ids: List[str] = []
            for sid in df["step_id"]:
                base = sid or "S"
                count = seen.get(base, 0) + 1
                seen[base] = count
                new_ids.append(f"{base}_{count}" if count > 1 else base)
            df["step_id"] = new_ids

        # Keep stable UI order
        if df["order"].isna().any():
            df["order"] = range(1, len(df) + 1)
        df = df.sort_values("order").reset_index(drop=True)

        # Derive internal DAG connectivity from the UI-facing successor_step
        df = _apply_successor_ui_to_graph(df)

        # ------------------------------------------------------------------
        # 2) Time base and annual volume target
        # ------------------------------------------------------------------
        avail = max(min(g.avail, 1.0), 0.0)
        hours_year = g.working_days * g.shifts_per_day * g.shift_hours
        avail_time_seconds = hours_year * 3600.0 * avail

        cell_wh = g.cell_wh if g.cell_wh > 0 else g.cell_ah * g.cell_voltage
        cell_wh = max(cell_wh, 1e-9)
        cell_kwh = cell_wh / 1000.0

        final_cells_required = (g.annual_output_gwh * 1_000_000.0) / cell_kwh

        # ------------------------------------------------------------------
        # 3) Build DAG & simulate unit flow from all roots
        # ------------------------------------------------------------------
        graph = _build_graph(
            df,
            id_col="step_id",
            succ_col="succ",
            frac_col="succ_frac",
        )
        active_roots = graph["roots"]

        (
            flow_in_unit,
            good_out_unit,
            scrap_unit,
            edge_good_unit,
            final_good_unit,
        ) = _simulate_unit_flow(
            df,
            graph,
            root_ids=active_roots,          # support multiple start points
            id_col="step_id",
            scrap_col="scrap_rate",
        )

        if final_good_unit <= 0:
            raise ValueError(
                "Final good output from the DAG configuration is zero; "
                "check scrap rates and DAG connections."
            )

        # Scale per-unit flows to annual volumes
        root_input_per_year = final_cells_required / final_good_unit

        # Pre-allocate output columns
        df["input_units_per_year"] = 0.0
        df["good_units_per_year"] = 0.0
        df["scrap_units_per_year"] = 0.0
        df["machines"] = 0
        df["capacity_cells_per_year"] = 0.0
        df["capacity_ratio_vs_demand"] = np.nan

        id_to_idx = {sid: i for i, sid in enumerate(df["step_id"])}

        bottleneck_ratio = math.inf
        bottleneck_sid: str | None = None

        # ------------------------------------------------------------------
        # 4) Size machines per step based on lead_time_s and DAG demand
        # ------------------------------------------------------------------
        #for sid in graph["topo"]:
           # idx = id_to_idx[sid]
           # fin_unit = float(flow_in_unit[sid])
           # good_unit_val = float(good_out_unit[sid])
           # scrap_unit_val = float(scrap_unit[sid])

        for sid in graph["topo"]:
            idx = id_to_idx[sid]

            good_unit_val = float(good_out_unit[sid])
            scrap_unit_val = float(scrap_unit[sid])
            fin_unit = good_unit_val + scrap_unit_val

            fin_year = fin_unit * root_input_per_year
            good_year = good_unit_val * root_input_per_year
            scrap_year = scrap_unit_val * root_input_per_year


            df.at[idx, "input_units_per_year"] = fin_year
            df.at[idx, "good_units_per_year"] = good_year
            df.at[idx, "scrap_units_per_year"] = scrap_year

            lead_time_s = float(df.at[idx, "lead_time_s"] or 0.0)
            lead_time_s = max(lead_time_s, 1e-6)  # avoid division by zero

            override_raw = df.at[idx, "machines_override"]
            try:
                override_val = float(override_raw) if override_raw is not None else 0.0
            except (TypeError, ValueError):(
                    override_val) = 0.0
            override_val = max(override_val, 0.0)

            if override_val > 0:
                    machines = int(math.ceil(override_val))

            elif fin_year > 0 and avail_time_seconds > 0:
                seconds_needed = fin_year * lead_time_s
                machines = int(math.ceil(seconds_needed / avail_time_seconds))
                machines = max(machines, 1)
            else:
                machines = 0

            df.at[idx, "machines"] = machines

            # Capacity purely from lead_time and available time; no throughput modes
            if machines > 0:
                cap_cells = machines * avail_time_seconds / lead_time_s
            else:
                cap_cells = 0.0
            df.at[idx, "capacity_cells_per_year"] = cap_cells

            if fin_year > 0:
                ratio = cap_cells / fin_year
                df.at[idx, "capacity_ratio_vs_demand"] = ratio
                if ratio < bottleneck_ratio:
                    bottleneck_ratio = ratio
                    bottleneck_sid = sid

        if bottleneck_sid is None:
            # Fallback: treat first step as bottleneck with ratio 1
            bottleneck_sid = graph["topo"][0]
            bottleneck_ratio = 1.0

        # Line capacity determined by bottleneck (using its capacity/demand ratio)
        line_capacity_cells = final_cells_required * bottleneck_ratio
        #utilization = final_cells_required / max(line_capacity_cells, 1.0)

        # Derived UI helper columns
        df["demand_cells_step_per_year"] = df["input_units_per_year"]
        df["required_cycle_time_s"] = np.where(
            df["demand_cells_step_per_year"] > 0,
            (df["machines"] * avail_time_seconds) / df["demand_cells_step_per_year"],
            np.nan,
        )
        df["step_utilization"] = df["demand_cells_step_per_year"] / df["capacity_cells_per_year"].replace(0, np.nan)

        total_output = float(df["demand_cells_step_per_year"].sum())
        total_capacity = float(df["capacity_cells_per_year"].sum())
        utilization = total_output / max(total_capacity, 1e-9)  # weighted average = sum(demand)/sum(capacity)


        # ------------------------------------------------------------------
        # 5) Materials: survival from intro_step via DAG re-simulation
        # ------------------------------------------------------------------
        mats = self.raw_materials.copy()

        if "intro_step" not in mats.columns:
            mats["intro_step"] = ""
        mats["intro_step"] = mats["intro_step"].astype(str)

        # Make sure all required columns exist and are numeric
        if "pricing_unit" not in mats.columns:
            mats["pricing_unit"] = "kg"
        if "g_per_cell" not in mats.columns:
            mats["g_per_cell"] = 0.0
        if "area_per_cell_m2" not in mats.columns:
            mats["area_per_cell_m2"] = 0.0
        if "eur_per_kg" not in mats.columns:
            mats["eur_per_kg"] = 0.0
        if "eur_per_m2" not in mats.columns:
            mats["eur_per_m2"] = 0.0

        mats["pricing_unit"] = mats["pricing_unit"].astype(str).str.lower()
        mats["kg_per_cell"] = mats["g_per_cell"].astype(float) / 1000.0
        mats["m2_per_cell"] = mats["area_per_cell_m2"].astype(float)
        mats["eur_per_kg"] = mats["eur_per_kg"].astype(float)
        mats["eur_per_m2"] = mats["eur_per_m2"].astype(float)

        # Map step name -> step_id (first occurrence)
        name_to_sid: Dict[str, str] = {}
        for _, row in df.iterrows():
            name_to_sid.setdefault(row["step"], row["step_id"])

        def _material_survival(step_name: str) -> float:
            """Survival from intro step to final good using main DAG flows."""
            sid = name_to_sid.get(step_name)
            if not sid:
                return 1.0
            entered = float(flow_in_unit.get(sid, 0.0))
            if entered <= 0:
                return 1e-6
            return max(final_good_unit / entered, 1e-6)

        mats["survival"] = mats["intro_step"].map(_material_survival).astype(float)

        is_m2 = mats["pricing_unit"].eq("m2")
        mats["net_cost_per_cell_eur"] = np.where(
            is_m2,
            mats["m2_per_cell"] * mats["eur_per_m2"],
            mats["kg_per_cell"] * mats["eur_per_kg"],
        )
        mats["procurement_cost_per_cell_eur"] = (
            mats["net_cost_per_cell_eur"] / mats["survival"]
        )

        materials_procurement_per_cell = float(
            mats["procurement_cost_per_cell_eur"].sum()
        )

        # Attribute intro-step materials cost to intro step for per-step reporting
        df["materials_cost_per_cell_total_eur"] = 0.0
        for _, r in mats.iterrows():
            idx = df.index[df["step"] == r["intro_step"]]
            if len(idx):
                df.loc[
                    idx[0], "materials_cost_per_cell_total_eur"
                ] += float(r["procurement_cost_per_cell_eur"])

        # ------------------------------------------------------------------
        # 6) Labor per cell (based on line takt, not per-step takt)
        # ------------------------------------------------------------------
        line_cycle_time_s = avail_time_seconds / max(final_cells_required, 1.0)
        cpm = max(final_cells_required, 1.0) * 60 / avail_time_seconds

        df["spec_hours_per_cell"] = (
            df["spec_workers_per_machine"] * df["machines"]
        ) * (line_cycle_time_s / 3600.0)
        df["supp_hours_per_cell"] = (
            df["supp_workers_per_machine"] * df["machines"]
        ) * (line_cycle_time_s / 3600.0)

        spec_labor_per_cell = float(
            (df["spec_hours_per_cell"] * g.specialist_labor_eur_per_h).sum()
        )
        supp_labor_per_cell = float(
            (df["supp_hours_per_cell"] * g.support_labor_eur_per_h).sum()
        )
        labor_per_cell = spec_labor_per_cell + supp_labor_per_cell

        # ------------------------------------------------------------------
        # 7) CAPEX & Area (incl. logistics/indirect investment)
        # ------------------------------------------------------------------
        df["step_capex_total_eur"] = (
            df["capex_meur_per_machine"] * 1_000_000.0 * df["machines"]
        )
        total_capital_equipment_base = float(df["step_capex_total_eur"].sum())

        logistics_capex = total_capital_equipment_base * max(
            e.logistics_investment_factor, 0.0
        )
        indirect_capex = total_capital_equipment_base * max(
            e.indirect_investment_factor, 0.0
        )
        total_capital_equipment = (
            total_capital_equipment_base + logistics_capex + indirect_capex
        )

        base_indoor_total = float(
            (df["footprint_m2"] * df["machines"]).sum()
        )
        env_series = df["env"].astype(str).str.lower()

        base_indoor_clean = float(
            (
                df.loc[env_series.eq("clean"), "footprint_m2"]
                * df.loc[env_series.eq("clean"), "machines"]
            ).sum()
        )
        base_indoor_dry = float(
            (
                df.loc[env_series.eq("dry"), "footprint_m2"]
                * df.loc[env_series.eq("dry"), "machines"]
            ).sum()
        )
        base_indoor_none = base_indoor_total - base_indoor_clean - base_indoor_dry

        indoor_none = base_indoor_none * max(g.indoor_area_factor, 0.0)
        indoor_clean_raw = base_indoor_clean * max(g.indoor_area_factor, 0.0)
        indoor_dry_raw = base_indoor_dry * max(g.indoor_area_factor, 0.0)
        indoor_clean = indoor_clean_raw * max(g.clean_area_multiplier, 0.0)
        indoor_dry = indoor_dry_raw * max(g.dry_area_multiplier, 0.0)

        required_indoor_area = indoor_none + indoor_clean + indoor_dry
        required_outdoor_area = (
            required_indoor_area * max(g.outdoor_area_factor, 0.0)
        )
        total_required_area = required_indoor_area + required_outdoor_area

        base_cost = max(g.building_cost_eur_per_m2, 0.0)
        cost_indoor_none = indoor_none * max(g.building_cost_eur_per_m2, 0.0)
        cost_indoor_clean = indoor_clean * ( max(g.clean_capex_eur_per_m2, 0.0) + max(g.building_cost_eur_per_m2, 0.0) )
        cost_indoor_dry = indoor_dry * ( max(g.dry_capex_eur_per_m2, 0.0) + max(g.building_cost_eur_per_m2, 0.0) )
        cost_outdoor = required_outdoor_area * max(g.outdoor_area_cost, 0.0)
        building_value = cost_indoor_none + cost_indoor_clean + cost_indoor_dry + cost_outdoor

        annual_depr_equipment = (total_capital_equipment / e.depreciation_years_equipment)
        annual_depr_building = building_value / e.depreciation_years_building
        annual_depreciation = annual_depr_equipment + annual_depr_building

        actual_cells_for_cost = final_cells_required
        depreciation_per_cell = annual_depreciation / max(actual_cells_for_cost, 1.0)



        # ------------------------------------------------------------------
        # 8) Energy per cell (process + building baseline)
        # ------------------------------------------------------------------
        df["kwh_per_cell_process"] = df["kw_per_unit"] * (df["lead_time_s"] / 3600.0)
        df["energy_cost_per_cell_process_eur"] = (
            df["kwh_per_cell_process"] * g.electricity_price_eur_per_kwh
        )
        process_energy_per_cell = float(
            df["energy_cost_per_cell_process_eur"].sum()
        )

        annual_building_kwh = g.baseline_building_kwh * required_indoor_area
        building_energy_kwh_per_cell = (
            annual_building_kwh / max(actual_cells_for_cost, 1.0)
        )

        hours_year = g.working_days * g.shifts_per_day * g.shift_hours
        opex_building_eur_per_year = hours_year * (
            (annual_building_kwh * g.electricity_price_eur_per_kwh / hours_year)
                + indoor_none * 0  #Could be added
                + indoor_clean * max(g.clean_opex_eur_per_h_per_m2, 0.0)
                + indoor_dry * max(g.dry_opex_eur_per_h_per_m2, 0.0)
        )
        building_opex_per_cell = opex_building_eur_per_year / max(final_cells_required, 1.0)

        energy_per_cell = process_energy_per_cell + building_opex_per_cell
        df["energy_cost_per_cell_eur"] = (df["energy_cost_per_cell_process_eur"] + building_opex_per_cell)

        # ------------------------------------------------------------------
        # 9) Overheads & unit cost
        # ------------------------------------------------------------------
        indirect_personnel_oh_per_cell = (
            labor_per_cell * max(e.indirect_personnel_factor, 0.0)
        )
        logistics_personnel_oh_per_cell = (
            labor_per_cell * max(e.logistics_personnel_factor, 0.0)
        )

        annual_building_maintenance = (
            building_value * max(e.building_maintenance_factor, 0.0)
        )
        building_maintenance_oh_per_cell = (
            annual_building_maintenance / max(actual_cells_for_cost, 1.0)
        )

        annual_machine_maintenance = (
            total_capital_equipment * max(e.machine_maintenance_factor, 0.0)
        )
        machine_maintenance_oh_per_cell = (
            annual_machine_maintenance / max(actual_cells_for_cost, 1.0)
        )

        unit_cost_build = (
            materials_procurement_per_cell
            + energy_per_cell
            + labor_per_cell
            + indirect_personnel_oh_per_cell
            + logistics_personnel_oh_per_cell
            + building_maintenance_oh_per_cell
            + machine_maintenance_oh_per_cell
            + depreciation_per_cell
        )

        cost_build_per_kwh = unit_cost_build / cell_kwh

        # ------------------------------------------------------------------
        # 10) Sensitivity (±25%) on major components
        # ------------------------------------------------------------------
        sens_items: List[Dict[str, Any]] = []

        def add_s(name: str, value: float) -> None:
            sens_items.append(
                {
                    "Parameter": name,
                    "Low": unit_cost_build - 0.25 * value,
                    "High": unit_cost_build + 0.25 * value,
                }
            )

        add_s("Materials", materials_procurement_per_cell)
        add_s("Energy (process + building)", energy_per_cell)
        add_s("Specialist labor", spec_labor_per_cell)
        add_s("Support labor", supp_labor_per_cell)
        add_s("Indirect personnel OH", indirect_personnel_oh_per_cell)
        add_s("Logistics personnel OH", logistics_personnel_oh_per_cell)
        add_s("Building maintenance", building_maintenance_oh_per_cell)
        add_s("Machine maintenance", machine_maintenance_oh_per_cell)
        add_s("Depreciation", depreciation_per_cell)

        sens_df = pd.DataFrame(sens_items)
        sens_df["Impact"] = sens_df["High"] - sens_df["Low"]
        sens_df = sens_df.sort_values("Impact", ascending=False)

        price_per_cell = unit_cost_build * (1.0 + max(e.desired_margin, 0.0))
        price_per_kwh = price_per_cell / cell_kwh

        # ------------------------------------------------------------------
        # 11) Project timeline & cashflows
        # ------------------------------------------------------------------
        years = list(range(0, e.project_years + 1))
        capex_total = total_capital_equipment + building_value

        capex_outflow_per_year = [0.0] * (e.project_years + 1)
        for y in range(min(e.construction_years, e.project_years)):
            capex_outflow_per_year[y] = capex_total / max(e.construction_years, 1)

        prod_cells_per_year = [0.0] * (e.project_years + 1)
        start_prod_year = e.construction_years

        output_list = list(getattr(e, "ramp_output_rates", []) or [])

        def _clamp_output(v: float) -> float:
            return max(min(float(v), 1.0), 0.0)

        def _output_for_year(idx: int) -> float:
            if e.ramp_years and idx >= e.ramp_years:
                return 1.0
            if not output_list:
                return 1.0
            if idx < len(output_list):
                return _clamp_output(output_list[idx])
            return _clamp_output(output_list[-1])



        scrap_list = list(getattr(e, "ramp_scrap_rates", []) or [])

        def _clamp_scrap(v: float) -> float:
            return max(min(float(v), 0.999), 0.0)

        def _scrap_for_year(idx: int) -> float:
            if e.ramp_years and idx >= e.ramp_years:
                return 0.0
            if not scrap_list:
                return 0.0
            if idx < len(scrap_list):
                return _clamp_scrap(scrap_list[idx])
            return _clamp_scrap(scrap_list[-1])

        prod_cells_per_year = [0.0] * (e.project_years + 1)
        good_cells_per_year = [0.0] * (e.project_years + 1)

        start_prod_year = e.construction_years
        for y in range(start_prod_year, e.project_years + 1):
            t = y - start_prod_year  # ramp year index
            output_mult = _output_for_year(t)
            scrap = _scrap_for_year(t)
            prod_cells_per_year[y] = final_cells_required * output_mult
            good_cells_per_year[y] = prod_cells_per_year[y] * (1.0 - scrap)

        # Opex excludes depreciation; overheads are already included above
        opex_per_cell = unit_cost_build - depreciation_per_cell

        cashflows: List[Dict[str, Any]] = []
        cum_cash: List[float] = []
        cf_cum = 0.0

        for y in years:
            revenue = good_cells_per_year[y] * price_per_cell  # revenue only from good cells
            opex = prod_cells_per_year[y] * opex_per_cell  # costs at steady-state volume (incl. scrap)
            deprec = (
                annual_depr_building + annual_depr_equipment
                if y >= start_prod_year
                else 0.0
            )
            capex_y = capex_outflow_per_year[y]
            ebit = revenue - opex - deprec
            tax = max(ebit, 0.0) * e.tax_rate  # Taxes only when EBIT > 0. No loss carry-forwards modeled.
            nopat = ebit - tax
            fcf = nopat + deprec - capex_y
            disc = (1.0 + e.capital_cost_wacc) ** y
            npv_y = fcf / disc

            cashflows.append(
                {
                    "year": y,
                    "revenue": revenue,
                    "opex": opex,
                    "depr": deprec,
                    "tax": tax,
                    "capex": capex_y,
                    "fcf": fcf,
                    "npv": npv_y,
                    "cum_cash": cf_cum + fcf,  # cumulative free cash flow
                }
            )
            cf_cum += fcf
            cum_cash.append(cf_cum)

        npv_total = sum(c["npv"] for c in cashflows)
        breakeven_year = next(
            (c["year"] for c, cum in zip(cashflows, cum_cash) if cum >= 0.0),
            None,
        )

        # ------------------------------------------------------------------
        # 12) KPIs & output bundle
        # ------------------------------------------------------------------
        kpis = {
            "materials_procurement_per_cell_eur": materials_procurement_per_cell,
            "process_energy_per_cell_eur": process_energy_per_cell,
            "building_opex_per_cell_eur": building_opex_per_cell,
            "energy_per_cell_eur": energy_per_cell,
            "spec_labor_per_cell_eur": spec_labor_per_cell,
            "supp_labor_per_cell_eur": supp_labor_per_cell,
            "direct_labor_per_cell_eur": labor_per_cell,
            "indirect_personnel_oh_per_cell_eur": indirect_personnel_oh_per_cell,
            "logistics_personnel_oh_per_cell_eur": logistics_personnel_oh_per_cell,
            "building_maintenance_oh_per_cell_eur": building_maintenance_oh_per_cell,
            "machine_maintenance_oh_per_cell_eur": machine_maintenance_oh_per_cell,
            "depreciation_per_cell_eur": depreciation_per_cell,
            "unit_cost_build_eur_per_cell": unit_cost_build,
            "cost_build_per_kwh_eur": cost_build_per_kwh,
            "price_per_cell_eur": price_per_cell,
            "price_per_kwh_eur": price_per_kwh,
            "cell_kwh": cell_kwh,
            "cell_wh": cell_wh,
            "cell_ah": g.cell_ah,
            "cell_voltage": g.cell_voltage,
            "final_cells_required": final_cells_required,
            "available_time_seconds": avail_time_seconds,
            "line_cycle_time_s": line_cycle_time_s,
            "cpm": cpm,
            "line_capacity_cells": line_capacity_cells,
            "actual_cells": actual_cells_for_cost,
            "bottleneck_step": df.loc[df["step_id"] == bottleneck_sid, "step"].iloc[0],
            "utilization": utilization,
            "capital_equipment_base_eur": total_capital_equipment_base,
            "logistics_capex_eur": logistics_capex,
            "indirect_capex_eur": indirect_capex,
            "capital_equipment_total_eur": total_capital_equipment,
            "indoor_area_none_m2": indoor_none,
            "indoor_area_clean_m2": indoor_clean,
            "indoor_area_dry_m2": indoor_dry,
            "required_outdoor_area_m2": required_outdoor_area,
            "required_indoor_area_m2": required_indoor_area,
            "total_required_area_m2": total_required_area,
            "area_cost_indoor_none_eur": cost_indoor_none,
            "area_cost_indoor_clean_eur": cost_indoor_clean,
            "area_cost_indoor_dry_eur": cost_indoor_dry,
            "area_cost_outdoor_eur": cost_outdoor,
            "building_value_eur": building_value,
            "annual_depreciation_eur": annual_depreciation,
            "npv_total_eur": npv_total,
            "breakeven_year": breakeven_year,
        }

        cash_df = pd.DataFrame(cashflows)

        dag_totals = {
            "graph": graph,
            "flow_in_unit": flow_in_unit,
            "good_out_unit": good_out_unit,
            "scrap_unit": scrap_unit,
            "edge_good_unit": edge_good_unit,
            "root_input_per_year": root_input_per_year,
            "final_good_unit": final_good_unit,
        }

        return {
            "kpis": kpis,
            "steps": df,
            "materials": mats,
            "cash": cash_df,
            "sens": sens_df,
            "dag": dag_totals,
        }

    # ========================================================
    # Visualization helpers (figures + Sankey + table view)
    # ========================================================
    @staticmethod
    def figs(
        df_steps: pd.DataFrame,
        mats: pd.DataFrame,
        kpis: Dict[str, Any],
        cash_df: pd.DataFrame,
        sens_df: pd.DataFrame,
    ) -> Tuple[go.Figure, go.Figure, go.Figure, go.Figure, go.Figure, go.Figure, go.Figure, go.Figure]:


        """
        Build the standard dashboard figures from model outputs.
        Kept intentionally lightweight; everything derived from kpis + DataFrames.
        """
        final_cells = float(kpis["final_cells_required"])

        # --- Annual steady-state cost (M€/yr) --------------------------------
        annual_costs = {
            "Materials": kpis["materials_procurement_per_cell_eur"] * final_cells,
            "Energy": kpis["energy_per_cell_eur"] * final_cells,
            "Direct labor": kpis["direct_labor_per_cell_eur"] * final_cells,
            "Indirect personnel OH": kpis["indirect_personnel_oh_per_cell_eur"] * final_cells,
            "Logistics personnel OH": kpis["logistics_personnel_oh_per_cell_eur"] * final_cells,
            "Building maintenance": kpis["building_maintenance_oh_per_cell_eur"] * final_cells,
            "Machine maintenance": kpis["machine_maintenance_oh_per_cell_eur"] * final_cells,
            "Depreciation": kpis["depreciation_per_cell_eur"] * final_cells,
        }
        fig_annual = px.bar(
            x=list(annual_costs.keys()),
            y=[v / 1e6 for v in annual_costs.values()],
            labels={"x": "Component", "y": "Annual cost (M€)"},
        )
        fig_annual.update_traces(marker_color="#00549f")
        fig_annual.update_layout(margin=dict(l=40, r=10, t=30, b=40))

        # --- Single-cell cost breakdown (€/cell) ------------------------------
        cell_costs = {
            "Materials": kpis["materials_procurement_per_cell_eur"],
            "Energy": kpis["building_opex_per_cell_eur"],
            "Direct labor": kpis["direct_labor_per_cell_eur"],
            "Indirect personnel OH": kpis["indirect_personnel_oh_per_cell_eur"],
            "Logistics personnel OH": kpis["logistics_personnel_oh_per_cell_eur"],
            "Building maintenance": kpis["building_maintenance_oh_per_cell_eur"],
            "Machine maintenance": kpis["machine_maintenance_oh_per_cell_eur"],
            "Depreciation": kpis["depreciation_per_cell_eur"],
        }
        fig_cell = px.bar(
            x=list(cell_costs.keys()),
            y=list(cell_costs.values()),
            labels={"x": "Component", "y": "Cost (€/cell)"},
        )
        fig_cell.update_traces(marker_color="#00549f")
        fig_cell.update_layout(margin=dict(l=40, r=10, t=30, b=40))

        # --- Step capacity vs. final target ----------------------------------
        df_cap = df_steps[["step", "capacity_cells_per_year", "demand_cells_step_per_year"]].copy()
        df_cap["step"] = df_cap["step"].astype(str)
        fig_cap = px.bar(
            df_cap,
            x="step",
            y="capacity_cells_per_year",
            labels={"step": "Process step", "capacity_cells_per_year": "Capacity (cells/yr)"},
        )
        fig_cap.update_traces(marker_color="#00549f")

        fig_cap.add_trace(
            go.Scatter(
                x=df_cap["step"],
                y=df_cap["demand_cells_step_per_year"],
                mode="lines+markers",
                name="Step demand",
                line={"color": "#d62728"},
            )
        )

        fig_cap.update_layout(margin=dict(l=40, r=10, t=30, b=80))

        fig_utilization = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=float(kpis.get("utilization", 0.0)) * 100,  # % utilization
                number={"suffix": "%"},
                gauge={"axis": {"range": [0, max(100, float(kpis.get('utilization', 0.0)) * 120)]},
                       "steps": [
                           {"range": [0, 80], "color": "#f3cdbb"},
                           {"range": [80, 100], "color": "#ddebce"}],
                       "bar": {"color": "#00549f"},},
                title={"text": "Production Utilization"},
            )
        )


        # --- Materials procurement breakdown (€/cell) ------------------------
        if "procurement_cost_per_cell_eur" in mats.columns:
            fig_mat = px.bar(
                mats,
                x="name",
                y="procurement_cost_per_cell_eur",
                labels={"name": "Material", "procurement_cost_per_cell_eur": "Cost (€/cell)"},

            )
            fig_mat.update_traces(marker_color="#00549f")

        else:
            fig_mat = go.Figure()
        fig_mat.update_layout(margin=dict(l=40, r=10, t=30, b=80))

        # --- Tornado sensitivity (impact in €/cell) ---------------------------
        if not sens_df.empty:
            fig_sens = px.bar(
                sens_df,
                x="Impact",
                y="Parameter",
                orientation="h",
                labels={"Impact": "Δ cost (€/cell) at ±25%", "Parameter": ""},

            )
            fig_sens.update_layout(margin=dict(l=80, r=10, t=30, b=40))
            fig_sens.update_traces(marker_color="#00549f")
        else:
            fig_sens = go.Figure()

        # --- Project timeline: costs (bars) & revenue (line) -----------------
        fig_time = go.Figure()
        if not cash_df.empty:
            years = cash_df["year"]
            fig_time.add_bar(
                x=years,
                y=-(cash_df["opex"] / 1e6),
                name="Opex (M€)",
                marker_color="#e69679",
            )
            if "capex" in cash_df.columns:
                fig_time.add_bar(
                    x=years,
                    y=-(cash_df["capex"] / 1e6),
                    name="Capex (M€)",
                    marker_color="#cd8b87",
                )
            fig_time.add_bar(
                x=years,
                y=(cash_df["revenue"] / 1e6),
                name="Revenue (M€)",
                marker_color="#b8d698",
            )
            fig_time.add_scatter(
                x=years,
                y=(cash_df["cum_cash"] / 1e6),
                mode="lines",
                name="Cumulative Cashflow",
                line=dict(color="#00549f"),
            )
            fig_time.update_layout(
                barmode="relative",
                xaxis_title="Year",
                yaxis_title="M€",
                margin=dict(l=50, r=10, t=30, b=40),
            )


        # --- Footprint treemap (m²) --------------------------------------------
        fig_footprint = BatteryCostModel.footprint(kpis)


        return fig_annual, fig_cell, fig_cap, fig_utilization, fig_mat, fig_sens, fig_time, fig_footprint

    @staticmethod
    def footprint(kpis: Dict[str, Any]) -> go.Figure:
        labels = [
            "Total Footprint", "Indoor", "Clean Room", "Dry Room", "Standard", "Outdoor"
        ]
        parents = ["", "Total Footprint", "Indoor", "Indoor", "Indoor", "Total Footprint"]
        values = [
            kpis.get("total_required_area_m2", 0.0),
            kpis.get("required_indoor_area_m2", 0.0),
            kpis.get("indoor_area_clean_m2", 0.0),
            kpis.get("indoor_area_dry_m2", 0.0),
            kpis.get("indoor_area_none_m2", 0.0),
            kpis.get("required_outdoor_area_m2", 0.0),
        ]
        fig = px.treemap(
            branchvalues="total",
            names=labels,
            parents=parents,
            values=values,
            color=labels,
            color_discrete_map={'Total Footprint':'#e8f1fa', 'Indoor':'#8ebae5', 'Clean Room':'#00549f', 'Dry Room':'#00549f', 'Standard':'#407fb7', 'Outdoor':'#c7ddf2'},
            labels={"values": "m²"},
        )
        fig.update_traces(marker=dict(cornerradius=5))


        return fig

    @staticmethod
    def sankey(
        df_steps: pd.DataFrame,
        dag_totals: Dict[str, Any],
        scale: float = 1e6,
    ) -> go.Figure:
        """
        Build a Sankey diagram of good flow and scrap based on the DAG totals.

        scale: unit scaling (default 1e6 => values in million cells/year)

        The diagram aggregates all sink steps into a *single* "Final good output" node,
        so the visual end point is unique even if the process has multiple sinks.
        """
        graph = dag_totals["graph"]
        good_out_unit = dag_totals["good_out_unit"]
        scrap_unit = dag_totals["scrap_unit"]
        edge_good_unit = dag_totals["edge_good_unit"]
        root_input_per_year = dag_totals["root_input_per_year"]

        ids = graph["ids"]

        labels: List[str] = []
        node_index: Dict[Tuple[str, str], int] = {}

        # Step nodes
        for sid in ids:
            row = df_steps.loc[df_steps["step_id"].astype(str) == sid].iloc[0]
            label = str(row.get("step", sid))
            node_index[("step", sid)] = len(labels)
            labels.append(label)

        # Scrap nodes
        for sid in ids:
            node_index[("scrap", sid)] = len(labels)
            labels.append(f"Scrap @ {sid}")

        # Final-good node (single endpoint)
        final_node_idx = len(labels)
        labels.append("Final good output")

        sources: List[int] = []
        targets: List[int] = []
        values: List[float] = []

        # Good flow along edges
        for (src, dst), val_unit in edge_good_unit.items():
            val = val_unit * root_input_per_year / max(scale, 1.0)
            if val <= 0:
                continue
            s_idx = node_index[("step", src)]
            t_idx = node_index[("step", dst)]
            sources.append(s_idx)
            targets.append(t_idx)
            values.append(val)

        # Scrap edges
        for sid in ids:
            val_unit = scrap_unit[sid]
            val = val_unit * root_input_per_year / max(scale, 1.0)
            if val <= 0:
                continue
            s_idx = node_index[("step", sid)]
            t_idx = node_index[("scrap", sid)]
            sources.append(s_idx)
            targets.append(t_idx)
            values.append(val)

        # Final-good edges from sink steps
        for sid in graph["sinks"]:
            good_val_unit = good_out_unit[sid]
            val = good_val_unit * root_input_per_year / max(scale, 1.0)
            if val <= 0:
                continue
            s_idx = node_index[("step", sid)]
            t_idx = final_node_idx
            sources.append(s_idx)
            targets.append(t_idx)
            values.append(val)

        fig = go.Figure(
            data=[
                go.Sankey(
                    node=dict(
                        pad=15,
                        thickness=20,
                        line=dict(width=0.5),
                        label=labels,
                        color="#00549f",
                    ),
                    link=dict(
                        source=sources,
                        target=targets,
                        value=values,
                    ),
                )
            ]
        )
        fig.update_layout(
            title_text="Process flow and scrap (million cells / year)",
            font_size=10,
        )
        return fig

    @staticmethod
    def table_view(df: pd.DataFrame):
        """
        Build read-only step summary table for the right-hand panel.
        Only lead time, scrap & derived fields are shown, plus successor.
        """
        df = df.copy()
        if "order" not in df.columns:
            df["order"] = range(1, len(df) + 1)
        if "successor_step" not in df.columns:
            df["successor_step"] = ""
        df = df.sort_values("order")

        show_cols = [
            "order",
            "step",
            "successor_step",
            "env",
            "machines_override",
            "machines",
            "lead_time_s",
            "scrap_rate",
            "capex_meur_per_machine",
            "footprint_m2",
            "kw_per_unit",
            "spec_workers_per_machine",
            "supp_workers_per_machine",
            "required_cycle_time_s",
            "demand_cells_step_per_year",
            "capacity_cells_per_year",
            "materials_cost_per_cell_total_eur",
            "energy_cost_per_cell_eur",
        ]

        for c in [
            "required_cycle_time_s",
            "demand_cells_step_per_year",
            "capacity_cells_per_year",
            "materials_cost_per_cell_total_eur",
            "energy_cost_per_cell_eur",
            "machines",
        ]:
            if c not in df.columns:
                df[c] = 0.0

        round_map = {
            "order": 0,
            "lead_time_s": 3,
            "scrap_rate": 4,
            "machines_override": 0,
            "machines": 0,
            "capex_meur_per_machine": 3,
            "footprint_m2": 0,
            "kw_per_unit": 3,
            "required_cycle_time_s": 3,
            "demand_cells_step_per_year": 0,
            "capacity_cells_per_year": 0,
            "materials_cost_per_cell_total_eur": 5,
            "energy_cost_per_cell_eur": 5,
            "spec_workers_per_machine": 2,
            "supp_workers_per_machine": 2,
        }

        df_show = df[show_cols].round(round_map)
        columns = [
            {"name": "Order", "id": "order", "type": "numeric"},
            {"name": "Process Step", "id": "step", "presentation": "input"},
            {"name": "Successor Step", "id": "successor_step"},
            {"name": "Environment", "id": "env", "presentation": "dropdown"},
            {"name": "Machines (override)", "id": "machines_override"},
            {"name": "Machines (calc)", "id": "machines"},
            {"name": "Lead Time (s/unit)", "id": "lead_time_s", "type": "numeric"},
            {"name": "Scrap Rate (0–1)", "id": "scrap_rate", "type": "numeric"},
            {"name": "CAPEX (M€ / machine)", "id": "capex_meur_per_machine", "type": "numeric"},
            {"name": "Footprint (m² / machine)", "id": "footprint_m2", "type": "numeric"},
            {"name": "Energy (kW / machine)", "id": "kw_per_unit", "type": "numeric"},
            {"name": "Specialists / machine", "id": "spec_workers_per_machine", "type": "numeric"},
            {"name": "Support / machine", "id": "supp_workers_per_machine", "type": "numeric"},
            {"name": "Req. Cycle (s)", "id": "required_cycle_time_s"},
            {"name": "Demand at Step (cells/yr)", "id": "demand_cells_step_per_year"},
            {"name": "Capacity (cells/yr)", "id": "capacity_cells_per_year"},
            {"name": "Intro-step Materials €/Cell", "id": "materials_cost_per_cell_total_eur"},
            {"name": "Energy €/Cell", "id": "energy_cost_per_cell_eur"},
        ]

        return columns, df_show.to_dict("records")


# ============================================================
# UI helpers
# ============================================================

app = Dash(__name__)


server = app.server
app.title = "Cell Cost Estimator"
available_presets_init = all_presets()


def logo_src() -> str:
    """Return logo path or fallback SVG data URI."""
    if os.path.exists(os.path.join("assets", "logo.png")):
        return "/assets/logo.png"
    if os.path.exists(os.path.join("assets", "logo.svg")):
        return "/assets/logo.svg"

    svg = """
    <svg xmlns='http://www.w3.org/2000/svg' width='200' height='44'>
        <rect width='100%' height='100%' fill='#2563eb'/>
        <text x='50%' y='50%' dominant-baseline='middle' text-anchor='middle'
              font-family='Inter, Arial, sans-serif' font-size='16' fill='white'>
            PEM RWTH
        </text>
    </svg>
    """
    return "data:image/svg+xml;utf8," + quote(svg)


def num_input(
    id_obj: str,
    value: Any,
    step: Any = "any",
    min_: float | None = None,
    max_: float | None = None,
    width: str = "90%",
):
    """Small helper to keep numeric input definitions compact."""
    return dcc.Input(
        id=id_obj,
        type="number",
        value=value,
        step=step,
        min=min_,
        max=max_,
        style={"width": width},
    )


def build_materials_columns(steps_rows):
    """Build materials table column definitions & dropdowns."""
    step_labels = [r["step"] for r in (steps_rows or []) if r.get("step")]
    options = (
        [{"label": s, "value": s} for s in step_labels]
        or [{"label": DEFAULT_STEPS[0]["step"], "value": DEFAULT_STEPS[0]["step"]}]
    )

    columns = [
        {"name": "Name", "id": "name", "presentation": "input"},
        {"name": "Intro Step", "id": "intro_step", "presentation": "dropdown"},
        {"name": "Pricing Unit", "id": "pricing_unit", "presentation": "dropdown"},
        {"name": "g/cell", "id": "g_per_cell", "type": "numeric", "format": {"specifier": ".3f"}},
        {"name": "€/kg", "id": "eur_per_kg", "type": "numeric", "format": {"specifier": ".3f"}},
        {"name": "Area per cell (m²)", "id": "area_per_cell_m2", "type": "numeric", "format": {"specifier": ".4f"}},
        {"name": "€/m²", "id": "eur_per_m2", "type": "numeric", "format": {"specifier": ".3f"}},
    ]
    dropdown = {
        "intro_step": {"options": options},
        "pricing_unit": {
            "options": [
                {"label": "€/kg", "value": "kg"},
                {"label": "€/m²", "value": "m2"},
            ]
        },
    }
    return columns, dropdown


_mat_columns_init, _mat_dropdown_init = build_materials_columns(DEFAULT_STEPS)

ENV_DROPDOWN_OPTIONS = [
    {"label": "None", "value": "none"},
    {"label": "Clean", "value": "clean"},
    {"label": "Dry", "value": "dry"},
]


# ============================================================
# Layout
# ============================================================

preset_controls = html.Div(
    [
        html.H3("Presets"),
        html.Div(
            [
                dcc.Dropdown(
                    id="preset_select",
                    options=[{"label": k, "value": k} for k in available_presets_init.keys()],
                    value=DEFAULT_PRESET_KEY,
                    clearable=False,
                    style={"width": "100%"},
                ),
                html.Button(
                    "Apply preset",
                    id="apply_preset",
                    n_clicks=0,
                    style={
                        "marginTop": "6px",
                        "width": "100%",
                        "padding": "8px",
                        "backgroundColor": "#111827",
                        "color": "white",
                        "border": "none",
                        "borderRadius": "6px",
                        "cursor": "pointer",
                    },
                ),
                html.Button(
                    "Export current as JSON",
                    id="export_preset",
                    n_clicks=0,
                    style={
                        "marginTop": "6px",
                        "width": "100%",
                        "padding": "8px",
                        "backgroundColor": "#00549f",
                        "color": "white",
                        "border": "none",
                        "borderRadius": "6px",
                        "cursor": "pointer",
                    },
                ),
                dcc.Upload(
                    id="preset_upload",
                    children=html.Div("Import preset JSON (drop or click)", style={"textAlign": "center"}),
                    style={
                        "marginTop": "6px",
                        "padding": "10px",
                        "border": "1px dashed #ccc",
                        "borderRadius": "6px",
                        "cursor": "pointer",
                    },
                    multiple=False,
                ),
                dcc.Download(id="preset_download"),
            ]
        ),
        html.Div(
            id="preset_note",
            style={"fontSize": "12px", "color": "#555", "marginTop": "6px"},
        ),
    ],
    style={"marginBottom": "8px"},
)

left_inputs = html.Div(
    [
        preset_controls,
        html.H3("General assumptions"),
        html.Div(
            [
                html.Div(
                    [
                        html.Label("Energy Price (€/kWh)"),
                        num_input(
                            "el_price",
                            DEFAULT_GENERAL["electricity_price_eur_per_kwh"],
                            step="any",
                            min_=0,
                        ),
                    ]
                ),
                html.Div(
                    [
                        html.Label("Working Days per year"),
                        num_input(
                            "days",
                            DEFAULT_GENERAL["working_days"],
                            step=0.01,
                            min_=50,
                        ),
                    ]
                ),
                html.Div(
                    [
                        html.Label("Working shifts per day"),
                        num_input(
                            "shifts",
                            DEFAULT_GENERAL["shifts_per_day"],
                            step=0.01,
                            min_=1,
                        ),
                    ]
                ),
                html.Div(
                    [
                        html.Label("Working hours per shift"),
                        num_input(
                            "hshift",
                            DEFAULT_GENERAL["shift_hours"],
                            step=0.01,
                            min_=0.5,
                        ),
                    ]
                ),
                html.Div(
                    [
                        html.Label("Production Availability (0–1)"),
                        num_input(
                            "avail",
                            DEFAULT_GENERAL["avail"],
                            step="any",
                            min_=0,
                            max_=1,
                        ),
                    ]
                ),
                html.Div(
                    [
                        html.Label("Annual Output (GWh)"),
                        num_input(
                            "gwh",
                            DEFAULT_GENERAL["annual_output_gwh"],
                            step="any",
                            min_=0.1,
                        ),
                    ]
                ),

            ],
            style={
                "display": "grid",
                "gridTemplateColumns": "repeat(2, minmax(220px, 1fr))",
                "gap": "8px",
            },
        ),
        html.Hr(),
        html.H3("Cell parameters"),
        html.Div(
            [
                html.Div(
                    [
                        html.Label("Single Cell Capacity (Ah)"),
                        num_input(
                            "cell_ah",
                            DEFAULT_GENERAL["cell_ah"],
                            step="any",
                            min_=0.001,
                        ),
                    ]
                ),
                html.Div(
                    [
                        html.Label("Single Cell Energy (Wh)"),
                        num_input(
                            "cell_wh",
                            DEFAULT_GENERAL["cell_wh"],
                            step="any",
                            min_=0.001,
                        ),
                    ]
                ),
                html.Div(
                    [
                        html.Label("Nominal Voltage (V)"),
                        num_input(
                            "cell_v",
                            DEFAULT_GENERAL["cell_voltage"],
                            step="any",
                            min_=0.5,
                        ),
                    ]
                ),
            ],
            style={
                "display": "grid",
                "gridTemplateColumns": "repeat(2, minmax(220px, 1fr))",
                "gap": "8px",
            },
        ),
        html.Hr(),
        html.H3("Building cost & area factors"),
        html.Div(
            [
                html.Div(
                    [
                        html.Label("Building CAPEX (€/m²)"),
                        num_input(
                            "bldg_cost",
                            DEFAULT_GENERAL["building_cost_eur_per_m2"],
                            step="any",
                            min_=0,
                        ),
                    ]
                ),
                html.Div(
                    [
                        html.Label("Building Baseline OPEX (kWh/m²/a)"),
                        num_input(
                            "building_baseline_kwh",
                            DEFAULT_GENERAL["baseline_building_kwh"],
                            step="any",
                            min_=0,
                        ),
                    ]
                ),
                html.Div(
                    [
                        html.Label("Indoor area factor (×)"),
                        num_input(
                            "indoor_factor",
                            DEFAULT_GENERAL["indoor_area_factor"],
                            step="any",
                            min_=0.1,
                        ),
                    ]
                ),
                html.Div(
                    [
                        html.Label("Outdoor area factor (× of indoor)"),
                        num_input(
                            "outdoor_factor",
                            DEFAULT_GENERAL["outdoor_area_factor"],
                            step="any",
                            min_=0.0,
                        ),
                    ]
                ),
                html.Div(
                    [
                        html.Label("Outdoor area CAPEX (€/m²)"),
                        num_input(
                            "outdoor_cost",
                            DEFAULT_GENERAL["outdoor_area_cost"],
                            step="any",
                            min_=0.0,
                        ),
                    ]
                ),
            ],
            style={
                "display": "grid",
                "gridTemplateColumns": "repeat(2, minmax(220px, 1fr))",
                "gap": "8px",
            },
        ),
        html.Div(
            [
                html.Div(
                    [
                        html.Label("Clean room area multiplier (×)"),
                        num_input(
                            "clean_mult",
                            DEFAULT_GENERAL["clean_area_multiplier"],
                            step="any",
                            min_=1.0,
                        ),
                    ]
                ),
                html.Div(
                    [
                        html.Label("Clean room CAPEX (€/m²)"),
                        num_input(
                            "clean_capex",
                            DEFAULT_GENERAL["clean_capex_eur_per_m2"],
                            step="any",
                            min_=0.0,
                        ),
                    ]
                ),
                html.Div(
                    [
                        html.Label("Clean room OPEX (€/h/m²)"),
                        num_input(
                            "clean_opex",
                            DEFAULT_GENERAL["clean_opex_eur_per_h_per_m2"],
                            step="any",
                            min_=0.0,
                        ),
                    ]
                ),
                html.Div(
                    [
                        html.Label("Dry room area multiplier (×)"),
                        num_input(
                            "dry_mult",
                            DEFAULT_GENERAL["dry_area_multiplier"],
                            step="any",
                            min_=1.0,
                        ),
                    ]
                ),
                html.Div(
                    [
                        html.Label("Dry room CAPEX (€/m²)"),
                        num_input(
                            "dry_capex",
                            DEFAULT_GENERAL["dry_capex_eur_per_m2"],
                            step="any",
                            min_=0.0,
                        ),
                    ]
                ),
                html.Div(
                    [
                        html.Label("Dry room OPEX (€/h/m²)"),
                        num_input(
                            "dry_opex",
                            DEFAULT_GENERAL["dry_opex_eur_per_h_per_m2"],
                            step="any",
                            min_=0.0,
                        ),
                    ]
                ),
            ],
            style={
                "display": "grid",
                "gridTemplateColumns": "repeat(2, minmax(220px, 1fr))",
                "gap": "8px",
            },
        ),
        html.Hr(),
        html.H3("Labour"),
        html.Div(
            [
                html.Div(
                    [
                        html.Label("Specialist Labor (€/h)"),
                        num_input(
                            "labor_spec",
                            DEFAULT_GENERAL["specialist_labor_eur_per_h"],
                            step="any",
                            min_=0,
                        ),
                    ]
                ),
                html.Div(
                    [
                        html.Label("Support Labor (€/h)"),
                        num_input(
                            "labor_sup",
                            DEFAULT_GENERAL["support_labor_eur_per_h"],
                            step="any",
                            min_=0,
                        ),
                    ]
                ),
            ],
            style={
                "display": "grid",
                "gridTemplateColumns": "repeat(2, minmax(220px, 1fr))",
                "gap": "8px",
            },
        ),
        html.Hr(),
        html.H3("Economics"),
        html.Div(
            [
                html.Div(
                    [
                        html.Label("Project Duration (years)"),
                        num_input(
                            "proj_years",
                            DEFAULT_ECON["project_years"],
                            step=1,
                            min_=1,
                        ),
                    ]
                ),
                html.Div(
                    [
                        html.Label("Overall Tax Rate (0–1)"),
                        num_input(
                            "tax",
                            DEFAULT_ECON["tax_rate"],
                            step="any",
                            min_=0,
                            max_=1,
                        ),
                    ]
                ),
                html.Div(
                    [
                        html.Label("Depreciation (equip) years"),
                        num_input(
                            "dep_equip",
                            DEFAULT_ECON["depreciation_years_equipment"],
                            step=1,
                            min_=1,
                        ),
                    ]
                ),
                html.Div(
                    [
                        html.Label("Depreciation (building) years"),
                        num_input(
                            "dep_bldg",
                            DEFAULT_ECON["depreciation_years_building"],
                            step=1,
                            min_=1,
                        ),
                    ]
                ),
                html.Div(
                    [
                        html.Label("Construction Duration (years)"),
                        num_input(
                            "build_years",
                            DEFAULT_ECON["construction_years"],
                            step=1,
                            min_=0,
                        ),
                    ]
                ),
                html.Div(
                    [
                        html.Button("Add ramp year",
                                    id="add_ramp_year",
                                    n_clicks=0,
                                    style={"display": "block",
                                           "backgroundColor": "#00549f",
                                           "color": "white",
                                           "borderRadius": "2px",
                                           "width": "100%",
                                           "fontSize": "14px",
                                           "height": "20px",
                                           }),
                    ]
                ),
                html.Div(id="ramp_scrap_rows",
                         style={"display": "grid", "gap": "8px", "gridColumn": "1 / -1"}
                         ),
                html.Div(
                    [
                        html.Label("Capital Cost / WACC (0–1)"),
                        num_input(
                            "wacc",
                            DEFAULT_ECON["capital_cost_wacc"],
                            step="any",
                            min_=0.0,
                            max_=1.0,
                        ),
                    ]
                ),
                html.Div(
                    [
                        html.Label("Desired Margin (0–1)"),
                        num_input(
                            "margin",
                            DEFAULT_ECON["desired_margin"],
                            step="any",
                            min_=0.0,
                            max_=1.0,
                        ),
                    ]
                ),
            ],
            style={
                "display": "grid",
                "gridTemplateColumns": "repeat(2, minmax(220px, 1fr))",
                "gap": "8px",
            },
        ),

        html.Hr(),
        html.H3("Overhead & Investment Factors"),
        html.Div(
            [
                html.Div(
                    [
                        html.Label("Indirect personnel OH on labor"),
                        num_input(
                            "oh_indirect_personnel",
                            DEFAULT_ECON["indirect_personnel_factor"],
                            step="any",
                            min_=0,
                            max_=5.0,
                        ),
                    ]
                ),
                html.Div(
                    [
                        html.Label("Logistics personnel OH on labor"),
                        num_input(
                            "oh_logistics_personnel",
                            DEFAULT_ECON["logistics_personnel_factor"],
                            step="any",
                            min_=0,
                            max_=5.0,
                        ),
                    ]
                ),
                html.Div(
                    [
                        html.Label("Building maintenance"),
                        num_input(
                            "oh_building_maintenance",
                            DEFAULT_ECON["building_maintenance_factor"],
                            step="any",
                            min_=0,
                            max_=1.0,
                        ),
                    ]
                ),
                html.Div(
                    [
                        html.Label("Machine maintenance"),
                        num_input(
                            "oh_machine_maintenance",
                            DEFAULT_ECON["machine_maintenance_factor"],
                            step="any",
                            min_=0,
                            max_=1.0,
                        ),
                    ]
                ),
                html.Div(
                    [
                        html.Label("Logistics investment factor"),
                        num_input(
                            "inv_logistics_factor",
                            DEFAULT_ECON["logistics_investment_factor"],
                            step="any",
                            min_=0,
                            max_=2.0,
                        ),
                    ]
                ),
                html.Div(
                    [
                        html.Label("Indirect investment factor"),
                        num_input(
                            "inv_indirect_factor",
                            DEFAULT_ECON["indirect_investment_factor"],
                            step="any",
                            min_=0,
                            max_=2.0,
                        ),
                    ]
                ),
            ],
            style={
                "display": "grid",
                "gridTemplateColumns": "repeat(2, minmax(220px, 1fr))",
                "gap": "8px",
            },
        ),
        html.Hr(),
        html.H3("Raw Materials"),
        dash_table.DataTable(
            id="materials_table",
            columns=_mat_columns_init,
            data=DEFAULT_RAW_MATERIALS,
            editable=True,
            dropdown=_mat_dropdown_init,
            row_deletable=True,
            style_table={
                "overflowX": "auto",
                "maxHeight": "400px",
                "overflowY": "auto",
            },
            style_cell={"padding": "6px", "minWidth": 120, "whiteSpace": "normal"},
            page_action="none",
        ),
        html.Button(
            "Add material",
            id="add_mat",
            n_clicks=0,
            style={"marginTop": "8px"},
        ),
        html.Hr(),
        html.H3("Process Steps"),
        html.Div(
            [
                html.Button(
                    "Move Up",
                    id="move_up",
                    n_clicks=0,
                    style={"marginRight": "6px"},
                ),
                html.Button("Move Down", id="move_down", n_clicks=0),
            ],
            style={"marginBottom": "6px"},
        ),
        dash_table.DataTable(
            id="steps_table",
            columns=[
                {"name": "Order", "id": "order", "type": "numeric"},
                {"name": "Process Step", "id": "step", "presentation": "input"},
                {"name": "Successor Step", "id": "successor_step", "presentation": "dropdown"},
                {"name": "Environment (None/Clean/Dry)", "id": "env", "presentation": "dropdown"},
                {"name": "Machines (override, optional)", "id": "machines_override", "type": "numeric"},
                {"name": "Lead Time (s/unit)", "id": "lead_time_s", "type": "numeric"},
                {"name": "Scrap Rate (0–1)", "id": "scrap_rate", "type": "numeric"},
                {"name": "CAPEX (M€ / machine)", "id": "capex_meur_per_machine", "type": "numeric"},
                {"name": "Footprint (m² per machine)", "id": "footprint_m2", "type": "numeric"},
                {"name": "Energy Consumption (kW per machine)", "id": "kw_per_unit", "type": "numeric"},
                {"name": "Specialists / machine", "id": "spec_workers_per_machine", "type": "numeric"},
                {"name": "Support / machine", "id": "supp_workers_per_machine", "type": "numeric"},
            ],
            data=DEFAULT_STEPS,
            editable=True,
            dropdown={
                "env": {
                    "options": ENV_DROPDOWN_OPTIONS,
                },
                "successor_step": {
                    "options": [{"label": "End (no successor)", "value": ""}]
                    + [{"label": s["step"], "value": s["step"]} for s in DEFAULT_STEPS],
                },
            },
            row_deletable=True,
            row_selectable="single",
            style_table={
                "overflowX": "auto",
                "maxHeight": "400px",
                "overflowY": "auto",
            },
            style_cell={"padding": "6px", "minWidth": 120, "whiteSpace": "normal"},
            page_action="none",
        ),
        html.Div(
            [html.Button("Add step", id="add_step", n_clicks=0)],
            style={"marginTop": "8px"},
        ),
        html.Button(
            "Calculate",
            id="run",
            n_clicks=0,
            style={
                "position": "sticky",
                "bottom": "0",
                "width": "100%",
                "padding": "14px 0",
                "fontSize": "18px",
                "backgroundColor": "#00549f",
                "color": "white",
                "border": "none",
                "borderRadius": "6px",
                "cursor": "pointer",
                "marginTop": "10px",
                "zIndex": "100",
            },
        ),
    ],
    style={
        "display": "flex",
        "flexDirection": "column",
        "height": "100%",
        "overflowY": "auto",
        "padding": "8px",
        "borderRight": "1px solid #eee",
        "backgroundImage": "url('/assets/background.jpg')",
        "backgroundSize": "cover",
        "backgroundPosition": "center",
    },
)

right_outputs = html.Div(
    [
        html.H2("Analysis"),
        html.Div(
            id="kpi_row",
            style={"display": "flex", "gap": "12px", "flexWrap": "wrap"},
        ),
        html.Div([
            dcc.Graph(id="fig_utilization",
                style={"height": "320px"},
                config={"responsive": True},
                      ),
                ],
                    style={
                        "marginTop": "8px",
                    },
                ),
         html.Div([
             html.H4("Scaling effect on Utilization and Cell Cost"),
             html.Div(
                 [
                     html.Div(
                         [
                             html.Label("Sweep min (GWh)"),
                             num_input(
                                 "gwh_sweep_min",
                                 max(
                                     GWH_SWEEP_MIN_FLOOR,
                                     DEFAULT_GENERAL["annual_output_gwh"] * GWH_SWEEP_MIN_FACTOR,
                                 ),
                                 step="any",
                                 min_=GWH_SWEEP_MIN_FLOOR,
                                 width="120px",
                             ),
                         ],
                         style={"display": "flex", "flexDirection": "column"},
                     ),
                     html.Div(
                         [
                             html.Label("Sweep max (GWh)"),
                             num_input(
                                 "gwh_sweep_max",
                                 max(
                                     DEFAULT_GENERAL["annual_output_gwh"],
                                     GWH_SWEEP_MIN_FLOOR,
                                 )
                                 * GWH_SWEEP_MAX_MULTIPLIER,
                                 step="any",
                                 min_=GWH_SWEEP_MIN_FLOOR,
                                 width="120px",
                             ),
                         ],
                         style={"display": "flex", "flexDirection": "column"},
                     ),
                     html.Div(
                         [
                             html.Label("Points"),
                             num_input(
                                 "gwh_sweep_points",
                                 GWH_SWEEP_DEFAULT_POINTS,
                                 step=1,
                                 min_=2,
                                 width="90px",
                             ),
                         ],
                         style={"display": "flex", "flexDirection": "column", "alignItems": "stretch",},
                     ),
                 ],
                 style={
                     "display": "flex",
                     "gap": "10px",
                     "flexWrap": "wrap",
                     "alignItems": "flex-end",
                     "marginBottom": "8px",
                 },
             ),
             dcc.Graph(id="fig_utilization_gwh",
                    style={"height": "320px"},
                    config={"responsive": True},
                ),
         ],
             style={
                 "marginTop": "8px",
                 "maxHeight": "450px",
             },
         ),
        html.Div(
            [
            html.H4("Unit Cost vs Annual Output"),
                dcc.Graph(id="fig_cost_gwh",
                    style={"height": "320px"},
                    config={"responsive": True},
                ),
            ],
            style={
                "marginTop": "30px",
                "maxHeight": "450px",
            },
        ),


        html.Div(
            [
                html.Div(
                    [
                        html.H4("Annual Steady-State Cost Breakdown (€/yr)"),
                        dcc.Graph(id="fig_annual",
                            style={"height": "360px"},
                            config={"responsive": True},
                        ),
                    ],
                    style={
                        "marginTop": "8px",
                        "flex": "1 1 420px",
                        "minWidth": "320px",
                        "maxWidth": "100%",
                        "maxHeight": "450px",
                    },
                ),
                html.Div(
                    [
                        html.H4("Single-Cell Cost Breakdown (€/cell)"),
                        dcc.Graph(id="fig_cell",
                            style={"height": "360px"},
                            config={"responsive": True},
                        ),
                    ],
                    style={
                        "marginTop": "8px",
                        "flex": "1 1 420px",
                        "minWidth": "320px",
                        "maxWidth": "100%",
                        "maxHeight": "450px",
                    },
                ),
            ],
            style={
                "marginTop": "16px",
                "display": "flex",
                "flexWrap": "wrap",
                "gap": "12px",
            },
        ),
        html.Div(
            [
                html.Div(
                    [
                        html.H4("Step Capacity (Cells/Year) & Final Target"),
                        dcc.Graph(id="fig_cap",
                            style={"height": "360px"},
                            config={"responsive": True},
                        ),
                    ],
                    style={
                        "marginTop": "8px",
                        "flex": "1 1 420px",
                        "minWidth": "320px",
                        "maxWidth": "100%",
                        "maxHeight": "450px",
                    },
                ),

                html.Div(
                    [
                        html.H4("Materials Procurement Breakdown (€/cell)"),
                        dcc.Graph(id="fig_mat",
                            style={"height": "360px"},
                            config={"responsive": True},
                        ),
                    ],
                    style={
                        "marginTop": "8px",
                        "flex": "1 1 420px",
                        "minWidth": "320px",
                        "maxWidth": "100%",
                        "maxHeight": "450px",
                    },
                ),
            ],
            style={
                "marginTop": "16px",
                "display": "flex",
                "flexWrap": "wrap",
                "gap": "12px",
            },
        ),
        html.Div(
            [
                html.Div(
                    [
                        html.H4("Tornado Sensitivity (±25% — €/cell)"),
                        dcc.Graph(id="fig_sens",
                            style={"height": "360px"},
                            config={"responsive": True},
                        ),
                    ],
                    style={
                        "marginTop": "8px",
                        "flex": "1 1 420px",
                        "minWidth": "320px",
                        "maxWidth": "100%",
                        "maxHeight": "450px",
                    },
                ),
                html.Div(
                    [
                        html.H4("Project Timeline: Costs (bars) & Revenue (line)"),
                        dcc.Graph(id="fig_time",
                            style={"height": "360px"},
                            config={"responsive": True},
                        ),
                    ],
                    style={
                        "marginTop": "8px",
                        "flex": "1 1 420px",
                        "minWidth": "320px",
                        "maxWidth": "100%",
                        "maxHeight": "450px",
                    },
                ),
            ],
            style={
                "marginTop": "16px",
                "display": "flex",
                "flexWrap": "wrap",
                "gap": "12px",
            },
        ),
        html.Div(
            [
                html.H3("Flow & Scrap Sankey (Millions of Cells / Year)"),
                dcc.Graph(id="fig_sankey"),
            ],
            style={"marginTop": "8px"},
        ),
        html.Div(
         [
             html.H4("Footprint (Indoor/Outdoor, m²)"),
             dcc.Graph(id="fig_footprint",
             ),
         ],
            style={
                "marginTop": "8px",
            },
     ),
        html.H4("Per-step Overview (takt sizing, demand, capacity)"),
        dash_table.DataTable(
            id="steps_table_main",
            page_size=15,
            style_table={
                         "overflowX": "auto",
                         "maxHeight": "400px",
                         "overflowY": "auto",
                         },
            style_cell={
                "padding": "6px",
                "minWidth": 90,
                "maxWidth": 280,
                "whiteSpace": "normal",
            },
            page_action="none",
        ),
        html.Details(
            [
                html.Summary("Notes: formulas & assumptions (click to expand)"),
                html.Ul(
                    [
                        html.Li(
                            "Process topology is defined via 'Step ID' and internal "
                            "'Successors'/'Successor Fractions' fields. A DAG is built "
                            "and simulated to determine per-step demand, scrap, and flow."
                        ),
                        html.Li(
                            "Line cycle (takt) = available_time_seconds / final_good_cells. "
                            "Available time = days × shifts × hours × 3600 × Availability."
                        ),
                        html.Li(
                            "Machines per step are sized purely from lead time: "
                            "seconds_needed = demand × lead_time_s; "
                            "machines = ceil(seconds_needed / available_time_seconds)."
                        ),
                        html.Li(
                            "Capacity at step = machines × available_time_seconds / lead_time_s; "
                            "the bottleneck step defines the overall line capacity."
                        ),
                        html.Li(
                            "Energy per cell at a step = kW_per_machine × (lead_time_s / 3600). "
                            "Baseline building load is added as extra kWh/cell."
                        ),
                        html.Li(
                            "Area and equipment CAPEX scale with the calculated machine counts "
                            "and environment (none/clean/dry)."
                        ),
                        html.Li(
                            "Overheads are applied via explicit factors for personnel and "
                            "maintenance instead of generic OH/G&A/R&D buckets."
                        ),
                        html.Li(
                            "Sankey: each step splits into good flow to successors "
                            "(or Final Good) and scrap lost at that step. All sinks "
                            "are aggregated into a single final-good node."
                        ),
                        html.Li(
                            "Raw material input per final cell is scaled using process scrap "
                            "from the intro step to the end (no separate yield input)."
                        ),
                    ]
                ),
            ],
            open=False,
        ),
    ],
    style={"height": "100%",
           "overflowY": "auto",
           "padding": "10px 12px",
           },
)

app.layout = html.Div(
    [
        dcc.Store(id="preset_store", data=available_presets_init),
        dcc.Store(id="ramp_scrap_store", data=[]),
        dcc.Store(id="ramp_output_store", data=[]),

        html.Div(
            [
                html.Img(
                    src=logo_src(),
                    alt="PEM",
                    style={
                        "height": "70px",
                        "marginRight": "12px",
                        "marginLeft": "8px",
                        "display": "block",
                    },
                    draggable="false",
                ),
                html.H2(
                    "Cell Manufacturing Cost",
                    style={"margin": 0, "alignSelf": "center","color": "white"},
                ),
            ],
            style={
                "display": "flex",
                "alignItems": "flex-start",
                "gap": "0px",
                "margin": "0px 0 0px",
                "backgroundColor": "#00549f",
                "boxShadow": "0 4px 12px rgba(0, 0, 0, 0.2)",
                "position": "fixed",
                "top": 0,
                "left": 0,
                "width": "100%",
                "zIndex": 1000,
            },
        ),
        html.Div(
            style={
                "height": "72px",
            }
        ),

        html.Div(
            [left_inputs, right_outputs],
            style={
                "display": "grid",
                "gridTemplateColumns": "clamp(380px, 32vw, 560px) 1fr",
                "gap": "12px",
                "height": "calc(100vh - 72px)",
                "alignItems": "stretch",
            },
        ),
    ],
    style={
        "fontFamily": "HelveticaNeue-Light, Helvetica Neue Light, Helvetica Neue, Helvetica, Arial, Lucida Grande, sans-serif",
        "width": "100%",
        "maxWidth": "100%",
        "minHeight": "100vh",
        "overflowX": "hidden",
        "overflowY": "auto",
        "margin": "0",
        "padding": "0 12px 12px",

    },
)


# ============================================================
# Callbacks
# ============================================================

@app.callback(
    Output("materials_table", "data"),
    Input("add_mat", "n_clicks"),
    State("materials_table", "data"),
    State("steps_table", "data"),
    prevent_initial_call=True,
)
def add_material(n, rows, steps_rows):
    """Append a new material row, defaulting intro_step to first process step."""
    rows = rows or []
    first_step = (steps_rows or DEFAULT_STEPS)[0]["step"]
    rows.append(
        {
            "name": "New Material",
            "intro_step": first_step,
            "pricing_unit": "kg",
            "g_per_cell": 0.0,
            "eur_per_kg": 0.0,
            "area_per_cell_m2": 0.0,
            "eur_per_m2": 0.0,
        }
    )
    return rows


@app.callback(
    Output("steps_table", "data"),
    Input("add_step", "n_clicks"),
    State("steps_table", "data"),
    prevent_initial_call=True,
)
def add_step(n, rows):
    """Append a new generic process step at the end."""
    rows = rows or []
    next_order = 1 + max(
        [r.get("order", i + 1) for i, r in enumerate(rows)],
        default=0,
    )
    rows.append(
        {
            "order": next_order,
            "step": "New Step",
            "successor_step": "",  # default to End (no successor)
            "env": "none",
            "lead_time_s": 10.0,
            "scrap_rate": 0.005,
            "capex_meur_per_machine": 0.10,
            "footprint_m2": 50,
            "kw_per_unit": 0.001,
            "spec_workers_per_machine": 0.2,
            "supp_workers_per_machine": 0.2,
            "machines": 1,
        }
    )
    rows = sorted(rows, key=lambda r: r.get("order", 9999))
    return rows


@app.callback(
    Output("steps_table", "data", allow_duplicate=True),
    Input("move_up", "n_clicks"),
    Input("move_down", "n_clicks"),
    State("steps_table", "data"),
    State("steps_table", "selected_rows"),
    prevent_initial_call=True,
)
def move_step(up_clicks, down_clicks, rows, sel_rows):
    """Move selected step up or down in order (visual order; logic uses DAG)."""
    if not rows or not sel_rows:
        return rows

    idx = sel_rows[0]
    rows = sorted(rows, key=lambda r: r.get("order", 9999))
    if idx < 0 or idx >= len(rows):
        return rows

    trig = ctx.triggered_id
    if trig == "move_up" and idx > 0:
        rows[idx]["order"], rows[idx - 1]["order"] = (
            rows[idx - 1]["order"],
            rows[idx]["order"],
        )
    elif trig == "move_down" and idx < len(rows) - 1:
        rows[idx]["order"], rows[idx + 1]["order"] = (
            rows[idx + 1]["order"],
            rows[idx]["order"],
        )

    rows = sorted(rows, key=lambda r: r.get("order", 9999))
    for i, r in enumerate(rows, start=1):
        r["order"] = i
    return rows


@app.callback(
    Output("steps_table", "dropdown"),
    Input("steps_table", "data"),
)
def update_steps_dropdown(steps_rows):
    """
    Keep the 'Successor Step' dropdown in sync with the current step list.

    DAG connectivity is still driven by the internal succ/succ_frac fields,
    which are derived from this UI-facing column.
    """
    step_labels = [r.get("step") for r in (steps_rows or []) if r.get("step")]
    successor_options = [{"label": "End (no successor)", "value": ""}] + [
        {"label": s, "value": s} for s in step_labels
    ]
    return {
        "env": {"options": ENV_DROPDOWN_OPTIONS},
        "successor_step": {"options": successor_options},
    }

    return {
        "env": {"options": ENV_DROPDOWN_OPTIONS},
        "successor_step": {"options": successor_options},
    }


@app.callback(
    Output("preset_select", "options"),
    Output("preset_select", "value", allow_duplicate=True),
    Input("preset_store", "data"),
    State("preset_select", "value"),
    prevent_initial_call=True,
)
def sync_preset_dropdown(store_data, current_value):
    """Keep the preset dropdown in sync with available presets (built-in + external/imported)."""
    store_data = store_data or {}
    options = [{"label": k, "value": k} for k in store_data.keys()]
    value = current_value if current_value in store_data else (options[0]["value"] if options else None)
    return options, value


@app.callback(
    Output("preset_store", "data"),
    Output("preset_select", "value", allow_duplicate=True),
    Output("preset_note", "children", allow_duplicate=True),
    Input("preset_upload", "contents"),
    State("preset_upload", "filename"),
    State("preset_store", "data"),
    prevent_initial_call=True,
)
def import_preset(contents, filename, store_data):
    """Handle preset import from JSON file."""
    store = store_data or {}
    if not contents or not filename:
        return store, dash.no_update, dash.no_update
    try:
        _header, data = contents.split(",", 1)
        decoded = base64.b64decode(data)
        preset_raw = json.loads(decoded.decode("utf-8"))
        preset = _validate_preset_dict(preset_raw)
        key = preset.get("name") or os.path.splitext(filename)[0]
        store[key] = preset
        note = f"Imported preset '{key}' from {filename}."
        return store, key, note
    except Exception as exc:  # pragma: no cover - defensive UI path
        return store, dash.no_update, f"Import failed: {exc}"

@app.callback(
    Output("ramp_scrap_rows", "children"),
    Output("ramp_scrap_store", "data"),
    Output("ramp_output_store", "data"),
    Input("add_ramp_year", "n_clicks"),
    State("ramp_scrap_store", "data"),
    State("ramp_output_store", "data"),
    prevent_initial_call=True,
)
def add_ramp_row(n, scrap_data, output_data):
    scrap = list(scrap_data or [])
    output = list(output_data or [])
    scrap.append(0.0)
    output.append(0.5)  # default 50% output in first added ramp year
    rows = []
    for i, (s, o) in enumerate(zip(scrap, output), start=1):
        rows.append(
            html.Div(
                [
                    html.Label(f"Ramp year {i} output"),
                    dcc.Slider(
                        id={"type": "ramp_output_slider", "index": i},
                        min=0.0,
                        max=1.0,
                        step=0.05,
                        value=float(o),
                        marks={0: "0%", 0.5: "50%", 1.0: "100%"},
                        tooltip={"placement": "bottom", "always_visible": True},
                    ),
                    html.Label(f"Ramp year {i} scrap"),
                    dcc.Slider(
                        id={"type": "ramp_scrap_slider", "index": i},
                        min=0.0,
                        max=0.99,
                        step=0.01,
                        value=float(s),
                        marks={0: "0", 0.5: "50%", 0.9: "90%"},
                        tooltip={"placement": "bottom", "always_visible": True},
                    ),
                ],
                style={"border": "1px solid #eee", "borderRadius": "8px", "padding": "8px"},
            )
        )
    return rows, scrap, output

@app.callback(
    Output("ramp_output_store", "data", allow_duplicate=True),
    Input({"type": "ramp_output_slider", "index": dash.ALL}, "value"),
    State("ramp_output_store", "data"),
    prevent_initial_call=True,
)
def update_ramp_output(values, data):
    if values is None:
        return dash.no_update
    current = list(data or [])
    new_vals = list(values)
    if len(new_vals) < len(current):
        new_vals.extend([1.0] * (len(current) - len(new_vals)))
    return [max(min(float(v), 1.0), 0.0) for v in new_vals[: len(current)]]

@app.callback(
    Output("ramp_scrap_store", "data", allow_duplicate=True),
    Input({"type": "ramp_scrap_slider", "index": dash.ALL}, "value"),
    State("ramp_scrap_store", "data"),
    prevent_initial_call=True,
)
def update_ramp_store(values, data):
    if values is None:
        return dash.no_update
    current = list(data or [])
    new_vals = list(values)
    if len(new_vals) < len(current):
        new_vals.extend([0.0] * (len(current) - len(new_vals)))
    return [max(min(float(v), 0.999), 0.0) for v in new_vals[: len(current)]]


@app.callback(
    Output("preset_download", "data"),
    Input("export_preset", "n_clicks"),
    # General / energy
    State("el_price", "value"),
    State("building_baseline_kwh", "value"),
    State("labor_spec", "value"),
    State("labor_sup", "value"),
    State("bldg_cost", "value"),
    State("indoor_factor", "value"),
    State("outdoor_factor", "value"),
    State("outdoor_cost", "value"),
    State("clean_mult", "value"),
    State("clean_capex", "value"),
    State("clean_opex", "value"),
    State("dry_mult", "value"),
    State("dry_capex", "value"),
    State("dry_opex", "value"),
    State("gwh", "value"),
    State("cell_ah", "value"),
    State("cell_wh", "value"),
    State("cell_v", "value"),
    State("days", "value"),
    State("shifts", "value"),
    State("hshift", "value"),
    State("avail", "value"),
    # Econ / overhead
    State("proj_years", "value"),
    State("tax", "value"),
    State("dep_equip", "value"),
    State("dep_bldg", "value"),
    State("build_years", "value"),
    State("ramp_scrap_store", "data"),
    State("wacc", "value"),
    State("margin", "value"),
    State("oh_indirect_personnel", "value"),
    State("oh_logistics_personnel", "value"),
    State("oh_building_maintenance", "value"),
    State("oh_machine_maintenance", "value"),
    State("inv_logistics_factor", "value"),
    State("inv_indirect_factor", "value"),
    # Tables
    State("materials_table", "data"),
    State("steps_table", "data"),
    State("preset_select", "value"),
    prevent_initial_call=True,
)
def export_preset(
    _,
    el_price,
    building_baseline_kwh,
    labor_spec,
    labor_sup,
    bldg_cost,
    indoor_factor,
    outdoor_factor,
    outdoor_cost,
    clean_mult,
    clean_capex,
    clean_opex,
    dry_mult,
    dry_capex,
    dry_opex,
    gwh,
    cell_ah,
    cell_wh,
    cell_v,
    days,
    shifts,
    hshift,
    avail,
    proj_years,
    tax,
    dep_equip,
    dep_bldg,
    build_years,
    ramp_scrap_store,
    ramp_output_store,
    wacc,
    margin,
    oh_indirect_personnel,
    oh_logistics_personnel,
    oh_building_maintenance,
    oh_machine_maintenance,
    inv_logistics_factor,
    inv_indirect_factor,
    materials_rows,
    steps_rows,
    preset_key,
):
    """Export current UI state as a JSON preset."""
    preset = {
        "name": preset_key or "exported_preset",
        "general": {
            "electricity_price_eur_per_kwh": el_price,
            "baseline_building_kwh": building_baseline_kwh,
            "specialist_labor_eur_per_h": labor_spec,
            "support_labor_eur_per_h": labor_sup,
            "building_cost_eur_per_m2": bldg_cost,
            "indoor_area_factor": indoor_factor,
            "outdoor_area_factor": outdoor_factor,
            "outdoor_area_cost": outdoor_cost,
            "clean_area_multiplier": clean_mult,
            "clean_capex_eur_per_m2": clean_capex,
            "clean_opex_eur_per_h_per_m2": clean_opex,
            "dry_area_multiplier": dry_mult,
            "dry_capex_eur_per_m2": dry_capex,
            "dry_opex_eur_per_h_per_m2": dry_opex,
            "annual_output_gwh": gwh,
            "cell_ah": cell_ah,
            "cell_voltage": cell_v,
            "cell_wh": cell_wh,
            "working_days": days,
            "shifts_per_day": shifts,
            "shift_hours": hshift,
            "avail": avail,
        },
        "econ": {
            "project_years": proj_years,
            "tax_rate": tax,
            "depreciation_years_equipment": dep_equip,
            "depreciation_years_building": dep_bldg,
            "construction_years": build_years,
            "ramp_years": len(ramp_scrap_store or []),
            "ramp_scrap_rates": ramp_scrap_store,
            "capital_cost_wacc": wacc,
            "desired_margin": margin,
            "indirect_personnel_factor": oh_indirect_personnel,
            "logistics_personnel_factor": oh_logistics_personnel,
            "building_maintenance_factor": oh_building_maintenance,
            "machine_maintenance_factor": oh_machine_maintenance,
            "logistics_investment_factor": inv_logistics_factor,
            "indirect_investment_factor": inv_indirect_factor,
        },
        "steps": steps_rows or [],
        "materials": materials_rows or [],
        "note": f"Exported from app preset '{preset_key or 'custom'}'.",
    }
    payload = json.dumps(preset, indent=2)
    return dcc.send_string(payload, filename=f"{preset_key or 'preset'}.json")


@app.callback(
    # General/Econ inputs:
    Output("el_price", "value", allow_duplicate=True),
    Output("building_baseline_kwh", "value", allow_duplicate=True),
    Output("labor_spec", "value", allow_duplicate=True),
    Output("labor_sup", "value", allow_duplicate=True),
    Output("bldg_cost", "value", allow_duplicate=True),
    Output("indoor_factor", "value", allow_duplicate=True),
    Output("outdoor_factor", "value", allow_duplicate=True),
    Output("outdoor_cost", "value", allow_duplicate=True),
    Output("clean_mult", "value", allow_duplicate=True),
    Output("clean_capex", "value", allow_duplicate=True),
    Output("clean_opex", "value", allow_duplicate=True),
    Output("dry_mult", "value", allow_duplicate=True),
    Output("dry_capex", "value", allow_duplicate=True),
    Output("dry_opex", "value", allow_duplicate=True),
    Output("gwh", "value", allow_duplicate=True),
    Output("gwh_sweep_min", "value", allow_duplicate=True),
    Output("gwh_sweep_max", "value", allow_duplicate=True),
    Output("gwh_sweep_points", "value", allow_duplicate=True),
    Output("cell_ah", "value", allow_duplicate=True),
    Output("cell_wh", "value", allow_duplicate=True),
    Output("cell_v", "value", allow_duplicate=True),
    Output("days", "value", allow_duplicate=True),
    Output("shifts", "value", allow_duplicate=True),
    Output("hshift", "value", allow_duplicate=True),
    Output("avail", "value", allow_duplicate=True),
    Output("proj_years", "value", allow_duplicate=True),
    Output("tax", "value", allow_duplicate=True),
    Output("dep_equip", "value", allow_duplicate=True),
    Output("dep_bldg", "value", allow_duplicate=True),
    Output("build_years", "value", allow_duplicate=True),
    Output("ramp_scrap_store", "data", allow_duplicate=True),
    Output("ramp_output_store", "data", allow_duplicate=True),
    Output("wacc", "value", allow_duplicate=True),
    Output("margin", "value", allow_duplicate=True),
    # Overhead & investment factors:
    Output("oh_indirect_personnel", "value", allow_duplicate=True),
    Output("oh_logistics_personnel", "value", allow_duplicate=True),
    Output("oh_building_maintenance", "value", allow_duplicate=True),
    Output("oh_machine_maintenance", "value", allow_duplicate=True),
    Output("inv_logistics_factor", "value", allow_duplicate=True),
    Output("inv_indirect_factor", "value", allow_duplicate=True),
    # Tables:
    Output("materials_table", "data", allow_duplicate=True),
    Output("steps_table", "data", allow_duplicate=True),
    # Note
    Output("preset_note", "children", allow_duplicate=True),
    Input("apply_preset", "n_clicks"),
    State("preset_select", "value"),
    State("preset_store", "data"),
    prevent_initial_call=True,
)
def apply_preset(n, preset_key, preset_store):
    """Apply selected preset to all inputs and tables."""
    all_p = current_app_presets(preset_store)
    p = all_p.get(preset_key, all_p.get(DEFAULT_PRESET_KEY, next(iter(all_p.values()))))
    g = p["general"]
    ec = p["econ"]
    ramp_scrap = list(ec.get("ramp_scrap_rates", []) or [])
    ramp_output = list(ec.get("ramp_output_rates", []) or [])

    steps = copy.deepcopy(p["steps"])
    for i, r in enumerate(steps, start=1):
        r["order"] = i
        r.setdefault("successor_step", r.get("successor_step", ""))
    steps = sorted(steps, key=lambda r: r["order"])

    mats = copy.deepcopy(p["materials"])
    step_names = [s["step"] for s in steps]
    fallback = step_names[0] if step_names else DEFAULT_STEPS[0]["step"]
    for m in mats:
        if m.get("intro_step") not in step_names:
            m["intro_step"] = fallback

    note = f"Preset applied: {preset_key}. {p.get('note', '')}"
    sweep_min = max(GWH_SWEEP_MIN_FLOOR, g["annual_output_gwh"] * GWH_SWEEP_MIN_FACTOR)
    sweep_max = (max(g["annual_output_gwh"], GWH_SWEEP_MIN_FLOOR) * GWH_SWEEP_MAX_MULTIPLIER)

    return (
        g["electricity_price_eur_per_kwh"],
        g["baseline_building_kwh"],
        g["specialist_labor_eur_per_h"],
        g["support_labor_eur_per_h"],
        g["building_cost_eur_per_m2"],
        g["indoor_area_factor"],
        g["outdoor_area_factor"],
        g["outdoor_area_cost"],
        g["clean_area_multiplier"],
        g["clean_capex_eur_per_m2"],
        g["clean_opex_eur_per_h_per_m2"],
        g["dry_area_multiplier"],
        g["dry_capex_eur_per_m2"],
        g["dry_opex_eur_per_h_per_m2"],
        g["annual_output_gwh"],
        sweep_min,
        sweep_max,
        GWH_SWEEP_DEFAULT_POINTS,
        g["cell_ah"],
        g["cell_wh"],
        g["cell_voltage"],
        g["working_days"],
        g["shifts_per_day"],
        g["shift_hours"],
        g["avail"],
        ec["project_years"],
        ec["tax_rate"],
        ec["depreciation_years_equipment"],
        ec["depreciation_years_building"],
        ec["construction_years"],
        ramp_scrap,
        ec["capital_cost_wacc"],
        ec["desired_margin"],
        ec["indirect_personnel_factor"],
        ec["logistics_personnel_factor"],
        ec["building_maintenance_factor"],
        ec["machine_maintenance_factor"],
        ec["logistics_investment_factor"],
        ec["indirect_investment_factor"],
        mats,
        steps,
        note,
    )


@app.callback(
    Output("materials_table", "columns", allow_duplicate=True),
    Output("materials_table", "dropdown", allow_duplicate=True),
    Output("materials_table", "data", allow_duplicate=True),
    Input("steps_table", "data"),
    State("materials_table", "data"),
    prevent_initial_call=True,
)
def sync_materials_intro_step(steps_rows, materials_rows):
    """Keep material intro_step dropdown in sync with current step list."""
    columns, dropdown = build_materials_columns(steps_rows)
    rows = materials_rows or []

    step_names = [r["step"] for r in (steps_rows or []) if r.get("step")]
    if not step_names:
        step_names = [DEFAULT_STEPS[0]["step"]]
    fallback = step_names[0]

    for r in rows:
        if r.get("intro_step") not in step_names:
            r["intro_step"] = fallback
        r["pricing_unit"] = (r.get("pricing_unit") or "kg").lower()

    return columns, dropdown, rows


@app.callback(
    Output("kpi_row", "children"),
    Output("fig_annual", "figure"),
    Output("fig_cell", "figure"),
    Output("fig_cap", "figure"),
    Output("fig_utilization", "figure"),
    Output("fig_utilization_gwh", "figure"),
    Output("fig_cost_gwh", "figure"),
    Output("fig_mat", "figure"),
    Output("fig_sens", "figure"),
    Output("fig_time", "figure"),
    Output("fig_sankey", "figure"),
    Output("fig_footprint", "figure"),
    Output("steps_table_main", "columns"),
    Output("steps_table_main", "data"),
    Input("run", "n_clicks"),
    # General / energy
    State("el_price", "value"),
    State("building_baseline_kwh", "value"),
    State("labor_spec", "value"),
    State("labor_sup", "value"),
    State("bldg_cost", "value"),
    State("indoor_factor", "value"),
    State("outdoor_factor", "value"),
    State("outdoor_cost", "value"),
    State("clean_mult", "value"),
    State("clean_capex", "value"),
    State("clean_opex", "value"),
    State("dry_mult", "value"),
    State("dry_capex", "value"),
    State("dry_opex", "value"),
    State("gwh", "value"),
    State("gwh_sweep_min", "value"),
    State("gwh_sweep_max", "value"),
    State("gwh_sweep_points", "value"),
    State("cell_ah", "value"),
    State("cell_wh", "value"),
    State("cell_v", "value"),
    State("days", "value"),
    State("shifts", "value"),
    State("hshift", "value"),
    State("avail", "value"),
    # Econ / overhead
    State("proj_years", "value"),
    State("tax", "value"),
    State("dep_equip", "value"),
    State("dep_bldg", "value"),
    State("build_years", "value"),
    State("ramp_scrap_store", "data"),
    State("ramp_output_store", "data"),
    State("wacc", "value"),
    State("margin", "value"),
    State("oh_indirect_personnel", "value"),
    State("oh_logistics_personnel", "value"),
    State("oh_building_maintenance", "value"),
    State("oh_machine_maintenance", "value"),
    State("inv_logistics_factor", "value"),
    State("inv_indirect_factor", "value"),
    # Tables
    State("materials_table", "data"),
    State("steps_table", "data"),
    prevent_initial_call=True,
)
def run_calc(
    _,
    el_price,
    building_baseline_kwh,
    labor_spec,
    labor_sup,
    bldg_cost,
    indoor_factor,
    outdoor_factor,
    outdoor_cost,
    clean_mult,
    clean_capex,
    clean_opex,
    dry_mult,
    dry_capex,
    dry_opex,
    gwh,
    gwh_sweep_min,
    gwh_sweep_max,
    gwh_sweep_points,
    cell_ah,
    cell_wh,
    cell_v,
    days,
    shifts,
    hshift,
    avail,
    proj_years,
    tax,
    dep_equip,
    dep_bldg,
    build_years,
    ramp_scrap_store,
    ramp_output_store,
    wacc,
    margin,
    oh_indirect_personnel,
    oh_logistics_personnel,
    oh_building_maintenance,
    oh_machine_maintenance,
    inv_logistics_factor,
    inv_indirect_factor,
    materials_rows,
    steps_rows,
):
    """Run the cost model with current UI state and update all outputs."""
    resolved_ah, resolved_wh, resolved_v = _resolve_cell_params(
        cell_ah, cell_wh, cell_v
    )

    general = GeneralAssumptions(
        electricity_price_eur_per_kwh=float(
            el_price or DEFAULT_GENERAL["electricity_price_eur_per_kwh"]
        ),
        baseline_building_kwh=float(
            building_baseline_kwh or DEFAULT_GENERAL["baseline_building_kwh"]
        ),
        specialist_labor_eur_per_h=float(
            labor_spec or DEFAULT_GENERAL["specialist_labor_eur_per_h"]
        ),
        support_labor_eur_per_h=float(
            labor_sup or DEFAULT_GENERAL["support_labor_eur_per_h"]
        ),
        building_cost_eur_per_m2=float(
            bldg_cost or DEFAULT_GENERAL["building_cost_eur_per_m2"]
        ),
        indoor_area_factor=float(
            indoor_factor or DEFAULT_GENERAL["indoor_area_factor"]
        ),
        outdoor_area_factor=float(
            outdoor_factor or DEFAULT_GENERAL["outdoor_area_factor"]
        ),
        outdoor_area_cost=float(
            outdoor_cost or DEFAULT_GENERAL["outdoor_area_cost"]
        ),
        clean_area_multiplier=float(
            clean_mult or DEFAULT_GENERAL["clean_area_multiplier"]
        ),
        clean_capex_eur_per_m2=float(
            clean_capex or DEFAULT_GENERAL["clean_capex_eur_per_m2"]
        ),
        clean_opex_eur_per_h_per_m2=float(
            clean_opex or DEFAULT_GENERAL["clean_opex_eur_per_h_per_m2"]
        ),
        dry_area_multiplier=float(
            dry_mult or DEFAULT_GENERAL["dry_area_multiplier"]
        ),
        dry_capex_eur_per_m2=float(
            dry_capex or DEFAULT_GENERAL["dry_capex_eur_per_m2"]
        ),
        dry_opex_eur_per_h_per_m2=float(
            dry_opex or DEFAULT_GENERAL["dry_opex_eur_per_h_per_m2"]
        ),
        annual_output_gwh=float(gwh or DEFAULT_GENERAL["annual_output_gwh"]),
        cell_ah=float(resolved_ah),
        cell_wh=float(resolved_wh),
        cell_voltage=float(resolved_v),
        working_days=float(days or DEFAULT_GENERAL["working_days"]),
        shifts_per_day=float(shifts or DEFAULT_GENERAL["shifts_per_day"]),
        shift_hours=float(hshift or DEFAULT_GENERAL["shift_hours"]),
        avail=float(avail or DEFAULT_GENERAL["avail"]),
    )

    ramp_scrap_list = list(ramp_scrap_store or [])
    ramp_output_list = list(ramp_output_store or [])
    ramp_years_val = (
        max(len(ramp_scrap_list), len(ramp_output_list))
        if (ramp_scrap_list or ramp_output_list)
        else DEFAULT_ECON["ramp_years"]
    )


    econ = Economics(
        project_years=int(proj_years or DEFAULT_ECON["project_years"]),
        tax_rate=float(tax or DEFAULT_ECON["tax_rate"]),
        depreciation_years_equipment=int(
            dep_equip or DEFAULT_ECON["depreciation_years_equipment"]
        ),
        depreciation_years_building=int(
            dep_bldg or DEFAULT_ECON["depreciation_years_building"]
        ),
        construction_years=int(build_years or DEFAULT_ECON["construction_years"]),
        ramp_years=ramp_years_val,
        capital_cost_wacc=float(wacc or DEFAULT_ECON["capital_cost_wacc"]),
        desired_margin=float(margin or DEFAULT_ECON["desired_margin"]),
        indirect_personnel_factor=float(
            oh_indirect_personnel or DEFAULT_ECON["indirect_personnel_factor"]
        ),
        logistics_personnel_factor=float(
            oh_logistics_personnel or DEFAULT_ECON["logistics_personnel_factor"]
        ),
        building_maintenance_factor=float(
            oh_building_maintenance or DEFAULT_ECON["building_maintenance_factor"]
        ),
        machine_maintenance_factor=float(
            oh_machine_maintenance or DEFAULT_ECON["machine_maintenance_factor"]
        ),
        logistics_investment_factor=float(
            inv_logistics_factor or DEFAULT_ECON["logistics_investment_factor"]
        ),
        indirect_investment_factor=float(
            inv_indirect_factor or DEFAULT_ECON["indirect_investment_factor"]
        ),
        ramp_scrap_rates=ramp_scrap_list,
        ramp_output_rates=ramp_output_list,
    )

    steps = pd.DataFrame(steps_rows or [])
    mats = pd.DataFrame(materials_rows or [])

    model = BatteryCostModel(
        general=general,
        econ=econ,
        steps=steps,
        raw_materials=mats,
    )
    result = model.compute()

    k = result["kpis"]
    df = result["steps"]
    mats = result["materials"]
    cash = result["cash"]
    sens_df = result["sens"]
    dag_totals = result["dag"]

    def card(label: str, value: str, suffix: str = ""):
        return html.Div(
            [
                html.Div(
                    label,
                    style={"fontSize": "12px", "color": "#666"},
                ),
                html.Div(
                    f"{value}{suffix}",
                    style={"fontSize": "20px", "fontWeight": 600},
                ),
            ],
            style={
                "border": "1px solid #eee",
                "borderRadius": "12px",
                "padding": "10px 12px",
                "minWidth": "210px",
            },
        )

    kpi_children = [
        card("Line Cycle / Takt", f"{k['line_cycle_time_s']:.3f}", " s"),
        card("CPM", f"{k['cpm']:.0f}", " 1/min"),
        card("Final Cells Target", f"{k['final_cells_required']:.0f}"),
        card("Line Capacity (Cells)", f"{k['line_capacity_cells']:.0f}"),
        card("Bottleneck", str(k["bottleneck_step"])),
        card(
            "Build Cost / Cell",
            f"{k['unit_cost_build_eur_per_cell']:.2f}",
            " €",
        ),
        card(
            "Build Cost / kWh",
            f"{k['cost_build_per_kwh_eur']:.2f}",
            " €/kWh",
        ),
        card("Price / Cell (margin)", f"{k['price_per_cell_eur']:.2f}", " €"),
        card(
            "Price / kWh (margin)",
            f"{k['price_per_kwh_eur']:.2f}",
            " €/kWh",
        ),
        card("Indoor (standard)", f"{k['indoor_area_none_m2']:.0f}", " m²"),
        card("Clean room", f"{k['indoor_area_clean_m2']:.0f}", " m²"),
        card("Dry room", f"{k['indoor_area_dry_m2']:.0f}", " m²"),
        card("Outdoor", f"{k['required_outdoor_area_m2']:.0f}", " m²"),
        card(
            "Total Required Area",
            f"{k['total_required_area_m2']:.0f}",
            " m²",
        ),
        card(
            "Cost indoor (non-dry)",
            f"{k['area_cost_indoor_none_eur'] / 1e6:.2f}",
            " M€",
        ),
        card(
            "Cost clean room",
            f"{k['area_cost_indoor_clean_eur'] / 1e6:.2f}",
            " M€",
        ),
        card(
            "Cost dry Room",
            f"{k['area_cost_indoor_dry_eur'] / 1e6:.2f}",
            " M€",
        ),
        card(
            "Cost outdoor",
            f"{k['area_cost_outdoor_eur'] / 1e6:.2f}",
            " M€",
        ),
        card(
            "Building CAPEX (total)",
            f"{k['building_value_eur'] / 1e6:.2f}",
            " M€",
        ),
        card(
            "Equipment CAPEX (incl. logistics & indirect)",
            f"{k['capital_equipment_total_eur'] / 1e6:.2f}",
            " M€",
        ),
        card("NPV (project)", f"{k['npv_total_eur'] / 1e6:.2f}", " M€"),
        card(
            "Breakeven Year",
            str(k["breakeven_year"] if k["breakeven_year"] is not None else "n/a"),
        ),
    ]

    fig_annual, fig_cell, fig_cap, fig_utilization, fig_mat, fig_sens, fig_time, fig_footprint, = BatteryCostModel.figs(
        df, mats, k, cash, sens_df
    )

    # Utilization vs GWh sweep
    try:
        gwh_input = float(gwh or general.annual_output_gwh)
    except Exception:
        gwh_input = general.annual_output_gwh
    sweep_min = max(0.1, gwh_input * 0.2)
    sweep_max = max(gwh_input, 0.1) * 5
    gwh_points = np.linspace(sweep_min, sweep_max, 10)
    sweep_min_default = max(GWH_SWEEP_MIN_FLOOR, gwh_input * GWH_SWEEP_MIN_FACTOR)
    sweep_max_default = max(gwh_input, GWH_SWEEP_MIN_FLOOR) * GWH_SWEEP_MAX_MULTIPLIER

    def _safe_float(val):
        try:
            return float(val)
        except Exception:
            return None

    sweep_min_val = _safe_float(gwh_sweep_min)
    sweep_max_val = _safe_float(gwh_sweep_max)
    sweep_points_val = _safe_float(gwh_sweep_points)

    sweep_min = sweep_min_val if sweep_min_val and sweep_min_val > 0 else sweep_min_default
    sweep_max = sweep_max_val if sweep_max_val and sweep_max_val > sweep_min else sweep_max_default
    if sweep_max <= sweep_min:
        sweep_max = sweep_min + max(sweep_min * 0.05, GWH_SWEEP_MIN_FLOOR)

    gwh_points_count = int(sweep_points_val) if sweep_points_val else GWH_SWEEP_DEFAULT_POINTS
    if gwh_points_count < 2:
        gwh_points_count = 2

    gwh_points = np.linspace(sweep_min, sweep_max, gwh_points_count)

    util_points = []
    cost_points = []

    for gwh_val in gwh_points:
        g_sweep = replace(general, annual_output_gwh=float(gwh_val))
        model_sweep = BatteryCostModel(
            general=g_sweep,
            econ=econ,
            steps=steps,
            raw_materials=mats,
        )
        k_sweep = model_sweep.compute()["kpis"]
        util_points.append(
            {"gwh": gwh_val,
             "utilization_pct": k_sweep.get("utilization", 0.0) * 100.0}
        )
        cost_points.append(
            {"gwh": gwh_val,
             "unit_cost": k_sweep.get("unit_cost_build_eur_per_cell")}
        )
    fig_utilization_gwh = px.line(
        util_points,
        x="gwh",
        y="utilization_pct",
        labels={"gwh": "Annual output (GWh)", "utilization_pct": "Utilization (%)"},
    )
    fig_utilization_gwh.update_traces(mode="lines+markers", line=dict(color="#00549f"))
    fig_utilization_gwh.update_layout(margin=dict(l=40, r=10, t=30, b=60))

    fig_cost_gwh = px.line(
        cost_points,
        x="gwh",
        y="unit_cost",
        labels={"gwh": "Annual output (GWh)", "unit_cost": "€/cell"},
    )
    fig_cost_gwh.update_traces(mode="lines+markers", line=dict(color="#cc071e"))
    fig_cost_gwh.update_layout(margin=dict(l=40, r=10, t=30, b=60))


    # Use DAG totals (multi-root aware) for Sankey
    fig_sankey = BatteryCostModel.sankey(df, dag_totals, scale=1e6)
    columns, data = BatteryCostModel.table_view(df)


    return (
        kpi_children,
        fig_annual,
        fig_cell,
        fig_cap,
        fig_utilization,
        fig_utilization_gwh,
        fig_cost_gwh,
        fig_mat,
        fig_sens,
        fig_time,
        fig_sankey,
        fig_footprint,
        columns,
        data,
    )


if __name__ == "__main__":
    # NOTE: Use debug=False in production deployments.
    app.run(debug=True)
