"""
Cell Manufacturing Cost Estimator (refactored)
"""

from dataclasses import dataclass
from typing import Dict, Tuple, List, Any
import copy
import math
import os
from urllib.parse import quote

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, html, dcc, Input, Output, State, dash_table, ctx

# ============================================================
# Defaults & Presets
# ============================================================

def _derive_ah_v_from_kwh(kwh: float, v: float = 3.7) -> Tuple[float, float]:
    """Helper to derive Ah from kWh and voltage (kept for backwards compatibility)."""
    v = v if v and v > 0 else 3.7
    ah = (kwh * 1000.0) / v
    return ah, v


# --- Default process steps (used as template for all presets) -----------------
DEFAULT_STEPS: List[Dict[str, Any]] = [
    {
        "step": "Mixing",
        "lead_time_s": 0.017,
        "scrap_rate": 0.01,
        "capex_meur_per_machine": 1.08,
        "footprint_m2": 24,
        "kw_per_unit": 20.0,
        "env": "none",
        "spec_workers_per_machine": 0.5,
        "supp_workers_per_machine": 0,
        "machines": 18,  # initial guess; will be recalculated in model
    },
    {
        "step": "Coating",
        "lead_time_s": 0.02325,
        "scrap_rate": 0.0,
        "capex_meur_per_machine": 37.8,
        "footprint_m2": 300,
        "kw_per_unit": 75.0,
        "env": "clean",
        "spec_workers_per_machine": 1,
        "supp_workers_per_machine": 0,
        "machines": 6,
    },
    {
        "step": "Calendering",
        "lead_time_s": 0.02325,
        "scrap_rate": 0.0,
        "capex_meur_per_machine": 2.9,
        "footprint_m2": 24,
        "kw_per_unit": 60.0,
        "env": "none",
        "spec_workers_per_machine": 0.5,
        "supp_workers_per_machine": 0,
        "machines": 6,
    },
    {
        "step": "Slitting",
        "lead_time_s": 0.02325,
        "scrap_rate": 0.0,
        "capex_meur_per_machine": 1.15,
        "footprint_m2": 24,
        "kw_per_unit": 45.0,
        "env": "none",
        "spec_workers_per_machine": 0,
        "supp_workers_per_machine": 0,
        "machines": 6,
    },
    {
        "step": "Vacuum Drying",
        "lead_time_s": 0.02325,
        "scrap_rate": 0.0,
        "capex_meur_per_machine": 1.2,
        "footprint_m2": 11,
        "kw_per_unit": 56.0,
        "env": "none",
        "spec_workers_per_machine": 0.1,
        "supp_workers_per_machine": 0,
        "machines": 12,
    },
    {
        "step": "Contacting",
        "lead_time_s": 3,
        "scrap_rate": 0.0010,
        "capex_meur_per_machine": 5.0,
        "footprint_m2": 12,
        "kw_per_unit": 5.0,
        "env": "none",
        "spec_workers_per_machine": 0.2,
        "supp_workers_per_machine": 0.0,
        "machines": 37,
    },
    {
        "step": "Winding",
        "lead_time_s": 1.50,
        "scrap_rate": 0.006,
        "capex_meur_per_machine": 0.80,
        "footprint_m2": 12,
        "kw_per_unit": 25.0,
        "env": "clean",
        "spec_workers_per_machine": 0.5,
        "supp_workers_per_machine": 0.0,
        "machines": 23,
    },
    {
        "step": "Insert in Housing",
        "lead_time_s": 0.60,
        "scrap_rate": 0.0015,
        "capex_meur_per_machine": 2.5,
        "footprint_m2": 93,
        "kw_per_unit": 117.50,
        "env": "clean",
        "spec_workers_per_machine": 1.25,
        "supp_workers_per_machine": 0.0,
        "machines": 12,
    },
    {
        "step": "Electrolyte Fill",
        "lead_time_s": 2,
        "scrap_rate": 0.001,
        "capex_meur_per_machine": 1.0,
        "footprint_m2": 60,
        "kw_per_unit": 22.0,
        "env": "dry",
        "spec_workers_per_machine": 0.66,
        "supp_workers_per_machine": 0.0,
        "machines": 25,
    },
    {
        "step": "Formation",
        "lead_time_s": 177.1875,
        "scrap_rate": 0.005,
        "capex_meur_per_machine": 0.01,
        "footprint_m2": 5,
        "kw_per_unit": 10.0,
        "env": "none",
        "spec_workers_per_machine": 0.005,
        "supp_workers_per_machine": 0.0,
        "machines": 167,
    },
    {
        "step": "Ageing",
        "lead_time_s": 270,
        "scrap_rate": 0.0,
        "capex_meur_per_machine": 0.005,
        "footprint_m2": 5,
        "kw_per_unit": 10.0,
        "env": "none",
        "spec_workers_per_machine": 0.002,
        "supp_workers_per_machine": 0.0,
        "machines": 225,
    },
    {
        "step": "Final EoL Test",
        "lead_time_s": 0.9375,
        "scrap_rate": 0.025,
        "capex_meur_per_machine": 0.005,
        "footprint_m2": 5,
        "kw_per_unit": 10.0,
        "env": "none",
        "spec_workers_per_machine": 0.005,
        "supp_workers_per_machine": 0.0,
        "machines": 60,
    },
]

for i, r in enumerate(DEFAULT_STEPS, start=1):
    r["order"] = i  # deterministic ordering


# --- Default raw materials (used as template for all presets) -----------------
DEFAULT_RAW_MATERIALS: List[Dict[str, Any]] = [
    {
        "name": "NMC",
        "intro_step": "Mixing",
        "pricing_unit": "kg",
        "g_per_cell": 128.0,
        "eur_per_kg": 25.0,
        "area_per_cell_m2": 0.0,
        "eur_per_m2": 0.0,
        "total_yield": 0.922,
    },
    {
        "name": "Graphite+Si",
        "intro_step": "Mixing",
        "pricing_unit": "kg",
        "g_per_cell": 65.38,
        "eur_per_kg": 5.42,
        "area_per_cell_m2": 0.0,
        "eur_per_m2": 0.0,
        "total_yield": 0.922,
    },
    {
        "name": "Conductive Carbon",
        "intro_step": "Mixing",
        "pricing_unit": "kg",
        "g_per_cell": 4.0,
        "eur_per_kg": 3.0,
        "area_per_cell_m2": 0.0,
        "eur_per_m2": 0.0,
        "total_yield": 0.922,
    },
    {
        "name": "Binder (PVDF/CMC/SBR)",
        "intro_step": "Mixing",
        "pricing_unit": "kg",
        "g_per_cell": 6.0,
        "eur_per_kg": 13.59,
        "area_per_cell_m2": 0.0,
        "eur_per_m2": 0.0,
        "total_yield": 0.922,
    },
    {
        "name": "Solvent (NMP)",
        "intro_step": "Mixing",
        "pricing_unit": "kg",
        "g_per_cell": 2.0,
        "eur_per_kg": 2.7,
        "area_per_cell_m2": 0.0,
        "eur_per_m2": 0.0,
        "total_yield": 0.995,
    },
    {
        "name": "Aluminum Foil (CC)",
        "intro_step": "Coating",
        "pricing_unit": "kg",
        "g_per_cell": 22.0,
        "eur_per_kg": 4.87,
        "area_per_cell_m2": 0.0,
        "eur_per_m2": 0.0,
        "total_yield": 0.865,
    },
    {
        "name": "Copper Foil (CC)",
        "intro_step": "Coating",
        "pricing_unit": "kg",
        "g_per_cell": 34.0,
        "eur_per_kg": 12.3,
        "area_per_cell_m2": 0.0,
        "eur_per_m2": 0.0,
        "total_yield": 0.865,
    },
    {
        "name": "Separator",
        "intro_step": "Winding",
        "pricing_unit": "m2",
        "g_per_cell": 0.0,
        "eur_per_kg": 0.0,
        "area_per_cell_m2": 0.07,
        "eur_per_m2": 0.26,
        "total_yield": 0.980,
    },
    {
        "name": "Cell Case",
        "intro_step": "Insert in Housing",
        "pricing_unit": "m2",
        "g_per_cell": 65.5,
        "eur_per_kg": 0.0,
        "area_per_cell_m2": 0.0,
        "eur_per_m2": 0.0,
        "total_yield": 0.990,
    },
    {
        "name": "Electrolyte",
        "intro_step": "Electrolyte Fill",
        "pricing_unit": "kg",
        "g_per_cell": 49.37,
        "eur_per_kg": 5.39,
        "area_per_cell_m2": 0.0,
        "eur_per_m2": 0.0,
        "total_yield": 0.990,
    },
]

# --- Presets (all defaults live here) ----------------------------------------

# Baseline NMC 4680
nmca_v = _derive_ah_v_from_kwh(0.0927, 3.7)
NMC_CELL_AH, NMC_CELL_V = nmca_v
NMC_CELL_WH = NMC_CELL_AH * NMC_CELL_V

PRESETS: Dict[str, Dict[str, Any]] = {
    "NMC 4680 (Baseline)": {
        "general": {
            "electricity_price_eur_per_kwh": 0.1589,
            "baseline_building_kwh": 101.8,  # baseline building load (kW)
            "specialist_labor_eur_per_h": 44.0,
            "support_labor_eur_per_h": 0.0,
            "building_cost_eur_per_m2": 3360.0,
            "indoor_area_factor": 3.0,
            "outdoor_area_factor": 0.0,
            "clean_area_multiplier": 1.0,
            "clean_add_cost_eur_per_m2": 2850.0,
            "dry_area_multiplier": 1.15,
            "dry_add_cost_eur_per_m2": 120.0,
            "annual_output_gwh": 10.0,
            "cell_ah": NMC_CELL_AH,
            "cell_voltage": NMC_CELL_V,
            "cell_wh": NMC_CELL_WH,
            "working_days": 365.0,
            "shifts_per_day": 3.0,
            "shift_hours": 8.0,
            "oee": 0.855,
        },
        "econ": {
            "project_years": 10,
            "tax_rate": 0.30,
            "depreciation_years_equipment": 7,
            "depreciation_years_building": 33,
            "construction_years": 2,
            "ramp_years": 1,
            "capital_cost_wacc": 0.06,
            "desired_margin": 0.15,
            # New overhead factors (relative to direct labor or assets)
            "indirect_personnel_factor": 0.25,
            "logistics_personnel_factor": 0.15,
            "building_maintenance_factor": 0.02,  # of building CAPEX per year
            "machine_maintenance_factor": 0.03,   # of equipment CAPEX per year
            # New investment factors (multipliers on equipment CAPEX)
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
            "indoor_area_factor": 1.25,
            "outdoor_area_factor": 0.35,
            "clean_area_multiplier": 1.35,
            "clean_add_cost_eur_per_m2": 260.0,
            "dry_area_multiplier": 1.20,
            "dry_add_cost_eur_per_m2": 130.0,
            "annual_output_gwh": 5.0,
            "cell_ah": 20.0,
            "cell_voltage": 3.8,
            "cell_wh": 20.0 * 3.8,
            "working_days": 330.0,
            "shifts_per_day": 3.0,
            "shift_hours": 8.0,
            "oee": 0.80,
        },
        "econ": {
            "project_years": 12,
            "tax_rate": 0.30,
            "depreciation_years_equipment": 8,
            "depreciation_years_building": 35,
            "construction_years": 3,
            "ramp_years": 2,
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
            "indoor_area_factor": 1.15,
            "outdoor_area_factor": 0.25,
            "clean_area_multiplier": 1.25,
            "clean_add_cost_eur_per_m2": 220.0,
            "dry_area_multiplier": 1.10,
            "dry_add_cost_eur_per_m2": 100.0,
            "annual_output_gwh": 8.0,
            "cell_ah": 12.0,
            "cell_voltage": 2.5,
            "cell_wh": 12.0 * 2.5,
            "working_days": 350.0,
            "shifts_per_day": 2.5,
            "shift_hours": 8.0,
            "oee": 0.82,
        },
        "econ": {
            "project_years": 10,
            "tax_rate": 0.28,
            "depreciation_years_equipment": 7,
            "depreciation_years_building": 30,
            "construction_years": 2,
            "ramp_years": 1,
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

# Ensure deterministic ordering on preset step lists
for preset in PRESETS.values():
    if "steps" in preset:
        for i, r in enumerate(preset["steps"], start=1):
            r.setdefault("order", i)

# Base preset used for initial UI defaults and fallback values
DEFAULT_PRESET_KEY = "NMC 4680 (Baseline)"
DEFAULT_GENERAL = PRESETS[DEFAULT_PRESET_KEY]["general"]
DEFAULT_ECON = PRESETS[DEFAULT_PRESET_KEY]["econ"]


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

    provided = {
        "ah": ah is not None,
        "wh": wh is not None,
        "v": v is not None,
    }
    count = sum(provided.values())

    # start from defaults
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
            # all three provided: keep Ah & V authoritative
            wh_res = ah_res * v_res

    return ah_res, wh_res, v_res


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
    clean_area_multiplier: float
    clean_add_cost_eur_per_m2: float
    dry_area_multiplier: float
    dry_add_cost_eur_per_m2: float
    annual_output_gwh: float
    cell_ah: float
    cell_wh: float
    cell_voltage: float
    working_days: float
    shifts_per_day: float
    shift_hours: float
    oee: float


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
    # New explicit overhead factors
    indirect_personnel_factor: float
    logistics_personnel_factor: float
    building_maintenance_factor: float
    machine_maintenance_factor: float
    # New investment factors
    logistics_investment_factor: float
    indirect_investment_factor: float


@dataclass
class BatteryCostModel:
    """
    Encapsulates the core cell manufacturing cost model.

    The model:
    - Computes line takt from annual output and available hours
    - Sizes machines by lead time & scrap
    - Allocates materials, labor, energy, area, and CAPEX
    - Applies explicit overhead & maintenance factors
    - Produces KPIs and time-series cashflows
    """

    general: GeneralAssumptions
    econ: Economics
    steps: pd.DataFrame
    raw_materials: pd.DataFrame

    def compute(self) -> Dict[str, Any]:
        g, e = self.general, self.econ
        df = self.steps.copy().reset_index(drop=True)

        if "order" in df.columns:
            df = df.sort_values("order").reset_index(drop=True)

        # ----------------------------------------------------------
        # 1) Available time & line cycle time ("takt")
        # ----------------------------------------------------------
        hours_year = g.working_days * g.shifts_per_day * g.shift_hours
        avail_time_seconds = hours_year * 3600.0 * g.oee

        # Prefer Wh from assumptions; fall back to Ah * V if needed
        cell_wh = g.cell_wh if g.cell_wh > 0 else g.cell_ah * g.cell_voltage
        cell_wh = max(cell_wh, 1e-9)
        cell_kwh = cell_wh / 1000.0

        final_cells_required = (g.annual_output_gwh * 1_000_000.0) / cell_kwh
        line_cycle_time_s = avail_time_seconds / max(final_cells_required, 1.0)

        # ----------------------------------------------------------
        # 2) Per-step survival, cumulative survival to end, demand
        # ----------------------------------------------------------
        df["scrap_rate"] = (
            df.get("scrap_rate", 0.0)
            .astype(float)
            .clip(lower=0.0, upper=0.999999)
        )
        df["survival"] = 1.0 - df["scrap_rate"]

        # cumulative survival from this step to the end (inclusive)
        survival_downstream: List[float] = []
        prod = 1.0
        for s in df["survival"][::-1]:
            prod *= s
            survival_downstream.append(prod)
        survival_downstream = list(reversed(survival_downstream))
        df["cumulative_survival_to_end"] = survival_downstream

        # demand at each step (units/year at entry)
        df["demand_cells_step_per_year"] = (
            final_cells_required
            / df["cumulative_survival_to_end"].replace(0, 1e-12)
        )

        # required cycle time at each step
        df["required_cycle_time_s"] = (
            line_cycle_time_s * df["cumulative_survival_to_end"]
        )

        # ----------------------------------------------------------
        # 3) Lead time, machine sizing, capacity
        # ----------------------------------------------------------
        df["lead_time_s"] = (
            pd.to_numeric(df.get("lead_time_s", 0.0), errors="coerce")
            .fillna(0.0)
            .clip(lower=1e-6)
        )

        # Number of Machines derived solely from lead_time & required cycle
        df["machines"] = np.ceil(
            df["lead_time_s"] / df["required_cycle_time_s"]
        ).astype(int).clip(lower=1)

        df["capacity_cells_per_year"] = (
            df["machines"] * avail_time_seconds / df["lead_time_s"]
        )
        df["capacity_ratio_vs_demand"] = (
            df["capacity_cells_per_year"]
            / df["demand_cells_step_per_year"].replace(0, 1e-12)
        )

        bottleneck_idx = df["capacity_ratio_vs_demand"].idxmin()
        bottleneck_row = df.loc[bottleneck_idx]

        line_capacity_cells = (
            float(df.loc[bottleneck_idx, "capacity_cells_per_year"])
            * float(df.loc[bottleneck_idx, "cumulative_survival_to_end"])
        )

        actual_cells_for_cost = final_cells_required
        utilization = min(
        actual_cells_for_cost / max(line_capacity_cells, 1.0), 1.0
        )

        # ----------------------------------------------------------
        # 4) Materials procurement
        # ----------------------------------------------------------
        mats = self.raw_materials.copy()
        mats["pricing_unit"] = mats.get("pricing_unit", "kg").astype(str).str.lower()
        mats["kg_per_cell"] = mats["g_per_cell"].astype(float) / 1000.0
        mats["m2_per_cell"] = mats.get("area_per_cell_m2", 0.0).astype(float)
        mats["eur_per_kg"] = mats.get("eur_per_kg", 0.0).astype(float)
        mats["eur_per_m2"] = mats.get("eur_per_m2", 0.0).astype(float)
        mats["survival"] = mats["total_yield"].clip(lower=1e-6)

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

        # Attribute intro-step materials cost to the right step (if names match)
        df["materials_cost_per_cell_total_eur"] = 0.0
        for _, r in mats.iterrows():
            idx = df.index[df["step"] == r["intro_step"]]
            if len(idx):
                df.loc[
                    idx[0], "materials_cost_per_cell_total_eur"
                ] += float(r["procurement_cost_per_cell_eur"])

        # ----------------------------------------------------------
        # 5) Labor per cell
        # ----------------------------------------------------------
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

   
        # ----------------------------------------------------------
        # 6) CAPEX & Area (incl. logistics/indirect investment)
        # ----------------------------------------------------------
        df["step_capex_total_eur"] = (
            df["capex_meur_per_machine"] * 1_000_000.0 * df["machines"]
        )
        total_capital_equipment_base = float(df["step_capex_total_eur"].sum())

        # Additional CAPEX for logistics & indirect investment
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
        cost_indoor_none = indoor_none * base_cost
        cost_indoor_clean = (
            indoor_clean * base_cost + indoor_clean * max(g.clean_add_cost_eur_per_m2, 0.0)
        )
        cost_indoor_dry = (
            indoor_dry * base_cost + indoor_dry * max(g.dry_add_cost_eur_per_m2, 0.0)
        )
        cost_outdoor = required_outdoor_area * base_cost

        building_value = (
            cost_indoor_none + cost_indoor_clean + cost_indoor_dry + cost_outdoor
        )

        annual_depr_equipment = (
            total_capital_equipment / e.depreciation_years_equipment
        )
        annual_depr_building = building_value / e.depreciation_years_building
        annual_depreciation = annual_depr_equipment + annual_depr_building

        depreciation_per_cell = annual_depreciation / max(actual_cells_for_cost, 1.0)

        # ----------------------------------------------------------
        # 7) Energy per cell (process + building baseline)
        # ----------------------------------------------------------
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
        building_energy_cost_per_cell = (
                building_energy_kwh_per_cell * g.electricity_price_eur_per_kwh
        )

        energy_per_cell = process_energy_per_cell + building_energy_cost_per_cell

        # ----------------------------------------------------------
        # 8) Overheads & unit cost (new factor model)
        # ----------------------------------------------------------
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

        # ----------------------------------------------------------
        # 9) Sensitivity (±25%) – now on explicit components
        # ----------------------------------------------------------
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

        # ----------------------------------------------------------
        # 10) Project timeline & cashflows
        # ----------------------------------------------------------
        years = list(range(0, e.project_years + 1))
        capex_total = total_capital_equipment + building_value

        capex_outflow_per_year = [0.0] * (e.project_years + 1)
        for y in range(min(e.construction_years, e.project_years)):
            capex_outflow_per_year[y] = capex_total / max(
                e.construction_years, 1
            )

        prod_cells_per_year = [0.0] * (e.project_years + 1)
        start_prod_year = e.construction_years
        for y in range(start_prod_year, e.project_years + 1):
            t = y - start_prod_year
            ramp_frac = (
                1.0
                if e.ramp_years <= 0
                else min(t / e.ramp_years, 1.0)
            )
            prod_cells_per_year[y] = final_cells_required * ramp_frac

        # Opex excludes depreciation; overheads are already included
        opex_per_cell = unit_cost_build - depreciation_per_cell

        cashflows: List[Dict[str, Any]] = []
        cum_cash: List[float] = []
        cf_cum = 0.0

        for y in years:
            revenue = prod_cells_per_year[y] * price_per_cell
            opex = prod_cells_per_year[y] * opex_per_cell
            deprec = (
                annual_depr_building + annual_depr_equipment
                if y >= start_prod_year
                else 0.0
            )
            ebit = revenue - opex - deprec
            tax = max(ebit, 0.0) * e.tax_rate
            nopat = ebit - tax
            fcf = nopat + deprec - capex_outflow_per_year[y]
            disc = (1.0 + e.capital_cost_wacc) ** y
            npv_y = fcf / disc

            cashflows.append(
                {
                    "year": y,
                    "revenue": revenue,
                    "opex": opex,
                    "depr": deprec,
                    "tax": tax,
                    "fcf": fcf,
                    "npv": npv_y,
                }
            )
            cf_cum += fcf
            cum_cash.append(cf_cum)

        npv_total = sum(c["npv"] for c in cashflows)
        breakeven_year = next(
            (c["year"] for c, cum in zip(cashflows, cum_cash) if cum >= 0.0),
            None,
        )

        kpis = {
            "materials_procurement_per_cell_eur": materials_procurement_per_cell,
            "process_energy_per_cell_eur": process_energy_per_cell,
            "building_energy_per_cell_eur": building_energy_cost_per_cell,
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
            "line_capacity_cells": line_capacity_cells,
            "actual_cells": actual_cells_for_cost,
            "bottleneck_step": bottleneck_row["step"],
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

        return {
            "kpis": kpis,
            "steps": df,
            "materials": mats,
            "cash": pd.DataFrame(cashflows),
            "sens": sens_df,
        }

    # ------------------------------------------------------------------
    # Figures
    # ------------------------------------------------------------------
    @staticmethod
    def figs(
        df: pd.DataFrame,
        mats: pd.DataFrame,
        k: Dict[str, Any],
        cash: pd.DataFrame,
        sens_df: pd.DataFrame,
    ):
        cells = max(k["actual_cells"], 1.0)

        parts = {
            "Materials": k["materials_procurement_per_cell_eur"] * cells,
            "Energy (incl. building)": k["energy_per_cell_eur"] * cells,
            "Specialist Labor": k["spec_labor_per_cell_eur"] * cells,
            "Support Labor": k["supp_labor_per_cell_eur"] * cells,
            "Indirect Personnel OH": k["indirect_personnel_oh_per_cell_eur"] * cells,
            "Logistics Personnel OH": k["logistics_personnel_oh_per_cell_eur"] * cells,
            "Building Maintenance": k["building_maintenance_oh_per_cell_eur"] * cells,
            "Machine Maintenance": k["machine_maintenance_oh_per_cell_eur"] * cells,
            "Depreciation": k["depreciation_per_cell_eur"] * cells,
        }

        fig_annual = px.bar(
            x=list(parts.keys()),
            y=list(parts.values()),
            labels={"x": "Component", "y": "€/year"},
        )
        fig_annual.update_layout(yaxis_tickprefix="€ ")

        parts_cell = {
            "Materials": k["materials_procurement_per_cell_eur"],
            "Energy (incl. building)": k["energy_per_cell_eur"],
            "Specialist Labor": k["spec_labor_per_cell_eur"],
            "Support Labor": k["supp_labor_per_cell_eur"],
            "Indirect Personnel OH": k["indirect_personnel_oh_per_cell_eur"],
            "Logistics Personnel OH": k["logistics_personnel_oh_per_cell_eur"],
            "Building Maintenance": k["building_maintenance_oh_per_cell_eur"],
            "Machine Maintenance": k["machine_maintenance_oh_per_cell_eur"],
            "Depreciation": k["depreciation_per_cell_eur"],
        }

        fig_cell = px.bar(
            x=list(parts_cell.keys()),
            y=list(parts_cell.values()),
            labels={"x": "Component", "y": "€/cell"},
        )
        fig_cell.add_hline(
            y=k["unit_cost_build_eur_per_cell"],
            line_dash="dash",
            annotation_text="Build cost",
            annotation_position="top left",
        )

        fig_cap = px.bar(
            df,
            x="step",
            y="capacity_cells_per_year",
            labels={
                "step": "Process Step",
                "capacity_cells_per_year": "Cells/Year",
            },
        )
        fig_cap.add_hline(
            y=k["final_cells_required"],
            line_dash="dash",
            annotation_text="Final Cells Target",
            annotation_position="top left",
        )

        fig_mat = px.bar(
            mats,
            x="name",
            y="procurement_cost_per_cell_eur",
            labels={
                "name": "Material",
                "procurement_cost_per_cell_eur": "€/cell (procurement)",
            },
        )

        base = k["unit_cost_build_eur_per_cell"]
        td = sens_df.copy()
        td["LowDelta"] = td["Low"] - base
        td["HighDelta"] = td["High"] - base
        td = td.sort_values("Impact", ascending=False)

        fig_sens = go.Figure()
        fig_sens.add_trace(
            go.Bar(
                y=td["Parameter"],
                x=td["LowDelta"],
                name="-25%",
                orientation="h",
            )
        )
        fig_sens.add_trace(
            go.Bar(
                y=td["Parameter"],
                x=td["HighDelta"],
                name="+25%",
                orientation="h",
            )
        )
        fig_sens.update_layout(
            barmode="overlay",
            xaxis_title="Δ €/cell vs base",
            legend_orientation="h",
        )

        cash = cash.copy()
        fig_time = go.Figure()
        fig_time.add_trace(
            go.Bar(
                x=cash["year"],
                y=-(cash["depr"] + cash["opex"]),
                name="Costs (excl. CAPEX)",
                offsetgroup=0,
            )
        )
        fig_time.add_trace(
            go.Bar(
                x=cash["year"],
                y=-cash["fcf"].where(cash["fcf"] < 0, 0.0),
                name="CAPEX/FCF<0",
                offsetgroup=1,
            )
        )
        fig_time.add_trace(
            go.Scatter(
                x=cash["year"],
                y=cash["revenue"],
                name="Revenue",
                mode="lines+markers",
                yaxis="y",
            )
        )

        be_year = k.get("breakeven_year")
        if isinstance(be_year, (int, float)):
            fig_time.add_vline(
                x=int(be_year),
                line_dash="dash",
                annotation_text=f"Breakeven Y{int(be_year)}",
            )
        fig_time.update_layout(yaxis_title="€ / year")

        return fig_annual, fig_cell, fig_cap, fig_mat, fig_sens, fig_time

    # ============================================================
    # Sankey: flow vs scrap
    # ============================================================
    @staticmethod
    def sankey(
        df: pd.DataFrame,
        k: Dict[str, Any],  # kept for future extensions if needed
        scale: float = 1e6,
    ) -> go.Figure:
        """
        Sankey diagram where each step splits into:
        - good flow to the next step (or Final Good for the last step)
        - scrap flow to a dedicated Scrap node for this step

        Values are scaled by 'scale' (default 1e6 -> millions of cells/year).
        """
        df = df.copy().reset_index(drop=True)

        steps = df["step"].tolist()
        n = len(steps)
        step_nodes = steps
        final_node_label = "Final Good Cells"
        scrap_nodes = [f"Scrap @ {s}" for s in steps]

        labels = step_nodes + [final_node_label] + scrap_nodes
        idx_final = n
        idx_scrap_start = n + 1

        sources: List[int] = []
        targets: List[int] = []
        values: List[float] = []
        link_labels: List[str] = []
        link_colors: List[str] = []

        good_color = "rgba(37,99,235,0.5)"
        scrap_color = "rgba(220,38,38,0.6)"

        for i in range(n):
            step_input = float(df.loc[i, "demand_cells_step_per_year"])
            survival = float(df.loc[i, "survival"])
            scrap_rate = float(df.loc[i, "scrap_rate"])  # kept for clarity
            good_out = step_input * survival
            scrap_out = step_input - good_out

            good_val = max(good_out / scale, 0.0)
            scrap_val = max(scrap_out / scale, 0.0)

            # good flow: step i -> next step or final
            if i < n - 1:
                sources.append(i)
                targets.append(i + 1)
            else:
                sources.append(i)
                targets.append(idx_final)
            values.append(good_val)
            link_labels.append(f"Good from {steps[i]}: {good_out:,.0f}/yr")
            link_colors.append(good_color)

            # scrap flow: step i -> its scrap node
            sources.append(i)
            targets.append(idx_scrap_start + i)
            values.append(scrap_val)
            link_labels.append(f"Scrap at {steps[i]}: {scrap_out:,.0f}/yr")
            link_colors.append(scrap_color)

        node_colors = (
            ["#93c5fd"] * n  # steps
            + ["#22c55e"]    # final good
            + ["#fca5a5"] * n  # scrap nodes
        )

        fig = go.Figure(
            data=[
                go.Sankey(
                    arrangement="snap",
                    textfont=dict(size=12),
                    node=dict(
                        label=labels,
                        color=node_colors,
                        pad=16,
                        thickness=22,
                    ),
                    link=dict(
                        source=sources,
                        target=targets,
                        value=values,
                        label=link_labels,
                        color=link_colors,
                    ),
                )
            ]
        )
        fig.update_layout(
            title=(
                "Flow & Scrap per Step — values in millions of cells/year "
                f"(÷ {int(scale):,})"
            ),
            margin=dict(l=10, r=10, t=50, b=10),
            height=540,
        )
        return fig

    @staticmethod
    def table_view(df: pd.DataFrame):
        """
        Build read-only step summary table for the right-hand panel.
        Only lead time, scrap & derived fields are shown.
        """
        df = df.copy()
        if "order" not in df.columns:
            df["order"] = range(1, len(df) + 1)
        df = df.sort_values("order")

        show_cols = [
            "order",
            "step",
            "env",
            "lead_time_s",
            "scrap_rate",
            "machines",
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
            {"name": "Environment", "id": "env", "presentation": "dropdown"},
            {
                "name": "Lead Time (s/unit)",
                "id": "lead_time_s",
                "type": "numeric",
            },
            {
                "name": "Scrap Rate (0–1)",
                "id": "scrap_rate",
                "type": "numeric",
            },
            {"name": "Machines (calc)", "id": "machines"},
            {
                "name": "CAPEX (M€ / machine)",
                "id": "capex_meur_per_machine",
                "type": "numeric",
            },
            {
                "name": "Footprint (m² / machine)",
                "id": "footprint_m2",
                "type": "numeric",
            },
            {
                "name": "Energy (kW / machine)",
                "id": "kw_per_unit",
                "type": "numeric",
            },
            {
                "name": "Specialists / machine",
                "id": "spec_workers_per_machine",
                "type": "numeric",
            },
            {
                "name": "Support / machine",
                "id": "supp_workers_per_machine",
                "type": "numeric",
            },
            {"name": "Req. Cycle (s)", "id": "required_cycle_time_s"},
            {
                "name": "Demand at Step (cells/yr)",
                "id": "demand_cells_step_per_year",
            },
            {
                "name": "Capacity (cells/yr)",
                "id": "capacity_cells_per_year",
            },
            {
                "name": "Intro-step Materials €/Cell",
                "id": "materials_cost_per_cell_total_eur",
            },
            {"name": "Energy €/Cell", "id": "energy_cost_per_cell_eur"},
        ]

        return columns, df_show.to_dict("records")


# ============================================================
# UI helpers
# ============================================================

app = Dash(__name__)
server = app.server
app.title = "Cell Cost Estimator"


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
    width: str = "120px",
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
        {
            "name": "g/cell",
            "id": "g_per_cell",
            "type": "numeric",
            "format": {"specifier": ".3f"},
        },
        {
            "name": "€/kg",
            "id": "eur_per_kg",
            "type": "numeric",
            "format": {"specifier": ".3f"},
        },
        {
            "name": "Area per cell (m²)",
            "id": "area_per_cell_m2",
            "type": "numeric",
            "format": {"specifier": ".4f"},
        },
        {
            "name": "€/m²",
            "id": "eur_per_m2",
            "type": "numeric",
            "format": {"specifier": ".3f"},
        },
        {
            "name": "Total Yield (0–1)",
            "id": "total_yield",
            "type": "numeric",
            "format": {"specifier": ".4f"},
        },
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


# Precomputed columns & dropdown for initial layout
_mat_columns_init, _mat_dropdown_init = build_materials_columns(DEFAULT_STEPS)

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
                    options=[{"label": k, "value": k} for k in PRESETS.keys()],
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
                        html.Label("Building Baseline Load (kWh/m²/a)"),
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
                        html.Label("Working Days / yr"),
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
                        html.Label("Shifts / day"),
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
                        html.Label("Hours / shift"),
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
                        html.Label("OEE (0–1)"),
                        num_input(
                            "oee",
                            DEFAULT_GENERAL["oee"],
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
        html.H3("Cell parameters (Ah / Wh / V)"),
        html.Div(
            [
                html.Div(
                    [
                        html.Label("Cell Capacity (Ah)"),
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
                        html.Label("Cell Energy (Wh)"),
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
                        html.Label("Base Building Cost (€/m²)"),
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
                        html.Label("Clean room add-on cost (€/m²)"),
                        num_input(
                            "clean_add",
                            DEFAULT_GENERAL["clean_add_cost_eur_per_m2"],
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
                        html.Label("Dry room add-on cost (€/m²)"),
                        num_input(
                            "dry_add",
                            DEFAULT_GENERAL["dry_add_cost_eur_per_m2"],
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
                        html.Label("Tax Rate (0–1)"),
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
                        html.Label("Ramp-up Duration (years)"),
                        num_input(
                            "ramp_years",
                            DEFAULT_ECON["ramp_years"],
                            step=1,
                            min_=0,
                        ),
                    ]
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
                        html.Label("Indirect personnel OH on labor (×)"),
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
                        html.Label("Logistics personnel OH on labor (×)"),
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
                        html.Label("Building maintenance (× CAPEX / yr)"),
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
                        html.Label("Machine maintenance (× CAPEX / yr)"),
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
                        html.Label("Logistics investment factor (× equip CAPEX)"),
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
                        html.Label("Indirect investment factor (× equip CAPEX)"),
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
        html.H3("Raw Materials — price by kg or m²"),
        dash_table.DataTable(
            id="materials_table",
            columns=_mat_columns_init,
            data=DEFAULT_RAW_MATERIALS,
            editable=True,
            dropdown=_mat_dropdown_init,
            row_deletable=True,
            style_table={"overflowX": "auto"},
            style_cell={"padding": "6px", "minWidth": 120, "whiteSpace": "normal"},
            page_size=12,
        ),
        html.Button(
            "Add material",
            id="add_mat",
            n_clicks=0,
            style={"marginTop": "8px"},
        ),
        html.Hr(),
        html.H3("Process Steps (lead time & scrap; add/remove & reorder)"),
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
                {
                    "name": "Environment (None/Clean/Dry)",
                    "id": "env",
                    "presentation": "dropdown",
                },
                {
                    "name": "Lead Time (s/unit)",
                    "id": "lead_time_s",
                    "type": "numeric",
                },
                {
                    "name": "Scrap Rate (0–1)",
                    "id": "scrap_rate",
                    "type": "numeric",
                },
                {
                    "name": "CAPEX (M€ / machine)",
                    "id": "capex_meur_per_machine",
                    "type": "numeric",
                },
                {
                    "name": "Footprint (m² per machine)",
                    "id": "footprint_m2",
                    "type": "numeric",
                },
                {
                    "name": "Energy Consumption (kW per machine)",
                    "id": "kw_per_unit",
                    "type": "numeric",
                },
                {
                    "name": "Specialists / machine",
                    "id": "spec_workers_per_machine",
                    "type": "numeric",
                },
                {
                    "name": "Support / machine",
                    "id": "supp_workers_per_machine",
                    "type": "numeric",
                },
            ],
            data=DEFAULT_STEPS,
            editable=True,
            dropdown={
                "env": {
                    "options": [
                        {"label": "None", "value": "none"},
                        {"label": "Clean", "value": "clean"},
                        {"label": "Dry", "value": "dry"},
                    ]
                },
            },
            row_deletable=True,
            row_selectable="single",
            style_table={"overflowX": "auto"},
            style_cell={"padding": "6px", "minWidth": 120, "whiteSpace": "normal"},
            page_size=12,
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
                "backgroundColor": "#2563eb",
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
        "height": "100%",
        "overflowY": "auto",
        "padding": "8px",
        "borderRight": "1px solid #eee",
    },
)

right_outputs = html.Div(
    [
        html.H2("Analysis (Cell only)"),
        html.Div(
            id="kpi_row",
            style={"display": "flex", "gap": "12px", "flexWrap": "wrap"},
        ),
        html.Div(
            [
                html.Div(
                    [
                        html.H4("Annual Steady-State Cost Breakdown (€/yr)"),
                        dcc.Graph(id="fig_annual"),
                    ],
                    style={
                        "width": "48%",
                        "display": "inline-block",
                        "verticalAlign": "top",
                    },
                ),
                html.Div(
                    [
                        html.H4("Single-Cell Cost Breakdown (€/cell)"),
                        dcc.Graph(id="fig_cell"),
                    ],
                    style={
                        "width": "48%",
                        "display": "inline-block",
                        "verticalAlign": "top",
                    },
                ),
            ],
            style={
                "display": "flex",
                "justifyContent": "space-between",
                "gap": "12px",
                "flexWrap": "wrap",
            },
        ),
        html.Div(
            [
                html.Div(
                    [
                        html.H4("Step Capacity (Cells/Year) & Final Target"),
                        dcc.Graph(id="fig_cap"),
                    ],
                    style={
                        "width": "48%",
                        "display": "inline-block",
                        "verticalAlign": "top",
                    },
                ),
                html.Div(
                    [
                        html.H4("Materials Procurement Breakdown (€/cell)"),
                        dcc.Graph(id="fig_mat"),
                    ],
                    style={
                        "width": "48%",
                        "display": "inline-block",
                        "verticalAlign": "top",
                    },
                ),
            ],
            style={
                "display": "flex",
                "justifyContent": "space-between",
                "gap": "12px",
                "flexWrap": "wrap",
            },
        ),
        html.Div(
            [
                html.Div(
                    [
                        html.H4("Tornado Sensitivity (±25% — €/cell)"),
                        dcc.Graph(id="fig_sens"),
                    ],
                    style={
                        "width": "48%",
                        "display": "inline-block",
                        "verticalAlign": "top",
                    },
                ),
                html.Div(
                    [
                        html.H4("Project Timeline: Costs (bars) & Revenue (line)"),
                        dcc.Graph(id="fig_time"),
                    ],
                    style={
                        "width": "48%",
                        "display": "inline-block",
                        "verticalAlign": "top",
                    },
                ),
            ],
            style={
                "display": "flex",
                "justifyContent": "space-between",
                "gap": "12px",
                "flexWrap": "wrap",
            },
        ),
        html.Div(
            [
                html.H4("Flow & Scrap Sankey (Millions of Cells / Year)"),
                dcc.Graph(id="fig_sankey"),
            ],
            style={"marginTop": "8px"},
        ),
        html.H4("Per-step Overview (takt sizing, demand, capacity)"),
        dash_table.DataTable(
            id="steps_table_main",
            page_size=15,
            style_table={"overflowX": "auto"},
            style_cell={
                "padding": "6px",
                "minWidth": 90,
                "maxWidth": 280,
                "whiteSpace": "normal",
            },
        ),
        html.Details(
            [
                html.Summary("Notes: formulas & assumptions (click to expand)"),
                html.Ul(
                    [
                        html.Li(
                            "Line cycle (takt) = available_time_seconds / final_good_cells. "
                            "Available time = days × shifts × hours × 3600 × OEE."
                        ),
                        html.Li(
                            "At step i: cumulative survival to end = Π (1 - scrap_j) "
                            "for j=i..end; demand at step = final_cells / cumulative_survival."
                        ),
                        html.Li(
                            "Required step cycle = line_takt × cumulative_survival_to_end "
                            "(faster than takt when there is scrap downstream)."
                        ),
                        html.Li(
                            "Machines = ceil(lead_time / required_step_cycle). "
                            "Capacity at step = machines × available_time / lead_time."
                        ),
                        html.Li(
                            "Energy per cell at a step = kW_per_machine × (lead_time_s / 3600). "
                            "Baseline building load is added as extra kWh/cell."
                        ),
                        html.Li(
                            "Area and equipment CAPEX scale with the calculated machine counts."
                        ),
                        html.Li(
                            "Overheads are applied via explicit factors for personnel and "
                            "maintenance instead of generic OH/G&A/R&D buckets."
                        ),
                        html.Li(
                            "Sankey: each step splits into good flow to the next step "
                            "(or Final Good) and scrap lost at that step."
                        ),
                    ]
                ),
            ],
            open=False,
        ),
    ],
    style={"height": "100%", "overflowY": "auto", "padding": "10px 12px"},
)

app.layout = html.Div(
    [
        html.Div(
            [
                html.Img(
                    src=logo_src(),
                    alt="PEM",
                    style={
                        "height": "44px",
                        "marginRight": "12px",
                        "display": "block",
                    },
                    draggable="false",
                ),
                html.H2(
                    "Cell Manufacturing Cost",
                    style={"margin": 0, "alignSelf": "center"},
                ),
            ],
            style={
                "display": "flex",
                "alignItems": "center",
                "gap": "8px",
                "margin": "8px 0 12px",
            },
        ),
        html.Div(
            [left_inputs, right_outputs],
            style={
                "display": "grid",
                "gridTemplateColumns": "clamp(380px, 32vw, 560px) 1fr",
                "gap": "12px",
                "height": "calc(100vh - 80px)",
                "alignItems": "stretch",
            },
        ),
    ],
    style={
        "fontFamily": "Inter, system-ui, Arial",
        "width": "100%",
        "maxWidth": "100%",
        "margin": "0 auto",
        "padding": "0 8px 8px",
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
            "total_yield": 1.0,
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
    """Move selected step up or down in order."""
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
    # General/Econ inputs:
    Output("el_price", "value", allow_duplicate=True),
    Output("building_baseline_kwh", "value", allow_duplicate=True),
    Output("labor_spec", "value", allow_duplicate=True),
    Output("labor_sup", "value", allow_duplicate=True),
    Output("bldg_cost", "value", allow_duplicate=True),
    Output("indoor_factor", "value", allow_duplicate=True),
    Output("outdoor_factor", "value", allow_duplicate=True),
    Output("clean_mult", "value", allow_duplicate=True),
    Output("clean_add", "value", allow_duplicate=True),
    Output("dry_mult", "value", allow_duplicate=True),
    Output("dry_add", "value", allow_duplicate=True),
    Output("gwh", "value", allow_duplicate=True),
    Output("cell_ah", "value", allow_duplicate=True),
    Output("cell_wh", "value", allow_duplicate=True),
    Output("cell_v", "value", allow_duplicate=True),
    Output("days", "value", allow_duplicate=True),
    Output("shifts", "value", allow_duplicate=True),
    Output("hshift", "value", allow_duplicate=True),
    Output("oee", "value", allow_duplicate=True),
    Output("proj_years", "value", allow_duplicate=True),
    Output("tax", "value", allow_duplicate=True),
    Output("dep_equip", "value", allow_duplicate=True),
    Output("dep_bldg", "value", allow_duplicate=True),
    Output("build_years", "value", allow_duplicate=True),
    Output("ramp_years", "value", allow_duplicate=True),
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
    prevent_initial_call=True,
)
def apply_preset(n, preset_key):
    """Apply selected preset to all inputs and tables."""
    p = PRESETS.get(preset_key, PRESETS[DEFAULT_PRESET_KEY])
    g = p["general"]
    ec = p["econ"]

    steps = copy.deepcopy(p["steps"])
    for i, r in enumerate(steps, start=1):
        r["order"] = i
    steps = sorted(steps, key=lambda r: r["order"])

    mats = copy.deepcopy(p["materials"])
    step_names = [s["step"] for s in steps]
    fallback = step_names[0] if step_names else DEFAULT_STEPS[0]["step"]
    for m in mats:
        if m.get("intro_step") not in step_names:
            m["intro_step"] = fallback

    note = f"Preset applied: {preset_key}. {p.get('note', '')}"

    return (
        g["electricity_price_eur_per_kwh"],
        g["baseline_building_kwh"],
        g["specialist_labor_eur_per_h"],
        g["support_labor_eur_per_h"],
        g["building_cost_eur_per_m2"],
        g["indoor_area_factor"],
        g["outdoor_area_factor"],
        g["clean_area_multiplier"],
        g["clean_add_cost_eur_per_m2"],
        g["dry_area_multiplier"],
        g["dry_add_cost_eur_per_m2"],
        g["annual_output_gwh"],
        g["cell_ah"],
        g["cell_wh"],
        g["cell_voltage"],
        g["working_days"],
        g["shifts_per_day"],
        g["shift_hours"],
        g["oee"],
        ec["project_years"],
        ec["tax_rate"],
        ec["depreciation_years_equipment"],
        ec["depreciation_years_building"],
        ec["construction_years"],
        ec["ramp_years"],
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
    Output("fig_mat", "figure"),
    Output("fig_sens", "figure"),
    Output("fig_time", "figure"),
    Output("fig_sankey", "figure"),
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
    State("clean_mult", "value"),
    State("clean_add", "value"),
    State("dry_mult", "value"),
    State("dry_add", "value"),
    State("gwh", "value"),
    State("cell_ah", "value"),
    State("cell_wh", "value"),
    State("cell_v", "value"),
    State("days", "value"),
    State("shifts", "value"),
    State("hshift", "value"),
    State("oee", "value"),
    # Econ / overhead
    State("proj_years", "value"),
    State("tax", "value"),
    State("dep_equip", "value"),
    State("dep_bldg", "value"),
    State("build_years", "value"),
    State("ramp_years", "value"),
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
    clean_mult,
    clean_add,
    dry_mult,
    dry_add,
    gwh,
    cell_ah,
    cell_wh,
    cell_v,
    days,
    shifts,
    hshift,
    oee,
    proj_years,
    tax,
    dep_equip,
    dep_bldg,
    build_years,
    ramp_years,
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
    # Resolve cell Ah/Wh/V consistently
    resolved_ah, resolved_wh, resolved_v = _resolve_cell_params(
        cell_ah, cell_wh, cell_v
    )

    general = GeneralAssumptions(
        electricity_price_eur_per_kwh=float(
            el_price or DEFAULT_GENERAL["electricity_price_eur_per_kwh"]
        ),
        baseline_building_kwh=float(
            building_baseline_kwh or DEFAULT_GENERAL["baseline_building_kw"]
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
        clean_area_multiplier=float(
            clean_mult or DEFAULT_GENERAL["clean_area_multiplier"]
        ),
        clean_add_cost_eur_per_m2=float(
            clean_add or DEFAULT_GENERAL["clean_add_cost_eur_per_m2"]
        ),
        dry_area_multiplier=float(
            dry_mult or DEFAULT_GENERAL["dry_area_multiplier"]
        ),
        dry_add_cost_eur_per_m2=float(
            dry_add or DEFAULT_GENERAL["dry_add_cost_eur_per_m2"]
        ),
        annual_output_gwh=float(gwh or DEFAULT_GENERAL["annual_output_gwh"]),
        cell_ah=float(resolved_ah),
        cell_wh=float(resolved_wh),
        cell_voltage=float(resolved_v),
        working_days=float(days or DEFAULT_GENERAL["working_days"]),
        shifts_per_day=float(shifts or DEFAULT_GENERAL["shifts_per_day"]),
        shift_hours=float(hshift or DEFAULT_GENERAL["shift_hours"]),
        oee=float(oee or DEFAULT_GENERAL["oee"]),
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
        ramp_years=int(ramp_years or DEFAULT_ECON["ramp_years"]),
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
    )

    steps = pd.DataFrame(steps_rows or DEFAULT_STEPS).copy()
    if "order" not in steps.columns:
        steps["order"] = range(1, len(steps) + 1)
    steps = steps.sort_values("order")

    mats = pd.DataFrame(materials_rows or DEFAULT_RAW_MATERIALS).copy()
    step_names = set(steps["step"].tolist())
    if len(steps):
        fallback_step = steps["step"].iloc[0]
        mats["intro_step"] = mats["intro_step"].apply(
            lambda x: x if x in step_names else fallback_step
        )

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
        card("Final Cells Target", f"{k['final_cells_required']:.0f}"),
        card("Line Capacity (Cells)", f"{k['line_capacity_cells']:.0f}"),
        card("Bottleneck", str(k["bottleneck_step"])),
        card(
            "Build Cost / Cell",
            f"{k['unit_cost_build_eur_per_cell']:.4f}",
            " €",
        ),
        card(
            "Build Cost / kWh",
            f"{k['cost_build_per_kwh_eur']:.2f}",
            " €/kWh",
        ),
        card("Price / Cell (margin)", f"{k['price_per_cell_eur']:.4f}", " €"),
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
            str(
                k["breakeven_year"]
                if k["breakeven_year"] is not None
                else "n/a"
            ),
        ),
    ]

    fig_annual, fig_cell, fig_cap, fig_mat, fig_sens, fig_time = BatteryCostModel.figs(
        df, mats, k, cash, sens_df
    )
    fig_sankey = BatteryCostModel.sankey(df, k, scale=1e6)
    columns, data = BatteryCostModel.table_view(df)

    return (
        kpi_children,
        fig_annual,
        fig_cell,
        fig_cap,
        fig_mat,
        fig_sens,
        fig_time,
        fig_sankey,
        columns,
        data,
    )


if __name__ == "__main__":
    # NOTE: set debug=False in production deployments.
    app.run(debug=True)



