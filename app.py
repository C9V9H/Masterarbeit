from dataclasses import dataclass
from typing import List, Dict, Tuple
from dash import Dash, html, dcc, Input, Output, State, dash_table, ctx
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os
from urllib.parse import quote
import math
import copy

# ============================================================
# Cell-only manufacturing cost model (v5.5)
# New in v5.5:
# - Per-cell Ah and Voltage inputs
# - kWh/cell derived = Ah * V / 1000
# - Target annual cells = (Annual GWh * 1e6) / (kWh/cell)
# - Use TARGET CELLS for all costs & cashflow
# - Keep bottleneck (line capacity) calculations as before
# ============================================================

# -----------------------------------------
# Default process steps (EU-leaning placeholders)
# throughput_cps is **per machine**
# -----------------------------------------
DEFAULT_STEPS = [
    {"step": "Mixing & Slurry Prep",          "throughput_cps": 0.60, "machines": 18,  "capex_meur_per_machine": 5.50, "footprint_m2": 500, "kw_per_unit": 20.0, "dry_room": False, "spec_workers_per_machine": 0.5, "supp_workers_per_machine": 1.0},
    {"step": "Coating & Drying",              "throughput_cps": 0.30, "machines": 6,  "capex_meur_per_machine": 18.0, "footprint_m2": 3500, "kw_per_unit": 1750.0, "dry_room": False, "spec_workers_per_machine": 0.4, "supp_workers_per_machine": 0.6},
    {"step": "Calendering & Slitting",        "throughput_cps": 0.70, "machines": 6,  "capex_meur_per_machine": 3.8,  "footprint_m2": 1000, "kw_per_unit": 60.0,  "dry_room": False, "spec_workers_per_machine": 0.2, "supp_workers_per_machine": 0.3},
    {"step": "Post/Vacuum Drying",            "throughput_cps": 0.35, "machines": 12, "capex_meur_per_machine": 0.30, "footprint_m2": 300,  "kw_per_unit": 56.0,  "dry_room": True,  "spec_workers_per_machine": 0.2, "supp_workers_per_machine": 0.3},
    {"step": "Stacking/Winding & Assembly",   "throughput_cps": 0.60, "machines": 37, "capex_meur_per_machine": 0.80, "footprint_m2": 500,  "kw_per_unit": 15.0,  "dry_room": True,  "spec_workers_per_machine": 0.7, "supp_workers_per_machine": 0.5},
    {"step": "Contacting/Welding",            "throughput_cps": 0.30, "machines": 23, "capex_meur_per_machine": 7.60, "footprint_m2": 1500, "kw_per_unit": 12.5,  "dry_room": True,  "spec_workers_per_machine": 0.4, "supp_workers_per_machine": 0.3},
    {"step": "Insert in Housing/Pouch",       "throughput_cps": 0.50, "machines": 12, "capex_meur_per_machine": 0.85, "footprint_m2": 600,  "kw_per_unit": 15.0,  "dry_room": True,  "spec_workers_per_machine": 0.3, "supp_workers_per_machine": 0.4},
    {"step": "Electrolyte Fill & First Seal", "throughput_cps": 0.28, "machines": 25, "capex_meur_per_machine": 3.50, "footprint_m2": 8000, "kw_per_unit": 15.0,  "dry_room": True,  "spec_workers_per_machine": 0.3, "supp_workers_per_machine": 0.4},
    {"step": "Wetting/Soak",                  "throughput_cps": 0.14, "machines": 167,"capex_meur_per_machine": 0.01, "footprint_m2": 1900, "kw_per_unit": 0.0,   "dry_room": True,  "spec_workers_per_machine": 0.1, "supp_workers_per_machine": 0.2},
    {"step": "Formation & Degassing",         "throughput_cps": 0.02, "machines": 225,"capex_meur_per_machine": 0.80, "footprint_m2": 2500, "kw_per_unit": 0.0,   "dry_room": False, "spec_workers_per_machine": 0.1, "supp_workers_per_machine": 0.8},
    {"step": "Final Seal, Aging & EoL Test",  "throughput_cps": 0.00347, "machines": 60, "capex_meur_per_machine": 0.85, "footprint_m2": 19000, "kw_per_unit": 100, "dry_room": True,  "spec_workers_per_machine": 0.3, "supp_workers_per_machine": 0.4},
]
for i, r in enumerate(DEFAULT_STEPS, start=1):
    r["order"] = i

# -----------------------------------------
# Default raw materials
# -----------------------------------------
DEFAULT_RAW_MATERIALS = [
    {"name": "Cathode Active", "intro_step": "Mixing & Slurry Prep", "g_per_cell": 128.0, "eur_per_kg": 9.0,  "total_yield": 0.922},
    {"name": "Anode Active",   "intro_step": "Mixing & Slurry Prep", "g_per_cell": 65.38, "eur_per_kg": 5.5,  "total_yield": 0.922},
    {"name": "Conductive Carbon", "intro_step": "Mixing & Slurry Prep", "g_per_cell": 4.0,   "eur_per_kg": 11.59, "total_yield": 0.922},
    {"name": "Binder (PVDF/CMC/SBR)", "intro_step": "Mixing & Slurry Prep","g_per_cell": 6.0, "eur_per_kg": 42.16,"total_yield": 0.922},
    {"name": "Solvent (NMP)",  "intro_step": "Coating & Drying",     "g_per_cell": 2.0,   "eur_per_kg": 3.2,  "total_yield": 0.995},
    {"name": "Aluminum Foil (CC)","intro_step": "Stacking/Winding & Assembly","g_per_cell": 22.0,"eur_per_kg": 1.5,"total_yield": 0.865},
    {"name": "Copper Foil (CC)","intro_step": "Stacking/Winding & Assembly","g_per_cell": 34.0,"eur_per_kg": 8.5,"total_yield": 0.865},
    {"name": "Separator",      "intro_step": "Stacking/Winding & Assembly","g_per_cell": 4.19,"eur_per_kg": 4.8,"total_yield": 0.980},
    {"name": "Pouch/Case Film","intro_step": "Insert in Housing/Pouch","g_per_cell": 249.03,"eur_per_kg": 1.5, "total_yield": 0.990},
    {"name": "Electrolyte",    "intro_step": "Electrolyte Fill & First Seal","g_per_cell": 49.37,"eur_per_kg": 11.5,"total_yield": 0.990},
]

# ------------------------------------------------
# Presets (now include overhead coefficients)
# For Ah/V: derive from prior kWh assumption using 3.7 V default.
# ------------------------------------------------
def _derive_ah_v_from_kwh(kwh: float, v: float = 3.7) -> Tuple[float, float]:
    v = v if v and v > 0 else 3.2
    ah = (kwh * 1000.0) / v
    return ah, v

PRESETS: Dict[str, Dict[str, object]] = {
    "NMC Pouch (Baseline)": {
        "general": {
            "electricity_price_eur_per_kwh": 0.15, "specialist_labor_eur_per_h": 47.39, "support_labor_eur_per_h": 37.62,
            "factory_area_m2": 120_000.0, "building_cost_eur_per_m2": 1234.0, "annual_output_gwh": 10.0,
            # v5.5: preset via Ah/V derived from 0.20 kWh @ 3.7 V
            **(lambda a_v=_derive_ah_v_from_kwh(0.20, 3.2): {"cell_ah": a_v[0], "cell_voltage": a_v[1]})(),
            "working_days": 208.0, "shifts_per_day": 3.0, "shift_hours": 8.0, "oee": 0.80,
        },
        "econ": {
            "project_years": 10, "tax_rate": 0.30, "depreciation_years_equipment": 10, "depreciation_years_building": 50,
            "construction_years": 2, "ramp_years": 1, "capital_cost_wacc": 0.08, "desired_margin": 0.15,
            # overhead coefficients
            "var_oh_on_labor": 0.40, "var_oh_on_depr": 0.20,
            "gsa_on_labor_oh": 0.25, "gsa_on_depr": 0.25,
            "rnd_on_depr": 0.40,
        },
        "steps": DEFAULT_STEPS,
        "materials": DEFAULT_RAW_MATERIALS,
        "note": "Baseline NMC pouch cell process with wet electrolyte."
    },
    "SSB (oxide, illustrative)": {
        "general": {
            "electricity_price_eur_per_kwh": 0.15, "specialist_labor_eur_per_h": 50.0, "support_labor_eur_per_h": 40.0,
            "factory_area_m2": 140_000.0, "building_cost_eur_per_m2": 1400.0, "annual_output_gwh": 8.0,
            **(lambda a_v=_derive_ah_v_from_kwh(0.20, 3.2): {"cell_ah": a_v[0], "cell_voltage": a_v[1]})(),
            "working_days": 208.0, "shifts_per_day": 3.0, "shift_hours": 8.0, "oee": 0.75,
        },
        "econ": {
            "project_years": 12, "tax_rate": 0.30, "depreciation_years_equipment": 10, "depreciation_years_building": 40,
            "construction_years": 2, "ramp_years": 2, "capital_cost_wacc": 0.10, "desired_margin": 0.18,
            "var_oh_on_labor": 0.40, "var_oh_on_depr": 0.20,
            "gsa_on_labor_oh": 0.25, "gsa_on_depr": 0.25,
            "rnd_on_depr": 0.40,
        },
        "steps": [
            {"step": "Dry Powder Mixing & Granulation", "throughput_cps": 0.40, "machines": 16, "capex_meur_per_machine": 6.0, "footprint_m2": 600, "kw_per_unit": 25.0, "dry_room": True, "spec_workers_per_machine": 0.6, "supp_workers_per_machine": 0.8},
            {"step": "Dry Electrode Sheet Forming",     "throughput_cps": 0.20, "machines": 10, "capex_meur_per_machine": 20.0, "footprint_m2": 4000, "kw_per_unit": 1200.0, "dry_room": True, "spec_workers_per_machine": 0.5, "supp_workers_per_machine": 0.6},
            {"step": "Calendering & Slitting",          "throughput_cps": 0.50, "machines": 6,  "capex_meur_per_machine": 4.2, "footprint_m2": 1000, "kw_per_unit": 70.0, "dry_room": True, "spec_workers_per_machine": 0.2, "supp_workers_per_machine": 0.3},
            {"step": "Stacking & Alignment",            "throughput_cps": 0.45, "machines": 30, "capex_meur_per_machine": 1.10, "footprint_m2": 600, "kw_per_unit": 18.0, "dry_room": True, "spec_workers_per_machine": 0.7, "supp_workers_per_machine": 0.5},
            {"step": "Isostatic/Plate Hot Pressing",    "throughput_cps": 0.08, "machines": 40, "capex_meur_per_machine": 3.5,  "footprint_m2": 1200, "kw_per_unit": 50.0, "dry_room": True, "spec_workers_per_machine": 0.5, "supp_workers_per_machine": 0.7},
            {"step": "Contacting/Welding",              "throughput_cps": 0.25, "machines": 20, "capex_meur_per_machine": 7.6,  "footprint_m2": 1500, "kw_per_unit": 12.5, "dry_room": True, "spec_workers_per_machine": 0.4, "supp_workers_per_machine": 0.3},
            {"step": "Insert in Housing/Pouch",         "throughput_cps": 0.45, "machines": 12, "capex_meur_per_machine": 0.9,  "footprint_m2": 600,  "kw_per_unit": 15.0, "dry_room": True, "spec_workers_per_machine": 0.3, "supp_workers_per_machine": 0.4},
            {"step": "Final Seal & EoL Test",           "throughput_cps": 0.01, "machines": 70, "capex_meur_per_machine": 0.9,  "footprint_m2": 20000,"kw_per_unit": 120.0,"dry_room": True, "spec_workers_per_machine": 0.3, "supp_workers_per_machine": 0.5},
        ],
        "materials": [
            {"name": "Cathode Active (SSB)",       "intro_step": "Dry Powder Mixing & Granulation", "g_per_cell": 125.0, "eur_per_kg": 10.0, "total_yield": 0.90},
            {"name": "Anode Active (SSB)",         "intro_step": "Dry Powder Mixing & Granulation", "g_per_cell": 60.0,  "eur_per_kg": 6.0,  "total_yield": 0.90},
            {"name": "Solid Electrolyte (oxide)",  "intro_step": "Dry Powder Mixing & Granulation", "g_per_cell": 40.0,  "eur_per_kg": 60.0, "total_yield": 0.88},
            {"name": "Ceramic/Polymer Separator",  "intro_step": "Dry Electrode Sheet Forming",     "g_per_cell": 6.0,   "eur_per_kg": 15.0, "total_yield": 0.95},
            {"name": "Aluminum Foil (CC)",         "intro_step": "Stacking & Alignment",            "g_per_cell": 22.0,  "eur_per_kg": 1.6,  "total_yield": 0.90},
            {"name": "Copper Foil (CC)",           "intro_step": "Stacking & Alignment",            "g_per_cell": 30.0,  "eur_per_kg": 8.8,  "total_yield": 0.90},
            {"name": "Pouch/Case Film",            "intro_step": "Insert in Housing/Pouch",         "g_per_cell": 240.0, "eur_per_kg": 1.6,  "total_yield": 0.99},
        ],
        "note": "Illustrative SSB process (oxide). Values are placeholders."
    }
}
for preset in PRESETS.values():
    if "steps" in preset:
        for i, r in enumerate(preset["steps"], start=1):
            r.setdefault("order", i)

# -----------------------------------------
# General assumptions & economics
# -----------------------------------------
@dataclass(frozen=True)
class GeneralAssumptions:
    electricity_price_eur_per_kwh: float
    specialist_labor_eur_per_h: float
    support_labor_eur_per_h: float
    factory_area_m2: float
    building_cost_eur_per_m2: float
    annual_output_gwh: float
    # v5.5: store Ah and Voltage, compute kWh/cell internally
    cell_ah: float
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
    # coefficients
    var_oh_on_labor: float
    var_oh_on_depr: float
    gsa_on_labor_oh: float
    gsa_on_depr: float
    rnd_on_depr: float

@dataclass
class BatteryCostModel:
    general: GeneralAssumptions
    econ: Economics
    steps: pd.DataFrame
    raw_materials: pd.DataFrame

    def compute(self) -> Dict[str, object]:
        g, e = self.general, self.econ
        df = self.steps.copy().reset_index(drop=True)

        if "order" in df.columns:
            df = df.sort_values("order").reset_index(drop=True)

        hours_year = g.working_days * g.shifts_per_day * g.shift_hours * g.oee

        # Bottleneck capacity (unchanged)
        df["capacity_cells_per_year"] = df["throughput_cps"] * df["machines"] * 3600.0 * hours_year
        bottleneck_row = df.loc[df["capacity_cells_per_year"].idxmin()]
        line_capacity_cells = float(df["capacity_cells_per_year"].min())

        # v5.5: derive cell kWh from Ah and Voltage
        cell_kwh = max((g.cell_ah * g.cell_voltage) / 1000.0, 1e-12)

        # v5.5: target cells is authoritative for costs/timeline
        annual_cells_target = (g.annual_output_gwh * 1_000_000.0) / cell_kwh
        actual_cells_for_cost = annual_cells_target  # <-- use target for costs
        utilization = min(annual_cells_target / max(line_capacity_cells, 1.0), 1.0)

        # Materials procurement
        mats = self.raw_materials.copy()
        mats["kg_per_cell"] = mats["g_per_cell"].astype(float) / 1000.0
        mats["net_cost_per_cell_eur"] = mats["kg_per_cell"] * mats["eur_per_kg"].astype(float)
        mats["survival"] = mats["total_yield"].clip(lower=1e-6)
        mats["procurement_cost_per_cell_eur"] = mats["net_cost_per_cell_eur"] / mats["survival"]
        materials_procurement_per_cell = float(mats["procurement_cost_per_cell_eur"].sum())

        # Attribute intro-step materials cost to the right step (for the table)
        df["materials_cost_per_cell_total_eur"] = 0.0
        for _, r in mats.iterrows():
            idx = df.index[df["step"] == r["intro_step"]]
            if len(idx):
                df.loc[idx[0], "materials_cost_per_cell_total_eur"] += float(r["procurement_cost_per_cell_eur"])

        # Labor
        df["spec_hours_per_cell"] = df["spec_workers_per_machine"] / (df["throughput_cps"] * 3600.0)
        df["supp_hours_per_cell"] = df["supp_workers_per_machine"] / (df["throughput_cps"] * 3600.0)
        spec_labor_per_cell = float((df["spec_hours_per_cell"] * g.specialist_labor_eur_per_h).sum())
        supp_labor_per_cell = float((df["supp_hours_per_cell"] * g.support_labor_eur_per_h).sum())
        labor_per_cell = spec_labor_per_cell + supp_labor_per_cell

        # Energy
        df["kwh_per_cell"] = df["kw_per_unit"] * df["throughput_cps"] / 3600.0
        df["energy_cost_per_cell_eur"] = df["kwh_per_cell"] * g.electricity_price_eur_per_kwh
        energy_per_cell = float(df["energy_cost_per_cell_eur"].sum())

        # CAPEX & depreciation
        df["step_capex_total_eur"] = df["capex_meur_per_machine"] * 1_000_000.0 * df["machines"]
        total_capital_equipment = float(df["step_capex_total_eur"].sum())
        building_value = g.factory_area_m2 * g.building_cost_eur_per_m2

        annual_depr_equipment = total_capital_equipment / e.depreciation_years_equipment
        annual_depr_building  = building_value / e.depreciation_years_building
        annual_depreciation   = annual_depr_equipment + annual_depr_building
        depreciation_per_cell = annual_depreciation / max(actual_cells_for_cost, 1)

        # Overheads (user coefficients)
        variable_overhead_per_cell = e.var_oh_on_labor * labor_per_cell + e.var_oh_on_depr * depreciation_per_cell
        gsa_per_cell = e.gsa_on_labor_oh * (labor_per_cell + variable_overhead_per_cell) + e.gsa_on_depr * depreciation_per_cell
        rnd_per_cell = e.rnd_on_depr * depreciation_per_cell

        unit_cost_build = (
            materials_procurement_per_cell + energy_per_cell + labor_per_cell +
            variable_overhead_per_cell + depreciation_per_cell + gsa_per_cell + rnd_per_cell
        )
        cost_build_per_kwh = unit_cost_build / cell_kwh

        # Sensitivity (±25%)
        sens_items = []
        def add_s(name, value):
            sens_items.append({"Parameter": name, "Low": unit_cost_build - 0.25*value, "High": unit_cost_build + 0.25*value})
        add_s("Materials", materials_procurement_per_cell)
        add_s("Energy", energy_per_cell)
        add_s("Specialist labor", spec_labor_per_cell)
        add_s("Support labor", supp_labor_per_cell)
        add_s("Variable OH", variable_overhead_per_cell)
        add_s("Depreciation", depreciation_per_cell)
        add_s("GSA", gsa_per_cell)
        add_s("R&D", rnd_per_cell)
        sens_df = pd.DataFrame(sens_items)
        sens_df["Impact"] = sens_df["High"] - sens_df["Low"]
        sens_df = sens_df.sort_values("Impact", ascending=False)

        price_per_cell = unit_cost_build * (1.0 + max(self.econ.desired_margin, 0.0))
        price_per_kwh  = price_per_cell / cell_kwh

        # Project timeline (v5.5 uses TARGET cells for revenue/opex ramp)
        years = list(range(0, self.econ.project_years + 1))
        capex_total = total_capital_equipment + building_value
        capex_outflow_per_year = [0.0]*(self.econ.project_years+1)
        for y in range(min(self.econ.construction_years, self.econ.project_years)):
            capex_outflow_per_year[y] = capex_total / max(self.econ.construction_years,1)

        prod_cells_per_year = [0.0]*(self.econ.project_years+1)
        start_prod_year = self.econ.construction_years
        for y in range(start_prod_year, self.econ.project_years+1):
            t = y - start_prod_year
            ramp_frac = 1.0 if self.econ.ramp_years <= 0 else min(t / self.econ.ramp_years, 1.0)
            prod_cells_per_year[y] = annual_cells_target * ramp_frac  # <-- target cells, not capped by line capacity

        opex_per_cell = unit_cost_build - depreciation_per_cell

        cashflows, cum_cash, cf_cum = [], [], 0.0
        for y in years:
            revenue = prod_cells_per_year[y] * price_per_cell
            opex    = prod_cells_per_year[y] * opex_per_cell
            deprec  = annual_depreciation if y >= start_prod_year else 0.0
            ebit    = revenue - opex - deprec
            tax     = max(ebit, 0.0) * self.econ.tax_rate
            nopat   = ebit - tax
            fcf     = nopat + deprec - capex_outflow_per_year[y]
            disc    = (1.0 + self.econ.capital_cost_wacc) ** y
            npv_y   = fcf / disc
            cashflows.append({"year": y, "revenue": revenue, "opex": opex, "depr": deprec, "tax": tax, "fcf": fcf, "npv": npv_y})
            cf_cum += fcf
            cum_cash.append(cf_cum)

        npv_total = sum(c["npv"] for c in cashflows)
        breakeven_year = next((c["year"] for c, cum in zip(cashflows, cum_cash) if cum >= 0.0), None)

        kpis = {
            "materials_procurement_per_cell_eur": materials_procurement_per_cell,
            "energy_per_cell_eur": energy_per_cell,
            "spec_labor_per_cell_eur": spec_labor_per_cell,
            "supp_labor_per_cell_eur": supp_labor_per_cell,
            "direct_labor_per_cell_eur": labor_per_cell,
            "variable_overhead_per_cell_eur": variable_overhead_per_cell,
            "depreciation_per_cell_eur": depreciation_per_cell,
            "gsa_per_cell_eur": gsa_per_cell,
            "rnd_per_cell_eur": rnd_per_cell,
            "unit_cost_build_eur_per_cell": unit_cost_build,
            "cost_build_per_kwh_eur": cost_build_per_kwh,
            "price_per_cell_eur": price_per_cell,
            "price_per_kwh_eur": price_per_kwh,
            "cell_kwh": cell_kwh,
            "cell_ah": g.cell_ah,
            "cell_voltage": g.cell_voltage,
            "annual_cells_target": annual_cells_target,
            "line_capacity_cells": line_capacity_cells,
            "actual_cells": actual_cells_for_cost,  # used for annual totals on figures
            "bottleneck_step": bottleneck_row["step"],
            "utilization": utilization,
            "capital_equipment_total_eur": total_capital_equipment,
            "building_value_eur": building_value,
            "annual_depreciation_eur": annual_depreciation,
            "npv_total_eur": npv_total,
            "breakeven_year": breakeven_year,
        }

        return {"kpis": kpis, "steps": df, "materials": mats, "cash": pd.DataFrame(cashflows), "sens": sens_df}

    @staticmethod
    def figs(df: pd.DataFrame, mats: pd.DataFrame, k: Dict[str, object], cash: pd.DataFrame, sens_df: pd.DataFrame):
        cells = max(k["actual_cells"], 1.0)
        parts = {
            "Materials": k["materials_procurement_per_cell_eur"] * cells,
            "Energy": k["energy_per_cell_eur"] * cells,
            "Specialist Labor": k["spec_labor_per_cell_eur"] * cells,
            "Support Labor": k["supp_labor_per_cell_eur"] * cells,
            "Variable OH": k["variable_overhead_per_cell_eur"] * cells,
            "GSA": k["gsa_per_cell_eur"] * cells,
            "R&D": k["rnd_per_cell_eur"] * cells,
            "Depreciation": k["depreciation_per_cell_eur"] * cells,
        }
        fig_annual = px.bar(x=list(parts.keys()), y=list(parts.values()), labels={"x": "Component", "y": "€/year"})
        fig_annual.update_layout(yaxis_tickprefix="€ ")

        parts_cell = {
            "Materials": k["materials_procurement_per_cell_eur"],
            "Energy": k["energy_per_cell_eur"],
            "Specialist Labor": k["spec_labor_per_cell_eur"],
            "Support Labor": k["supp_labor_per_cell_eur"],
            "Variable OH": k["variable_overhead_per_cell_eur"],
            "GSA": k["gsa_per_cell_eur"],
            "R&D": k["rnd_per_cell_eur"],
            "Depreciation": k["depreciation_per_cell_eur"],
        }
        fig_cell = px.bar(x=list(parts_cell.keys()), y=list(parts_cell.values()), labels={"x": "Component", "y": "€/cell"})
        fig_cell.add_hline(y=k["unit_cost_build_eur_per_cell"], line_dash="dash", annotation_text="Build cost", annotation_position="top left")

        fig_cap = px.bar(df, x="step", y="capacity_cells_per_year", labels={"step": "Process Step", "capacity_cells_per_year": "Cells/Year"})
        fig_cap.add_hline(y=k["annual_cells_target"], line_dash="dash", annotation_text="Annual Target", annotation_position="top left")

        fig_mat = px.bar(mats, x="name", y="procurement_cost_per_cell_eur", labels={"name": "Material", "procurement_cost_per_cell_eur": "€/cell (procurement)"})

        base = k["unit_cost_build_eur_per_cell"]
        td = sens_df.copy()
        td["LowDelta"] = td["Low"] - base
        td["HighDelta"] = td["High"] - base
        td = td.sort_values("Impact", ascending=False)
        fig_sens = go.Figure()
        fig_sens.add_trace(go.Bar(y=td["Parameter"], x=td["LowDelta"], name="-25%", orientation='h'))
        fig_sens.add_trace(go.Bar(y=td["Parameter"], x=td["HighDelta"], name="+25%", orientation='h'))
        fig_sens.update_layout(barmode="overlay", xaxis_title="Δ €/cell vs base", legend_orientation='h')

        cash = cash.copy()
        fig_time = go.Figure()
        fig_time.add_trace(go.Bar(x=cash["year"], y=-(cash["depr"]+cash["opex"]), name="Costs (excl. CAPEX)", offsetgroup=0))
        fig_time.add_trace(go.Bar(x=cash["year"], y=-cash["fcf"].where(cash["fcf"]<0, 0.0), name="CAPEX/FCF<0", offsetgroup=1))
        fig_time.add_trace(go.Scatter(x=cash["year"], y=cash["revenue"], name="Revenue", mode="lines+markers", yaxis="y"))
        if not math.isnan(k.get("breakeven_year") or float("nan")) and k.get("breakeven_year") is not None:
            be = int(k["breakeven_year"])
            fig_time.add_vline(x=be, line_dash="dash", annotation_text=f"Breakeven Y{be}")
        fig_time.update_layout(yaxis_title="€ / year")

        return fig_annual, fig_cell, fig_cap, fig_mat, fig_sens, fig_time

    @staticmethod
    def table_view(df: pd.DataFrame) -> Tuple[List[Dict[str, str]], List[Dict[str, object]]]:
        df = df.copy()
        if "order" not in df.columns:
            df["order"] = range(1, len(df)+1)
        df = df.sort_values("order")
        show_cols = [
            "order","step","throughput_cps","machines","capex_meur_per_machine","footprint_m2","kw_per_unit","dry_room",
            "spec_workers_per_machine","supp_workers_per_machine",
            "capacity_cells_per_year","materials_cost_per_cell_total_eur","energy_cost_per_cell_eur"
        ]
        for c in ["capacity_cells_per_year", "materials_cost_per_cell_total_eur", "energy_cost_per_cell_eur"]:
            if c not in df.columns: df[c] = 0.0
        round_map = {"order":0, "throughput_cps": 3, "machines":0, "capex_meur_per_machine": 3, "footprint_m2": 0, "kw_per_unit": 5,
                     "capacity_cells_per_year": 0, "materials_cost_per_cell_total_eur": 5, "energy_cost_per_cell_eur":5,
                     "spec_workers_per_machine":2, "supp_workers_per_machine":2}
        df_show = df[show_cols].round(round_map)
        columns = [
            {"name": "Order", "id": "order", "type": "numeric"},
            {"name": "Process Step", "id": "step", "presentation": "input"},
            {"name": "Throughput (cells/s per machine)", "id": "throughput_cps", "type": "numeric"},
            {"name": "Machines (count)", "id": "machines", "type": "numeric"},
            {"name": "CAPEX (M€ / machine)", "id": "capex_meur_per_machine", "type": "numeric"},
            {"name": "Footprint (m² per machine)", "id": "footprint_m2", "type": "numeric"},
            {"name": "Energy Consumption (kW per machine)", "id": "kw_per_unit", "type": "numeric"},
            {"name": "Dry Room", "id": "dry_room", "presentation": "dropdown"},
            {"name": "Specialists / machine", "id": "spec_workers_per_machine", "type": "numeric"},
            {"name": "Support / machine", "id": "supp_workers_per_machine", "type": "numeric"},
            {"name": "Capacity (Cells/Year)", "id": "capacity_cells_per_year"},
            {"name": "Intro-step Materials €/Cell", "id": "materials_cost_per_cell_total_eur"},
            {"name": "Energy €/Cell", "id": "energy_cost_per_cell_eur"},
        ]
        return columns, df_show.to_dict("records")

# =========================================
# UI
# =========================================
app = Dash(__name__)
server = app.server
app.title = "Cell Cost Estimator v5.5 (EU)"

def logo_src() -> str:
    if os.path.exists(os.path.join("assets", "logo.png")):
        return "/assets/logo.png"
    if os.path.exists(os.path.join("assets", "logo.svg")):
        return "/assets/logo.svg"
    svg = """
    <svg xmlns='http://www.w3.org/2000/svg' width='200' height='44'>
      <rect width='100%' height='100%' fill='#2563eb'/>
      <text x='50%' y='50%' dominant-baseline='middle' text-anchor='middle'
            font-family='Inter, Arial, sans-serif' font-size='16' fill='white'>Cell Cost Estimator v5.5</text>
    </svg>
    """
    return "data:image/svg+xml;utf8," + quote(svg)

def num_input(id_obj, value, step="any", min_=None, max_=None, width="120px"):
    return dcc.Input(id=id_obj, type="number", value=value, step=step, min=min_, max=max_, style={"width": width})

# ---- Preset selector UI ----
preset_controls = html.Div([
    html.H3("Presets"),
    html.Div([
        dcc.Dropdown(
            id="preset_select",
            options=[{"label": k, "value": k} for k in PRESETS.keys()],
            value="NMC Pouch (Baseline)",
            clearable=False,
            style={"width": "100%"}
        ),
        html.Button("Apply preset", id="apply_preset", n_clicks=0, style={
            "marginTop": "6px", "width": "100%", "padding": "8px",
            "backgroundColor": "#111827", "color": "white", "border": "none", "borderRadius": "6px", "cursor": "pointer"
        }),
    ]),
    html.Div(id="preset_note", style={"fontSize": "12px", "color": "#555", "marginTop": "6px"})
], style={"marginBottom": "8px"})

def build_materials_columns(steps_rows):
    step_labels = [r["step"] for r in (steps_rows or []) if r.get("step")]
    options = [{"label": s, "value": s} for s in step_labels] or [{"label": DEFAULT_STEPS[0]["step"], "value": DEFAULT_STEPS[0]["step"]}]
    return [
        {"name": "Name", "id": "name", "presentation": "input"},
        {"name": "Intro Step", "id": "intro_step", "presentation": "dropdown"},
        {"name": "g/cell", "id": "g_per_cell", "type": "numeric", "format": {"specifier": ".3f"}},
        {"name": "€/kg", "id": "eur_per_kg", "type": "numeric", "format": {"specifier": ".3f"}},
        {"name": "Total Yield (0–1)", "id": "total_yield", "type": "numeric", "format": {"specifier": ".4f"}},
    ], {"intro_step": {"options": options}}

# Left panel with reorder & new cost allocation inputs
left_inputs = html.Div([
    preset_controls,

    html.H3("General Assumptions (EU, Cell line)"),
    html.Div([
        html.Div([html.Label("Energy Price (€/kWh)"),                 num_input("el_price", 0.15, step=0.01, min_=0)]),
        html.Div([html.Label("Specialist Labor (€/h)"),               num_input("labor_spec", 47.39, step=0.01, min_=0)]),
        html.Div([html.Label("Support Labor (€/h)"),                  num_input("labor_sup", 37.62, step=0.01, min_=0)]),
        html.Div([html.Label("Plant Area (m²)"),                      num_input("area", 120000, step=0.01, min_=0)]),
        html.Div([html.Label("Building Cost (€/m²)"),                 num_input("bldg_cost", 1234.0, step=0.01, min_=0)]),
        html.Div([html.Label("Annual Output (GWh)"),                  num_input("gwh", 10.0, step=0.01, min_=0.1)]),

        # v5.5: New capacity inputs
        html.Div([html.Label("Cell Capacity (Ah)"),                   num_input("cell_ah", 17.92, step=0.001, min_=0.001)]),
        html.Div([html.Label("Nominal Voltage (V)"),                  num_input("cell_v", 3.7, step=0.01, min_=0.5)]),

        html.Div([html.Label("Working Days / yr"),                    num_input("days", 208, step=0.01, min_=50)]),
        html.Div([html.Label("Shifts / day"),                         num_input("shifts", 3, step=0.01, min_=1)]),
        html.Div([html.Label("Hours / shift"),                        num_input("hshift", 8, step=0.01, min_=0.5)]),
        html.Div([html.Label("OEE (0–1)"),                            num_input("oee", 0.80, step=0.01, min_=0, max_=1)]),
    ], style={"display": "grid", "gridTemplateColumns": "repeat(2, minmax(200px, 1fr))", "gap": "8px"}),

    html.Hr(),
    html.H3("Economics"),
    html.Div([
        html.Div([html.Label("Project Duration (years)"),             num_input("proj_years", 10, step=1, min_=1)]),
        html.Div([html.Label("Tax Rate (0–1)"),                        num_input("tax", 0.30, step=0.01, min_=0, max_=1)]),
        html.Div([html.Label("Depreciation (equip) years"),            num_input("dep_equip", 10, step=1, min_=1)]),
        html.Div([html.Label("Depreciation (building) years"),         num_input("dep_bldg", 50, step=1, min_=1)]),
        html.Div([html.Label("Construction Duration (years)"),         num_input("build_years", 2, step=1, min_=0)]),
        html.Div([html.Label("Ramp-up Duration (years)"),              num_input("ramp_years", 1, step=1, min_=0)]),
        html.Div([html.Label("Capital Cost / WACC (0–1)"),             num_input("wacc", 0.08, step=0.005, min_=0.0, max_=1.0)]),
        html.Div([html.Label("Desired Margin (0–1)"),                  num_input("margin", 0.15, step=0.01, min_=0.0, max_=1.0)]),
    ], style={"display": "grid", "gridTemplateColumns": "repeat(2, minmax(200px, 1fr))", "gap": "8px"}),

    html.Hr(),
    html.H3("Cost Allocation Coefficients"),
    html.Div([
        html.Div([html.Label("Variable OH on Labor (0–1)"),           num_input("var_oh_lab", 0.40, step=0.01, min_=0, max_=2.0)]),
        html.Div([html.Label("Variable OH on Depreciation (0–1)"),     num_input("var_oh_depr", 0.20, step=0.01, min_=0, max_=2.0)]),
        html.Div([html.Label("GSA on (Labor + OH) (0–1)"),             num_input("gsa_labor_oh", 0.25, step=0.01, min_=0, max_=2.0)]),
        html.Div([html.Label("GSA on Depreciation (0–1)"),             num_input("gsa_depr", 0.25, step=0.01, min_=0, max_=2.0)]),
        html.Div([html.Label("R&D on Depreciation (0–1)"),             num_input("rnd_depr", 0.40, step=0.01, min_=0, max_=2.0)]),
    ], style={"display": "grid", "gridTemplateColumns": "repeat(2, minmax(200px, 1fr))", "gap": "8px"}),

    html.Hr(),
    html.H3("Raw Materials (g/cell, €/kg, Total Yield) – cell only"),
    dash_table.DataTable(
        id="materials_table",
        columns=build_materials_columns(DEFAULT_STEPS)[0],
        data=DEFAULT_RAW_MATERIALS,
        editable=True,
        dropdown=build_materials_columns(DEFAULT_STEPS)[1],
        row_deletable=True,
        style_table={"overflowX": "auto"},
        style_cell={"padding": "6px", "minWidth": 120, "whiteSpace": "normal"},
        page_size=12,
    ),
    html.Button("Add material", id="add_mat", n_clicks=0, style={"marginTop": "8px"}),

    html.Hr(),
    html.H3("Process Steps (add/remove & reorder)"),
    html.Div([
        html.Button("Move Up", id="move_up", n_clicks=0, style={"marginRight": "6px"}),
        html.Button("Move Down", id="move_down", n_clicks=0),
    ], style={"marginBottom": "6px"}),
    dash_table.DataTable(
        id="steps_table",
        columns=[
            {"name": "Order", "id": "order", "type": "numeric"},
            {"name": "Process Step", "id": "step", "presentation": "input"},
            {"name": "Throughput (cells/s per machine)", "id": "throughput_cps", "type": "numeric"},
            {"name": "Machines (count)", "id": "machines", "type": "numeric"},
            {"name": "CAPEX (M€ / machine)", "id": "capex_meur_per_machine", "type": "numeric"},
            {"name": "Footprint (m² per machine)", "id": "footprint_m2", "type": "numeric"},
            {"name": "Energy Consumption (kW per machine)", "id": "kw_per_unit", "type": "numeric"},
            {"name": "Dry Room", "id": "dry_room", "presentation": "dropdown"},
            {"name": "Specialists / machine", "id": "spec_workers_per_machine", "type": "numeric"},
            {"name": "Support / machine", "id": "supp_workers_per_machine", "type": "numeric"},
        ],
        data=DEFAULT_STEPS,
        editable=True,
        dropdown={"dry_room": {"options": [{"label": "Yes", "value": True}, {"label": "No", "value": False}]}},
        row_deletable=True,
        row_selectable="single",
        style_table={"overflowX": "auto"},
        style_cell={"padding": "6px", "minWidth": 120, "whiteSpace": "normal"},
        page_size=12,
    ),
    html.Div([html.Button("Add step", id="add_step", n_clicks=0)], style={"marginTop": "8px"}),

    html.Button(
        "Calculate", id="run", n_clicks=0,
        style={"position": "sticky", "bottom": "0", "width": "100%", "padding": "14px 0", "fontSize": "18px",
               "backgroundColor": "#2563eb", "color": "white", "border": "none", "borderRadius": "6px",
               "cursor": "pointer", "marginTop": "10px", "zIndex": "100"}
    ),
], style={"height": "100%", "overflowY": "auto", "padding": "8px", "borderRight": "1px solid #eee"})

# Right panel – Outputs
right_outputs = html.Div([
    html.H2("Analysis (Cell only)"),
    html.Div(id="kpi_row", style={"display": "flex", "gap": "12px", "flexWrap": "wrap"}),

    html.Div([
        html.Div([html.H4("Annual Steady-State Cost Breakdown (€/yr)"), dcc.Graph(id="fig_annual")], style={"width": "48%", "display": "inline-block", "verticalAlign": "top"}),
        html.Div([html.H4("Single-Cell Cost Breakdown (€/cell)"), dcc.Graph(id="fig_cell")], style={"width": "48%", "display": "inline-block", "verticalAlign": "top"}),
    ], style={"display": "flex", "justifyContent": "space-between", "gap": "12px", "flexWrap": "wrap"}),

    html.Div([
        html.Div([html.H4("Production Capacity (Cells/Year) & Annual Target"), dcc.Graph(id="fig_cap")], style={"width": "48%", "display": "inline-block", "verticalAlign": "top"}),
        html.Div([html.H4("Materials Procurement Breakdown (€/cell)"), dcc.Graph(id="fig_mat")], style={"width": "48%", "display": "inline-block", "verticalAlign": "top"}),
    ], style={"display": "flex", "justifyContent": "space-between", "gap": "12px", "flexWrap": "wrap"}),

    html.Div([
        html.Div([html.H4("Tornado Sensitivity (±25% — €/cell)"), dcc.Graph(id="fig_sens")], style={"width": "48%", "display": "inline-block", "verticalAlign": "top"}),
        html.Div([html.H4("Project Timeline: Costs (bars) & Revenue (line)"), dcc.Graph(id="fig_time")], style={"width": "48%", "display": "inline-block", "verticalAlign": "top"}),
    ], style={"display": "flex", "justifyContent": "space-between", "gap": "12px", "flexWrap": "wrap"}),

    html.H4("Per-step Overview (incl. intro-step materials)"),
    dash_table.DataTable(
        id="steps_table_main",
        page_size=15,
        style_table={"overflowX": "auto"},
        style_cell={"padding": "6px", "minWidth": 90, "maxWidth": 280, "whiteSpace": "normal"}
    ),

    html.Details([
        html.Summary("Notes: formulas & assumptions (click to expand)"),
        html.Ul([
            html.Li("kWh per cell = Ah × V / 1000 (derived). Annual target cells = Annual GWh × 1e6 / kWh per cell."),
            html.Li("All annualized costs and timeline use the TARGET annual cells (not capped by bottleneck)."),
            html.Li("Capacity by step = throughput_cps × machines × 3600 × hours_year; bottleneck limits line capacity (for display)."),
            html.Li("Labor per cell (step) = (specialists_per_machine × wage_spec + support_per_machine × wage_support) / (throughput_cps × 3600)."),
            html.Li("Equipment CAPEX = capex_per_machine × machines. Building = area × €/m². Straight-line depreciation (equip, building)."),
            html.Li("Variable OH = var_oh_lab × Labor + var_oh_depr × Depreciation (user-set)."),
            html.Li("GSA = gsa_on_labor_oh × (Labor + OH) + gsa_on_depr × Depreciation (user-set)."),
            html.Li("R&D = rnd_on_depr × Depreciation (user-set)."),
            html.Li("Materials intro-step options always reflect the current steps table; steps can be reordered."),
        ])
    ], open=False)
], style={"height": "100%", "overflowY": "auto", "padding": "10px 12px"})

# Layout
app.layout = html.Div([
    html.Div([
        html.Img(
            src=logo_src(),
            alt="PEM",
            style={"height": "44px", "marginRight": "12px", "display": "block"},
            draggable="false"
        ),
        html.H2("Cell Manufacturing Cost (EU) — v5.5", style={"margin": 0, "alignSelf": "center"}),
    ], style={"display": "flex", "alignItems": "center", "gap": "8px", "margin": "8px 0 12px"}),

    html.Div([left_inputs, right_outputs], style={
        "display": "grid", "gridTemplateColumns": "560px 1fr", "gap": "12px",
        "height": "calc(100vh - 80px)", "alignItems": "stretch"
    })
], style={"fontFamily": "Inter, system-ui, Arial", "maxWidth": "1500px", "margin": "0 auto", "padding": "0 8px 8px"})

# =========================================
# Callbacks
# =========================================

@app.callback(
    Output("materials_table", "data"),
    Input("add_mat", "n_clicks"),
    State("materials_table", "data"),
    State("steps_table", "data"),
    prevent_initial_call=True,
)
def add_material(n, rows, steps_rows):
    rows = rows or []
    first_step = (steps_rows or DEFAULT_STEPS)[0]["step"]
    rows.append({"name": "New Material", "intro_step": first_step, "g_per_cell": 0.0, "eur_per_kg": 0.0, "total_yield": 1.0})
    return rows

@app.callback(
    Output("steps_table", "data"),
    Input("add_step", "n_clicks"),
    State("steps_table", "data"),
    prevent_initial_call=True,
)
def add_step(n, rows):
    rows = rows or []
    next_order = 1 + max([r.get("order", i+1) for i, r in enumerate(rows)], default=0)
    rows.append({"order": next_order, "step": "New Step", "throughput_cps": 0.10, "machines": 1, "capex_meur_per_machine": 0.10,
                 "footprint_m2": 50, "kw_per_unit": 0.001, "dry_room": False, "spec_workers_per_machine": 0.2, "supp_workers_per_machine": 0.2})
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
    if not rows or not sel_rows:
        return rows
    idx = sel_rows[0]
    rows = sorted(rows, key=lambda r: r.get("order", 9999))
    if idx < 0 or idx >= len(rows):
        return rows

    trig = ctx.triggered_id
    if trig == "move_up" and idx > 0:
        rows[idx]["order"], rows[idx-1]["order"] = rows[idx-1]["order"], rows[idx]["order"]
    elif trig == "move_down" and idx < len(rows)-1:
        rows[idx]["order"], rows[idx+1]["order"] = rows[idx+1]["order"], rows[idx]["order"]

    rows = sorted(rows, key=lambda r: r.get("order", 9999))
    for i, r in enumerate(rows, start=1):
        r["order"] = i
    return rows

# ---- Apply Preset (now sets Ah & V instead of kWh) ----
@app.callback(
    # General/Econ inputs:
    Output("el_price", "value", allow_duplicate=True),
    Output("labor_spec", "value", allow_duplicate=True),
    Output("labor_sup", "value", allow_duplicate=True),
    Output("area", "value", allow_duplicate=True),
    Output("bldg_cost", "value", allow_duplicate=True),
    Output("gwh", "value", allow_duplicate=True),
    Output("cell_ah", "value", allow_duplicate=True),
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
    # coeff inputs:
    Output("var_oh_lab", "value", allow_duplicate=True),
    Output("var_oh_depr", "value", allow_duplicate=True),
    Output("gsa_labor_oh", "value", allow_duplicate=True),
    Output("gsa_depr", "value", allow_duplicate=True),
    Output("rnd_depr", "value", allow_duplicate=True),
    # Tables:
    Output("materials_table", "data", allow_duplicate=True),
    Output("steps_table", "data", allow_duplicate=True),
    # Note
    Output("preset_note", "children", allow_duplicate=True),
    Input("apply_preset", "n_clicks"),
    State("preset_select", "value"),
    prevent_initial_call=True
)
def apply_preset(n, preset_key):
    p = PRESETS.get(preset_key, PRESETS["NMC Pouch (Baseline)"])
    g = p["general"]; ec = p["econ"]
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
    note = f"Preset applied: {preset_key}. {p.get('note','')}"

    return (
        g["electricity_price_eur_per_kwh"],
        g["specialist_labor_eur_per_h"],
        g["support_labor_eur_per_h"],
        g["factory_area_m2"],
        g["building_cost_eur_per_m2"],
        g["annual_output_gwh"],
        g["cell_ah"],
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
        # coeffs
        ec["var_oh_on_labor"],
        ec["var_oh_on_depr"],
        ec["gsa_on_labor_oh"],
        ec["gsa_on_depr"],
        ec["rnd_on_depr"],
        mats,
        steps,
        note
    )

# --- Keep materials intro-step dropdown in sync with steps, and sanitize data ---
@app.callback(
    Output("materials_table", "columns", allow_duplicate=True),
    Output("materials_table", "dropdown", allow_duplicate=True),
    Output("materials_table", "data", allow_duplicate=True),
    Input("steps_table", "data"),
    State("materials_table", "data"),
    prevent_initial_call=True
)
def sync_materials_intro_step(steps_rows, materials_rows):
    columns, dropdown = build_materials_columns(steps_rows)
    rows = materials_rows or []
    step_names = [r["step"] for r in (steps_rows or []) if r.get("step")]
    if not step_names:
        step_names = [DEFAULT_STEPS[0]["step"]]
    fallback = step_names[0]
    for r in rows:
        if r.get("intro_step") not in step_names:
            r["intro_step"] = fallback
    return columns, dropdown, rows

@app.callback(
    Output("kpi_row", "children"),
    Output("fig_annual", "figure"),
    Output("fig_cell", "figure"),
    Output("fig_cap", "figure"),
    Output("fig_mat", "figure"),
    Output("fig_sens", "figure"),
    Output("fig_time", "figure"),
    Output("steps_table_main", "columns"),
    Output("steps_table_main", "data"),
    Input("run", "n_clicks"),
    State("el_price", "value"),
    State("labor_spec", "value"),
    State("labor_sup", "value"),
    State("area", "value"),
    State("bldg_cost", "value"),
    State("gwh", "value"),
    State("cell_ah", "value"),
    State("cell_v", "value"),
    State("days", "value"),
    State("shifts", "value"),
    State("hshift", "value"),
    State("oee", "value"),
    State("proj_years", "value"),
    State("tax", "value"),
    State("dep_equip", "value"),
    State("dep_bldg", "value"),
    State("build_years", "value"),
    State("ramp_years", "value"),
    State("wacc", "value"),
    State("margin", "value"),
    # coeff states
    State("var_oh_lab", "value"),
    State("var_oh_depr", "value"),
    State("gsa_labor_oh", "value"),
    State("gsa_depr", "value"),
    State("rnd_depr", "value"),
    State("materials_table", "data"),
    State("steps_table", "data"),
    prevent_initial_call=True
)
def run_calc(_, el_price, labor_spec, labor_sup, area, bldg_cost, gwh, cell_ah, cell_v, days, shifts, hshift, oee,
             proj_years, tax, dep_equip, dep_bldg, build_years, ramp_years, wacc, margin,
             var_oh_lab, var_oh_depr, gsa_labor_oh, gsa_depr, rnd_depr,
             materials_rows, steps_rows):

    general = GeneralAssumptions(
        electricity_price_eur_per_kwh=float(el_price or 0.09),
        specialist_labor_eur_per_h=float(labor_spec or 47.39),
        support_labor_eur_per_h=float(labor_sup or 37.62),
        factory_area_m2=float(area or 120000.0),
        building_cost_eur_per_m2=float(bldg_cost or 1234.23),
        annual_output_gwh=float(gwh or 10.0),
        cell_ah=float(cell_ah or 17.92),
        cell_voltage=float(cell_v or 3.2),
        working_days=float(days or 208.0),
        shifts_per_day=float(shifts or 3.0),
        shift_hours=float(hshift or 8.0),
        oee=float(oee or 0.80),
    )

    econ = Economics(
        project_years=int(proj_years or 15),
        tax_rate=float(tax or 0.30),
        depreciation_years_equipment=int(dep_equip or 10),
        depreciation_years_building=int(dep_bldg or 20),
        construction_years=int(build_years or 2),
        ramp_years=int(ramp_years or 1),
        capital_cost_wacc=float(wacc or 0.08),
        desired_margin=float(margin or 0.15),
        var_oh_on_labor=float(var_oh_lab or 0.40),
        var_oh_on_depr=float(var_oh_depr or 0.20),
        gsa_on_labor_oh=float(gsa_labor_oh or 0.25),
        gsa_on_depr=float(gsa_depr or 0.25),
        rnd_on_depr=float(rnd_depr or 0.40),
    )

    steps = pd.DataFrame(steps_rows or DEFAULT_STEPS).copy()
    if "order" not in steps.columns:
        steps["order"] = range(1, len(steps)+1)
    steps = steps.sort_values("order")

    mats = pd.DataFrame(materials_rows or DEFAULT_RAW_MATERIALS).copy()
    step_names = set(steps["step"].tolist())
    if len(steps):
        fallback = steps["step"].iloc[0]
        mats["intro_step"] = mats["intro_step"].apply(lambda x: x if x in step_names else fallback)

    model = BatteryCostModel(general=general, econ=econ, steps=steps, raw_materials=mats)
    result = model.compute()
    k = result["kpis"]; df = result["steps"]; mats = result["materials"]; cash = result["cash"]; sens_df = result["sens"]

    def card(label, value, suffix=""):
        return html.Div([
            html.Div(label, style={"fontSize": "12px", "color": "#666"}),
            html.Div(f"{value}{suffix}", style({"fontSize": "20px", "fontWeight": 600})),
        ], style={"border": "1px solid #eee", "borderRadius": "12px", "padding": "10px 12px", "minWidth": "210px"})

    # Fix small typo above (style call)
    def card(label, value, suffix=""):
        return html.Div([
            html.Div(label, style={"fontSize": "12px", "color": "#666"}),
            html.Div(f"{value}{suffix}", style={"fontSize": "20px", "fontWeight": 600}),
        ], style={"border": "1px solid #eee", "borderRadius": "12px", "padding": "10px 12px", "minWidth": "210px"})

    kpi_children = [
        card("Build Cost / Cell", f"{k['unit_cost_build_eur_per_cell']:.4f}", " €"),
        card("Build Cost / kWh", f"{k['cost_build_per_kwh_eur']:.2f}", " €/kWh"),
        card("Price / Cell (margin)", f"{k['price_per_cell_eur']:.4f}", " €"),
        card("Price / kWh (margin)", f"{k['price_per_kwh_eur']:.2f}", " €/kWh"),
        card("Materials / Cell", f"{k['materials_procurement_per_cell_eur']:.4f}", " €"),
        card("Spec Labor / Cell", f"{k['spec_labor_per_cell_eur']:.4f}", " €"),
        card("Support Labor / Cell", f"{k['supp_labor_per_cell_eur']:.4f}", " €"),
        card("Variable OH / Cell", f"{k['variable_overhead_per_cell_eur']:.4f}", " €"),
        card("GSA / Cell", f"{k['gsa_per_cell_eur']:.4f}", " €"),
        card("R&D / Cell", f"{k['rnd_per_cell_eur']:.4f}", " €"),
        card("Depreciation / Cell", f"{k['depreciation_per_cell_eur']:.4f}", " €"),
        card("Annual Cells Target", f"{k['annual_cells_target']:.0f}"),
        card("Line Capacity (Cells)", f"{k['line_capacity_cells']:.0f}"),
        card("Bottleneck", str(k['bottleneck_step'])),
        card("NPV (project)", f"{k['npv_total_eur']/1e6:.2f}", " M€"),
        card("Breakeven Year", str(k['breakeven_year'] if k['breakeven_year'] is not None else 'n/a')),
    ]

    fig_annual, fig_cell, fig_cap, fig_mat, fig_sens, fig_time = BatteryCostModel.figs(df, mats, k, cash, sens_df)
    columns, data = BatteryCostModel.table_view(df)
    return kpi_children, fig_annual, fig_cell, fig_cap, fig_mat, fig_sens, fig_time, columns, data


if __name__ == "__main__":
    app.run(debug=True)
