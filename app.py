from dataclasses import dataclass, field
from typing import List, Dict, Tuple
from dash import Dash, html, dcc, Input, Output, State, dash_table
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os
from urllib.parse import quote

# =========================================
# Domain Model (OO)
# =========================================

STEPS = [
    "Mixing",
    "Coating & Drying",
    "Calendering",
    "Slitting",
    "Post-Drying",
    "Winding",
    "Contacting",
    "Inserting & Closing of lid",
    "Electrolyte Filling",
    "Wetting",
    "Forming & Degassing",
    "Closing",
    "Aging",
    "End-of-Line Test",
]

# Include defaults for capex_m_eur_per_gwh, life_years, dry_room
# (Set CAPEX defaults to 0 to avoid surprising totals; adjust as you like.)
DEFAULT_STEP_PARAMS: Dict[str, Dict[str, float]] = {
    "Mixing":                 {"time_s": 3.0,  "kwh": 0.003, "yield": 0.995, "throughput": 1200, "area_share": 6,  "capex_m_eur_per_gwh": 2.3, "life_years": 10.0, "dry_room": 0},
    "Coating & Drying":      {"time_s": 9.0,  "kwh": 0.020, "yield": 0.990, "throughput": 900,  "area_share": 12, "capex_m_eur_per_gwh": 5.0, "life_years": 10.0, "dry_room": 1},
    "Calendering":           {"time_s": 2.0,  "kwh": 0.002, "yield": 0.998, "throughput": 2000, "area_share": 7,  "capex_m_eur_per_gwh": 3.0, "life_years": 10.0, "dry_room": 0},
    "Slitting":              {"time_s": 1.2,  "kwh": 0.001, "yield": 0.998, "throughput": 2500, "area_share": 5,  "capex_m_eur_per_gwh": 1.2, "life_years": 10.0, "dry_room": 0},
    "Post-Drying":           {"time_s": 4.5,  "kwh": 0.006, "yield": 0.996, "throughput": 1100, "area_share": 7,  "capex_m_eur_per_gwh": 1.9, "life_years": 10.0, "dry_room": 1},
    "Winding":               {"time_s": 6.0,  "kwh": 0.004, "yield": 0.995, "throughput": 1000, "area_share": 8,  "capex_m_eur_per_gwh": 6.0, "life_years": 10.0, "dry_room": 0},
    "Contacting":            {"time_s": 2.5,  "kwh": 0.001, "yield": 0.998, "throughput": 1800, "area_share": 5,  "capex_m_eur_per_gwh": 2.0, "life_years": 10.0, "dry_room": 0},
    "Inserting & Closing of lid": {"time_s": 3.0,  "kwh": 0.001, "yield": 0.997, "throughput": 1500, "area_share": 7,  "capex_m_eur_per_gwh": 4.0, "life_years": 10.0, "dry_room": 0},
    "Electrolyte Filling":   {"time_s": 4.0,  "kwh": 0.003, "yield": 0.996, "throughput": 1200, "area_share": 6,  "capex_m_eur_per_gwh": 1.1, "life_years": 10.0, "dry_room": 1},
    "Wetting":               {"time_s": 8.0,  "kwh": 0.002, "yield": 0.995, "throughput": 800,  "area_share": 7,  "capex_m_eur_per_gwh": 1.0, "life_years": 10.0, "dry_room": 1},
    "Forming & Degassing":   {"time_s": 30.0, "kwh": 0.030, "yield": 0.990, "throughput": 500,  "area_share": 20, "capex_m_eur_per_gwh": 15.0, "life_years": 10.0, "dry_room": 1},
    "Closing":               {"time_s": 1.0,  "kwh": 0.001, "yield": 0.999, "throughput": 2600, "area_share": 5,  "capex_m_eur_per_gwh": 5.1, "life_years": 10.0, "dry_room": 0},
    "Aging":                 {"time_s": 15.0, "kwh": 0.010, "yield": 0.995, "throughput": 700,  "area_share": 4,  "capex_m_eur_per_gwh": 1.7, "life_years": 10.0, "dry_room": 0},
    "End-of-Line Test":      {"time_s": 5.0,  "kwh": 0.002, "yield": 0.998, "throughput": 1400, "area_share": 6,  "capex_m_eur_per_gwh": 0.5, "life_years": 10.0, "dry_room": 0},
}

DEFAULTS = dict(
    electricity_price_eur_per_kwh = 0.22,
    labor_rate_eur_per_h           = 35.0,
    overhead_rate                  = 0.30,
    factory_area_m2                = 10000,
    rent_eur_per_m2_month          = 12.0,
    annual_output_gwh              = 1.0,
    cell_capacity_wh               = 60.0,
    working_days                   = 330,
    shifts_per_day                 = 3,
    shift_hours                    = 7.5,
    oee                            = 0.80,
    raw_mat_cost_eur_per_kwh       = 45.0,
    # NEW economics
    margin_rate                    = 0.20,   # 20% markup over cost incl. depreciation
    discount_rate                  = 0.10,   # 10% discount rate
    project_years                  = 10,     # horizon for NPV/payback
    dry_room_cost_eur_per_m2_year  = 150.0,  # annual dry-room opex per m²
)

# -----------------------------------------
# Dataclasses
# -----------------------------------------

@dataclass(frozen=True)
class StepParam:
    name: str
    time_s: float
    kwh: float
    yield_: float       # 0..1
    throughput: float
    area_share: float
    capex_m_eur_per_gwh: float  # NEW: M€/GWh
    life_years: float
    dry_room: bool

@dataclass(frozen=True)
class GeneralAssumptions:
    electricity_price_eur_per_kwh: float
    labor_rate_eur_per_h: float
    overhead_rate: float
    factory_area_m2: float
    rent_eur_per_m2_month: float
    annual_output_gwh: float
    cell_capacity_wh: float
    working_days: float
    shifts_per_day: float
    shift_hours: float
    oee: float
    raw_mat_cost_eur_per_kwh: float
    margin_rate: float
    discount_rate: float
    project_years: int
    dry_room_cost_eur_per_m2_year: float

@dataclass
class BatteryCostModel:
    general: GeneralAssumptions
    steps: List[StepParam] = field(default_factory=list)

    def steps_df(self) -> pd.DataFrame:
        rows = []
        for s in self.steps:
            rows.append({
                "step": s.name,
                "time_s": float(s.time_s),
                "kwh": float(s.kwh),
                "yield": float(s.yield_),
                "throughput": float(s.throughput),
                "area_share": float(s.area_share),
                "capex_m_eur_per_gwh": float(s.capex_m_eur_per_gwh),
                "life_years": float(s.life_years),
                "dry_room": bool(s.dry_room),
            })
        return pd.DataFrame(rows)

    def compute(self) -> Dict[str, object]:
        g = self.general
        df = self.steps_df().copy()

        # Derived volumes
        cell_kwh = g.cell_capacity_wh / 1000.0
        annual_cells_target = (g.annual_output_gwh * 1_000_000.0) / max(cell_kwh, 1e-12)
        hours_year = g.working_days * g.shifts_per_day * g.shift_hours * g.oee

        # Capacity + bottleneck
        df["capacity_cells_per_year"] = df["throughput"] * hours_year
        bottleneck_row = df.loc[df["capacity_cells_per_year"].idxmin()]
        line_capacity_cells = float(df["capacity_cells_per_year"].min())
        actual_cells = min(annual_cells_target, max(line_capacity_cells, 1.0))

        # Yield
        df["cum_yield"] = df["yield"].cumprod()
        total_yield = float(df["yield"].prod())

        # Base costs per cell
        time_h = df["time_s"] / 3600.0
        df["labor_cost_per_cell"]  = time_h * g.labor_rate_eur_per_h
        df["energy_cost_per_cell"] = df["kwh"] * g.electricity_price_eur_per_kwh
        labor_cost   = float(df["labor_cost_per_cell"].sum())
        energy_cost  = float(df["energy_cost_per_cell"].sum())
        overhead_cost = g.overhead_rate * labor_cost

        # Facility rent per cell (area-weighted)
        rent_year = g.factory_area_m2 * g.rent_eur_per_m2_month * 12.0
        df["area_weight"] = df["area_share"] / df["area_share"].sum()
        df["facility_cost_per_cell"] = (rent_year * df["area_weight"]) / max(actual_cells, 1.0)
        facility_cost = float(df["facility_cost_per_cell"].sum())

        # Dry-room opex per cell (only if dry_room True)
        df["dryroom_cost_per_cell"] = df.apply(
            lambda r: (g.factory_area_m2 * r["area_weight"] * g.dry_room_cost_eur_per_m2_year) / max(actual_cells, 1.0)
                      if bool(r["dry_room"]) else 0.0,
            axis=1
        )
        dry_room_cost = float(df["dryroom_cost_per_cell"].sum())

        # Raw materials per cell (adjusted by total yield)
        raw_mat_unit_cost = g.raw_mat_cost_eur_per_kwh * cell_kwh / max(total_yield, 1e-12)

        # CAPEX scaling (M€/GWh → €) per step, then depreciation per cell
        df["step_capex_total_eur"] = df["capex_m_eur_per_gwh"] * 1_000_000.0 * g.annual_output_gwh
        df["depr_per_cell"] = df.apply(
            lambda r: (r["step_capex_total_eur"] / max(r["life_years"], 1e-9)) / max(actual_cells, 1.0),
            axis=1
        )
        depreciation_cost = float(df["depr_per_cell"].sum())

        # Unit costs
        unit_total_cash = labor_cost + overhead_cost + energy_cost + facility_cost + dry_room_cost + raw_mat_unit_cost
        unit_total_with_depr = unit_total_cash + depreciation_cost

        utilization = min(annual_cells_target / max(line_capacity_cells, 1.0), 1.0)

        # Pricing & revenue
        selling_price = unit_total_with_depr * (1.0 + g.margin_rate)
        annual_revenue = selling_price * actual_cells
        annual_cash_cost = unit_total_cash * actual_cells
        annual_gross_margin = annual_revenue - annual_cash_cost  # ≈ EBITDA (no SG&A/taxes modeled)

        # Investment & returns
        initial_capex = float(df["step_capex_total_eur"].sum())
        r = g.discount_rate
        n = int(g.project_years)
        pv_annuity = n if r == 0 else (1 - (1 + r) ** (-n)) / r
        npv = -initial_capex + annual_gross_margin * pv_annuity
        payback_years = (initial_capex / annual_gross_margin) if annual_gross_margin > 0 else float("inf")

        kpis = {
            "unit_total_with_depr": unit_total_with_depr,
            "unit_total_cash": unit_total_cash,
            "raw_mat_unit_cost": raw_mat_unit_cost,
            "labor_cost": labor_cost,
            "overhead_cost": overhead_cost,
            "energy_cost": energy_cost,
            "facility_cost": facility_cost,
            "dry_room_cost": dry_room_cost,
            "depreciation_cost": depreciation_cost,
            "total_yield": total_yield,
            "annual_cells_target": annual_cells_target,
            "line_capacity_cells": line_capacity_cells,
            "actual_cells": actual_cells,
            "bottleneck_step": bottleneck_row["step"],
            "utilization": utilization,
            "selling_price": selling_price,
            "annual_revenue": annual_revenue,
            "annual_cash_cost": annual_cash_cost,
            "annual_gross_margin": annual_gross_margin,
            "initial_capex": initial_capex,
            "discount_rate": r,
            "project_years": n,
            "npv": npv,
            "payback_years": payback_years,
            "margin_rate": g.margin_rate,
        }
        return {"kpis": kpis, "steps": df}

    @staticmethod
    def figures(df: pd.DataFrame, k: Dict[str, float]) -> Tuple[go.Figure, go.Figure, go.Figure, go.Figure, go.Figure]:
        # Cost structure per cell (includes depreciation and dry room)
        cost_parts = {
            "Material": k["raw_mat_unit_cost"],
            "Energy": k["energy_cost"],
            "Labor": k["labor_cost"],
            "Overhead": k["overhead_cost"],
            "Facility": k["facility_cost"],
            "Dry Room": k["dry_room_cost"],
            "Depreciation": k["depreciation_cost"],
        }
        fig_cost = px.bar(x=list(cost_parts.keys()), y=list(cost_parts.values()),
                          labels={"x":"Component","y":"€/Cell"})

        # Energy by step
        fig_energy = px.bar(df, x="step", y="kwh", labels={"step":"Process Step","kwh":"kWh/Cell"})

        # Yield waterfall
        wf = []
        prev = 1.0
        for _, r in df.iterrows():
            wf.append({"name": r["step"], "delta": prev * (1 - r["yield"])})
            prev *= r["yield"]
        fig_yield = go.Figure(go.Waterfall(
            measure=["relative"] * len(wf) + ["total"],
            x=[w["name"] for w in wf] + ["Total"],
            y=[-w["delta"] for w in wf] + [prev],
            textposition="outside",
        ))
        fig_yield.update_layout(yaxis_title="Yield")

        # Capacity bar
        fig_cap = px.bar(df, x="step", y="capacity_cells_per_year",
                         labels={"step":"Process Step","capacity_cells_per_year":"Cells/Year"})
        fig_cap.add_hline(y=k["annual_cells_target"], line_dash="dash",
                          annotation_text="Annual Target", annotation_position="top left")

        # Revenue vs Cost (annual)
        rev_cost_df = pd.DataFrame({
            "Metric": ["Revenue", "Cash Cost", "EBITDA≈Gross Margin"],
            "EUR_per_year": [k["annual_revenue"], k["annual_cash_cost"], k["annual_gross_margin"]],
        })
        fig_rev_cost = px.bar(rev_cost_df, x="Metric", y="EUR_per_year",
                              labels={"Metric":"Annual Metric","EUR_per_year":"€ / year"})

        return fig_cost, fig_energy, fig_yield, fig_cap, fig_rev_cost

    @staticmethod
    def table_view(df: pd.DataFrame) -> Tuple[List[Dict[str, str]], List[Dict[str, object]]]:
        df = df.copy()
        df["yield_pct"] = df["yield"] * 100.0
        show_cols = [
            "step","time_s","kwh","yield_pct","throughput","capacity_cells_per_year","cum_yield",
            "labor_cost_per_cell","energy_cost_per_cell","facility_cost_per_cell",
            "depr_per_cell","dryroom_cost_per_cell"
        ]
        df_show = df[show_cols].round({
            "time_s":3,"kwh":6,"yield_pct":3,"throughput":0,"capacity_cells_per_year":0,
            "cum_yield":5,"labor_cost_per_cell":5,"energy_cost_per_cell":5,"facility_cost_per_cell":5,
            "depr_per_cell":5,"dryroom_cost_per_cell":5
        })
        columns = [
            {"name":"Process Step","id":"step"},
            {"name":"Time (s/Cell)","id":"time_s"},
            {"name":"Energy (kWh/Cell)","id":"kwh"},
            {"name":"Yield (%)","id":"yield_pct"},
            {"name":"Throughput (Cells/h)","id":"throughput"},
            {"name":"Capacity (Cells/Year)","id":"capacity_cells_per_year"},
            {"name":"Cumulative Yield","id":"cum_yield"},
            {"name":"Labor Cost/Cell","id":"labor_cost_per_cell"},
            {"name":"Energy Cost/Cell","id":"energy_cost_per_cell"},
            {"name":"Facility Cost/Cell","id":"facility_cost_per_cell"},
            {"name":"Depreciation/Cell","id":"depr_per_cell"},
            {"name":"Dry Room Cost/Cell","id":"dryroom_cost_per_cell"},
        ]
        return columns, df_show.to_dict("records")

# =========================================
# UI (functional) – side-by-side, independent scroll, logo in header
# =========================================

app = Dash(__name__)
server = app.server
app.title = "Battery Cost Estimator"

def num_input(id_obj, value, step="any", min_=None, max_=None, width="110px"):
    return dcc.Input(
        id=id_obj, type="number", value=value,
        step=step, min=min_, max=max_,
        style={"width": width}
    )

def step_controls(step_name: str, defaults: Dict[str, float]):
    p = defaults[step_name]
    dry_default = ["on"] if p.get("dry_room", 0) else []
    return html.Details([
        html.Summary(step_name),
        html.Div([
            html.Div([html.Label("Yield (%)"),
                      num_input({"type":"step_in","metric":"yield","index":step_name},
                                p["yield"]*100, step=0.01, min_=0, max_=100)]),
            html.Div([html.Label("Time/Cell (s)"),
                      num_input({"type":"step_in","metric":"time_s","index":step_name},
                                p["time_s"], step=0.1, min_=0)]),
            html.Div([html.Label("Energy Consumption/Cell (kWh)"),
                      num_input({"type":"step_in","metric":"kwh","index":step_name},
                                p["kwh"], step=0.0005, min_=0)]),
            html.Div([html.Label("Throughput (Cells/h)"),
                      num_input({"type":"step_in","metric":"throughput","index":step_name},
                                p["throughput"], step="any", min_=1)]),
            html.Div([html.Label("Area Share (%)"),
                      num_input({"type":"step_in","metric":"area_share","index":step_name},
                                p["area_share"], step=1, min_=1)]),

            html.Div([html.Label("Machine CAPEX (M€/GWh)"),
                      num_input({"type":"step_in","metric":"capex_m_eur_per_gwh","index":step_name},
                                p.get("capex_m_eur_per_gwh", 0.0), step=0.05, min_=0)]),
            html.Div([html.Label("Machine Life (years)"),
                      num_input({"type":"step_in","metric":"life_years","index":step_name},
                                p.get("life_years", 10.0), step=1, min_=1)]),
            html.Div([
                html.Label("Dry Room?"),
                dcc.Checklist(
                    options=[{"label": " Applies Dry Room Cost", "value":"on"}],
                    value=dry_default,
                    id={"type":"step_in","metric":"dry_room","index":step_name},
                    style={"marginTop":"6px"}
                )
            ], style={"gridColumn":"1 / span 2"}),
        ], style={"display":"grid","gridTemplateColumns":"repeat(2, minmax(180px, 1fr))","gap":"8px","padding":"6px 0"})
    ], open=False, style={"marginBottom":"6px"})

def logo_src() -> str:
    # Use asset if present; otherwise inline SVG fallback to avoid 404
    for name in ("logo.png", "logo.svg", "logo.jpg", "logo.jpeg"):
        if os.path.exists(os.path.join("assets", name)):
            return f"/assets/{name}"
    svg = """<svg xmlns='http://www.w3.org/2000/svg' width='160' height='44'>
      <rect width='100%' height='100%' fill='#0d6efd'/>
      <text x='50%' y='50%' dominant-baseline='middle' text-anchor='middle'
            font-family='Inter, Arial, sans-serif' font-size='18' fill='white'>
        Battery Cost Estimator
      </text>
    </svg>"""
    return "data:image/svg+xml;utf8," + quote(svg)

left_inputs = html.Div([
    html.H3("General Assumptions"),
    html.Div([
        html.Div([html.Label("Energy Price (€/kWh)"),                 num_input("el_price", DEFAULTS["electricity_price_eur_per_kwh"], step=0.01, min_=0)]),
        html.Div([html.Label("Salary (€/h)"),                         num_input("labor_rate", DEFAULTS["labor_rate_eur_per_h"], step=0.5, min_=0)]),
        html.Div([html.Label("Overhead Rate"),                        num_input("oh_rate", DEFAULTS["overhead_rate"], step=0.01, min_=0)]),
        html.Div([html.Label("Production Area (m²)"),                 num_input("area", DEFAULTS["factory_area_m2"], step=10, min_=0)]),
        html.Div([html.Label("Rent (€/m²/Month)"),                    num_input("rent", DEFAULTS["rent_eur_per_m2_month"], step=0.5, min_=0)]),
        html.Div([html.Label("Annual Output (GWh)"),                  num_input("gwh", DEFAULTS["annual_output_gwh"], step="any", min_=0.001)]),
        html.Div([html.Label("Cell Capacity (Wh/Cell)"),              num_input("cell_wh", DEFAULTS["cell_capacity_wh"], step=1, min_=1)]),
        html.Div([html.Label("Working Days per Year"),                num_input("days", DEFAULTS["working_days"], step=1, min_=1)]),
        html.Div([html.Label("Shifts per Day"),                       num_input("shifts", DEFAULTS["shifts_per_day"], step=1, min_=1)]),
        html.Div([html.Label("Hours per Shift"),                      num_input("hshift", DEFAULTS["shift_hours"], step=0.5, min_=0.5)]),
        html.Div([html.Label("Overall Equipment Efficiency (OEE) (0–1)"),
                  num_input("oee", DEFAULTS["oee"], step=0.01, min_=0, max_=1)]),
        html.Div([html.Label("Raw Material Cost (€/kWh)"),            num_input("rm_cost", DEFAULTS["raw_mat_cost_eur_per_kwh"], step=0.5, min_=0)]),

        # Economics inputs
        html.Div([html.Label("Margin (0–1)"),
                  num_input("margin", DEFAULTS["margin_rate"], step=0.01, min_=0, max_=1)]),
        html.Div([html.Label("Discount Rate (0–1)"),
                  num_input("disc_rate", DEFAULTS["discount_rate"], step=0.005, min_=0, max_=1)]),
        html.Div([html.Label("Project Years"),
                  num_input("proj_years", DEFAULTS["project_years"], step=1, min_=1)]),
        html.Div([html.Label("Dry Room Cost (€/m²/year)"),
                  num_input("dry_cost", DEFAULTS["dry_room_cost_eur_per_m2_year"], step=1, min_=0)]),
    ], style={"display":"grid","gridTemplateColumns":"repeat(2, minmax(180px, 1fr))","gap":"8px"}),
    html.Hr(),
    html.H3("Process Step Parameters"),
    html.Div([step_controls(s, DEFAULT_STEP_PARAMS) for s in STEPS],
             style={"paddingRight":"6px","border":"1px solid #eee","borderRadius":"8px","padding":"8px"}),
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
            "backgroundColor": "#007BFF",
            "color": "white",
            "border": "none",
            "borderRadius": "6px",
            "cursor": "pointer",
            "marginTop": "10px",
            "zIndex": "100",
        }
    ),
], style={
    "height": "100%",
    "overflowY": "auto",
    "padding": "8px",
    "borderRight": "1px solid #eee",
})

right_outputs = html.Div([
    html.H2("Analysis"),
    html.Div(id="kpi_row", style={"display":"flex","gap":"12px","flexWrap":"wrap"}),

    html.Div([
        html.Div([
            html.H4("Cost Structure per Cell (€/Cell)"),
            dcc.Graph(id="cost_bar"),
        ], style={"width":"48%","display":"inline-block","verticalAlign":"top"}),
        html.Div([
            html.H4("Energy Consumption per Process Step (kWh/Cell)"),
            dcc.Graph(id="energy_by_step"),
        ], style={"width":"48%","display":"inline-block","verticalAlign":"top"}),
    ], style={"display":"flex","justifyContent":"space-between","gap":"12px","flexWrap":"wrap"}),

    html.Div([
        html.Div([
            html.H4("Yield Waterfall"),
            dcc.Graph(id="yield_waterfall"),
        ], style={"width":"48%","display":"inline-block","verticalAlign":"top"}),
        html.Div([
            html.H4("Production Capacity (Cells/Year) & Annual Target"),
            dcc.Graph(id="capacity_bar"),
        ], style={"width":"48%","display":"inline-block","verticalAlign":"top"}),
    ], style={"display":"flex","justifyContent":"space-between","gap":"12px","flexWrap":"wrap"}),

    html.Div([
        html.Div([
            html.H4("Revenue vs Cost (Annual)"),
            dcc.Graph(id="rev_cost_bar"),
        ], style={"width":"48%","display":"inline-block","verticalAlign":"top"}),
    ], style={"display":"flex","justifyContent":"flex-start","gap":"12px","flexWrap":"wrap"}),

    html.H4("Overview"),
    dash_table.DataTable(
        id="steps_table",
        page_size=15,
        style_table={"overflowX":"auto"},
        style_cell={"padding":"6px","minWidth":90,"maxWidth":240,"whiteSpace":"normal"}
    ),
], style={
    "height": "100%",
    "overflowY": "auto",
    "padding": "10px 12px",
})

# Two-column frame with a header that includes the logo
app.layout = html.Div([
    # Header with logo + title
    html.Div([
        html.Img(
            src=app.get_asset_url("logo.png"),
            alt="PEM",
            style={
                "height": "44px",  # slightly larger than typical H2 text
                "marginRight": "12px",
                "display": "block"
            },
            draggable="false"
        ),
        html.H2("Battery Production Cost Calculator", style={"margin": 0, "alignSelf": "center"}),
    ], style={"display": "flex","alignItems": "center","gap": "8px","margin": "8px 0 12px"}),

    # Side-by-side scrollable columns
    html.Div([left_inputs, right_outputs], style={
        "display": "grid",
        "gridTemplateColumns": "420px 1fr",
        "gap": "12px",
        "height": "calc(100vh - 80px)",
        "alignItems": "stretch",
    })
], style={"fontFamily":"Inter, system-ui, Arial","maxWidth":"1400px","margin":"0 auto","padding":"0 8px 8px"})

# =========================================
# Callback
# =========================================

# Include new CAPEX metric key
metrics = ["yield","time_s","kwh","throughput","area_share","capex_m_eur_per_gwh","life_years","dry_room"]
step_states = [State({"type":"step_in","metric":m,"index":s}, "value") for s in STEPS for m in metrics]

@app.callback(
    Output("kpi_row", "children"),
    Output("cost_bar", "figure"),
    Output("energy_by_step", "figure"),
    Output("yield_waterfall", "figure"),
    Output("capacity_bar", "figure"),
    Output("rev_cost_bar", "figure"),
    Output("steps_table", "columns"),
    Output("steps_table", "data"),
    Input("run", "n_clicks"),
    State("el_price", "value"), State("labor_rate", "value"), State("oh_rate", "value"),
    State("area", "value"), State("rent", "value"), State("gwh", "value"), State("cell_wh", "value"),
    State("days", "value"), State("shifts", "value"), State("hshift", "value"), State("oee", "value"),
    State("rm_cost", "value"),
    State("margin", "value"), State("disc_rate", "value"), State("proj_years", "value"),
    State("dry_cost", "value"),
    *step_states,
    prevent_initial_call=True
)
def run_calc(_, el_price, labor_rate, oh_rate, area, rent, gwh, cell_wh, days, shifts, hshift, oee, rm_cost,
             margin, disc_rate, proj_years, dry_cost, *step_values):

    # Map State values back to per-step params
    step_params: Dict[str, StepParam] = {}
    idx = 0
    for s in STEPS:
        vals: Dict[str, object] = {}
        for m in metrics:
            val = step_values[idx]; idx += 1
            if m == "yield":
                val = float(val) / 100.0
            elif m in ("time_s","kwh","throughput","area_share","capex_m_eur_per_gwh","life_years"):
                val = float(val)
            elif m == "dry_room":
                # Checklist returns [] or ["on"]
                val = bool(val) and (len(val) > 0)
            vals[m] = val
        step_params[s] = StepParam(
            name=s,
            time_s=vals["time_s"],
            kwh=vals["kwh"],
            yield_=vals["yield"],
            throughput=vals["throughput"],
            area_share=vals["area_share"],
            capex_m_eur_per_gwh=vals["capex_m_eur_per_gwh"],
            life_years=vals["life_years"],
            dry_room=vals["dry_room"],
        )

    general = GeneralAssumptions(
        electricity_price_eur_per_kwh = float(el_price or DEFAULTS["electricity_price_eur_per_kwh"]),
        labor_rate_eur_per_h          = float(labor_rate or DEFAULTS["labor_rate_eur_per_h"]),
        overhead_rate                 = float(oh_rate or DEFAULTS["overhead_rate"]),
        factory_area_m2               = float(area or DEFAULTS["factory_area_m2"]),
        rent_eur_per_m2_month         = float(rent or DEFAULTS["rent_eur_per_m2_month"]),
        annual_output_gwh             = float(gwh or DEFAULTS["annual_output_gwh"]),
        cell_capacity_wh              = float(cell_wh or DEFAULTS["cell_capacity_wh"]),
        working_days                  = float(days or DEFAULTS["working_days"]),
        shifts_per_day                = float(shifts or DEFAULTS["shifts_per_day"]),
        shift_hours                   = float(hshift or DEFAULTS["shift_hours"]),
        oee                           = float(oee or DEFAULTS["oee"]),
        raw_mat_cost_eur_per_kwh      = float(rm_cost or DEFAULTS["raw_mat_cost_eur_per_kwh"]),
        margin_rate                   = float(margin or DEFAULTS["margin_rate"]),
        discount_rate                 = float(disc_rate or DEFAULTS["discount_rate"]),
        project_years                 = int(proj_years or DEFAULTS["project_years"]),
        dry_room_cost_eur_per_m2_year = float(dry_cost or DEFAULTS["dry_room_cost_eur_per_m2_year"]),
    )

    model = BatteryCostModel(general=general, steps=list(step_params.values()))
    result = model.compute()
    k = result["kpis"]
    df = result["steps"]

    def card(label, value, suffix=""):
        return html.Div([
            html.Div(label, style={"fontSize":"12px","color":"#666"}),
            html.Div(f"{value}{suffix}", style={"fontSize":"20px","fontWeight":600}),
        ], style={"border":"1px solid #eee","borderRadius":"12px","padding":"10px 12px","minWidth":"180px"})

    kpi_children = [
        card("Cost/Cell (Cash)", f"{k['unit_total_cash']:.4f}", " €"),
        card("Depreciation/Cell", f"{k['depreciation_cost']:.4f}", " €"),
        card("Cost/Cell (Incl. Depr.)", f"{k['unit_total_with_depr']:.4f}", " €"),
        card("Selling Price/Cell", f"{k['selling_price']:.4f}", " €"),
        card("Margin (%)", f"{k['margin_rate']*100:.1f}", " %"),
        card("Annual Units", f"{k['actual_cells']:.0f}"),
        card("Annual Revenue", f"{k['annual_revenue']:.0f}", " €"),
        card("Annual Cash Cost", f"{k['annual_cash_cost']:.0f}", " €"),
        card("Annual EBITDA (≈GM)", f"{k['annual_gross_margin']:.0f}", " €"),
        card("NPV", f"{k['npv']:.0f}", " €"),
        card("Payback (yrs)", f"{k['payback_years']:.2f}"),
        card("Total Yield", f"{k['total_yield']*100:.2f}", " %"),
        card("Annual Target (Cells)", f"{k['annual_cells_target']:.0f}"),
        card("Line Capacity (Cells)", f"{k['line_capacity_cells']:.0f}"),
        card("Bottleneck", k['bottleneck_step']),
    ]

    fig_cost, fig_energy, fig_yield, fig_cap, fig_rev_cost = BatteryCostModel.figures(df, k)
    columns, data = BatteryCostModel.table_view(df)
    return kpi_children, fig_cost, fig_energy, fig_yield, fig_cap, fig_rev_cost, columns, data

# =========================================
# Entrypoint
# =========================================
if __name__ == "__main__":
    app.run(debug=True)

