from dash import Dash, html, dcc, Input, Output, State, dash_table
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

app = Dash(__name__)
server = app.server  # <-- wichtig für Gunicorn/Render
app.title = "Battery Cost Estimator"

# ------------------------------
# Konfiguration & Defaults
# ------------------------------
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

# Default-Parameter je Schritt (intern weiterhin als Anteile 0–1 gespeichert)
# Zeit/Zelle (s), Energie/Zelle (kWh), Yield (0-1), Durchsatz (Zellen/h), Flächenanteil (%)
DEFAULT_STEP_PARAMS = {
    "Mixing":                 {"time_s": 3.0,  "kwh": 0.003, "yield": 0.995, "throughput": 1200, "area_share": 6},
    "Coating & Drying":      {"time_s": 9.0,  "kwh": 0.020, "yield": 0.990, "throughput": 900,  "area_share": 12},
    "Calendering":           {"time_s": 2.0,  "kwh": 0.002, "yield": 0.998, "throughput": 2000, "area_share": 7},
    "Slitting":              {"time_s": 1.2,  "kwh": 0.001, "yield": 0.998, "throughput": 2500, "area_share": 5},
    "Post-Drying":           {"time_s": 4.5,  "kwh": 0.006, "yield": 0.996, "throughput": 1100, "area_share": 7},
    "Winding":               {"time_s": 6.0,  "kwh": 0.004, "yield": 0.995, "throughput": 1000, "area_share": 8},
    "Contacting":            {"time_s": 2.5,  "kwh": 0.001, "yield": 0.998, "throughput": 1800, "area_share": 5},
    "Inserting & Closing of lid": {"time_s": 3.0,  "kwh": 0.001, "yield": 0.997, "throughput": 1500, "area_share": 7},
    "Electrolyte Filling":   {"time_s": 4.0,  "kwh": 0.003, "yield": 0.996, "throughput": 1200, "area_share": 6},
    "Wetting":               {"time_s": 8.0,  "kwh": 0.002, "yield": 0.995, "throughput": 800,  "area_share": 7},
    "Forming & Degassing":   {"time_s": 30.0, "kwh": 0.030, "yield": 0.990, "throughput": 500,  "area_share": 20},
    "Closing":               {"time_s": 1.0,  "kwh": 0.001, "yield": 0.999, "throughput": 2600, "area_share": 5},
    "Aging":                 {"time_s": 15.0, "kwh": 0.010, "yield": 0.995, "throughput": 700,  "area_share": 4},
    "End-of-Line Test":      {"time_s": 5.0,  "kwh": 0.002, "yield": 0.998, "throughput": 1400, "area_share": 6},
}

# Allgemeine Defaultannahmen
DEFAULTS = dict(
    electricity_price_eur_per_kwh = 0.22,   # €/kWh
    labor_rate_eur_per_h           = 35.0,  # €/h
    overhead_rate                  = 0.30,  # auf Lohnkosten
    factory_area_m2                = 10000, # m²
    rent_eur_per_m2_month          = 12.0,  # €/m²/Monat
    annual_output_gwh              = 1.0,   # GWh/Jahr
    cell_capacity_wh               = 60.0,  # Wh/Zelle
    working_days                   = 330,   # Tage/Jahr
    shifts_per_day                 = 3,
    shift_hours                    = 7.5,
    oee                            = 0.80,
    raw_mat_cost_eur_per_kwh       = 45.0,  # €/kWh Material
)

# ------------------------------
# Hilfsfunktionen
# ------------------------------
def steps_df_from_inputs(values: dict) -> pd.DataFrame:
    rows = []
    for step in STEPS:
        p = values.get(step, DEFAULT_STEP_PARAMS[step])
        rows.append({
            "step": step,
            "time_s": float(p["time_s"]),
            "kwh": float(p["kwh"]),
            "yield": float(p["yield"]),                # Anteil 0–1
            "throughput": float(p["throughput"]),
            "area_share": float(p["area_share"]),
        })
    return pd.DataFrame(rows)

def compute_model(general: dict, step_df: pd.DataFrame) -> dict:
    cell_kwh = general["cell_capacity_wh"] / 1000.0
    annual_cells_target = (general["annual_output_gwh"] * 1_000_000.0) / max(cell_kwh, 1e-12)

    hours_year = general["working_days"] * general["shifts_per_day"] * general["shift_hours"] * general["oee"]

    df = step_df.copy()
    df["capacity_cells_per_year"] = df["throughput"] * hours_year
    bottleneck_row = df.loc[df["capacity_cells_per_year"].idxmin()]

    df["cum_yield"] = df["yield"].cumprod()
    total_yield = float(df["yield"].prod())

    time_h = df["time_s"] / 3600.0
    labor_cost = (time_h * general["labor_rate_eur_per_h"]).sum()
    overhead_cost = general["overhead_rate"] * labor_cost
    energy_cost = (df["kwh"] * general["electricity_price_eur_per_kwh"]).sum()

    rent_year = general["factory_area_m2"] * general["rent_eur_per_m2_month"] * 12.0
    annual_cells_safe = max(annual_cells_target, 1.0)
    df["facility_cost_per_cell"] = (rent_year * (df["area_share"] / df["area_share"].sum())) / annual_cells_safe
    facility_cost = df["facility_cost_per_cell"].sum()

    raw_mat_unit_cost = general["raw_mat_cost_eur_per_kwh"] * cell_kwh / max(total_yield, 1e-12)

    unit_total = labor_cost + overhead_cost + energy_cost + facility_cost + raw_mat_unit_cost

    df["labor_cost_per_cell"]  = time_h * general["labor_rate_eur_per_h"]
    df["energy_cost_per_cell"] = df["kwh"] * general["electricity_price_eur_per_kwh"]

    line_capacity_cells = float(df["capacity_cells_per_year"].min())
    utilization = min(annual_cells_target / max(line_capacity_cells, 1.0), 1.0)

    kpis = {
        "unit_total": unit_total,
        "raw_mat_unit_cost": raw_mat_unit_cost,
        "labor_cost": labor_cost,
        "overhead_cost": overhead_cost,
        "energy_cost": energy_cost,
        "facility_cost": facility_cost,
        "total_yield": total_yield,
        "annual_cells_target": annual_cells_target,
        "line_capacity_cells": line_capacity_cells,
        "bottleneck_step": bottleneck_row["step"],
        "utilization": utilization,
    }
    return {"kpis": kpis, "steps": df}

# ------------------------------
# UI – Layout (Eingabefelder)
# ------------------------------
def num_input(id_obj, value, step=None, min_=None, max_=None, width="110px"):
    return dcc.Input(
        id=id_obj, type="number", value=value,
        step=step, min=min_, max=max_,
        style={"width": width}
    )

def step_controls(step_name: str, defaults: dict):
    p = defaults[step_name]
    return html.Details([
        html.Summary(step_name),
        html.Div([
            html.Div([html.Label("Yield (%)"),
                      num_input({"type":"step_in","metric":"yield","index":step_name},
                                p["yield"]*100, step=0.01, min_=0, max_=100)]),
            html.Div([html.Label("Zeit/Zelle (s)"),
                      num_input({"type":"step_in","metric":"time_s","index":step_name},
                                p["time_s"], step=0.1, min_=0)]),
            html.Div([html.Label("Energie/Zelle (kWh)"),
                      num_input({"type":"step_in","metric":"kwh","index":step_name},
                                p["kwh"], step=0.0005, min_=0)]),
            html.Div([html.Label("Durchsatz (Zellen/h)"),
                      num_input({"type":"step_in","metric":"throughput","index":step_name},
                                p["throughput"], step=10, min_=1)]),
            html.Div([html.Label("Flächenanteil (%)"),
                      num_input({"type":"step_in","metric":"area_share","index":step_name},
                                p["area_share"], step=1, min_=1)]),
        ], style={"display":"grid","gridTemplateColumns":"repeat(2, minmax(180px, 1fr))","gap":"8px","padding":"6px 0"})
    ], open=False, style={"marginBottom":"6px"})

left_inputs = html.Div([
    html.H3("Allgemeine Annahmen"),
    html.Div([
        html.Div([html.Label("Strompreis (€/kWh)"),                 num_input("el_price", DEFAULTS["electricity_price_eur_per_kwh"], step=0.01, min_=0)]),
        html.Div([html.Label("Lohn (€/h)"),                         num_input("labor_rate", DEFAULTS["labor_rate_eur_per_h"], step=0.5, min_=0)]),
        html.Div([html.Label("Overhead-Satz (auf Lohn)"),           num_input("oh_rate", DEFAULTS["overhead_rate"], step=0.01, min_=0)]),
        html.Div([html.Label("Fabrikfläche (m²)"),                  num_input("area", DEFAULTS["factory_area_m2"], step=10, min_=0)]),
        html.Div([html.Label("Miete (€/m²/Monat)"),                 num_input("rent", DEFAULTS["rent_eur_per_m2_month"], step=0.5, min_=0)]),
        html.Div([html.Label("Jahresausstoß (GWh)"),                num_input("gwh", DEFAULTS["annual_output_gwh"], step=0.1, min_=0.001)]),
        html.Div([html.Label("Zellkapazität (Wh/Zelle)"),           num_input("cell_wh", DEFAULTS["cell_capacity_wh"], step=1, min_=1)]),
        html.Div([html.Label("Arbeitstage/Jahr"),                   num_input("days", DEFAULTS["working_days"], step=1, min_=1)]),
        html.Div([html.Label("Schichten/Tag"),                      num_input("shifts", DEFAULTS["shifts_per_day"], step=1, min_=1)]),
        html.Div([html.Label("Stunden/Schicht"),                    num_input("hshift", DEFAULTS["shift_hours"], step=0.5, min_=0.5)]),
        html.Div([html.Label("OEE (0–1)"),                          num_input("oee", DEFAULTS["oee"], step=0.01, min_=0, max_=1)]),
        html.Div([html.Label("Rohstoffkosten (€/kWh)"),             num_input("rm_cost", DEFAULTS["raw_mat_cost_eur_per_kwh"], step=0.5, min_=0)]),
    ], style={"display":"grid","gridTemplateColumns":"repeat(2, minmax(180px, 1fr))","gap":"8px"}),
    html.Hr(),
    html.H3("Produktionsschritte – Parameter"),
    html.Div([step_controls(s, DEFAULT_STEP_PARAMS) for s in STEPS],
             style={"maxHeight":"65vh","overflow":"auto","paddingRight":"6px","border":"1px solid #eee","borderRadius":"8px","padding":"8px"}),
    html.Button("Berechnen", id="run", n_clicks=0, style={"marginTop":"8px"}),
], style={"width":"32%","display":"inline-block","verticalAlign":"top","padding":"8px","borderRight":"1px solid #eee"})

right_outputs = html.Div([
    html.H2("Auswertung"),
    html.Div(id="kpi_row", style={"display":"flex","gap":"12px","flexWrap":"wrap"}),

    html.Div([
        html.Div([
            html.H4("Kostenstruktur je Zelle (€/Zelle)"),
            dcc.Graph(id="cost_bar"),
        ], style={"width":"48%","display":"inline-block","verticalAlign":"top"}),
        html.Div([
            html.H4("Energie je Schritt (kWh/Zelle)"),
            dcc.Graph(id="energy_by_step"),
        ], style={"width":"48%","display":"inline-block","verticalAlign":"top"}),
    ], style={"display":"flex","justifyContent":"space-between","gap":"12px","flexWrap":"wrap"}),

    html.Div([
        html.Div([
            html.H4("Yield-Kaskade"),
            dcc.Graph(id="yield_waterfall"),
        ], style={"width":"48%","display":"inline-block","verticalAlign":"top"}),
        html.Div([
            html.H4("Kapazität je Schritt (Zellen/Jahr) & Jahresziel"),
            dcc.Graph(id="capacity_bar"),
        ], style={"width":"48%","display":"inline-block","verticalAlign":"top"}),
    ], style={"display":"flex","justifyContent":"space-between","gap":"12px","flexWrap":"wrap"}),

    html.H4("Tabellarische Übersicht je Schritt"),
    dash_table.DataTable(
        id="steps_table",
        page_size=15,
        style_table={"overflowX":"auto"},
        style_cell={"padding":"6px","minWidth":90,"maxWidth":240,"whiteSpace":"normal"}
    ),
], style={"width":"66%","display":"inline-block","padding":"10px 12px"})

app.layout = html.Div([
    html.H2("Batterie-Produktionskosten – Prototyp (Yield-Eingabe in %)"),
    html.Div([left_inputs, right_outputs])
], style={"fontFamily":"Inter, system-ui, Arial","maxWidth":"1400px","margin":"0 auto"})

# ------------------------------
# Callbacks
# ------------------------------
metrics = ["yield","time_s","kwh","throughput","area_share"]
step_states = [State({"type":"step_in","metric":m,"index":s}, "value") for s in STEPS for m in metrics]

@app.callback(
    Output("kpi_row", "children"),
    Output("cost_bar", "figure"),
    Output("energy_by_step", "figure"),
    Output("yield_waterfall", "figure"),
    Output("capacity_bar", "figure"),
    Output("steps_table", "columns"),
    Output("steps_table", "data"),
    Input("run", "n_clicks"),
    State("el_price", "value"), State("labor_rate", "value"), State("oh_rate", "value"),
    State("area", "value"), State("rent", "value"), State("gwh", "value"), State("cell_wh", "value"),
    State("days", "value"), State("shifts", "value"), State("hshift", "value"), State("oee", "value"),
    State("rm_cost", "value"),
    *step_states,
    prevent_initial_call=True
)
def run_calc(_, el_price, labor_rate, oh_rate, area, rent, gwh, cell_wh, days, shifts, hshift, oee, rm_cost, *step_values):
    # States zurück auf Schritte mappen
    step_params = {}
    idx = 0
    for s in STEPS:
        vals = {}
        for m in metrics:
            val = float(step_values[idx]); idx += 1
            if m == "yield":
                val = val / 100.0  # Prozent -> Anteil 0–1 (intern)
            vals[m] = val
        step_params[s] = vals

    general = dict(
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
    )

    df_steps = steps_df_from_inputs(step_params)
    result = compute_model(general, df_steps)
    k = result["kpis"]
    df = result["steps"].copy()

    # KPI-Cards
    def card(label, value, suffix=""):
        return html.Div([
            html.Div(label, style={"fontSize":"12px","color":"#666"}),
            html.Div(f"{value}{suffix}", style={"fontSize":"20px","fontWeight":600}),
        ], style={"border":"1px solid #eee","borderRadius":"12px","padding":"10px 12px","minWidth":"180px"})

    kpi_children = [
        card("Gesamtkosten je Zelle", f"{k['unit_total']:.4f}", " €"),
        card("Material je Zelle", f"{k['raw_mat_unit_cost']:.4f}", " €"),
        card("Energie je Zelle", f"{k['energy_cost']:.4f}", " €"),
        card("Lohn je Zelle", f"{k['labor_cost']:.4f}", " €"),
        card("Overhead je Zelle", f"{k['overhead_cost']:.4f}", " €"),
        card("Facility je Zelle", f"{k['facility_cost']:.4f}", " €"),
        card("Gesamtyield", f"{k['total_yield']*100:.2f}", " %"),
        card("Jahresziel (Zellen)", f"{k['annual_cells_target']:.0f}"),
        card("Linienkapazität (Zellen)", f"{k['line_capacity_cells']:.0f}"),
        card("Bottleneck", k['bottleneck_step']),
    ]

    # Kostenstruktur
    cost_parts = {
        "Material": k["raw_mat_unit_cost"],
        "Energie": k["energy_cost"],
        "Lohn": k["labor_cost"],
        "Overhead": k["overhead_cost"],
        "Facility": k["facility_cost"],
    }
    fig_cost = px.bar(x=list(cost_parts.keys()), y=list(cost_parts.values()),
                      labels={"x":"Komponente","y":"€/Zelle"})

    # Energie je Schritt
    fig_energy = px.bar(df, x="step", y="kwh", labels={"step":"Schritt","kwh":"kWh/Zelle"})

    # Yield-Wasserfall (kumulativ)
    wf = []
    prev = 1.0
    for _, r in df.iterrows():
        wf.append({"name": r["step"], "delta": prev * (1 - r["yield"])})
        prev *= r["yield"]
    fig_yield = go.Figure(go.Waterfall(
        measure=["relative"] * len(wf) + ["total"],
        x=[w["name"] for w in wf] + ["Gesamt"],
        y=[-w["delta"] for w in wf] + [prev],
        textposition="outside",
    ))
    fig_yield.update_layout(yaxis_title="Ausbeute (Anteil)")

    # Kapazität je Schritt
    fig_cap = px.bar(df, x="step", y="capacity_cells_per_year",
                     labels={"step":"Schritt","capacity_cells_per_year":"Zellen/Jahr"})
    fig_cap.add_hline(y=k["annual_cells_target"], line_dash="dash",
                      annotation_text="Jahresziel", annotation_position="top left")

    # Tabelle – Yield in % anzeigen
    df["yield_pct"] = df["yield"] * 100.0
    show_cols = [
        "step","time_s","kwh","yield_pct","throughput","capacity_cells_per_year","cum_yield",
        "labor_cost_per_cell","energy_cost_per_cell","facility_cost_per_cell"
    ]
    df_show = df[show_cols].round({
        "time_s":3,"kwh":6,"yield_pct":3,"throughput":0,"capacity_cells_per_year":0,
        "cum_yield":5,"labor_cost_per_cell":5,"energy_cost_per_cell":5,"facility_cost_per_cell":5
    })
    columns = [
        {"name":"Schritt","id":"step"},
        {"name":"Zeit (s/Zelle)","id":"time_s"},
        {"name":"Energie (kWh/Zelle)","id":"kwh"},
        {"name":"Yield (%)","id":"yield_pct"},
        {"name":"Durchsatz (Zellen/h)","id":"throughput"},
        {"name":"Kapazität (Zellen/Jahr)","id":"capacity_cells_per_year"},
        {"name":"kumul. Yield (Anteil)","id":"cum_yield"},
        {"name":"Lohn (€/Zelle)","id":"labor_cost_per_cell"},
        {"name":"Energie (€/Zelle)","id":"energy_cost_per_cell"},
        {"name":"Facility (€/Zelle)","id":"facility_cost_per_cell"},
    ]

    return kpi_children, fig_cost, fig_energy, fig_yield, fig_cap, columns, df_show.to_dict("records")

if __name__ == "__main__":
    app.run(debug=True)