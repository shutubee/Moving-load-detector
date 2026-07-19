"""
Moving Load Detector & Influence Line Response Engine
Author: Pushpa Ramakrishnan

Run:
    streamlit run app.py

Dependencies:
    streamlit
    numpy
    pandas
    plotly
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


ResponseType = Literal[
    "Left support reaction",
    "Right support reaction",
    "Shear at section",
    "Bending moment at section",
]


@dataclass(frozen=True)
class Vehicle:
    axle_loads: np.ndarray
    axle_spacings: np.ndarray

    @property
    def axle_offsets(self) -> np.ndarray:
        if self.axle_loads.size == 1:
            return np.array([0.0], dtype=float)
        return np.concatenate(([0.0], np.cumsum(self.axle_spacings)))

    @property
    def length(self) -> float:
        return float(self.axle_offsets[-1])

    @property
    def total_load(self) -> float:
        return float(np.sum(self.axle_loads))


def validate_inputs(
    span: float,
    section: float,
    step: float,
    loads: np.ndarray,
    spacings: np.ndarray,
) -> list[str]:
    errors: list[str] = []

    if not np.isfinite(span) or span <= 0:
        errors.append("Beam span must be a finite positive number.")

    if not np.isfinite(step) or step <= 0:
        errors.append("Movement step must be a finite positive number.")

    if loads.size == 0:
        errors.append("At least one axle load is required.")
    elif np.any(~np.isfinite(loads)) or np.any(loads < 0):
        errors.append("Axle loads must be finite and non-negative.")

    required_spacing_count = max(loads.size - 1, 0)
    if spacings.size != required_spacing_count:
        errors.append(
            f"A {loads.size}-axle vehicle requires exactly "
            f"{required_spacing_count} axle spacings."
        )
    elif spacings.size and (
        np.any(~np.isfinite(spacings)) or np.any(spacings <= 0)
    ):
        errors.append("All axle spacings must be finite and greater than zero.")

    if np.isfinite(span) and span > 0:
        if not np.isfinite(section) or not (0 < section < span):
            errors.append("The response section must lie strictly inside the beam.")

    return errors


def influence_left_reaction(x: np.ndarray, span: float, _: float) -> np.ndarray:
    return (span - x) / span


def influence_right_reaction(x: np.ndarray, span: float, _: float) -> np.ndarray:
    return x / span


def influence_shear(
    x: np.ndarray,
    span: float,
    section: float,
    shear_side: str = "Right face",
) -> np.ndarray:
    """
    Shear influence line using:
      x < c : -x/L
      x > c : (L-x)/L

    At x == c, the ordinate depends on the selected convention.
    """
    ordinate = np.where(x < section, -x / span, (span - x) / span)

    at_section = np.isclose(x, section, rtol=0.0, atol=1e-10)
    if shear_side == "Left face":
        ordinate = np.where(at_section, -section / span, ordinate)
    else:
        ordinate = np.where(at_section, (span - section) / span, ordinate)

    return ordinate


def influence_moment(x: np.ndarray, span: float, section: float) -> np.ndarray:
    left = x * (span - section) / span
    right = section * (span - x) / span
    return np.where(x <= section, left, right)


def get_influence_function(
    response_type: ResponseType,
    shear_side: str,
) -> Callable[[np.ndarray, float, float], np.ndarray]:
    if response_type == "Left support reaction":
        return influence_left_reaction
    if response_type == "Right support reaction":
        return influence_right_reaction
    if response_type == "Bending moment at section":
        return influence_moment

    return lambda x, span, section: influence_shear(
        x, span, section, shear_side
    )


def build_position_grid(
    span: float,
    offsets: np.ndarray,
    section: float,
    step: float,
    include_section_events: bool,
) -> np.ndarray:
    """
    Vehicle reference coordinate s is the leading axle position for left-to-right travel.
    Axle i lies at x_i = s - d_i.
    """
    s_min = 0.0
    s_max = span + float(offsets[-1])

    uniform = np.arange(s_min, s_max + 0.5 * step, step, dtype=float)
    if uniform[-1] < s_max:
        uniform = np.append(uniform, s_max)

    events = [s_min, s_max]
    events.extend(offsets.tolist())                 # axle entry: s = d_i
    events.extend((span + offsets).tolist())        # axle exit: s = L + d_i

    if include_section_events:
        events.extend((section + offsets).tolist()) # axle crosses section: s = c + d_i

    all_positions = np.concatenate((uniform, np.asarray(events, dtype=float)))
    all_positions = all_positions[
        (all_positions >= s_min - 1e-12)
        & (all_positions <= s_max + 1e-12)
    ]

    return np.unique(np.round(all_positions, 12))


def solve_response_history(
    span: float,
    section: float,
    vehicle: Vehicle,
    positions: np.ndarray,
    influence_function: Callable[[np.ndarray, float, float], np.ndarray],
    load_factor: float,
    dynamic_factor: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    loads = vehicle.axle_loads * load_factor * dynamic_factor
    offsets = vehicle.axle_offsets

    rows: list[dict[str, object]] = []
    contribution_rows: list[dict[str, float | int | bool]] = []

    for s in positions:
        axle_positions = s - offsets
        active = (axle_positions >= -1e-10) & (axle_positions <= span + 1e-10)

        ordinates = np.zeros_like(axle_positions, dtype=float)
        contributions = np.zeros_like(axle_positions, dtype=float)

        if np.any(active):
            x_active = np.clip(axle_positions[active], 0.0, span)
            ordinates[active] = influence_function(x_active, span, section)
            contributions[active] = loads[active] * ordinates[active]

        response = float(np.sum(contributions))
        abs_contributions = np.abs(contributions)
        dominant_axle = (
            int(np.argmax(abs_contributions) + 1)
            if np.any(abs_contributions > 0)
            else None
        )

        rows.append(
            {
                "vehicle_position_m": float(s),
                "response": response,
                "positive_contribution": float(np.sum(np.maximum(contributions, 0.0))),
                "negative_contribution": float(np.sum(np.minimum(contributions, 0.0))),
                "active_axles": int(np.sum(active)),
                "dominant_axle": dominant_axle,
                "active_load_kN": float(np.sum(loads[active])),
            }
        )

        for i, (load, offset, x, is_active, ordinate, contribution) in enumerate(
            zip(loads, offsets, axle_positions, active, ordinates, contributions),
            start=1,
        ):
            contribution_rows.append(
                {
                    "vehicle_position_m": float(s),
                    "axle": i,
                    "factored_axle_load_kN": float(load),
                    "axle_offset_m": float(offset),
                    "axle_position_m": float(x),
                    "active": bool(is_active),
                    "influence_ordinate": float(ordinate),
                    "contribution": float(contribution),
                }
            )

    return pd.DataFrame(rows), pd.DataFrame(contribution_rows)


def refine_peak_parabolically(
    history: pd.DataFrame,
    target: Literal["max", "min"],
) -> tuple[float, float] | None:
    y = history["response"].to_numpy(dtype=float)
    x = history["vehicle_position_m"].to_numpy(dtype=float)

    idx = int(np.argmax(y) if target == "max" else np.argmin(y))
    if idx == 0 or idx == len(y) - 1:
        return None

    x3 = x[idx - 1 : idx + 2]
    y3 = y[idx - 1 : idx + 2]

    if np.unique(x3).size < 3:
        return None

    coeff = np.polyfit(x3, y3, 2)
    a, b, c = coeff
    if np.isclose(a, 0.0):
        return None

    xv = -b / (2.0 * a)
    if not (x3[0] <= xv <= x3[-1]):
        return None

    yv = a * xv**2 + b * xv + c
    return float(xv), float(yv)


def make_influence_figure(
    span: float,
    section: float,
    influence_function: Callable[[np.ndarray, float, float], np.ndarray],
    response_type: str,
) -> go.Figure:
    x = np.linspace(0.0, span, 801)
    y = influence_function(x, span, section)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x,
            y=y,
            mode="lines",
            name="Influence line",
            hovertemplate="x=%{x:.3f} m<br>ordinate=%{y:.5g}<extra></extra>",
        )
    )
    fig.add_hline(y=0.0, line_width=1)
    if "section" in response_type.lower():
        fig.add_vline(
            x=section,
            line_dash="dash",
            annotation_text=f"Section c = {section:.3f} m",
        )
    fig.update_layout(
        title=f"Influence Line — {response_type}",
        xaxis_title="Unit-load position x (m)",
        yaxis_title="Influence ordinate",
        height=420,
        margin=dict(l=20, r=20, t=60, b=20),
    )
    return fig


def make_response_figure(
    history: pd.DataFrame,
    max_row: pd.Series,
    min_row: pd.Series,
) -> go.Figure:
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=history["vehicle_position_m"],
            y=history["response"],
            mode="lines",
            name="Response history",
            hovertemplate=(
                "Vehicle position=%{x:.3f} m<br>"
                "Response=%{y:.6g}<extra></extra>"
            ),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=[max_row["vehicle_position_m"]],
            y=[max_row["response"]],
            mode="markers+text",
            text=["Maximum"],
            textposition="top center",
            name="Maximum",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=[min_row["vehicle_position_m"]],
            y=[min_row["response"]],
            mode="markers+text",
            text=["Minimum"],
            textposition="bottom center",
            name="Minimum",
        )
    )

    fig.add_hline(y=0.0, line_width=1)
    fig.update_layout(
        title="Response History and Critical Vehicle Positions",
        xaxis_title="Leading-axle position s (m)",
        yaxis_title="Structural response",
        height=460,
        margin=dict(l=20, r=20, t=60, b=20),
        hovermode="x unified",
    )
    return fig


def make_envelope_figure(history: pd.DataFrame) -> go.Figure:
    response = history["response"].to_numpy(dtype=float)
    x = history["vehicle_position_m"].to_numpy(dtype=float)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x,
            y=np.maximum(response, 0.0),
            mode="lines",
            name="Positive envelope",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=np.minimum(response, 0.0),
            mode="lines",
            name="Negative envelope",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=np.abs(response),
            mode="lines",
            name="Absolute envelope",
        )
    )

    fig.update_layout(
        title="Positive, Negative, and Absolute Response Envelopes",
        xaxis_title="Leading-axle position s (m)",
        yaxis_title="Envelope value",
        height=430,
        margin=dict(l=20, r=20, t=60, b=20),
        hovermode="x unified",
    )
    return fig


def make_vehicle_figure(
    span: float,
    section: float,
    vehicle: Vehicle,
    critical_s: float,
    factored_loads: np.ndarray,
    title: str,
) -> go.Figure:
    offsets = vehicle.axle_offsets
    axle_positions = critical_s - offsets
    active = (axle_positions >= -1e-10) & (axle_positions <= span + 1e-10)

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=[0.0, span],
            y=[0.0, 0.0],
            mode="lines",
            line=dict(width=8),
            name="Beam",
            hoverinfo="skip",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=[0.0, span],
            y=[-0.08, -0.08],
            mode="markers",
            marker=dict(size=15, symbol=["triangle-up", "triangle-up"]),
            name="Supports",
            hoverinfo="skip",
        )
    )

    for i, (x, load, is_active) in enumerate(
        zip(axle_positions, factored_loads, active), start=1
    ):
        if not is_active:
            continue

        fig.add_annotation(
            x=float(x),
            y=0.0,
            ax=float(x),
            ay=0.65,
            xref="x",
            yref="y",
            axref="x",
            ayref="y",
            showarrow=True,
            arrowhead=2,
            arrowsize=1,
            arrowwidth=2,
            text=f"Axle {i}<br>{load:.2f} kN",
        )

    fig.add_vline(
        x=section,
        line_dash="dash",
        annotation_text=f"c={section:.3f} m",
    )

    x_margin = max(0.05 * span, 0.5)
    fig.update_xaxes(range=[-x_margin, span + x_margin])
    fig.update_yaxes(range=[-0.3, 0.9], visible=False)
    fig.update_layout(
        title=title,
        xaxis_title="Beam coordinate x (m)",
        height=330,
        margin=dict(l=20, r=20, t=60, b=20),
        showlegend=False,
    )
    return fig


def format_response_unit(response_type: str) -> str:
    if "reaction" in response_type.lower() or "shear" in response_type.lower():
        return "kN"
    if "moment" in response_type.lower():
        return "kN·m"
    return ""


def load_vehicle_preset(name: str) -> pd.DataFrame:
    presets = {
        "Three-axle reference": pd.DataFrame(
            {
                "Axle load (kN)": [80.0, 120.0, 120.0],
                "Spacing to next axle (m)": [4.0, 1.5, np.nan],
            }
        ),
        "Two-axle truck": pd.DataFrame(
            {
                "Axle load (kN)": [70.0, 140.0],
                "Spacing to next axle (m)": [4.5, np.nan],
            }
        ),
        "Four-axle vehicle": pd.DataFrame(
            {
                "Axle load (kN)": [60.0, 100.0, 100.0, 100.0],
                "Spacing to next axle (m)": [3.6, 1.3, 1.3, np.nan],
            }
        ),
        "Single axle": pd.DataFrame(
            {
                "Axle load (kN)": [100.0],
                "Spacing to next axle (m)": [np.nan],
            }
        ),
    }
    return presets[name].copy()


st.set_page_config(
    page_title="Moving Load Detector",
    page_icon="🚛",
    layout="wide",
)

st.title("Moving Load Detector & Influence Line Response Engine")
st.caption(
    "Quasi-static multi-axle vehicle analysis for simply supported beam structures."
)

with st.sidebar:
    st.header("Beam and response")

    span = st.number_input(
        "Beam span L (m)",
        min_value=0.1,
        value=20.0,
        step=0.5,
        format="%.3f",
    )

    response_type: ResponseType = st.selectbox(
        "Response type",
        [
            "Left support reaction",
            "Right support reaction",
            "Shear at section",
            "Bending moment at section",
        ],
        index=3,
    )

    section_disabled = "reaction" in response_type.lower()
    section = st.number_input(
        "Evaluation section c (m)",
        min_value=0.001,
        max_value=max(float(span) - 0.001, 0.001),
        value=min(float(span) / 2.0, max(float(span) - 0.001, 0.001)),
        step=0.25,
        format="%.3f",
        disabled=section_disabled,
    )
    if section_disabled:
        section = float(span) / 2.0

    shear_side = "Right face"
    if response_type == "Shear at section":
        shear_side = st.radio(
            "Shear value when an axle is exactly at c",
            ["Right face", "Left face"],
            horizontal=True,
        )

    st.header("Analysis controls")

    step = st.number_input(
        "Movement step Δs (m)",
        min_value=0.001,
        value=0.05,
        step=0.01,
        format="%.4f",
    )

    include_events = st.checkbox(
        "Add axle entry, exit, and section-crossing positions",
        value=True,
    )

    load_factor = st.number_input(
        "Load factor γ",
        min_value=0.0,
        value=1.0,
        step=0.05,
        format="%.3f",
    )

    dynamic_factor = st.number_input(
        "Dynamic amplification factor",
        min_value=0.0,
        value=1.0,
        step=0.05,
        format="%.3f",
    )

st.subheader("Vehicle definition")

preset = st.selectbox(
    "Vehicle preset",
    [
        "Three-axle reference",
        "Two-axle truck",
        "Four-axle vehicle",
        "Single axle",
    ],
)

preset_key = f"preset::{preset}"
if st.session_state.get("active_preset") != preset:
    st.session_state["vehicle_table"] = load_vehicle_preset(preset)
    st.session_state["active_preset"] = preset

vehicle_table = st.data_editor(
    st.session_state["vehicle_table"],
    num_rows="dynamic",
    use_container_width=True,
    hide_index=True,
    column_config={
        "Axle load (kN)": st.column_config.NumberColumn(
            min_value=0.0,
            format="%.3f",
            required=True,
        ),
        "Spacing to next axle (m)": st.column_config.NumberColumn(
            min_value=0.001,
            format="%.3f",
            help="The final row should remain blank because the last axle has no following axle.",
        ),
    },
    key="vehicle_editor",
)

loads = pd.to_numeric(
    vehicle_table["Axle load (kN)"], errors="coerce"
).dropna().to_numpy(dtype=float)

spacing_column = pd.to_numeric(
    vehicle_table["Spacing to next axle (m)"], errors="coerce"
)
spacings = spacing_column.iloc[: max(len(loads) - 1, 0)].dropna().to_numpy(dtype=float)

errors = validate_inputs(
    float(span),
    float(section),
    float(step),
    loads,
    spacings,
)

if errors:
    for error in errors:
        st.error(error)
    st.stop()

vehicle = Vehicle(axle_loads=loads, axle_spacings=spacings)
influence_function = get_influence_function(response_type, shear_side)

positions = build_position_grid(
    span=float(span),
    offsets=vehicle.axle_offsets,
    section=float(section),
    step=float(step),
    include_section_events=include_events,
)

history, contributions = solve_response_history(
    span=float(span),
    section=float(section),
    vehicle=vehicle,
    positions=positions,
    influence_function=influence_function,
    load_factor=float(load_factor),
    dynamic_factor=float(dynamic_factor),
)

max_idx = int(history["response"].idxmax())
min_idx = int(history["response"].idxmin())
abs_idx = int(history["response"].abs().idxmax())

max_row = history.loc[max_idx]
min_row = history.loc[min_idx]
abs_row = history.loc[abs_idx]

response_unit = format_response_unit(response_type)
factored_loads = vehicle.axle_loads * float(load_factor) * float(dynamic_factor)

metric_cols = st.columns(5)
metric_cols[0].metric("Axles", len(vehicle.axle_loads))
metric_cols[1].metric("Vehicle load", f"{vehicle.total_load:.3f} kN")
metric_cols[2].metric("Axle-train length", f"{vehicle.length:.3f} m")
metric_cols[3].metric("Analysis positions", f"{len(history):,}")
metric_cols[4].metric("Factored total load", f"{factored_loads.sum():.3f} kN")

tab_results, tab_influence, tab_configuration, tab_data = st.tabs(
    [
        "Results",
        "Influence line",
        "Critical configuration",
        "Data",
    ]
)

with tab_results:
    summary_cols = st.columns(3)
    summary_cols[0].metric(
        "Maximum response",
        f"{max_row['response']:.6g} {response_unit}",
        help=f"At s = {max_row['vehicle_position_m']:.4f} m",
    )
    summary_cols[1].metric(
        "Minimum response",
        f"{min_row['response']:.6g} {response_unit}",
        help=f"At s = {min_row['vehicle_position_m']:.4f} m",
    )
    summary_cols[2].metric(
        "Maximum absolute response",
        f"{abs_row['response']:.6g} {response_unit}",
        help=f"At s = {abs_row['vehicle_position_m']:.4f} m",
    )

    st.plotly_chart(
        make_response_figure(history, max_row, min_row),
        use_container_width=True,
    )
    st.plotly_chart(
        make_envelope_figure(history),
        use_container_width=True,
    )

    with st.expander("Optional parabolic peak estimates"):
        max_refined = refine_peak_parabolically(history, "max")
        min_refined = refine_peak_parabolically(history, "min")

        if max_refined:
            st.write(
                f"Estimated maximum near **s = {max_refined[0]:.6f} m**, "
                f"response = **{max_refined[1]:.6g} {response_unit}**."
            )
        else:
            st.write("Maximum could not be refined using the neighboring three points.")

        if min_refined:
            st.write(
                f"Estimated minimum near **s = {min_refined[0]:.6f} m**, "
                f"response = **{min_refined[1]:.6g} {response_unit}**."
            )
        else:
            st.write("Minimum could not be refined using the neighboring three points.")

with tab_influence:
    st.plotly_chart(
        make_influence_figure(
            float(span),
            float(section),
            influence_function,
            response_type,
        ),
        use_container_width=True,
    )

    x_check = np.array([0.0, float(section), float(span)])
    y_check = influence_function(x_check, float(span), float(section))
    st.dataframe(
        pd.DataFrame(
            {
                "x (m)": x_check,
                "Influence ordinate": y_check,
            }
        ),
        use_container_width=True,
        hide_index=True,
    )

with tab_configuration:
    critical_choice = st.radio(
        "Show configuration for",
        ["Maximum", "Minimum", "Maximum absolute"],
        horizontal=True,
    )

    selected_row = {
        "Maximum": max_row,
        "Minimum": min_row,
        "Maximum absolute": abs_row,
    }[critical_choice]

    selected_s = float(selected_row["vehicle_position_m"])

    st.plotly_chart(
        make_vehicle_figure(
            float(span),
            float(section),
            vehicle,
            selected_s,
            factored_loads,
            title=(
                f"{critical_choice} configuration at "
                f"s = {selected_s:.4f} m"
            ),
        ),
        use_container_width=True,
    )

    critical_contributions = contributions[
        np.isclose(
            contributions["vehicle_position_m"],
            selected_s,
            rtol=0.0,
            atol=1e-10,
        )
    ].copy()

    critical_contributions["contribution_percent_abs"] = 0.0
    denominator = critical_contributions["contribution"].abs().sum()
    if denominator > 0:
        critical_contributions["contribution_percent_abs"] = (
            critical_contributions["contribution"].abs() / denominator * 100.0
        )

    critical_contributions = critical_contributions[
        [
            "axle",
            "factored_axle_load_kN",
            "axle_position_m",
            "active",
            "influence_ordinate",
            "contribution",
            "contribution_percent_abs",
        ]
    ]

    st.dataframe(
        critical_contributions,
        use_container_width=True,
        hide_index=True,
        column_config={
            "axle": "Axle",
            "factored_axle_load_kN": st.column_config.NumberColumn(
                "Factored load (kN)", format="%.4f"
            ),
            "axle_position_m": st.column_config.NumberColumn(
                "Position x (m)", format="%.5f"
            ),
            "active": "Active",
            "influence_ordinate": st.column_config.NumberColumn(
                "IL ordinate", format="%.7f"
            ),
            "contribution": st.column_config.NumberColumn(
                f"Contribution ({response_unit})", format="%.7g"
            ),
            "contribution_percent_abs": st.column_config.ProgressColumn(
                "Absolute contribution share",
                min_value=0.0,
                max_value=100.0,
                format="%.2f%%",
            ),
        },
    )

with tab_data:
    st.write("Response history")
    st.dataframe(history, use_container_width=True, hide_index=True)

    st.write("Axle-wise contribution records")
    st.dataframe(contributions, use_container_width=True, hide_index=True)

    history_csv = history.to_csv(index=False).encode("utf-8")
    contribution_csv = contributions.to_csv(index=False).encode("utf-8")

    download_cols = st.columns(2)
    download_cols[0].download_button(
        "Download response history CSV",
        data=history_csv,
        file_name="moving_load_response_history.csv",
        mime="text/csv",
        use_container_width=True,
    )
    download_cols[1].download_button(
        "Download axle contributions CSV",
        data=contribution_csv,
        file_name="moving_load_axle_contributions.csv",
        mime="text/csv",
        use_container_width=True,
    )

st.divider()
st.caption(
    "Assumptions: linear elastic response, quasi-static concentrated axle loads, "
    "simply supported beam, and influence-line superposition."
)
