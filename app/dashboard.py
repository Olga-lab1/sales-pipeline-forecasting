"""
Sales Pipeline Health & Revenue Forecasting Dashboard
Run:  streamlit run app/dashboard.py
"""

import sys
from pathlib import Path

import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from forecasting import backtest_mape, forecast_bookings, quarter_projection  # noqa: E402
from metrics import (funnel_conversion, headline_kpis,  # noqa: E402
                     load_opportunities, loss_reasons, monthly_bookings,
                     pipeline_coverage, rep_leaderboard, win_rate_by)

# ---------------------------------------------------------------- page setup
st.set_page_config(page_title="Pipeline Command Center",
                   page_icon="📈", layout="wide")

INK = "#1B2A41"       # deep navy — primary
ACCENT = "#00A6A6"    # teal — bookings / positive
WARN = "#F26419"      # signal orange — risk
MUTE = "#8A94A6"

PLOT_LAYOUT = dict(
    font=dict(family="Source Sans Pro, sans-serif", color=INK, size=13),
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    margin=dict(l=10, r=10, t=40, b=10), hovermode="x unified",
)


def money(x: float) -> str:
    if abs(x) >= 1e6:
        return f"${x/1e6:,.1f}M"
    return f"${x/1e3:,.0f}K"


@st.cache_data
def get_data():
    df = load_opportunities(str(ROOT / "data" / "opportunities.csv"))
    return df


df = get_data()

# ---------------------------------------------------------------- sidebar
st.sidebar.title("Filters")
seg = st.sidebar.multiselect("Segment", sorted(df.segment.unique()),
                             default=sorted(df.segment.unique()))
reg = st.sidebar.multiselect("Region", sorted(df.region.unique()),
                             default=sorted(df.region.unique()))
quota = st.sidebar.number_input("Quarterly quota ($M)", min_value=1.0,
                                value=9.0, step=0.5) * 1e6
horizon = st.sidebar.slider("Forecast horizon (months)", 3, 12, 6)

f = df[df.segment.isin(seg) & df.region.isin(reg)]
if f.empty:
    st.warning("No opportunities match the current filters.")
    st.stop()

# ---------------------------------------------------------------- header KPIs
st.title("Pipeline Command Center")
st.caption("Synthetic B2B SaaS CRM data · Jan 2023 – Jun 2026 · all figures illustrative")

k = headline_kpis(f)
cov = pipeline_coverage(f, quota)

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Open pipeline", money(k["open_pipeline"]),
          f"{k['open_deals']} deals")
c2.metric("Weighted pipeline", money(k["weighted_pipeline"]),
          f"{cov['weighted_coverage']:.1f}x quota coverage")
c3.metric("Win rate", f"{k['win_rate']:.0%}",
          f"avg cycle {k['avg_cycle_days']:.0f} days", delta_color="off")
c4.metric("Bookings this quarter", money(k["bookings_this_qtr"]))
c5.metric("Deals with pushed close dates", f"{k['slipped_deals_pct']:.0%}",
          "slippage risk", delta_color="inverse")

st.divider()

# ---------------------------------------------------------------- forecast
left, right = st.columns([3, 2])

monthly = monthly_bookings(f)
fc = forecast_bookings(monthly, horizon=horizon)
mape = backtest_mape(monthly)
qp = quarter_projection(monthly, fc, f.created_date.max())

with left:
    st.subheader("Revenue forecast")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=monthly.month, y=monthly.bookings,
                             name="Actual bookings", mode="lines",
                             line=dict(color=INK, width=2.5)))
    fig.add_trace(go.Scatter(x=fc.month, y=fc.hi80, mode="lines",
                             line=dict(width=0), showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=fc.month, y=fc.lo80, mode="lines",
                             line=dict(width=0), fill="tonexty",
                             fillcolor="rgba(0,166,166,0.18)",
                             name="80% interval", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=fc.month, y=fc.forecast, name="Forecast",
                             mode="lines", line=dict(color=ACCENT, width=2.5, dash="dash")))
    fig.update_layout(**PLOT_LAYOUT, height=380,
                      yaxis=dict(tickprefix="$", gridcolor="#EDEFF3"),
                      legend=dict(orientation="h", y=1.12))
    st.plotly_chart(fig, use_container_width=True)
    st.caption(f"Holt-Winters (trend + yearly seasonality) · rolling-origin backtest "
               f"MAPE {mape:.0%} over 3-month horizons")

with right:
    st.subheader(f"{qp['quarter']} projection")
    gap = qp["projected_total"] - quota
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=qp["projected_total"],
        delta={"reference": quota, "valueformat": "$,.0f"},
        number={"valueformat": "$,.0f"},
        gauge={
            "axis": {"range": [0, max(quota * 1.4, qp["projected_hi"])],
                     "tickformat": "$,.0s"},
            "bar": {"color": ACCENT if gap >= 0 else WARN},
            "threshold": {"line": {"color": INK, "width": 3},
                          "value": quota},
            "steps": [{"range": [qp["projected_lo"], qp["projected_hi"]],
                       "color": "#EDEFF3"}],
        },
    ))
    fig.update_layout(**PLOT_LAYOUT, height=280)
    st.plotly_chart(fig, use_container_width=True)
    verdict = ("**On track.** Projected bookings clear quota; interval floor is "
               f"{money(qp['projected_lo'])}." if qp["projected_lo"] >= quota else
               ("**At risk.** Central projection clears quota but the downside "
                f"scenario ({money(qp['projected_lo'])}) falls short — protect "
                "late-stage deals from slipping." if gap >= 0 else
                f"**Gap to quota: {money(-gap)}.** Prioritize Negotiation-stage "
                "deals and pull forward next quarter's best-qualified pipeline."))
    st.markdown(verdict)

st.divider()

# ---------------------------------------------------------------- funnel + mix
a, b = st.columns(2)

with a:
    st.subheader("Funnel conversion")
    fn = funnel_conversion(f)
    fig = go.Figure(go.Funnel(
        y=fn.stage, x=fn.deals,
        textinfo="value+percent initial",
        marker={"color": [INK, "#33506B", "#4F7396", ACCENT, "#00C4C4"][:len(fn)]},
    ))
    fig.update_layout(**PLOT_LAYOUT, height=360)
    st.plotly_chart(fig, use_container_width=True)

with b:
    st.subheader("Win rate by segment")
    wr = win_rate_by(f, "segment")
    fig = go.Figure()
    fig.add_trace(go.Bar(x=wr.segment, y=wr.win_rate, name="Win rate",
                         marker_color=ACCENT,
                         text=[f"{v:.0%}" for v in wr.win_rate],
                         textposition="outside"))
    fig.update_layout(**PLOT_LAYOUT, height=360,
                      yaxis=dict(tickformat=".0%", range=[0, wr.win_rate.max() * 1.3],
                                 gridcolor="#EDEFF3"))
    st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------- reps + losses
a, b = st.columns(2)

with a:
    st.subheader("Rep leaderboard (closed revenue)")
    lb = rep_leaderboard(f)
    lb_disp = lb.assign(
        revenue=lb.revenue.map(money),
        win_rate=lb.win_rate.map("{:.0%}".format),
        avg_cycle=lb.avg_cycle.map("{:.0f}d".format),
    ).rename(columns={"owner": "Rep", "closed_deals": "Closed",
                      "win_rate": "Win rate", "revenue": "Revenue",
                      "avg_cycle": "Avg cycle"})
    st.dataframe(lb_disp, use_container_width=True, hide_index=True)

with b:
    st.subheader("Why we lose")
    lr = loss_reasons(f)
    fig = go.Figure(go.Bar(
        x=lr.lost_revenue, y=lr.loss_reason, orientation="h",
        marker_color=WARN, text=[money(v) for v in lr.lost_revenue],
        textposition="outside",
    ))
    fig.update_layout(**PLOT_LAYOUT, height=360,
                      xaxis=dict(tickprefix="$", gridcolor="#EDEFF3"),
                      yaxis=dict(autorange="reversed"))
    st.plotly_chart(fig, use_container_width=True)

st.divider()
st.caption("Built with Streamlit, Plotly, and statsmodels · data is fully synthetic "
           "(see src/generate_data.py) · MIT license")
