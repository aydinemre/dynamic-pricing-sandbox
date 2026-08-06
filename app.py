"""
AutoPricing Experimenter — Streamlit app.

An interactive surge-pricing experimentation console:
  * a real H3 map of a synthetic Dubai marketplace,
  * A/B and switchback experiments between two optimizers,
  * live tracking of GMV & trips as the experiment runs,
  * a Difference-in-Differences readout of GMV lift and Trip lift.

Run:  streamlit run app.py
"""
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import streamlit as st
import pydeck as pdk

import engine as e
import experiment as x

st.set_page_config(page_title="AutoPricing Experimenter", page_icon="🚗",
                   layout="wide")


# ---------------------------------------------------------------------------
# Data (cached)
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner="Building synthetic Dubai marketplace…")
def get_data(res, k, days, seed):
    city = e.generate_city(res=res, k=k, seed=seed)
    market = e.generate_market(city, days=days, seed=seed)
    return city, market


def value_to_rgb(vals, cmap="viridis", vmin=None, vmax=None):
    vals = np.asarray(vals, float)
    vmin = float(np.nanmin(vals)) if vmin is None else vmin
    vmax = float(np.nanmax(vals)) if vmax is None else vmax
    norm = mcolors.Normalize(vmin, vmax if vmax > vmin else vmin + 1e-6)
    rgba = plt.get_cmap(cmap)(norm(vals))
    return [[int(r * 255), int(g * 255), int(b * 255)] for r, g, b, _ in rgba]


def ab_timeline(records: pd.DataFrame):
    g = records.groupby(["day", "group"])[["gmv", "trips"]].mean().reset_index()
    gmv = g.pivot(index="day", columns="group", values="gmv").rename(
        columns={"T": "GMV · treatment", "C": "GMV · control"})
    trip = g.pivot(index="day", columns="group", values="trips").rename(
        columns={"T": "Trips · treatment", "C": "Trips · control"})
    return gmv, trip


def sb_timeline(records: pd.DataFrame):
    s = records.sort_values("t").copy()
    s["GMV · treatment"] = np.where(s["treated"], s["gmv"], np.nan)
    s["GMV · control"] = np.where(~s["treated"], s["gmv"], np.nan)
    s["Trips · treatment"] = np.where(s["treated"], s["trips"], np.nan)
    s["Trips · control"] = np.where(~s["treated"], s["trips"], np.nan)
    roll = (s.set_index("t")[["GMV · treatment", "GMV · control",
                              "Trips · treatment", "Trips · control"]]
              .rolling(16, min_periods=1).mean())
    return roll[["GMV · treatment", "GMV · control"]], \
        roll[["Trips · treatment", "Trips · control"]]


def hex_deck(slot: pd.DataFrame, value_col: str, cmap: str):
    d = slot.copy()
    d["color"] = value_to_rgb(d[value_col], cmap=cmap)
    d["ratio_s"] = d["ratio"].round(2)
    d["surge_s"] = d["surge"].round(2)
    d["rides_s"] = d["rides"].round(0)
    layer = pdk.Layer(
        "H3HexagonLayer", d, get_hexagon="h3", get_fill_color="color",
        pickable=True, opacity=0.55, stroked=True, filled=True,
        extruded=False, get_line_color=[255, 255, 255], line_width_min_pixels=1,
    )
    view = pdk.ViewState(latitude=e.CITY_CENTER[0], longitude=e.CITY_CENTER[1],
                         zoom=10.2, pitch=0)
    tooltip = {"text": "zone: {zone}\nimbalance D/S: {ratio_s}\n"
                       "surge: {surge_s}x\nrides: {rides_s}"}
    return pdk.Deck(layers=[layer], initial_view_state=view,
                    map_provider="carto", map_style="light", tooltip=tooltip)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
st.sidebar.title("🚗 AutoPricing Experimenter")
st.sidebar.caption("Synthetic ride-hailing dynamic-pricing lab · real H3 · DiD")

names = e.OPTIMIZER_NAMES
treat_name = st.sidebar.selectbox("Treatment optimizer", names,
                                  index=names.index("Optimizer · dynamic (max rides)"))
ctrl_name = st.sidebar.selectbox("Control optimizer", names,
                                 index=names.index("Flat (no surge)"))
design = st.sidebar.radio("Experiment design", ["A/B", "Switchback"], horizontal=True)
seed = st.sidebar.number_input("Random seed", 1, 9999, 7)
k = st.sidebar.slider("City size (H3 rings)", 3, 7, e.CITY_K)
days = st.sidebar.slider("Days simulated", 21, 60, e.N_DAYS)

city, market = get_data(e.H3_RES, k, days, seed)
treat, ctrl = e.build_optimizer(treat_name), e.build_optimizer(ctrl_name)

st.sidebar.markdown("---")
st.sidebar.metric("Hex cells", len(city))
st.sidebar.metric("(hex,day,hour) cells", f"{len(market):,}")

tab_map, tab_exp, tab_res, tab_doc = st.tabs(
    ["🗺️ Live map", "🧪 Experiment (live)", "📊 Results (DiD)", "📖 How it works"])


# ---------------------------------------------------------------------------
# Map tab
# ---------------------------------------------------------------------------
with tab_map:
    st.subheader("Where surge fires — real H3 grid over Dubai")
    c1, c2, c3 = st.columns(3)
    viz_opt = c1.selectbox("Optimizer shown on map", names,
                           index=names.index(treat_name))
    day_sel = c2.slider("Day", 0, int(market["day"].max()), int(market["day"].max()))
    hour_sel = c3.slider("Hour", min(e.HOURS), max(e.HOURS), 18)
    metric = st.radio("Colour by", ["Demand/supply imbalance", "Surge multiplier"],
                      horizontal=True)
    slot = e.map_frame(city, market, e.build_optimizer(viz_opt), day_sel, hour_sel)
    col, cmap = (("ratio", "magma") if metric.startswith("Demand")
                 else ("surge", "viridis"))
    st.pydeck_chart(hex_deck(slot, col, cmap), use_container_width=True)
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Avg imbalance D/S", f"{slot['ratio'].mean():.2f}")
    m2.metric("Avg surge", f"{slot['surge'].mean():.2f}×")
    m3.metric("Completed rides (slot)", f"{slot['rides'].sum():.0f}")
    m4.metric("Hexes surging (>1.05×)", int((slot['surge'] > 1.05).sum()))
    st.caption("At rush hour the busiest hexes show the biggest demand/supply gap; "
               "switch 'Colour by' to Surge multiplier to see the optimizer put "
               "surge exactly there.")


# ---------------------------------------------------------------------------
# Experiment tab (live)
# ---------------------------------------------------------------------------
with tab_exp:
    st.subheader(f"{design} experiment · {treat_name}  vs  {ctrl_name}")
    st.caption("Treatment and control run side by side; watch GMV and trips diverge, "
               "then read the DiD lift. Results persist in the Results tab.")
    go = st.button("▶ Run experiment", type="primary")

    if go:
        try:
            res = x.run_experiment(city, market, treat, ctrl, design=design, seed=seed)
            did = res["did_fn"](res["records"])
            # persist BEFORE the (interruptible) animation so results always survive
            st.session_state["exp"] = {
                "design": design, "records": res["records"], "meta": res["meta"],
                "did": did, "treat": treat_name, "ctrl": ctrl_name}
        except Exception as ex:                     # surface any failure in-app
            st.exception(ex)

    if "exp" in st.session_state:
        exp = st.session_state["exp"]
        recs = exp["records"]
        gmv_df, trip_df = (ab_timeline(recs) if exp["design"] == "A/B"
                           else sb_timeline(recs))

        k1, k2, k3 = st.columns(3)
        k1.metric("GMV lift (DiD)", f"{exp['did']['gmv']['lift_pct']:+.1f}%")
        k2.metric("Trip lift (DiD)", f"{exp['did']['trips']['lift_pct']:+.1f}%")
        k3.metric("Design", exp["design"])

        st.markdown("**GMV** — treatment vs control")
        g_ph = st.empty()
        st.markdown("**Trips** — treatment vs control")
        t_ph = st.empty()

        if go:                                       # animated reveal on this run
            n = len(gmv_df)
            step = max(1, n // 24)
            for j in range(step, n + step, step):
                g_ph.line_chart(gmv_df.iloc[:min(j, n)], height=230)
                t_ph.line_chart(trip_df.iloc[:min(j, n)], height=230)
                time.sleep(0.04)
        g_ph.line_chart(gmv_df, height=230)
        t_ph.line_chart(trip_df, height=230)
        st.caption("Treatment (LP optimizer) pulls ahead of control after the "
                   "experiment starts — quantified as DiD lift in the Results tab.")


# ---------------------------------------------------------------------------
# Results tab
# ---------------------------------------------------------------------------
with tab_res:
    st.subheader("Difference-in-Differences readout")
    if "exp" not in st.session_state:
        st.info("Run an experiment first (🧪 Experiment tab).")
    else:
        s = st.session_state["exp"]
        did, meta = s["did"], s["meta"]
        st.markdown(f"**Design:** {s['design']}  ·  **Treatment:** {s['treat']}  "
                    f"·  **Control:** {s['ctrl']}")

        def card(colobj, title, r):
            sig = "✅ significant" if r["p_value"] < 0.05 else "⚠️ not significant"
            colobj.metric(title, f"{r['lift_pct']:+.1f}%",
                          help=f"95% CI [{r['ci'][0]:+.1f}%, {r['ci'][1]:+.1f}%]")
            colobj.caption(f"95% CI [{r['ci'][0]:+.1f}%, {r['ci'][1]:+.1f}%] · "
                           f"p={r['p_value']:.3f} · {sig}")

        c1, c2 = st.columns(2)
        card(c1, "GMV lift", did["gmv"])
        card(c2, "Trip lift", did["trips"])

        st.markdown("#### Estimate table")
        tbl = pd.DataFrame({
            "metric": ["GMV", "Trips"],
            "lift %": [round(did["gmv"]["lift_pct"], 2), round(did["trips"]["lift_pct"], 2)],
            "CI low %": [round(did["gmv"]["ci"][0], 2), round(did["trips"]["ci"][0], 2)],
            "CI high %": [round(did["gmv"]["ci"][1], 2), round(did["trips"]["ci"][1], 2)],
            "p-value": [round(did["gmv"]["p_value"], 3), round(did["trips"]["p_value"], 3)],
        }).set_index("metric")
        st.table(tbl)

        method = ("Classic 2×2 DiD on per-hex/day outcomes: "
                  "(Tₚₒₛₜ−Tₚᵣₑ)−(Cₚₒₛₜ−Cₚᵣₑ); CI by bootstrap over hexes."
                  if s["design"] == "A/B" else
                  "Two-way (day × hour-of-day) fixed-effects difference on city "
                  "totals — the switchback analogue of DiD; CI by block bootstrap "
                  "over days.")
        st.info(f"**Method.** {method}")
        gl = did["gmv"]["lift_pct"]; tl = did["trips"]["lift_pct"]
        verdict = ("Ship it — both GMV and trips improve with tight CIs."
                   if did["gmv"]["p_value"] < 0.05 and did["trips"]["p_value"] < 0.05
                   and gl > 0 and tl > 0 else "Mixed / inconclusive — iterate.")
        st.markdown(f"**Verdict:** {verdict}")

        st.markdown("#### 🧠 AI Experiment Narrator")
        st.caption("An LLM (or deterministic fallback) turns the DiD numbers into an "
                   "executive conclusion.")
        st.success(x.narrate_did(s["design"], s["treat"], s["ctrl"], did))


# ---------------------------------------------------------------------------
# Docs tab
# ---------------------------------------------------------------------------
with tab_doc:
    st.subheader("How it works — five small modules")
    st.markdown(r"""
The pricing system is deliberately simple and split into **five independent
modules** (the standard, public decomposition):

1. **Demand forecast** — makes up a demand $D$ per hex-hour (random draw around an
   hourly base level).
2. **Supply forecast** — makes up a supply $S$ per hex-hour (damped hourly shape,
   so supply lags demand at the peaks).
3. **Demand elasticity** — how riders react to price, **by hour of day**: rush
   hours are *inelastic* (people must ride), off-peak is *elastic*.
4. **Supply elasticity** — how drivers react to price, **by hour**: drivers chase
   surge most in the evening / late hours.
5. **Optimizer** — for each hex it just tries a grid of multipliers and keeps the
   best. Predicted response at multiplier $m$:
""")
    st.latex(r"D(m) = D\,[\,1-\varepsilon^{d}_{h}\,(m-1)\,],\qquad "
             r"S(m) = S\,[\,1+\varepsilon^{s}_{h}\,(m-1)\,]")
    st.latex(r"m^{*} = \arg\max_{m\in[1,\,M]}\ "
             r"\text{objective}\big(\min(D(m),\,S(m))\big)")
    st.markdown("No solver, no LP — a plain per-hex grid search. Maximising **rides** "
                "clears the supply/demand gap; a **GMV** objective pushes surge higher "
                "and trades trips for revenue.")

    st.markdown("#### The dynamic (hour-of-day) elasticities")
    el_df = pd.DataFrame({
        "demand elasticity": e.DEMAND_ELASTICITY_BY_HOUR,
        "supply elasticity": e.SUPPLY_ELASTICITY_BY_HOUR,
    }, index=pd.Index(e.HOURS, name="hour"))
    st.line_chart(el_df, height=240)

    st.markdown("#### Optimizers available")
    st.table(pd.DataFrame({
        "optimizer": names,
        "what it does": [
            "No surge — baseline (m = 1).",
            "Grid-search per hex using the hourly elasticities; maximises rides.",
            "Same, but one average elasticity (ignores the hour) — a bit worse.",
            "Maximises GMV instead of rides → over-surges, trades trips for revenue.",
        ],
    }))
