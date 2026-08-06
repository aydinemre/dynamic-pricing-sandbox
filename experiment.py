"""
experiment.py — A/B and switchback experiment simulation + DiD analysis.

Given a treatment optimizer and a control optimizer, we simulate a randomized
experiment on the synthetic marketplace and estimate the GMV lift and Trip lift
with Difference-in-Differences logic:

  * A/B  (spatial): hexes are randomized to treatment / control. A pre-period
    (everyone on control) enables a classic 2x2 DiD:
        DiD = (T_post - T_pre) - (C_post - C_pre).

  * Switchback (temporal): the whole city alternates treatment / control by time
    block. The effect is a two-way (day x hour-of-day) fixed-effects difference
    -- the switchback analogue of DiD -- which removes time-of-day and day
    confounding before comparing treated vs control blocks.

Both return a chronological timeline (for live tracking) and a DiD result dict
(lift %, bootstrap CI, p-value) for GMV and trips.
"""

from __future__ import annotations

import os
import json
import textwrap

import numpy as np
import pandas as pd

from engine import simulate_outcomes, temporal_split


PRE_FRAC = 0.5          # first half of days = pre-period (baseline on control)
N_BOOT = 400            # bootstrap resamples for CIs


# ---------------------------------------------------------------------------
# Outcome helper
# ---------------------------------------------------------------------------
def _apply(market: pd.DataFrame, mask, treat, ctrl) -> np.ndarray:
    """Multiplier per row: treatment optimizer where mask, else control."""
    m = np.ones(len(market))
    if mask.any():
        m[mask.to_numpy()] = treat.price(market[mask])
    if (~mask).any():
        m[(~mask).to_numpy()] = ctrl.price(market[~mask])
    return m


def _outcomes(market: pd.DataFrame, m: np.ndarray) -> pd.DataFrame:
    out = simulate_outcomes(market["demand"].to_numpy(),
                            market["supply"].to_numpy(), m,
                            market["hour"].to_numpy())
    df = market[["hex_id", "day", "hour"]].copy()
    df["gmv"] = out["revenue"]
    df["trips"] = out["rides"]
    df["surge"] = m
    return df


# ---------------------------------------------------------------------------
# A/B (spatial randomization)  ->  classic 2x2 DiD
# ---------------------------------------------------------------------------
def run_ab(city, market, treat, ctrl, seed=7):
    rng = np.random.default_rng(seed)
    hex_ids = city["hex_id"].to_numpy()
    treat_hexes = set(rng.choice(hex_ids, size=len(hex_ids) // 2, replace=False))
    grp = {h: ("T" if h in treat_hexes else "C") for h in hex_ids}

    pre_cut = int(market["day"].max() * PRE_FRAC)
    market = market.copy()
    market["group"] = market["hex_id"].map(grp)
    market["period"] = np.where(market["day"] <= pre_cut, "pre", "exp")
    # treatment applied only to T hexes in the exp period
    mask = (market["period"] == "exp") & (market["group"] == "T")
    m = _apply(market, mask, treat, ctrl)
    out = _outcomes(market, m)
    out["group"] = market["group"].to_numpy()
    out["period"] = market["period"].to_numpy()

    # per (hex, day) totals
    hd = (out.groupby(["hex_id", "day", "group", "period"], as_index=False)
             [["gmv", "trips"]].sum())

    # timeline: per-day group means, chronological (pre then exp)
    tl = (hd.groupby(["day", "group"], as_index=False)[["gmv", "trips"]].mean()
            .sort_values(["day", "group"]))
    tl["period"] = np.where(tl["day"] <= pre_cut, "pre", "exp")
    meta = {"design": "A/B", "pre_cut": pre_cut,
            "n_treat_hex": len(treat_hexes), "n_ctrl_hex": len(hex_ids) - len(treat_hexes)}
    return hd, tl, meta


def _ab_point(frame: pd.DataFrame, col: str):
    """Classic 2x2 DiD on per-hex-day means. Returns (abs_effect, lift_pct)."""
    g = frame.groupby(["group", "period"])[col].mean()
    t_pre, t_post = g.get(("T", "pre"), np.nan), g.get(("T", "exp"), np.nan)
    c_pre, c_post = g.get(("C", "pre"), np.nan), g.get(("C", "exp"), np.nan)
    did_abs = (t_post - t_pre) - (c_post - c_pre)
    counterfactual = t_pre + (c_post - c_pre)     # what T would have been
    lift = 100 * did_abs / counterfactual if counterfactual else np.nan
    return did_abs, lift


def point_lift_ab(hd: pd.DataFrame, col: str) -> float:
    return _ab_point(hd, col)[1]


def _did_2x2(hd: pd.DataFrame, col: str, rng):
    """Classic DiD on per-hex-day means, with bootstrap over hexes."""
    def did(frame):
        return _ab_point(frame, col)

    did_abs, lift = did(hd)
    # bootstrap: resample hexes within each group
    hexes = hd[["hex_id", "group"]].drop_duplicates()
    t_h = hexes[hexes.group == "T"]["hex_id"].to_numpy()
    c_h = hexes[hexes.group == "C"]["hex_id"].to_numpy()
    boots = []
    for _ in range(N_BOOT):
        samp = np.concatenate([rng.choice(t_h, len(t_h)), rng.choice(c_h, len(c_h))])
        sub = pd.concat([hd[hd.hex_id == h] for h in samp], ignore_index=True) \
            if False else hd.set_index("hex_id").loc[samp].reset_index()
        boots.append(did(sub)[1])
    boots = np.array([b for b in boots if np.isfinite(b)])
    lo, hi = np.percentile(boots, [2.5, 97.5])
    p = 2 * min((boots <= 0).mean(), (boots >= 0).mean())
    return {"lift_pct": float(lift), "ci": (float(lo), float(hi)),
            "p_value": float(p), "abs": float(did_abs)}


def did_ab(hd: pd.DataFrame, seed=11):
    rng = np.random.default_rng(seed)
    return {"gmv": _did_2x2(hd, "gmv", rng), "trips": _did_2x2(hd, "trips", rng)}


# ---------------------------------------------------------------------------
# Switchback (temporal randomization)  ->  two-way FE difference
# ---------------------------------------------------------------------------
def run_switchback(city, market, treat, ctrl, seed=7):
    rng = np.random.default_rng(seed)
    pre_cut = int(market["day"].max() * PRE_FRAC)
    market = market.copy()
    market["period"] = np.where(market["day"] <= pre_cut, "pre", "exp")

    # assign each (day,hour) exp slot to treatment/control, balanced within day
    exp = market[market["period"] == "exp"]
    slots = exp[["day", "hour"]].drop_duplicates().sort_values(["day", "hour"])
    treated = {}
    for day, g in slots.groupby("day"):
        hrs = g["hour"].to_numpy()
        pick = rng.permutation(hrs)[: len(hrs) // 2]
        for h in hrs:
            treated[(day, h)] = h in set(pick)
    market["treated"] = [
        treated.get((d, h), False) if p == "exp" else False
        for d, h, p in zip(market["day"], market["hour"], market["period"])
    ]
    mask = market["treated"] & (market["period"] == "exp")
    m = _apply(market, mask, treat, ctrl)
    out = _outcomes(market, m)
    out["treated"] = market["treated"].to_numpy()
    out["period"] = market["period"].to_numpy()

    # city totals per (day, hour) slot
    slot = (out.groupby(["day", "hour", "treated", "period"], as_index=False)
               [["gmv", "trips"]].sum().sort_values(["day", "hour"]))
    slot["t"] = range(len(slot))
    meta = {"design": "Switchback", "pre_cut": pre_cut,
            "n_treat_slot": int((slot.period == "exp").sum() and slot[slot.period == "exp"]["treated"].sum()),
            "n_ctrl_slot": int((slot.period == "exp").sum() and (~slot[slot.period == "exp"]["treated"]).sum())}
    return slot, slot, meta


def _twfe_point(frame: pd.DataFrame, col: str):
    """Two-way (day, hour-of-day) demeaned treated-vs-control diff. (abs, lift%)."""
    grand = frame[col].mean()
    day_m = frame.groupby("day")[col].transform("mean")
    hr_m = frame.groupby("hour")[col].transform("mean")
    resid = frame[col] - day_m - hr_m + grand
    eff = resid[frame.treated].mean() - resid[~frame.treated].mean()
    base = frame[~frame.treated][col].mean()
    return eff, (100 * eff / base if base else np.nan)


def point_lift_sb(slot: pd.DataFrame, col: str) -> float:
    exp = slot[slot["period"] == "exp"]
    if exp["treated"].nunique() < 2:
        return np.nan
    return _twfe_point(exp, col)[1]


def _did_twfe(slot: pd.DataFrame, col: str, rng):
    """Two-way (day, hour-of-day) demeaned treated-vs-control diff, bootstrap over days."""
    exp = slot[slot["period"] == "exp"].copy()
    if exp["treated"].nunique() < 2:
        return {"lift_pct": np.nan, "ci": (np.nan, np.nan), "p_value": np.nan, "abs": np.nan}

    def effect(frame):
        return _twfe_point(frame, col)

    eff_abs, lift = effect(exp)
    days = exp["day"].unique()
    boots = []
    for _ in range(N_BOOT):
        samp = rng.choice(days, len(days))
        sub = exp.set_index("day").loc[samp].reset_index()
        boots.append(effect(sub)[1])
    boots = np.array([b for b in boots if np.isfinite(b)])
    lo, hi = np.percentile(boots, [2.5, 97.5])
    p = 2 * min((boots <= 0).mean(), (boots >= 0).mean())
    return {"lift_pct": float(lift), "ci": (float(lo), float(hi)),
            "p_value": float(p), "abs": float(eff_abs)}


def did_switchback(slot: pd.DataFrame, seed=11):
    rng = np.random.default_rng(seed)
    return {"gmv": _did_twfe(slot, "gmv", rng), "trips": _did_twfe(slot, "trips", rng)}


# ---------------------------------------------------------------------------
# Unified entry point
# ---------------------------------------------------------------------------
def run_experiment(city, market, treat, ctrl, design="A/B", seed=7):
    if design == "A/B":
        hd, tl, meta = run_ab(city, market, treat, ctrl, seed)
        return {"records": hd, "timeline": tl, "meta": meta, "did_fn": did_ab}
    slot, tl, meta = run_switchback(city, market, treat, ctrl, seed)
    return {"records": slot, "timeline": tl, "meta": meta, "did_fn": did_switchback}


# ---------------------------------------------------------------------------
# Experiment Narrator (LLM)  — turns the DiD readout into an exec conclusion
# ---------------------------------------------------------------------------
NARRATOR_PROMPT = textwrap.dedent("""\
    You are a senior data scientist writing the conclusion of a ride-hailing
    dynamic-pricing experiment. Given the Difference-in-Differences results below,
    write <=120 words that: (1) state the GMV and Trip lift with significance,
    (2) name the key trade-off, (3) give ONE ship/iterate recommendation with a
    guardrail to watch, (4) flag one next experiment. Be specific, no preamble.

    EXPERIMENT (JSON):
    {payload}
    """)


def _fallback_narrative(design, treat, ctrl, did):
    g, t = did["gmv"], did["trips"]
    sig = "significant" if g["p_value"] < 0.05 and t["p_value"] < 0.05 else "not significant"
    ship = (g["lift_pct"] > 0 and t["lift_pct"] > 0 and g["p_value"] < 0.05)
    rec = ("Ship the treatment and ramp gradually" if ship
           else "Hold — iterate before shipping")
    gmv_trade = ("GMV and trips both rise" if g["lift_pct"] > 0 and t["lift_pct"] > 0
                 else "trips rise but GMV slips — the optimizer trades revenue for "
                      "fulfilment" if t["lift_pct"] > 0 > g["lift_pct"]
                 else "mixed movement across GMV and trips")
    return textwrap.dedent(f"""\
        In a {design} test of **{treat}** vs **{ctrl}**, DiD estimates a
        **GMV lift of {g['lift_pct']:+.1f}%** (95% CI [{g['ci'][0]:+.1f}, {g['ci'][1]:+.1f}])
        and a **Trip lift of {t['lift_pct']:+.1f}%** (95% CI [{t['ci'][0]:+.1f}, {t['ci'][1]:+.1f}])
        — {sig}. Here {gmv_trade}; the trade-off to watch is average surge vs rider
        price sensitivity. **Recommendation: {rec}**, with 95th-percentile rider wait
        and average surge as guardrails. Next experiment: re-run as a switchback per
        zone-cluster and test an online/streaming surge update that re-solves within
        the day rather than a daily batch.""")


def narrate_did(design, treat, ctrl, did):
    """Exec conclusion from a DiD result. Uses an LLM if a key is set, else a
    deterministic template (so the app always produces a narrative)."""
    payload = {"design": design, "treatment": treat, "control": ctrl,
               "gmv": did["gmv"], "trips": did["trips"]}
    prompt = NARRATOR_PROMPT.format(payload=json.dumps(payload, indent=2))
    try:
        if os.environ.get("OPENAI_API_KEY"):
            from openai import OpenAI
            r = OpenAI().chat.completions.create(
                model="gpt-4o-mini", temperature=0.3,
                messages=[{"role": "user", "content": prompt}])
            return "🧠 " + r.choices[0].message.content.strip()
        if os.environ.get("ANTHROPIC_API_KEY"):
            import anthropic
            r = anthropic.Anthropic().messages.create(
                model="claude-sonnet-4-5", max_tokens=400,
                messages=[{"role": "user", "content": prompt}])
            return "🧠 " + r.content[0].text.strip()
    except Exception:
        pass
    return _fallback_narrative(design, treat, ctrl, did)
