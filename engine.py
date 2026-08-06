"""
engine.py — a deliberately simple, modular dynamic-pricing demo on a real H3 grid.

Five small, independent modules (the standard, public decomposition):

    1. Demand forecast   — makes up a (random) demand per hex-hour   [DemandForecast]
    2. Supply forecast   — makes up a (random) supply per hex-hour   [SupplyForecast]
    3. Demand elasticity — how demand reacts to price, BY HOUR        [DemandElasticity]
    4. Supply elasticity — how supply reacts to price, BY HOUR        [SupplyElasticity]
    5. Optimizer         — grid-searches the surge that's best        [SurgeOptimizer]

The optimizer is trivial and written from scratch: for each hex it tries a grid of
surge multipliers, predicts rides/GMV from the (hour-dependent) elasticities, and
keeps the best. No solver, no LP. Everything is synthetic; nothing here is
confidential. Geography is Dubai only so the map looks real.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
import h3


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SEED = 7
CITY_CENTER = (25.20, 55.28)      # central Dubai (lat, lng)
H3_RES = 7
CITY_K = 6                        # -> 127 hexes
N_DAYS = 42
HOURS = list(range(7, 23))        # operating hours 07:00-22:00
BASE_FARE = 8.0
MAX_SURGE = 2.5

_H = np.array(HOURS, dtype=float)
# Demand is peaky (morning + evening rush); supply is a flatter, damped version
# so it lags demand at the peaks (that's where surge matters).
DEMAND_SHAPE = 0.4 + np.exp(-((_H - 8) ** 2) / 3) + np.exp(-((_H - 18) ** 2) / 4)
SUPPLY_SHAPE = 0.55 * DEMAND_SHAPE + 0.45 * DEMAND_SHAPE.mean()

# --- Hour-dependent elasticities (Modules 3 & 4) --------------------------
# Demand: inelastic at the peaks (people must ride), elastic off-peak.
_peak = (DEMAND_SHAPE - DEMAND_SHAPE.min()) / (DEMAND_SHAPE.max() - DEMAND_SHAPE.min())
DEMAND_ELASTICITY_BY_HOUR = 0.25 + 0.40 * (1 - _peak)
# Supply: drivers chase surge most in the evening / late hours.
SUPPLY_ELASTICITY_BY_HOUR = 0.30 + 0.45 * np.exp(-((_H - 20) ** 2) / 8)


def _hidx(hour):
    return np.asarray(hour) - HOURS[0]


# ---------------------------------------------------------------------------
# 1 & 2. Forecast modules  (they just make up numbers)
# ---------------------------------------------------------------------------
class DemandForecast:
    """Module 1 — random demand around an hourly base level."""
    def sample(self, base, rng):
        return base * DEMAND_SHAPE * rng.uniform(0.7, 1.3, size=len(HOURS))


class SupplyForecast:
    """Module 2 — random supply; damped hourly shape so it lags demand at peaks."""
    def sample(self, base, rng):
        return base * 0.9 * SUPPLY_SHAPE * rng.uniform(0.7, 1.3, size=len(HOURS))


# ---------------------------------------------------------------------------
# 3 & 4. Elasticity modules  (dynamic by hour)
# ---------------------------------------------------------------------------
@dataclass
class DemandElasticity:
    static: bool = False          # if True, ignore the hour (use the daily mean)
    def at(self, hour):
        i = _hidx(hour)
        if self.static:
            return np.full(len(i), DEMAND_ELASTICITY_BY_HOUR.mean())
        return DEMAND_ELASTICITY_BY_HOUR[i]


@dataclass
class SupplyElasticity:
    static: bool = False
    def at(self, hour):
        i = _hidx(hour)
        if self.static:
            return np.full(len(i), SUPPLY_ELASTICITY_BY_HOUR.mean())
        return SUPPLY_ELASTICITY_BY_HOUR[i]


# ---------------------------------------------------------------------------
# The world: outcome of charging surge `m` given hour-dependent elasticities
# ---------------------------------------------------------------------------
def simulate_outcomes(demand, supply, m, hour):
    ed = DEMAND_ELASTICITY_BY_HOUR[_hidx(hour)]
    es = SUPPLY_ELASTICITY_BY_HOUR[_hidx(hour)]
    realized_demand = demand * np.clip(1 - ed * (m - 1), 0.05, None)
    effective_supply = supply * (1 + es * (m - 1))
    rides = np.minimum(realized_demand, effective_supply)
    unmet = np.maximum(realized_demand - effective_supply, 0.0)
    wait = 2.0 + 6.0 * np.clip(realized_demand / (effective_supply + 1e-6) - 0.8, 0, None)
    revenue = rides * BASE_FARE * m
    return dict(rides=rides, unmet=unmet, wait=wait, revenue=revenue,
                realized_demand=realized_demand, effective_supply=effective_supply)


# ---------------------------------------------------------------------------
# 5. Optimizer  (grid-search the best surge per hex)
# ---------------------------------------------------------------------------
class Policy:
    name = "policy"
    def price(self, df: pd.DataFrame) -> np.ndarray:
        raise NotImplementedError


class FlatPricing(Policy):
    name = "Flat (no surge)"
    def price(self, df):
        return np.ones(len(df))


@dataclass
class SurgeOptimizer(Policy):
    """Module 5 — try a grid of surge multipliers, keep the one that maximises the
    objective (predicted rides or GMV) using the hour-dependent elasticities."""
    objective: str = "rides"          # "rides" or "gmv"
    dem_el: DemandElasticity = None
    sup_el: SupplyElasticity = None
    cap: float = MAX_SURGE
    grid_n: int = 25
    label: str = "Optimizer"

    @property
    def name(self):
        return self.label

    def price(self, df):
        D = df["demand"].to_numpy()
        S = df["supply"].to_numpy()
        ed = self.dem_el.at(df["hour"].to_numpy())[:, None]
        es = self.sup_el.at(df["hour"].to_numpy())[:, None]
        grid = np.linspace(1.0, self.cap, self.grid_n)[None, :]
        rd = D[:, None] * np.clip(1 - ed * (grid - 1), 0.05, None)
        rs = S[:, None] * (1 + es * (grid - 1))
        rides = np.minimum(rd, rs)
        value = BASE_FARE * grid * rides if self.objective == "gmv" else rides
        return grid[0, np.argmax(value, axis=1)]


OPTIMIZER_NAMES = [
    "Flat (no surge)",
    "Optimizer · dynamic (max rides)",
    "Optimizer · static elasticity",
    "Optimizer · max GMV",
]


def build_optimizer(name: str, train=None) -> Policy:
    if name == "Flat (no surge)":
        return FlatPricing()
    if name == "Optimizer · dynamic (max rides)":
        return SurgeOptimizer("rides", DemandElasticity(), SupplyElasticity(), label=name)
    if name == "Optimizer · static elasticity":
        return SurgeOptimizer("rides", DemandElasticity(static=True),
                              SupplyElasticity(static=True), label=name)
    if name == "Optimizer · max GMV":
        return SurgeOptimizer("gmv", DemandElasticity(), SupplyElasticity(), label=name)
    raise KeyError(name)


# ---------------------------------------------------------------------------
# City + panel
# ---------------------------------------------------------------------------
def _haversine_km(lat1, lng1, lat2, lng2):
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = math.radians(lat2 - lat1), math.radians(lng2 - lng1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def generate_city(center=CITY_CENTER, res=H3_RES, k=CITY_K, seed=SEED) -> pd.DataFrame:
    center_cell = h3.latlng_to_cell(center[0], center[1], res)
    cells = sorted(h3.grid_disk(center_cell, k))
    rng = np.random.default_rng(seed)
    rows = []
    for i, c in enumerate(cells):
        lat, lng = h3.cell_to_latlng(c)
        ring = h3.grid_distance(center_cell, c)
        boundary = [[b[1], b[0]] for b in h3.cell_to_boundary(c)]
        zone = "core" if ring <= 1 else "inner" if ring <= 3 else "outer"
        rows.append({"hex_id": i, "h3": c, "lat": lat, "lng": lng,
                     "ring": ring, "zone": zone,
                     "base": float(rng.uniform(15, 55)), "boundary": boundary})
    return pd.DataFrame(rows)


def generate_market(city: pd.DataFrame, days=N_DAYS, seed=SEED) -> pd.DataFrame:
    """One row per (hex, day, hour) with forecast demand & supply (random)."""
    rng = np.random.default_rng(seed + 1)
    dfc, sfc = DemandForecast(), SupplyForecast()
    frames = []
    for _, hx in city.iterrows():
        for day in range(days):
            dow = day % 7
            frames.append(pd.DataFrame({
                "hex_id": hx["hex_id"], "h3": hx["h3"], "zone": hx["zone"],
                "day": day, "dow": dow, "is_weekend": int(dow >= 5), "hour": HOURS,
                "demand": dfc.sample(hx["base"], rng),
                "supply": sfc.sample(hx["base"], rng),
            }))
    df = pd.concat(frames, ignore_index=True)
    df["ratio"] = df["demand"] / (df["supply"] + 1e-6)
    return df


# ---------------------------------------------------------------------------
# Evaluation & map helper
# ---------------------------------------------------------------------------
def evaluate(opt: Policy, df: pd.DataFrame) -> dict:
    m = opt.price(df)
    out = simulate_outcomes(df["demand"].to_numpy(), df["supply"].to_numpy(),
                            m, df["hour"].to_numpy())
    total_demand = out["realized_demand"].sum()
    return dict(
        rides=float(out["rides"].sum()),
        gmv=float(out["revenue"].sum()),
        avg_wait=float(np.average(out["wait"], weights=out["rides"] + 1e-6)),
        unmet_rate=float(out["unmet"].sum() / (total_demand + 1e-6)),
        avg_surge=float(np.average(m, weights=df["demand"].to_numpy())),
    )


def temporal_split(df, train_frac=0.7):
    cutoff = int(df["day"].max() * train_frac)
    return df[df["day"] <= cutoff].copy(), df[df["day"] > cutoff].copy()


def map_frame(city, market, optimizer, day, hour) -> pd.DataFrame:
    slot = market[(market["day"] == day) & (market["hour"] == hour)].copy()
    slot = slot.sort_values("hex_id")
    m = optimizer.price(slot)
    out = simulate_outcomes(slot["demand"].to_numpy(), slot["supply"].to_numpy(),
                            m, slot["hour"].to_numpy())
    slot["surge"] = m
    slot["rides"] = out["rides"]
    return slot.merge(city[["hex_id", "lat", "lng", "boundary", "zone"]],
                      on="hex_id", suffixes=("", "_c"))
