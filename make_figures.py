"""Generate static PNGs for the README / GitHub (the live app is the star).

  outputs/map_surge.png   real H3 hex map over Dubai, coloured by chosen surge
  outputs/did_ab.png      A/B divergence: treatment vs control GMV over days
  outputs/did_bars.png    GMV & Trip lift with 95% CI
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection

import engine as e
import experiment as x

os.makedirs("outputs", exist_ok=True)
R = 6378137.0


def merc(lng, lat):
    return np.radians(lng) * R, np.log(np.tan(np.pi / 4 + np.radians(lat) / 2)) * R


def fig_map():
    city = e.generate_city()
    market = e.generate_market(city)
    train,_=e.temporal_split(market); opt = e.build_optimizer("Optimizer · dynamic (max rides)", train)
    slot = e.map_frame(city, market, opt, day=int(market["day"].max()), hour=18)

    fig, ax = plt.subplots(figsize=(8, 8))
    patches, vals = [], []
    for _, r in slot.iterrows():
        xs, ys = merc(np.array([p[0] for p in r["boundary"]]),
                      np.array([p[1] for p in r["boundary"]]))
        patches.append(Polygon(np.column_stack([xs, ys])))
        vals.append(r["surge"])
    pc = PatchCollection(patches, cmap="viridis", edgecolor="white", linewidth=0.8, alpha=0.7)
    pc.set_array(np.array(vals))
    ax.add_collection(pc)
    ax.autoscale()
    ax.set_aspect("equal"); ax.axis("off")
    cb = fig.colorbar(pc, ax=ax, fraction=0.04, pad=0.02); cb.set_label("surge ×")
    ax.set_title("Optimizer surge on a real H3 grid over Dubai (18:00)")
    try:
        import contextily as ctx
        ctx.add_basemap(ax, crs="EPSG:3857", source=ctx.providers.CartoDB.Positron)
    except Exception as ex:
        print(f"[map] basemap skipped ({ex})")
    fig.tight_layout(); fig.savefig("outputs/map_surge.png", dpi=130); plt.close(fig)
    print("wrote outputs/map_surge.png")


def fig_did():
    city = e.generate_city(); market = e.generate_market(city)
    train,_=e.temporal_split(market)
    treat = e.build_optimizer("Optimizer · dynamic (max rides)", train)
    ctrl = e.build_optimizer("Flat (no surge)", train)
    res = x.run_experiment(city, market, treat, ctrl, design="A/B", seed=7)
    hd = res["records"]; did = res["did_fn"](hd)
    pre_cut = res["meta"]["pre_cut"]

    g = hd.groupby(["day", "group"])["gmv"].mean().unstack()
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.plot(g.index, g["T"], color="#1f9e6e", label="treatment (optimizer)")
    ax.plot(g.index, g["C"], color="#d1495b", label="control (flat)")
    ax.axvline(pre_cut + 0.5, color="#888", ls="--", lw=1)
    ax.text(pre_cut + 1, ax.get_ylim()[0], " experiment starts", color="#555", fontsize=8)
    ax.set_xlabel("day"); ax.set_ylabel("GMV per hex-day")
    ax.set_title("A/B divergence — treatment pulls ahead post-launch (DiD)")
    ax.legend(); ax.grid(alpha=0.3); fig.tight_layout()
    fig.savefig("outputs/did_ab.png", dpi=130); plt.close(fig)
    print("wrote outputs/did_ab.png")

    fig, ax = plt.subplots(figsize=(6, 4))
    labels = ["GMV lift", "Trip lift"]
    lifts = [did["gmv"]["lift_pct"], did["trips"]["lift_pct"]]
    los = [lifts[0] - did["gmv"]["ci"][0], lifts[1] - did["trips"]["ci"][0]]
    his = [did["gmv"]["ci"][1] - lifts[0], did["trips"]["ci"][1] - lifts[1]]
    ax.bar(labels, lifts, yerr=[los, his], capsize=6, color=["#1f9e6e", "#3a7ca5"])
    for i, v in enumerate(lifts):
        ax.text(i, v, f" {v:+.1f}%", va="bottom", ha="center", fontweight="bold")
    ax.axhline(0, color="#333", lw=0.8)
    ax.set_ylabel("lift % vs control"); ax.set_title("DiD lift with 95% CI (held-out)")
    ax.grid(alpha=0.3, axis="y"); fig.tight_layout()
    fig.savefig("outputs/did_bars.png", dpi=130); plt.close(fig)
    print("wrote outputs/did_bars.png")


if __name__ == "__main__":
    fig_map()
    fig_did()
