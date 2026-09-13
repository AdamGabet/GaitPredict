"""Extended Data Figure 2 — duration scaling with intermittent-sampling overlay.

Overlays intermittent-sampling points on the continuous duration-scaling
curves for the age/BMI/VAT core-target panel (treadmill 3 km/h activity).

Reads two de-identified, aggregate (seed-level, no subject data) CSVs from the
repo's top-level results/ dir:
  - duration_scaling_plot_data.csv            (continuous curves)
  - duration_scaling_intermittent_plot_data.csv (intermittent overlay points)

Writes PDF + PNG to plotting/extended_figure_2/output/.

Intermittent datasets (age_bmi_vat panel, treadmill):
  intermittent_10s_5x2s   → 5 × 2s windows = 10s total  → placed at x = 10
  intermittent_30s_6x5s   → 6 × 5s        = 30s total  → placed at x = 30
  intermittent_30s_2x15s  → 2 × 15s       = 30s total  → placed at x = 30
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd


RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "results")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")
CONTINUOUS_CSV = os.path.join(RESULTS_DIR, "duration_scaling_plot_data.csv")
INTERMITTENT_CSV = os.path.join(RESULTS_DIR, "duration_scaling_intermittent_plot_data.csv")

# Intermittent points sit ON the matched-duration tick (plot_x); `dodge` is a
# multiplicative offset applied only at draw time so co-located points (the two
# 30s sub-protocols) don't overlap on the log axis. `filled` toggles solid vs
# open marker so the two 30s diamonds read as two distinct symbols.
# `groups` scopes a model to plot groups where its downstream results exist.
INTERMITTENT_MODELS = [
    {
        "name": "intermittent_10s_5x2s",
        "total_seconds": 10,
        "plot_x": 10,
        "dodge": 1.0,
        "label": "Intermittent 5×2s (10s)",
        "marker": "^",
        "markersize": 9,
        "filled": True,
        "groups": ["age_bmi_vat"],
    },
    {
        "name": "intermittent_30s_6x5s",
        "total_seconds": 30,
        "plot_x": 30,
        "dodge": 0.92,
        "label": "Intermittent 6×5s (30s)",
        "marker": "D",
        "markersize": 8,
        "filled": True,
        "groups": ["age_bmi_vat"],
    },
    {
        "name": "intermittent_30s_2x15s",
        "total_seconds": 30,
        "plot_x": 30,
        "dodge": 1.085,
        "label": "Intermittent 2×15s (30s)",
        "marker": "s",
        "markersize": 6.5,
        "filled": True,
        "groups": ["age_bmi_vat"],
    },
]


def models_for_group(group_name):
    return [m for m in INTERMITTENT_MODELS if group_name in m.get("groups", [])]


TARGETS = [
    {
        "name": "age",
        "display": "Age",
        "color": "#2f6fbb",
        "group": "age_bmi_vat",
    },
    {
        "name": "bmi",
        "display": "BMI",
        "color": "#d8842f",
        "group": "age_bmi_vat",
    },
    {
        "name": "total_scan_vat_area",
        "display": "VAT area",
        "color": "#7b4ab2",
        "group": "age_bmi_vat",
    },
    {
        "name": "hr_bpm",
        "display": "Heart rate (BPM)",
        "color": "#e05252",
        "group": "age_bmi_vat",
    },
    {
        "name": "liver_elasticity",
        "display": "Liver elasticity",
        "color": "#4f9a57",
        "group": "age_bmi_vat",
    },
]

PLOT_GROUP = {
    "name": "age_bmi_vat",
    "title": "Age, BMI, and VAT area",
    "first_tick": "Height only",
    "note": "Absolute Pearson r (gait embeddings + height)",
    "baseline_model": "Height",
    "genders": ["all"],
    "absolute_scores": True,
}

# Nature: every piece of figure text must print between 5 and 7 pt at the 180 mm
# two-column width. This figure's canvas is 8 in (203 mm) wide, so printed pt =
# fontsize x 180/203 = x0.886 -- almost 1:1, which is why the old 9-13 pt sizes
# printed at 8.0-11.5 pt. Ticks previously inherited the 10 pt rcParams default.
FS_TITLE = 7.5    # prints at 6.6 pt
FS_TICK = 7.0     # prints at 6.2 pt
FS_LEGEND = 7.0   # prints at 6.2 pt
FS_LABEL = 7.5    # axis labels; prints at 6.6 pt

ACTIVITY_DISPLAY = {
    "tm_3kmh": "Treadmill 3 km/h",
    "self_selected_gait_speed": "Self-selected gait speed",
    "ensemble": "Motor battery (ensemble)",
}


def log(message):
    print(message, flush=True)


def load_intermittent_rows(plot_group):
    """Return seed-level rows for all intermittent models × targets, scoped to
    this plot group's targets."""
    if not os.path.isfile(INTERMITTENT_CSV):
        log(f"Missing CSV: {INTERMITTENT_CSV}")
        return pd.DataFrame()
    df = pd.read_csv(INTERMITTENT_CSV)
    group_targets = {t["name"] for t in TARGETS if t["group"] == plot_group["name"]}
    group_models = {m["name"] for m in models_for_group(plot_group["name"])}
    return df[df["target"].isin(group_targets) & df["interm_name"].isin(group_models)].copy()


def style_axis(ax):
    ax.set_axisbelow(True)
    ax.grid(True, color="#e8e8e8", linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#bdbdbd")
    ax.spines["bottom"].set_color("#bdbdbd")


def plot_overlay(continuous_df, interm_df, activity, plot_group, x_order, x_labels,
                 output_dir, suffix, full_protocol_df=None):
    """Draw continuous curves + intermittent overlay for a single activity/mode."""
    targets = [t for t in TARGETS if t["group"] == plot_group["name"]]
    absolute = plot_group.get("absolute_scores", False)
    genders = plot_group["genders"]

    if not absolute:
        baseline_rows = [
            {"activity": activity, "gender": g, "target": t["name"],
             "duration_seconds": 0, "pearson_r": 0.0}
            for g in genders for t in targets
        ]
        continuous_df = pd.concat(
            [pd.DataFrame(baseline_rows), continuous_df], ignore_index=True, sort=False
        )
        continuous_df["plot_seconds"] = continuous_df["duration_seconds"].replace(0, 1)
    else:
        continuous_df = continuous_df.copy()
        continuous_df["plot_seconds"] = continuous_df["duration_seconds"]

    gender_titles = {"male": "Male", "female": "Female", "all": "All subjects"}
    legend_extra = 2.5
    fig_width = 5.5 * len(genders) + legend_extra
    fig, axes = plt.subplots(1, len(genders), figsize=(fig_width, 4.8), sharey=True)
    if len(genders) == 1:
        axes = [axes]
    fig.patch.set_facecolor("white")

    for ax, gender in zip(axes, genders):
        style_axis(ax)
        gender_df = continuous_df[continuous_df["gender"] == gender]
        interm_gender = interm_df[interm_df["gender"] == gender] if not interm_df.empty else pd.DataFrame()

        if not absolute:
            ax.axhline(0, color="#6f6f6f", linewidth=1.0, linestyle="--", alpha=0.8)

        for target in targets:
            color = target["color"]
            target_df = gender_df[gender_df["target"] == target["name"]]
            grouped = (
                target_df.groupby("plot_seconds")["pearson_r"]
                .agg(["mean", "std"])
                .reindex(x_order)
            )
            y = grouped["mean"].to_numpy(dtype=np.float64)
            sd = grouped["std"].fillna(0).to_numpy(dtype=np.float64)
            x = np.array(x_order, dtype=np.float64)
            valid = ~np.isnan(y)

            # Use T-shaped error bars instead of a shaded corridor so the
            # seed-level uncertainty remains legible when several curves overlap.
            ax.errorbar(
                x[valid], y[valid], yerr=sd[valid],
                fmt="none", ecolor=color, elinewidth=1.2, capsize=4.0,
                capthick=1.2, alpha=0.4, barsabove=True, zorder=5,
            )
            ax.plot(x[valid], y[valid], color=color, linewidth=2.4,
                    marker="o", markersize=5.2, markerfacecolor="white",
                    markeredgewidth=1.6, label=target["display"], zorder=4)

            # Full protocol point (ensemble only)
            if full_protocol_df is not None and "full_protocol_seconds" in (full_protocol_df or {}):
                pass  # handled below via pre-computed x_order

            # Intermittent overlay points
            if not interm_gender.empty:
                for interm in models_for_group(plot_group["name"]):
                    rows = interm_gender[
                        (interm_gender["target"] == target["name"]) &
                        (interm_gender["interm_name"] == interm["name"])
                    ]
                    if rows.empty:
                        continue
                    xi = float(interm.get("plot_x", interm["total_seconds"]))
                    xi *= interm.get("dodge", 1.0)
                    vals = rows["pearson_r"].to_numpy(dtype=np.float64)
                    yi_mean = vals.mean()
                    yi_lo, yi_hi = vals.min(), vals.max()

                    # Seed interval: min–max whisker across seeds
                    ax.errorbar(
                        xi, yi_mean,
                        yerr=[[yi_mean - yi_lo], [yi_hi - yi_mean]],
                        color=color, elinewidth=1.4, capsize=4,
                        capthick=1.4, alpha=0.45, barsabove=True,
                        linestyle="none", zorder=11,
                    )
                    filled = interm.get("filled", True)
                    ax.plot(
                        xi, yi_mean,
                        marker=interm["marker"],
                        markersize=interm["markersize"],
                        markerfacecolor=color if filled else "white",
                        markeredgecolor="white" if filled else color,
                        markeredgewidth=1.3,
                        linestyle="none",
                        zorder=10,
                    )

        ax.set_xscale("log")
        ax.set_xticks(x_order)
        ax.set_xticklabels(x_labels)
        ax.xaxis.set_minor_locator(mticker.NullLocator())
        ax.set_xlabel("Recording duration", fontsize=FS_LABEL)
        ax.tick_params(axis="both", labelsize=FS_TICK)
        ax.set_title(gender_titles[gender], fontsize=FS_TITLE, fontweight="bold")
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))

    if absolute:
        axes[0].set_ylabel("Pearson r", fontsize=FS_LABEL)
        # Nature policy: y-axis must start at 0 for absolute-magnitude panels.
        # (Delta panels are exempt -- they are centred on a zero reference line
        # and can legitimately go negative, so their limits are left to autoscale.)
        for ax in axes:
            ax.set_ylim(0, ax.get_ylim()[1])
    else:
        axes[0].set_ylabel("Improvement over covariate-only model (Δ Pearson r)", fontsize=FS_LABEL)

    # Build legend
    target_handles = [
        mlines.Line2D([], [], color=t["color"], linewidth=2.4, marker="o",
                      markersize=5.2, markerfacecolor="white",
                      markeredgewidth=1.6, label=t["display"])
        for t in targets
    ]
    interm_handles = [
        mlines.Line2D([], [], color="#555555", linestyle="none",
                      marker=im["marker"], markersize=im["markersize"],
                      markerfacecolor="#555555" if im.get("filled", True) else "white",
                      markeredgecolor="white" if im.get("filled", True) else "#555555",
                      markeredgewidth=1.3, label=im["label"])
        for im in models_for_group(plot_group["name"])
    ]
    axes[-1].legend(
        handles=target_handles + interm_handles,
        frameon=True, loc="upper left", fontsize=FS_LEGEND,
        bbox_to_anchor=(1.02, 1.0), borderaxespad=0,
    )

    fig.suptitle(
        f"{plot_group['title']} — {ACTIVITY_DISPLAY.get(activity, activity)} (with intermittent sampling)",
        fontsize=FS_TITLE, fontweight="bold",
    )

    right_margin = (5.5 * len(genders)) / fig_width - 0.02
    fig.subplots_adjust(left=0.08, right=right_margin, bottom=0.20, top=0.84, wspace=0.08)

    safe_activity = activity.replace("/", "_")
    stem = f"duration_scaling_intermittent_{plot_group['name']}_{safe_activity}"
    for ext, dpi in [("png", 220), ("pdf", 300)]:
        out_path = os.path.join(output_dir, f"{stem}.{ext}")
        fig.savefig(out_path, dpi=dpi)
        log(f"Saved {out_path}")
    plt.close(fig)


def run_per_activity(plot_group, activity, output_dir):
    """Per-activity version: x = [2, 5, 10, 15, 30] (+ 180 for tm)."""
    if not os.path.isfile(CONTINUOUS_CSV):
        log(f"Missing CSV: {CONTINUOUS_CSV} — skipping per-activity {activity}")
        return
    all_df = pd.read_csv(CONTINUOUS_CSV)
    act_df = all_df[
        (all_df["activity"] == activity) &
        (all_df["target"].isin([t["name"] for t in TARGETS if t["group"] == plot_group["name"]]))
    ].copy()

    absolute = plot_group.get("absolute_scores", False)
    fmt = (lambda s: f"{s}s") if absolute else (lambda s: f"+{s}")
    x_order = [2, 5, 10, 15, 30]  # 60s point intentionally excluded
    x_labels = [fmt(s) for s in x_order]
    if activity == "tm_3kmh" and (all_df["duration_seconds"] == 180).any():
        x_order.append(180)
        x_labels.append(fmt(180))
    # Add any intermittent plot_x positions that don't already have a tick
    for im in INTERMITTENT_MODELS:
        px = im.get("plot_x", im["total_seconds"])
        if px not in x_order:
            x_order.append(px)
            x_labels.append(f"{im['total_seconds']}s*")
    x_order_sorted = sorted(range(len(x_order)), key=lambda i: x_order[i])
    x_order = [x_order[i] for i in x_order_sorted]
    x_labels = [x_labels[i] for i in x_order_sorted]

    interm_df = load_intermittent_rows(plot_group)
    interm_df = interm_df[interm_df.apply(
        lambda r: r["target"] in [t["name"] for t in TARGETS if t["group"] == plot_group["name"]], axis=1
    )]

    plot_overlay(act_df, interm_df, activity, plot_group, x_order, x_labels, output_dir, "per_activity")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    log(f"Per-activity: {PLOT_GROUP['name']} / tm_3kmh")
    run_per_activity(PLOT_GROUP, "tm_3kmh", OUTPUT_DIR)


if __name__ == "__main__":
    main()
