# -*- coding: utf-8 -*-
"""
=========================================
make_figures.py
=========================================
Main figure-generation entry point for the FTT-GMP model. Each figure is
drawn from the combined results pickle (``Output/Results.pickle``) written by
``Utilities/monte_carlo.py run`` / ``run_file.py``. The pickle holds one
variable dictionary per scenario: ``{scenario: {var: np.ndarray}}``.

When Monte Carlo scenarios are present (scenarios named ``<prefix>NNN``), the
MC spread is drawn as a shaded envelope (min-max range) around the chosen
base-scenario line (default S0). The base and prefix are selected with
``--base-scenario`` and ``--prefix`` so that e.g. a nat_scale Monte Carlo run
(``--base-scenario nat_scale --prefix nat_scale_MC_``) plots nat_scale only --
S0 and nat_scale are never drawn on the same figure. If only the base scenario
is present, just its line is drawn.

Available figures (key: description)
------------------------------------
doc_vs_dac      DOC vs DAC levelised costs over time (same axis)
lcoh_vs_lcom    LCOH vs LCOM levelised costs over time (same axis)
pathway_lcoe    Levelised cost of dispatchable electricity (GBP/kWh) by
                pathway + MC envelope on each pathway

Usage
-----
python figures/make_figures.py [--figures all|lcoh,pathway_lcoe,...]
                               [--results Output/Results.pickle]
                               [--base-scenario S0,nat_scale] [--prefix MC_,nat_scale_MC_]
                               [--out-dir figures]
                               [--start-year 2025]
                               [--format png] [--dpi 150] [--show]
--base-scenario (and --prefix) accept comma-separated lists, one figure set
per base scenario (mirroring Utilities/monte_carlo.py). If --prefix is omitted
it is derived per base scenario: MC_ for S0, otherwise <base>_MC_.
"""

# Standard library imports
import argparse
import configparser
import pickle
import sys
from pathlib import Path

# Third party imports
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Inter"

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RESULTS = PROJECT_ROOT / "Output" / "Results.pickle"
DEFAULT_OUT_DIR = PROJECT_ROOT / "figures"

sys.path.insert(0, str(PROJECT_ROOT))
from Utilities.monte_carlo import parse_pairs
SETTINGS_PATH = PROJECT_ROOT / "settings.ini"

# Friendly display names for base scenarios. Scenarios not listed here fall
# back to their raw scenario name.
SCENARIO_LABELS = {
    "S0": "Local scale",
    "nat_scale": "National scale",
}


def base_scenario_label(base_scenario):
    """Return the friendly display name for a base scenario."""
    return SCENARIO_LABELS.get(base_scenario, base_scenario)


# --------------------------------------------------------------------------
# Data loading
# --------------------------------------------------------------------------
def load_results(path):
    """Load the combined results pickle into {scenario: variables}."""
    path = Path(path)
    if not path.is_file():
        raise SystemExit(f"Results pickle not found: {path}")
    with open(path, "rb") as f:
        return pickle.load(f)


def get_simulation_years():
    """Return the full simulation year axis from settings.ini."""
    config = configparser.ConfigParser()
    config.read(str(SETTINGS_PATH))
    start = int(config.get("settings", "simulation_start"))
    end = int(config.get("settings", "simulation_end"))
    return np.arange(start, end + 1)


def split_scenarios(results, base_scenario, prefix):
    """Return (base, mc) dicts of variable dicts, dropping non-MC extras.

    ``base`` is the single scenario named ``base_scenario`` (e.g. ``S0`` or
    ``nat_scale``); ``mc`` are the scenarios named ``<prefix>NNN``. This keeps
    one chosen base per figure -- S0 and nat_scale are never mixed.
    """
    base = {name: vars_ for name, vars_ in results.items()
            if name == base_scenario}
    mc = {name: vars_ for name, vars_ in results.items()
          if name.startswith(prefix)}
    return base, mc


def extract_series(vars_, var_name, pathway=None, keep=None):
    """Return the (years,) series for a variable, optionally one pathway."""
    arr = vars_[var_name]
    if arr.ndim == 4:
        if pathway is None:
            series = arr[0, 0, 0, :]
        else:
            series = arr[0, pathway, 0, :]
    else:
        series = np.asarray(arr).ravel()
    series = np.asarray(series, dtype=float)
    if keep is not None:
        series = series[keep]
    return series


def envelope(mc_array):
    """Return (low, high, mean) across the MC scenarios (axis 0)."""
    low = np.min(mc_array, axis=0)
    high = np.max(mc_array, axis=0)
    mean = np.mean(mc_array, axis=0)
    return low, high, mean


# --------------------------------------------------------------------------
# Plotting helpers
# --------------------------------------------------------------------------
def annotate_base(ax, base_scenario):
    """Write a friendly base-scenario label in the top-left corner."""
    ax.text(0.04, 0.96, base_scenario_label(base_scenario),
            transform=ax.transAxes, ha="left", va="top",
            fontsize=11, color="dimgray")


def plot_two_costs(results, years, keep, series, unit, title, fname,
                   out_dir, dpi, show, fmt, base_scenario, prefix):
    """Draw two cost variables (each with MC envelope) on one axis."""
    base, mc = split_scenarios(results, base_scenario, prefix)
    fig, ax = plt.subplots(figsize=(6, 4))

    for var_name, label, color in series:
        drawn = False
        if base:
            ax.plot(years,
                    extract_series(list(base.values())[0], var_name, keep=keep),
                    color=color, linewidth=2.5, label=label)
            drawn = True
        if mc:
            mc_array = np.stack(
                [extract_series(v, var_name, keep=keep) for v in mc.values()])
            low, high, _ = envelope(mc_array)
            ax.fill_between(years, low, high, color=color, alpha=0.20,
                            label=label if not drawn else "_nolegend_")

    ax.set_ylabel(f"Levelised cost (£/{unit})")
    ax.set_title(title)
    ax.margins(x=0)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    if base:
        annotate_base(ax, base_scenario)
    save_or_show(fig, out_dir, fname, dpi, show, fmt, base_scenario)


def plot_doc_vs_dac(results, years, keep, out_dir, dpi, show, fmt,
                    base_scenario, prefix):
    """Draw DOC and DAC levelised costs (each with MC envelope) on one axis."""
    series = (("gm_lcodac", "DAC", "#ee6677"),
              ("gm_lcodoc", "DOC", "#4477AA"))
    plot_two_costs(results, years, keep, series, "tCO$_2$",
                   "", "gm_lcodoc_vs_gm_lcodac",
                   out_dir, dpi, show, fmt, base_scenario, prefix)


def plot_lcoh_vs_lcom(results, years, keep, out_dir, dpi, show, fmt,
                      base_scenario, prefix):
    """Draw LCOH and LCOM (each with MC envelope) on one axis."""
    series = (("gm_lcom_ccs", "E-methane", "#AA3377"),
              ("gm_lcoh", "Hydrogen", "#228833"))
    plot_two_costs(results, years, keep, series, "kWh",
                   "", "gm_lcoh_vs_gm_lcom_ccs",
                   out_dir, dpi, show, fmt, base_scenario, prefix)


def plot_pathway_lcoe(results, years, keep, out_dir, dpi, show, fmt,
                      base_scenario, prefix):
    """Draw all dispatchable pathways on one axis, each with an MC envelope."""
    base, mc = split_scenarios(results, base_scenario, prefix)

    sys.path.insert(0, str(PROJECT_ROOT))
    from SourceCode.support.titles_functions import load_titles
    pathways = load_titles()["titles_gm_pathways"]

    fig, ax = plt.subplots(figsize=(6, 4))
    # Manually picked high-contrast palette (Paul Tol "bright").
    colors = ["#4477AA", "#66CCEE", "#228833", "#CCBB44", "#EE6677",
              "#AA3377", "#BBBBBB"]

    for pathway_idx, name in enumerate(pathways):
        color = colors[pathway_idx % len(colors)]
        if base:
            ax.plot(years,
                    extract_series(list(base.values())[0], "gm_pathway_lcoe",
                                   pathway=pathway_idx, keep=keep),
                    color=color, linewidth=2.2, label=name)
        if mc:
            mc_array = np.stack([
                extract_series(v, "gm_pathway_lcoe", pathway=pathway_idx,
                               keep=keep)
                for v in mc.values()])
            low, high, _ = envelope(mc_array)
            ax.fill_between(years, low, high, color=color, alpha=0.22)

    ax.set_ylabel("Levelised cost (£/kWh)")
    ax.margins(x=0)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=7)
    if base:
        annotate_base(ax, base_scenario)
    save_or_show(fig, out_dir, "gm_pathway_lcoe_mc", dpi, show, fmt,
                 base_scenario)


def save_or_show(fig, out_dir, fname, dpi, show, fmt, base_scenario):
    """Save the figure to out_dir and optionally open the interactive window.

    The base scenario is appended to the filename (e.g. ``gm_lcoh_mc_S0.png``
    vs ``gm_lcoh_mc_nat_scale.png``) so figures from different runs do not
    overwrite each other.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{fname}_{base_scenario}.{fmt}"
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    print(f"Figure saved to {path}")
    if show:
        plt.show()
    else:
        plt.close(fig)


# --------------------------------------------------------------------------
# Figure registry
# --------------------------------------------------------------------------
FIGURES = {
    "doc_vs_dac": plot_doc_vs_dac,
    "lcoh_vs_lcom": plot_lcoh_vs_lcom,
    "pathway_lcoe": plot_pathway_lcoe,
}


def main():
    """Parse arguments and draw the requested figures."""
    parser = argparse.ArgumentParser(
        description="Generate figures from the FTT-GMP results pickle."
    )
    parser.add_argument(
        "--figures", type=str, default="all",
        help="Comma-separated figure keys, or 'all' (default: all). "
             f"Valid keys: {', '.join(FIGURES)}")
    parser.add_argument("--results", type=str, default=str(DEFAULT_RESULTS),
                        help="Combined results pickle path "
                             "(default: Output/Results.pickle)")
    parser.add_argument("--base-scenario", type=str, default="S0",
                        help="Comma-separated scenarios to draw as the base "
                             "line (default: S0; e.g. S0,nat_scale)")
    parser.add_argument("--prefix", type=str, default=None,
                        help="Scenario name prefix(es) for the MC envelope, "
                             "one per base scenario (default: derived from "
                             "--base-scenario: MC_ for S0, else <base>_MC_)")
    parser.add_argument("--out-dir", type=str, default=str(DEFAULT_OUT_DIR),
                        help="Output directory for figures (default: figures)")
    parser.add_argument("--start-year", type=int, default=2025,
                        help="First year plotted (default: 2025)")
    parser.add_argument("--format", type=str, default="png",
                        help="Figure format (default: png)")
    parser.add_argument("--dpi", type=int, default=150,
                        help="Figure resolution (default: 150)")
    parser.add_argument("--show", action="store_true",
                        help="Open the interactive plot window(s) "
                             "(default: off)")
    args = parser.parse_args()

    requested = [key.strip().casefold() for key in args.figures.split(",")]
    if "all" in requested:
        requested = list(FIGURES)
    unknown = [key for key in requested if key not in FIGURES]
    if unknown:
        raise SystemExit(f"Unknown figure keys: {', '.join(unknown)}")

    results = load_results(args.results)
    years_full = get_simulation_years()
    keep = years_full >= args.start_year
    years = years_full[keep]

    pairs = parse_pairs(args.base_scenario, args.prefix)

    for base_scenario, prefix in pairs:
        for key in requested:
            draw = FIGURES[key]
            draw(results, years, keep, args.out_dir, args.dpi, args.show,
                 args.format, base_scenario, prefix)


if __name__ == "__main__":
    main()
