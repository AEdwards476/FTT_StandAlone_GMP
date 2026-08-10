# -*- coding: utf-8 -*-
"""
=========================================
monte_carlo.py
=========================================
Monte Carlo scenario generation for FTT-GMP.

Subcommands
-----------
generate
    Read a parameter spec CSV, sample N draws, write one scenario
    folder per draw (Inputs/MC_001/FTT-GMP/...).
run
    Set the scenarios in settings.ini from the existing MC_* folders, run the
    model, and save the combined output to Output/Results.pickle.
split
    Split Output/Results.pickle into one pickle per scenario under
    Output/monte_carlo/. (might be helpful for plotting)

Usage
-----
python Utilities/monte_carlo.py generate --n 100 --seed 42
python Utilities/monte_carlo.py run
python Utilities/monte_carlo.py split
"""

# Standard library imports
import argparse
import configparser
import os
import pickle
import shutil
import sys
from pathlib import Path

# Third party imports
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SPEC = PROJECT_ROOT / "Inputs" / "_monte_carlo_params.csv"
MODULE = "FTT-GMP"
SCENARIO_PREFIX = "MC_"


def scenario_name(i):
    """Return the scenario name for draw index i (3 digits, zero-padded)."""
    return f"{SCENARIO_PREFIX}{i:03d}"


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Monte Carlo scenario generation for FTT-GMP."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    gen = subparsers.add_parser("generate", help="Generate scenario folders")
    gen.add_argument("--n", type=int, required=True, help="Number of Monte Carlo draws")
    gen.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    gen.add_argument("--spec", type=str, default=str(DEFAULT_SPEC), help="Path to parameter spec CSV")

    run = subparsers.add_parser("run", help="Set scenarios in settings.ini, run the model, and save Output/Results.pickle")
    run.add_argument("--results", type=str, default="Output/Results.pickle", help="Output pickle path")

    split = subparsers.add_parser("split", help="Split a results pickle into per-scenario pickles")
    split.add_argument("--results", type=str, default="Output/Results.pickle", help="Input pickle path")
    split.add_argument("--out-dir", type=str, default="Output/monte_carlo", help="Output directory")

    return parser.parse_args()


def is_time_series_frame(frame):
    """Return True if the frame is a wide time-series (first column 'RTI').

    Such files (e.g. ``gm_NESO_av_electricity_price.csv``) have a second
    coordinate column (option/technology label) and year columns as the wide
    axis, unlike the cost matrices whose first column holds the labels.
    """
    return str(frame.columns[0]).strip().casefold() == "rti"


def read_base_frame(variable_file, s0_dir):
    """Read the S0 cost matrix CSV and return the DataFrame.

    Accepts either the bare variable name (e.g. ``gm_costs_removal``) or the
    filename (``gm_costs_removal.csv``).
    """
    name = variable_file
    if name.lower().endswith(".csv"):
        name = name[:-4]
    path = s0_dir / f"{name}.csv"
    if not path.is_file():
        raise ValueError(f"Variable file not found in S0: {variable_file}")
    frame = pd.read_csv(path)
    if frame.columns[0].startswith("Unnamed"):
        frame.rename(columns={frame.columns[0]: ""}, inplace=True)
    value_columns = list(frame.columns[2:]) if is_time_series_frame(frame) else list(frame.columns[1:])
    frame[value_columns] = frame[value_columns].astype(float)
    return frame


def find_row_index(df, technology, label_col=None):
    """Return the index of the row whose label matches technology."""
    if label_col is None:
        label_col = df.columns[0]
    labels = df[label_col].astype(str).str.strip()
    exact = labels == technology.strip()
    if exact.any():
        return labels[exact].index[0]
    folded = labels.str.casefold()
    match = folded == technology.strip().casefold()
    if match.any():
        return labels[match].index[0]
    return None


def find_column(df, cost_column):
    """Return the column name matching cost_column (case-insensitive)."""
    if cost_column in df.columns:
        return cost_column
    fold = {str(c).strip().casefold(): c for c in df.columns}
    return fold.get(cost_column.strip().casefold())


def validate_spec(spec, s0_dir):
    """Validate the spec and return a list of parameter dicts."""
    required = ["variable_file", "technology", "cost_column", "mode", "lower", "upper"]
    for column in required:
        if column not in spec.columns:
            raise SystemExit(f"Spec file is missing column: {column}")

    base_frames = {}
    params = []
    errors = []
    warnings = []

    for row_index, row in spec.iterrows():
        variable_file = str(row["variable_file"]).strip()
        if not variable_file.lower().endswith(".csv"):
            variable_file = f"{variable_file}.csv"
        technology = str(row["technology"]).strip()
        cost_column = str(row["cost_column"]).strip()
        mode = str(row["mode"]).strip().lower()

        try:
            lower = float(row["lower"])
            upper = float(row["upper"])
        except (TypeError, ValueError):
            errors.append(
                f"Row {row_index}: lower/upper must be numeric for {variable_file} / {technology} / {cost_column}"
            )
            continue
        
        # factor is multiplicative, absolute is additive -- only options for now
        if mode not in ("factor", "absolute"):
            errors.append(f"Row {row_index}: mode must be 'factor' or 'absolute', got '{mode}'")
            continue

        if variable_file not in base_frames:
            try:
                base_frames[variable_file] = read_base_frame(variable_file, s0_dir)
            except ValueError as exc:
                errors.append(f"Row {row_index}: {exc}")
                continue

        frame = base_frames[variable_file]
        is_ts = is_time_series_frame(frame)
        label_col = frame.columns[1] if is_ts else None
        row_idx = find_row_index(frame, technology, label_col=label_col)
        if row_idx is None:
            errors.append(
                f"Row {row_index}: technology '{technology}' not found in {variable_file}"
            )
            continue

        if is_ts:
            col_name = "ALL"
        else:
            col_name = find_column(frame, cost_column)
            if col_name is None:
                errors.append(
                    f"Row {row_index}: cost column '{cost_column}' not found in {variable_file}"
                )
                continue

        if is_ts:
            value_cols = list(frame.columns[2:])
            base_value = float(frame.loc[row_idx, value_cols].mean())
        else:
            value_cols = None
            base_value = float(frame.loc[row_idx, col_name])
        if mode == "factor" and base_value == 0:
            warnings.append(
                f"Row {row_index}: factor mode on zero base value for "
                f"{variable_file} / {technology} / {cost_column} - skipping"
            )
            continue

        params.append({
            "variable_file": variable_file,
            "technology": technology,
            "cost_column": col_name,
            "mode": mode,
            "lower": lower,
            "upper": upper,
            "base_value": base_value,
            "row_index": row_idx,
            "time_series": is_ts,
            "value_columns": value_cols,
        })

    if errors:
        raise SystemExit("Invalid spec rows:\n  " + "\n  ".join(errors))
    for warning in warnings:
        print(f"Warning: {warning}")

    return params, base_frames


def sample_draws(rng, n, params):
    """Sample uniform draws for each scenario and return drawn values per parameter."""
    quantiles = {param_index: np.zeros(n) for param_index in range(len(params))}

    for draw_index in range(n):
        for param_index in range(len(params)):
            quantiles[param_index][draw_index] = rng.random()

    draws = {}
    for param_index, param in enumerate(params):
        q = param["lower"] + quantiles[param_index] * (param["upper"] - param["lower"])
        if param["mode"] == "factor":
            draws[param_index] = param["base_value"] * (1 + q)
        else:
            draws[param_index] = param["base_value"] + q

    return draws


def write_scenario_folder(scenario, params, draws, draw_index, base_frames, s0_dir):
    """Write the FTT-GMP cost matrix files for one scenario folder."""
    scenario_dir = s0_dir.parent.parent / scenario / MODULE
    scenario_dir.mkdir(parents=True, exist_ok=True)

    for variable_file in base_frames:
        frame = base_frames[variable_file].copy()
        for param_index, param in enumerate(params):
            if param["variable_file"] == variable_file:
                drawn = draws[param_index][draw_index]
                if param["time_series"]:
                    if param["mode"] == "factor":
                        multiplier = drawn / param["base_value"]
                        frame.loc[param["row_index"], param["value_columns"]] *= multiplier
                    else:
                        addend = drawn - param["base_value"]
                        frame.loc[param["row_index"], param["value_columns"]] += addend
                else:
                    frame.loc[param["row_index"], param["cost_column"]] = drawn
        frame.to_csv(scenario_dir / variable_file, index=False)


def update_settings(n, settings_path):
    """Set the scenarios entry in settings.ini."""
    config = configparser.ConfigParser()
    config.read(str(settings_path))
    scenarios = ["S0"] + [scenario_name(i) for i in range(1, n + 1)]
    config.set("settings", "scenarios", ", ".join(scenarios))
    with open(settings_path, "w") as configfile:
        config.write(configfile)


def cmd_generate(args):
    """Generate scenario folders."""
    spec_path = Path(args.spec)
    if not spec_path.is_file():
        raise SystemExit(f"Spec file not found: {spec_path}")

    n = args.n
    if n < 1:
        raise SystemExit("Number of draws must be >= 1")

    s0_dir = PROJECT_ROOT / "Inputs" / "S0" / MODULE
    spec = pd.read_csv(spec_path)
    params, base_frames = validate_spec(spec, s0_dir)

    inputs_root = PROJECT_ROOT / "Inputs"
    existing = sorted(inputs_root.glob(f"{SCENARIO_PREFIX}*"))
    for folder in existing:
        shutil.rmtree(folder)
        print(f"Removed {folder}")

    rng = np.random.default_rng(args.seed)
    draws = sample_draws(rng, n, params)

    draw_records = []
    for draw_index in range(n):
        scenario = scenario_name(draw_index + 1)
        write_scenario_folder(scenario, params, draws, draw_index, base_frames, s0_dir)
        for param_index, param in enumerate(params):
            draw_records.append({
                "scenario": scenario,
                "variable_file": param["variable_file"],
                "technology": param["technology"],
                "cost_column": param["cost_column"],
                "mode": param["mode"],
                "base_value": param["base_value"],
                "drawn_value": draws[param_index][draw_index],
            })
        print(f"Generated {scenario}")

    out_dir = PROJECT_ROOT / "Output" / "monte_carlo"
    out_dir.mkdir(parents=True, exist_ok=True)
    draws_path = out_dir / "draws.csv"
    pd.DataFrame(draw_records).to_csv(draws_path, index=False)
    print(f"Draw log written to {draws_path}")


def cmd_run(args):
    """Set the scenarios in settings.ini, run the model, and save the combined output pickle."""
    inputs_root = PROJECT_ROOT / "Inputs"
    mc_folders = sorted(inputs_root.glob(f"{SCENARIO_PREFIX}*"))
    if not mc_folders:
        print("Warning: no MC_* scenario folders found in Inputs; running S0 only.")
    update_settings(len(mc_folders), PROJECT_ROOT / "settings.ini")

    sys.path.insert(0, str(PROJECT_ROOT))
    from SourceCode.model_class import RunFTT

    model = RunFTT()
    model.run()

    results_path = PROJECT_ROOT / args.results
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, "wb") as f:
        pickle.dump(model.output, f)
    print(f"Results saved to {results_path}")


def cmd_split(args):
    """Split a results pickle into one pickle per scenario."""
    results_path = Path(args.results)
    if not results_path.is_file():
        raise SystemExit(f"Results pickle not found: {results_path}")

    with open(results_path, "rb") as f:
        data = pickle.load(f)

    out_dir = PROJECT_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    for scenario, variables in data.items():
        with open(out_dir / f"{scenario}.pickle", "wb") as f:
            pickle.dump({scenario: variables}, f)
        print(f"Wrote {scenario}.pickle")


def main():
    """Run the requested subcommand."""
    os.chdir(PROJECT_ROOT)
    args = parse_args()
    if args.command == "generate":
        cmd_generate(args)
    elif args.command == "run":
        cmd_run(args)
    elif args.command == "split":
        cmd_split(args)


if __name__ == "__main__":
    main()