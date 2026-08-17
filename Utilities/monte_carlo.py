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
    folder per draw (Inputs/<prefix>001/FTT-GMP/...). The base scenario
    defaults to S0; use --base-scenario (e.g. nat_scale) to change a
    scenario. Variables the base scenario does not override fall back to S0, 
    matching the model loader. --base-scenario/--prefix accept comma-separated
    lists so that e.g. the S0 and nat_scale Monte Carlo folders can be
    generated in one command; each base scenario gets its own prefix and
    draw log.
run
    Set the scenarios in settings.ini from the existing <prefix>* folders,
    run the model, and save the combined output to Output/Results.pickle.
    --base-scenario/--prefix accept comma-separated lists so that e.g. the
    S0 and nat_scale Monte Carlo runs can be combined into one model run.
split
    Split Output/Results.pickle into one pickle per scenario under
    Output/monte_carlo/. (might be helpful for plotting)

Usage
-----
python Utilities/monte_carlo.py generate --n 100 --seed 42
python Utilities/monte_carlo.py generate --n 100 --seed 42 --base-scenario nat_scale
python Utilities/monte_carlo.py generate --n 100 --seed 42 --base-scenario S0,nat_scale
python Utilities/monte_carlo.py run
python Utilities/monte_carlo.py run --base-scenario S0,nat_scale
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
PERTURB_START_YEAR = 2026


def scenario_name(i, prefix):
    """Return the scenario name for draw index i (3 digits, zero-padded)."""
    return f"{prefix}{i:03d}"


def parse_pairs(base_scenarios_arg, prefixes_arg):
    """Return a list of (base_scenario, prefix) pairs from comma-separated args.

    When ``--prefix`` is omitted, each prefix is derived from its base
    scenario: ``MC_`` for S0, otherwise ``<base>_MC_``.
    """
    bases = [b.strip() for b in base_scenarios_arg.split(",") if b.strip()]
    if prefixes_arg is None:
        prefixes = ["MC_" if b == "S0" else f"{b}_MC_" for b in bases]
    else:
        prefixes = [p.strip() for p in prefixes_arg.split(",") if p.strip()]
    if len(bases) != len(prefixes):
        raise SystemExit(
            "--base-scenario and --prefix must list the same number of entries"
        )
    return list(zip(bases, prefixes))


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Monte Carlo scenario generation for FTT-GMP."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    gen = subparsers.add_parser("generate", help="Generate scenario folders")
    gen.add_argument("--n", type=int, default=100, help="Number of Monte Carlo draws (default: 100)")
    gen.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    gen.add_argument("--spec", type=str, default=str(DEFAULT_SPEC), help="Path to parameter spec CSV")
    gen.add_argument("--base-scenario", type=str, default="S0",
                     help="Comma-separated base scenario folders to perturb (default: S0)")
    gen.add_argument("--prefix", type=str, default=None,
                     help="Comma-separated scenario prefixes (default: derived per "
                          "base scenario: MC_ for S0, else <base>_MC_)")

    run = subparsers.add_parser("run", help="Set scenarios in settings.ini, run the model, and save Output/Results.pickle")
    run.add_argument("--results", type=str, default="Output/Results.pickle", help="Output pickle path")
    run.add_argument("--base-scenario", type=str, default="S0",
                     help="Comma-separated base scenario names (default: S0)")
    run.add_argument("--prefix", type=str, default=None,
                     help="Comma-separated scenario prefixes (default: derived "
                          "per base scenario: MC_ for S0, else <base>_MC_)")

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


def time_series_columns(frame):
    """Return (label_col, value_cols) for a wide time-series frame.

    Two layouts exist for variables whose first column is 'RTI':

      - Layout A (e.g. ``gm_NESO_av_electricity_price.csv``): the file has a
        second coordinate column holding the labels, and the year columns
        start at the third column: ``RTI, <label>, <year>, ...``.
      - Layout B (e.g. ``gm_CO2_intensity_upstream.csv``): the RTI column
        itself holds the labels and the year columns start at the second
        column: ``RTI, <year>, ...``.

    Layout B is detected when the second column header is a year (numeric),
    matching how the model loader derives coordinate columns from the
    VariableListing dimensions.
    """
    columns = list(frame.columns)
    if len(columns) < 2:
        return columns[0], []
    if str(columns[1]).strip().lstrip("-").isdigit():
        return columns[0], columns[1:]
    return columns[1], columns[2:]


def resolve_base_dir(variable_file, base_dir, s0_dir):
    """Return the directory holding the base frame for a variable.

    Prefer the base scenario's folder; fall back to S0. This mirrors the
    model loader, which copies S0 values and overlays scenario-specific
    files, so a scenario such as ``nat_scale`` that only overrides some
    variables still gets its non-overridden variables from S0.
    """
    name = variable_file
    if name.lower().endswith(".csv"):
        name = name[:-4]
    if (base_dir / f"{name}.csv").is_file():
        return base_dir
    return s0_dir


def read_base_frame(variable_file, base_dir):
    """Read the base cost matrix CSV and return the DataFrame.

    Accepts either the bare variable name (e.g. ``gm_costs_removal``) or the
    filename (``gm_costs_removal.csv``).
    """
    name = variable_file
    if name.lower().endswith(".csv"):
        name = name[:-4]
    path = base_dir / f"{name}.csv"
    if not path.is_file():
        raise ValueError(f"Variable file not found in {base_dir.name} or S0: {variable_file}")
    frame = pd.read_csv(path)
    if frame.columns[0].startswith("Unnamed"):
        frame.rename(columns={frame.columns[0]: ""}, inplace=True)
    if is_time_series_frame(frame):
        _, value_columns = time_series_columns(frame)
    else:
        value_columns = list(frame.columns[1:])
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



def _year_columns(frame):
    """Return a list of the year-value columns in a non-RTI frame."""
    return [c for c in frame.columns[1:] if str(c).strip().lstrip("-").isdigit()]


def sample_target_trajectory(rng, base_values, lower, upper,
                              perturb_start_idx, years):
    """Draw a capacity trajectory by scaling the base by a random 2050 target.

    A single random value is drawn at year 2050 within ``[lower, upper]``.
    The ratio to the base 2050 value gives a scale factor that is applied
    uniformly to all years from *perturb_start_idx* onward — the trajectory
    shape is preserved exactly; only the overall level changes.

    Parameters
    ----------
    rng : numpy.random.Generator
    base_values : array, shape (n_years,)
    lower, upper : float
        Absolute lower/upper bounds for the 2050 target value.
    perturb_start_idx : int
        First index that is perturbed; earlier years stay at base values.
    years : array of int, shape (n_years,)
        Calendar year corresponding to each element.

    Returns
    -------
    trajectory : ndarray, shape (n_years,)
    """
    n_years = len(base_values)
    trajectory = base_values.copy()

    idx_2050 = int(np.searchsorted(years, 2050))
    if idx_2050 >= n_years or years[idx_2050] != 2050:
        return trajectory

    base_2050 = base_values[idx_2050]
    if base_2050 < 1e-10:
        return trajectory

    drawn_2050 = rng.uniform(lower, upper)
    scale_factor = drawn_2050 / base_2050

    trajectory[perturb_start_idx:] = base_values[perturb_start_idx:] * scale_factor
    trajectory = np.maximum(trajectory, 0.0)

    return trajectory


# ---------------------------------------------------------------------------
# Core MC functions
# ---------------------------------------------------------------------------

def validate_spec(spec, s0_dir, base_dir):
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

        if mode not in ("factor", "absolute", "yearly_smooth"):
            errors.append(
                f"Row {row_index}: mode must be 'factor', 'absolute', or 'yearly_smooth', got '{mode}'"
            )
            continue

        # --- yearly_smooth: separate path -----------------------------------
        if mode == "yearly_smooth":
            if variable_file not in base_frames:
                try:
                    resolved_dir = resolve_base_dir(variable_file, base_dir, s0_dir)
                    base_frames[variable_file] = read_base_frame(variable_file, resolved_dir)
                except ValueError as exc:
                    errors.append(f"Row {row_index}: {exc}")
                    continue

            frame = base_frames[variable_file]
            label_col = frame.columns[0]
            row_idx = find_row_index(frame, technology, label_col=label_col)
            if row_idx is None:
                errors.append(
                    f"Row {row_index}: technology '{technology}' not found in {variable_file}"
                )
                continue

            year_cols = _year_columns(frame)
            all_years = np.array([int(c) for c in year_cols])
            base_trajectory = np.asarray(frame.loc[row_idx, year_cols], dtype=float)

            first_year = int(all_years[0])
            perturb_start_idx = PERTURB_START_YEAR - first_year

            params.append({
                "variable_file": variable_file,
                "technology": technology,
                "cost_column": "ALL",
                "mode": "yearly_smooth",
                "lower": lower,
                "upper": upper,
                "base_value": float(base_trajectory.mean()),
                "row_index": row_idx,
                "time_series": False,
                "value_columns": None,
                "base_trajectory": base_trajectory,
                "perturb_start_idx": perturb_start_idx,
                "years": all_years,
                "year_columns": year_cols,
            })
            continue

        # --- factor / absolute: original path --------------------------------
        if variable_file not in base_frames:
            try:
                resolved_dir = resolve_base_dir(variable_file, base_dir, s0_dir)
                base_frames[variable_file] = read_base_frame(variable_file, resolved_dir)
            except ValueError as exc:
                errors.append(f"Row {row_index}: {exc}")
                continue

        frame = base_frames[variable_file]
        is_ts = is_time_series_frame(frame)
        if is_ts:
            label_col, value_cols = time_series_columns(frame)
        else:
            label_col, value_cols = None, None
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
    """Sample uniform draws for each scenario and return drawn values per parameter.

    For ``factor``/``absolute`` modes the result is a 1-D array of shape
    ``(n,)`` per parameter.  For ``yearly_smooth`` the result is a 2-D array
    of shape ``(n, n_years)`` representing a scaled trajectory per draw.
    """
    draws = {}

    for param_index, param in enumerate(params):
        if param["mode"] == "yearly_smooth":
            trajectories = np.zeros((n, len(param["years"])))
            for draw_index in range(n):
                trajectories[draw_index] = sample_target_trajectory(
                    rng,
                    param["base_trajectory"],
                    param["lower"],
                    param["upper"],
                    param["perturb_start_idx"],
                    param["years"],
                )
            draws[param_index] = trajectories
        else:
            q = np.empty(n)
            for draw_index in range(n):
                q[draw_index] = rng.random()
            q = param["lower"] + q * (param["upper"] - param["lower"])
            if param["mode"] == "factor":
                draws[param_index] = param["base_value"] * (1.0 + q)
            else:
                draws[param_index] = param["base_value"] + q

    return draws


def write_scenario_folder(scenario, params, draws, draw_index, base_frames,
                         base_dir=None):
    """Write the FTT-GMP cost matrix files for one scenario folder.

    Perturbed files are written from *base_frames*.  Any CSV in *base_dir*
    that is **not** in *base_frames* is copied as-is so that base-scenario-
    specific variables (e.g. a gas-price spike) survive into the MC draws.
    """
    scenario_dir = PROJECT_ROOT / "Inputs" / scenario / MODULE
    scenario_dir.mkdir(parents=True, exist_ok=True)

    for variable_file in base_frames:
        frame = base_frames[variable_file].copy()
        for param_index, param in enumerate(params):
            if param["variable_file"] == variable_file:
                drawn_array = draws[param_index]
                if drawn_array.ndim == 2:
                    # yearly_smooth: write the full trajectory row
                    trajectory = drawn_array[draw_index]
                    frame.loc[param["row_index"], param["year_columns"]] = trajectory
                else:
                    drawn = drawn_array[draw_index]
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

    if base_dir is not None and base_dir.is_dir():
        perturbed = {vf.lower() for vf in base_frames}
        for csv_file in sorted(base_dir.glob("*.csv")):
            if csv_file.name.lower() not in perturbed:
                shutil.copy2(csv_file, scenario_dir / csv_file.name)


def update_settings(scenarios, settings_path):
    """Set the scenarios entry in settings.ini."""
    config = configparser.ConfigParser()
    config.read(str(settings_path))
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

    pairs = parse_pairs(args.base_scenario, args.prefix)

    spec = pd.read_csv(spec_path)
    inputs_root = PROJECT_ROOT / "Inputs"
    s0_dir = inputs_root / "S0" / MODULE
    rng = np.random.default_rng(args.seed)

    for base_scenario, prefix in pairs:
        base_dir = inputs_root / base_scenario / MODULE
        if not base_dir.is_dir():
            raise SystemExit(f"Base scenario folder not found: {base_dir}")

        params, base_frames = validate_spec(spec, s0_dir, base_dir)

        existing = sorted(inputs_root.glob(f"{prefix}*"))
        for folder in existing:
            shutil.rmtree(folder)
            print(f"Removed {folder}")

        draws = sample_draws(rng, n, params)

        draw_records = []
        for draw_index in range(n):
            scenario = scenario_name(draw_index + 1, prefix)
            write_scenario_folder(scenario, params, draws, draw_index, base_frames,
                                 base_dir=base_dir)
            for param_index, param in enumerate(params):
                drawn_array = draws[param_index]
                if drawn_array.ndim == 2:
                    drawn_value = float(drawn_array[draw_index].mean())
                else:
                    drawn_value = float(drawn_array[draw_index])
                draw_records.append({
                    "scenario": scenario,
                    "variable_file": param["variable_file"],
                    "technology": param["technology"],
                    "cost_column": param["cost_column"],
                    "mode": param["mode"],
                    "base_value": param["base_value"],
                    "drawn_value": drawn_value,
                })
            print(f"Generated {scenario}")

        out_dir = PROJECT_ROOT / "Output" / "monte_carlo"
        out_dir.mkdir(parents=True, exist_ok=True)
        draws_path = out_dir / f"draws_{prefix.rstrip('_')}.csv"
        pd.DataFrame(draw_records).to_csv(draws_path, index=False)
        print(f"Draw log written to {draws_path}")


def cmd_run(args):
    """Set the scenarios in settings.ini, run the model, and save the combined output pickle."""
    pairs = parse_pairs(args.base_scenario, args.prefix)

    inputs_root = PROJECT_ROOT / "Inputs"
    scenarios = []
    for base_scenario, prefix in pairs:
        folders = sorted(inputs_root.glob(f"{prefix}*"))
        if not folders:
            print(f"Warning: no {prefix}* scenario folders found in Inputs; "
                  f"skipping base scenario '{base_scenario}'.")
            continue
        scenarios.append(base_scenario)
        for i in range(1, len(folders) + 1):
            scenarios.append(scenario_name(i, prefix))

    if not scenarios:
        print("Warning: no MC scenario folders found in Inputs; running S0 only.")
        scenarios = ["S0"]
    update_settings(scenarios, PROJECT_ROOT / "settings.ini")

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
