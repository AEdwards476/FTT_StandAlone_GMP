"""Convert Plexos indicative electricity price curves to S0-style CSV.

Expected output shape:
    - Column 1: RTI region label (fixed to "1 UK")
    - Column 2: Hour index (1..N, ordered from cheapest electricity)
    - Remaining columns: sorted electricity prices by year

The workbook sheet ``P1-YearSummary`` only exposes one selected year at a time.
To create a full by-year matrix, this script uses ``P1-RawData`` and rebuilds
the same ordered price curves for each year.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_INPUT = Path("Inputs/IndicativePriceCurves-PlexosModels v1.0.xlsx")
DEFAULT_OUTPUT = Path("Inputs/S0/FTT-GMP/gm_elec_price_curve.csv")
DEFAULT_SHEET = "P1-RawData"
DEFAULT_HOURS = 8760
DEFAULT_REGION = "1 UK"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert Plexos electricity prices into S0 curve format."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="Path to IndicativePriceCurves workbook.",
    )
    parser.add_argument(
        "--sheet",
        default=DEFAULT_SHEET,
        help="Workbook sheet containing Year + Price columns (default: P1-RawData).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output CSV path in Inputs/S0 format.",
    )
    parser.add_argument(
        "--hours",
        type=int,
        default=DEFAULT_HOURS,
        help="Number of ordered hours in output (default: 8760).",
    )
    parser.add_argument(
        "--region",
        default=DEFAULT_REGION,
        help='Region label written in first column (default: "1 UK").',
    )
    return parser.parse_args()


def _resample_sorted_prices(sorted_prices: np.ndarray, target_hours: int) -> np.ndarray:
    """Resample sorted prices to target_hours using quantile interpolation."""
    source_hours = sorted_prices.size
    if source_hours == target_hours:
        return sorted_prices
    if source_hours < target_hours:
        raise ValueError(
            f"Year has {source_hours} rows, fewer than requested {target_hours}."
        )

    source_q = (np.arange(source_hours, dtype=np.float64) + 0.5) / source_hours
    target_q = (np.arange(target_hours, dtype=np.float64) + 0.5) / target_hours
    return np.interp(target_q, source_q, sorted_prices)


def build_curve_table(df: pd.DataFrame, hours: int, region: str) -> tuple[pd.DataFrame, list[int]]:
    """Build an S0-style matrix of sorted electricity prices by year."""
    required = {"Year", "Price (€/MWh)"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    work = df[["Year", "Price (€/MWh)"]].copy()
    work["Year"] = pd.to_numeric(work["Year"], errors="coerce")
    work["Price (€/MWh)"] = pd.to_numeric(work["Price (€/MWh)"], errors="coerce")
    work = work.dropna(subset=["Year", "Price (€/MWh)"])
    work["Year"] = work["Year"].astype(int)

    output = pd.DataFrame(
        {
            "RTI": [region] * hours,
            "Hour": np.arange(1, hours + 1, dtype=int),
        }
    )

    skipped_years: list[int] = []
    for year in sorted(work["Year"].unique()):
        prices = np.sort(work.loc[work["Year"] == year, "Price (€/MWh)"].to_numpy())
        if prices.size < hours:
            skipped_years.append(year)
            continue
        output[str(year)] = _resample_sorted_prices(prices, hours)

    if output.shape[1] <= 2:
        raise ValueError("No years were converted. Check sheet content and --hours.")

    return output, skipped_years


def main() -> None:
    args = parse_args()

    if args.hours <= 0:
        raise ValueError("--hours must be a positive integer.")
    if not args.input.exists():
        raise FileNotFoundError(f"Input workbook not found: {args.input}")

    source = pd.read_excel(args.input, sheet_name=args.sheet)
    converted, skipped = build_curve_table(source, hours=args.hours, region=args.region)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    converted.to_csv(args.output, index=False)

    years = converted.columns[2:].tolist()
    print(f"Wrote {args.output}")
    print(f"Rows: {len(converted)}")
    print(f"Years converted ({len(years)}): {years[0]} ... {years[-1]}")
    if skipped:
        print(f"Skipped incomplete years: {[int(year) for year in skipped]}")


if __name__ == "__main__":
    main()
