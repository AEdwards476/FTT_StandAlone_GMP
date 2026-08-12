# -*- coding: utf-8 -*-
"""
============================================================
ftt_gmp_hydrogen.py
============================================================
Functions for the calculation of hydrogen costs.


Functions included:
    - get_hydrogen_lc
        Returns current year levelised costs of hydrogen production

"""

# global imports
import numpy as np

from SourceCode.Green_Molecules.ftt_gmp_prices import (
    get_electricity_price,
    get_electricity_price_sd,
)

# local library imports

HYDROGEN_OPTIMISATION_STEP_HOURS = 10
NESO_PRICE_OPTION_INDEX = 1
HYDROGEN_MAX_CAPACITY_FACTOR = 0.95
HYDROGEN_DIAGNOSTIC_EVERY_YEARS = 5


def get_hydrogen_lc(data, year, mol_cost_titles, molecule_titles):
    """
    Returns current year levelised costs of hydrogen production in GBP/kWh.
    Modifies the input 'data' dictionary in-place.

    Unit conventions used in this function:
    - Hydrogen energy content and process efficiency are on an LHV basis.
    - Molecule CAPEX/OPEX fields are in GBP/kW and GBP/(kW.year),
      converted to absolute GBP using installed capacity (kW).
    - Stack lead time is intentionally treated as zero (fast replacement).

    Parameters
    -----------
    data: dictionary of NumPy arrays
        Model variables for given year of solution
    year: int
        Current year
    mol_cost_titles: dictionary
        Dictionary containing the indices for the molecule cost titles
    molecule_titles: dictionary
        Dictionary containing the indices for the molecule titles

    Returns
    ----------
    data: Global model data dictionary, updated with new levelised costs of
        hydrogen production
    """

    molecule_costs = data['gm_costs_molecules'][0, :, :].copy()
    price_curve_data = data['gm_elec_price_curve']

    capacity_kw = 6500      # Placeholder

    bop_idx = molecule_titles['3 Hydrogen BOP']
    stack_idx = molecule_titles['4 Hydrogen stack']

    capex_idx = mol_cost_titles['Capex (GBP/kW)']
    capex_sd_idx = mol_cost_titles['Capex SD']
    opex_idx = mol_cost_titles['Opex (GBP/kW)']
    opex_sd_idx = mol_cost_titles['Opex SD']
    lifetime_idx = mol_cost_titles['Lifetime']
    lead_time_idx = mol_cost_titles['Lead time']
    discount_idx = mol_cost_titles['Discount rate']
    cap_factor_idx = mol_cost_titles['Capacity factor']
    eff_idx = mol_cost_titles['Efficiency (MWh/t)']
    energy_idx = mol_cost_titles['Energy content (kWh/t)']

    default_capacity_factor = float(molecule_costs[bop_idx, cap_factor_idx])

    # During model runtime this variable is cross-sectioned to one year,
    # so the active curve is always in index 0.
    price_curve = np.asarray(price_curve_data[0, :, 0], dtype=float)
    price_curve = price_curve[np.isfinite(price_curve)]
    has_curve_data = price_curve.size > 0 and np.any(np.abs(price_curve) > 0)

    max_capacity_factor = HYDROGEN_MAX_CAPACITY_FACTOR
    max_operating_hours = int(np.floor(max_capacity_factor * 8760.0))

    if has_curve_data:
        # Align the curve level to the selected NESO annual average while
        # preserving the intra-year hourly profile.
        neso_av_prices = np.asarray(data['gm_NESO_av_electricity_price'][0, :, 0], dtype=float)
        neso_target_price = float(neso_av_prices[NESO_PRICE_OPTION_INDEX])

        curve_average = float(np.mean(price_curve))
        price_curve = price_curve + (neso_target_price - curve_average)

        # Input curve is in price-per-MWh, but hydrogen cash-flow math is in kWh.
        price_curve = price_curve / 1000.0
        price_curve = np.sort(price_curve)
        n_hours = min(price_curve.size, 8760, max_operating_hours)
        candidate_hours = np.arange(
            HYDROGEN_OPTIMISATION_STEP_HOURS,
            n_hours + 1,
            HYDROGEN_OPTIMISATION_STEP_HOURS,
            dtype=int,
        )
        if candidate_hours.size == 0 or candidate_hours[-1] != n_hours:
            candidate_hours = np.append(candidate_hours, n_hours)
        candidate_hours = np.unique(candidate_hours)

        cumulative_prices = np.cumsum(price_curve[:n_hours])

    mwh_per_t = float(molecule_costs[bop_idx, eff_idx])
    kwh_per_t_h2 = float(molecule_costs[bop_idx, energy_idx])
    lead_time = int(molecule_costs[bop_idx, lead_time_idx])
    t_project = int(molecule_costs[bop_idx, lifetime_idx])
    t_stack = int(molecule_costs[stack_idx, lifetime_idx])
    dr = float(molecule_costs[bop_idx, discount_idx])

    raw_capex_bop = float(molecule_costs[bop_idx, capex_idx]) * capacity_kw
    raw_capex_stack = float(molecule_costs[stack_idx, capex_idx]) * capacity_kw
    raw_capex_bop_sd = float(molecule_costs[bop_idx, capex_sd_idx]) * capacity_kw
    raw_capex_stack_sd = float(molecule_costs[stack_idx, capex_sd_idx]) * capacity_kw
    raw_opex_bop = float(molecule_costs[bop_idx, opex_idx]) * capacity_kw
    raw_opex_stack = float(molecule_costs[stack_idx, opex_idx]) * capacity_kw
    raw_opex_bop_sd = float(molecule_costs[bop_idx, opex_sd_idx]) * capacity_kw
    raw_opex_stack_sd = float(molecule_costs[stack_idx, opex_sd_idx]) * capacity_kw

    capex_frac = 1.0 / lead_time
    if not has_curve_data:
        capacity_factor = min(default_capacity_factor, max_capacity_factor)
        optimal_hours = int(round(capacity_factor * 8760))
        optimal_hours = min(optimal_hours, max_operating_hours)
        capacity_factor = optimal_hours / 8760.0
        
    else:
        discount_years = np.arange(lead_time, lead_time + t_project, dtype=float)
        discount_factors = (1.0 + dr) ** discount_years
        discount_sum = np.sum(1.0 / discount_factors)

        annual_elec_input_by_hours = capacity_kw * candidate_hours
        annual_h2_output_by_hours = (
            annual_elec_input_by_hours / (mwh_per_t * 1000.0)
        ) * kwh_per_t_h2
        annual_elec_cost_by_hours = capacity_kw * cumulative_prices[candidate_hours - 1]

        fixed_capex_npv = 0.0
        for c in range(lead_time):
            df_c = (1 + dr) ** c
            fixed_capex_npv += (raw_capex_bop + raw_capex_stack) * capex_frac / df_c

        fixed_operating_npv = 0.0
        for age in range(lead_time, lead_time + t_project):
            op_age = age - lead_time
            lifetime_year_costs = raw_opex_bop + raw_opex_stack
            if op_age > 0 and op_age % t_stack == 0 and op_age < t_project:
                lifetime_year_costs += raw_capex_stack
            fixed_operating_npv += lifetime_year_costs / ((1 + dr) ** age)

        candidate_npv_costs = fixed_capex_npv + fixed_operating_npv + (
            annual_elec_cost_by_hours * discount_sum
        )
        candidate_npv_generation = annual_h2_output_by_hours * discount_sum
        candidate_lcoh = candidate_npv_costs / candidate_npv_generation

        best_index = int(np.argmin(candidate_lcoh))
        optimal_hours = int(candidate_hours[best_index])
        optimal_hours = min(optimal_hours, max_operating_hours)
        capacity_factor = optimal_hours / 8760.0

     

    data['gm_costs_molecules'][0, bop_idx, cap_factor_idx] = capacity_factor
    molecule_costs[bop_idx, cap_factor_idx] = capacity_factor

    # Electricity price in £/kWh
    elec_price = get_electricity_price(data)
    elec_price_sd = get_electricity_price_sd(data)

    # 1. Production and Resource Inputs (LHV basis)
    annual_elec_input_kwh = capacity_kw * (capacity_factor * 8760)
    # Convert MWh/t to kWh/t by multiplying by 1000
    annual_capacity_tonnes = annual_elec_input_kwh / (mwh_per_t * 1000)
    annual_h2_output_kwh = annual_capacity_tonnes * kwh_per_t_h2

    # 2. Financial Metrics Extraction
    # Molecule CapEx and OpEx are input as GBP/kW and annual GBP/(kW.year).

    # Absolute annual electricity cost and variance
    annual_elec_cost_gbp = annual_elec_input_kwh * elec_price
    if has_curve_data:
        annual_elec_cost_gbp = annual_elec_cost_by_hours[best_index]
    annual_elec_variance = (annual_elec_input_kwh * elec_price_sd) ** 2

    # 3. Cash Flow & Discounting Setup
    # Capex spread evenly over the construction period (years 0 to lead_time-1)
    npv_costs = 0.0
    npv_capex_costs = 0.0
    npv_capex_bop_costs = 0.0
    npv_capex_stack_upfront_costs = 0.0
    npv_capex_stack_replacement_costs = 0.0
    npv_opex_costs = 0.0
    npv_electricity_costs = 0.0
    npv_costs_variance = 0.0
    for c in range(lead_time):
        df_c = (1 + dr) ** c
        capex_bop_cost_c = raw_capex_bop * capex_frac / df_c
        capex_stack_cost_c = raw_capex_stack * capex_frac / df_c
        capex_cost_c = capex_bop_cost_c + capex_stack_cost_c
        npv_costs += capex_cost_c
        npv_capex_costs += capex_cost_c
        npv_capex_bop_costs += capex_bop_cost_c
        npv_capex_stack_upfront_costs += capex_stack_cost_c
        npv_costs_variance += ((raw_capex_bop_sd * capex_frac / df_c) ** 2 +
                               (raw_capex_stack_sd * capex_frac / df_c) ** 2)

    npv_generation = 0.0
    costs_no_dr = raw_capex_bop + raw_capex_stack
    generation_no_dr = 0.0

    # 4. Lifetime Loop (operational years start after lead time)
    for age in range(lead_time, lead_time + t_project):
        op_age = age - lead_time
        opex_cost_year = raw_opex_bop + raw_opex_stack
        electricity_cost_year = annual_elec_cost_gbp
        lifetime_year_costs = opex_cost_year + electricity_cost_year
        year_variance = ((raw_opex_bop_sd ** 2) + (raw_opex_stack_sd ** 2) +
                         annual_elec_variance)

        # Stack replacement at every t_stack operational years (not at start
        # or end). No additional lead time is applied for replacement.
        stack_replacement = op_age > 0 and op_age % t_stack == 0 and op_age < t_project
        if stack_replacement:
            lifetime_year_costs += raw_capex_stack
            year_variance += (raw_capex_stack_sd ** 2)

        discount_factor = (1 + dr) ** age

        npv_costs += lifetime_year_costs / discount_factor
        npv_opex_costs += opex_cost_year / discount_factor
        npv_electricity_costs += electricity_cost_year / discount_factor
        if stack_replacement:
            replacement_capex_discounted = raw_capex_stack / discount_factor
            npv_capex_costs += replacement_capex_discounted
            npv_capex_stack_replacement_costs += replacement_capex_discounted
        npv_generation += annual_h2_output_kwh / discount_factor

        npv_costs_variance += year_variance / (discount_factor ** 2)

        costs_no_dr += lifetime_year_costs
        generation_no_dr += annual_h2_output_kwh

    # 5. LCOH Calculation & Assignment
    lcoh = npv_costs / npv_generation
    lcoh_capex_component = npv_capex_costs / npv_generation
    lcoh_capex_bop_component = npv_capex_bop_costs / npv_generation
    lcoh_capex_stack_component = (
        (npv_capex_stack_upfront_costs + npv_capex_stack_replacement_costs) /
        npv_generation
    )
    lcoh_opex_component = npv_opex_costs / npv_generation
    lcoh_electricity_component = npv_electricity_costs / npv_generation
    efficiency_pct = 100.0 * (kwh_per_t_h2 / 1000.0) / mwh_per_t
    lcoh_no_dr = costs_no_dr / generation_no_dr
    total_costs_sd = np.sqrt(npv_costs_variance)
    lcoh_sd = total_costs_sd / npv_generation
    
    if not hasattr(get_hydrogen_lc, "_diag_logged_years"):
        get_hydrogen_lc._diag_logged_years = set()

    if year % HYDROGEN_DIAGNOSTIC_EVERY_YEARS == 0 and year not in get_hydrogen_lc._diag_logged_years:
        print(
            f"year={year:4d}  "
            f"cf={capacity_factor:6.3f}  "
            f"in={mwh_per_t:6.2f}  "
            f"eff={efficiency_pct:5.3f}%  "
            f"capex_tot={lcoh_capex_component:7.3f}  "
            f"capex_bop={lcoh_capex_bop_component:7.3f}  "
            f"capex_stk={lcoh_capex_stack_component:7.3f}  "
            f"opex={lcoh_opex_component:7.3f}  "
            f"elec={lcoh_electricity_component:7.3f}"
        )
        get_hydrogen_lc._diag_logged_years.add(year)

    data['gm_lcoh'][:, 0, 0] = lcoh
    data['gm_lcoh_capex_component'][:, 0, 0] = lcoh_capex_component
    data['gm_lcoh_opex_component'][:, 0, 0] = lcoh_opex_component
    data['gm_lcoh_electricity_component'][:, 0, 0] = lcoh_electricity_component
    data['gm_lcoh_no_dr'][:, 0, 0] = lcoh_no_dr
    data['gm_lcoh_sd'][:, 0, 0] = lcoh_sd

    return data
