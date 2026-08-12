# -*- coding: utf-8 -*-
"""
============================================================
ftt_gmp_lcoe.py
============================================================
Functions for the calculation of combustion costs.


Functions included:
    - get_lcoe
        Returns current year levelised costs of producing electricity for each
        pathway in £/kwh.

"""

# global imports
import numpy as np

# local library imports

FOSSIL_GAS_PRICE_SD_FRACTION = 0.25

def get_lcoe(data, year, molecule_titles,
             comb_cost_titles, combustion_titles, pathway_titles,
             rem_cost_titles, removal_titles):
    """
    Returns current year levelised costs of producing electricity for each
    pathway in £/kwh. Modifies the input 'data' dictionary in-place.

    Each pathway combines a molecule feedstock (fossil gas, e-methane, or
    hydrogen), a combustion route (conventional CCGT, CCS retrofit, H2 ready,
    or H2 retrofit), and an optional CO2 removal step (DAC or DOC). The
    applicable technologies for each pathway are determined by the interaction
    matrices stored in ``data``.

    The LCOE is computed via NPV discounting of combustion plant capex/opex,
    feedstock fuel costs, and CO2 removal costs, normalised against the
    discounted electricity generation over the plant lifetime.

    Parameters
    -----------
    data: dictionary of NumPy arrays
        Model variables for given year of solution
    year: int
        Current year
    molecule_titles: dictionary
        Dictionary containing the indices for the molecule titles
    comb_cost_titles: dictionary
        Dictionary containing the indices for the combustion cost titles
    combustion_titles: dictionary
        Dictionary containing the indices for the combustion technology titles
    pathway_titles: dictionary
        Dictionary containing the indices for the GMP pathway titles
    rem_cost_titles: dictionary
        Dictionary containing the indices for the removal cost titles
    removal_titles: dictionary
        Dictionary containing the indices for the CO2 removal technology titles

    Returns
    ----------
    data: Global model data dictionary, updated with new levelised costs of 
        producing electricity for each pathway stored in
        ``data['gm_pathway_lcoe']`` and ``data['gm_pathway_lcoe_sd']``
        (GBP/kWh).
    """
    # Extract cost matrices
    # shape: (n_combustion_tech, n_comb_cost_titles)
    comb_costs = data['gm_costs_combustion'][0, :, :]

    # Extract interaction matrices (shape: (n_pathways, n_tech))
    inter_comb = data['gm_interaction_combustion'][0, :, :]
    inter_mol = data['gm_interaction_molecules'][0, :, :]
    inter_rem = data['gm_interaction_removal'][0, :, :]

    n_pathways = len(pathway_titles)
    n_molecules = len(molecule_titles)
    n_removals = len(removal_titles)

    # Build molecule fuel price vector (GBP/kWh_fuel)
    # Hydrogen and methane: use pre-computed LCOH/LCOM from data
    molecule_fuel_prices = np.zeros(n_molecules)
    molecule_fuel_sd = np.zeros(n_molecules)

    fg_idx = molecule_titles['1 Fossil gas']
    molecule_fuel_prices[fg_idx] = float(
        data['gm_fossil_gas_price'][0, 0, 0])
    molecule_fuel_sd[fg_idx] = (
        molecule_fuel_prices[fg_idx] * FOSSIL_GAS_PRICE_SD_FRACTION)

    ch4_idx = molecule_titles['2 Synthetic methane']
    # Note: methane LCOM selection (DAC vs DOC) is handled per-pathway in the
    # loop below via the removal interaction matrix; no single price set here.

    h2_idx = molecule_titles['3 Hydrogen BOP']
    molecule_fuel_prices[h2_idx] = data['gm_lcoh'][0, 0, 0]
    molecule_fuel_sd[h2_idx] = data['gm_lcoh_sd'][0, 0, 0]

    # Build removal cost vectors (GBP/tCO2)
    removal_prices = np.zeros(n_removals)
    removal_prices_sd = np.zeros(n_removals)

    dac_idx = removal_titles['1 DAC']
    removal_prices[dac_idx] = data['gm_lcodac'][0, 0, 0]
    removal_prices_sd[dac_idx] = data['gm_lcodac_sd'][0, 0, 0]

    doc_idx = removal_titles['2 DOC']
    removal_prices[doc_idx] = data['gm_lcodoc'][0, 0, 0]
    removal_prices_sd[doc_idx] = data['gm_lcodoc_sd'][0, 0, 0]

    # Capacity = 1 kW (costs are in GBP/kW so this normalisation unit
    # cancels out cleanly in the LCOE ratio)
    capacity_kw = 1.0

    for p_idx in range(n_pathways):
        # Find applicable combustion technology (first match in row)
        t_vec = inter_comb[p_idx, :]
        if not np.any(t_vec):
            continue
        t_idx = int(np.argmax(t_vec))

        # Combustion technology parameters
        capacity_factor = float(
            comb_costs[t_idx, comb_cost_titles['Capacity factor']])
        efficiency = float(
            comb_costs[t_idx, comb_cost_titles['Efficiency (%)']])
        co2_g_per_kwh = float(
            comb_costs[t_idx, comb_cost_titles['CO2 emissions (gCO2/kWh)']])

        raw_capex = (
            float(comb_costs[t_idx, comb_cost_titles['Capex (GBP/kW)']]) *
            capacity_kw)
        raw_capex_sd = (
            float(comb_costs[t_idx, comb_cost_titles['Capex SD']]) *
            capacity_kw)
        raw_opex = (
            float(comb_costs[t_idx, comb_cost_titles['Opex (GBP/kW)']]) *
            capacity_kw)
        raw_opex_sd = (
            float(comb_costs[t_idx, comb_cost_titles['Opex SD']]) *
            capacity_kw)
        t_project = int(
            float(comb_costs[t_idx, comb_cost_titles['Lifetime']]))
        dr = float(comb_costs[t_idx, comb_cost_titles['Discount rate']])

        # Annual electricity output (kWh/year per kW installed)
        annual_elec_kwh = capacity_kw * capacity_factor * 8760

        # Annual fuel input (kWh_fuel/year)
        annual_fuel_kwh = annual_elec_kwh / efficiency if efficiency > 0 else 0.0

        # Fuel cost and removal cost — handling differs for methane pathways.
        # For methane: the molecule feedstock is produced from CCS-recycled
        # CO2, so the LCOM (which only includes the CO2 transport/storage cost)
        # is used as the fuel price. DAC or DOC still removes the combustion
        # leakage emissions.
        # For fossil gas / hydrogen: standard fuel price, and DAC/DOC removes
        # the combustion leakage emissions.
        m_vec = inter_mol[p_idx, :]
        r_vec = inter_rem[p_idx, :]

        co2_per_kwh_tco2 = co2_g_per_kwh * 1e-6
        removal_price    = float(np.dot(r_vec, removal_prices))
        removal_price_sd = float(np.dot(r_vec, removal_prices_sd))

        if float(m_vec[ch4_idx]) > 0:
            # Synthetic methane pathway: fuel produced from CCS-recycled CO2
            fuel_price = float(data['gm_lcom_ccs'][0, 0, 0])
            fuel_sd    = float(data['gm_lcom_ccs_sd'][0, 0, 0])
        else:
            # Fossil gas / hydrogen: standard fuel price
            fuel_price = float(np.dot(m_vec, molecule_fuel_prices))
            fuel_sd    = float(np.dot(m_vec, molecule_fuel_sd))

        annual_removal_cost     = annual_elec_kwh * co2_per_kwh_tco2 * removal_price
        annual_removal_variance = (
            annual_elec_kwh * co2_per_kwh_tco2 * removal_price_sd) ** 2

        # Fossil gas pathways need extra DAC/DOC to offset upstream CO2e emissions
        # so both fossil-gas routes remain net zero.
        if float(m_vec[fg_idx]) > 0:
            upstream_intensity_gco2e_per_mj = float(data['gm_CO2_intensity_upstream'][0, 0, 0])
            # Convert MJ-based intensity to kWh_fuel and use the actual fuel
            # input, which already reflects the combustion efficiency loss.
            extra_removal_tco2 = (
                annual_fuel_kwh * 3.6 * upstream_intensity_gco2e_per_mj * 1e-6
            )
            annual_removal_cost += extra_removal_tco2 * removal_prices[dac_idx]
            annual_removal_variance += (
                extra_removal_tco2 * removal_prices_sd[dac_idx]
            ) ** 2

        annual_fuel_cost     = annual_fuel_kwh * fuel_price
        annual_fuel_variance = (annual_fuel_kwh * fuel_sd) ** 2

        # NPV calculation — year 0: upfront capex
        npv_costs = raw_capex
        npv_generation = 0.0
        npv_costs_variance = raw_capex_sd ** 2

        for age in range(1, t_project + 1):
            year_costs = raw_opex + annual_fuel_cost + annual_removal_cost
            year_variance = (raw_opex_sd ** 2 +
                             annual_fuel_variance +
                             annual_removal_variance)
            discount_factor = (1 + dr) ** age
            npv_costs += year_costs / discount_factor
            npv_generation += annual_elec_kwh / discount_factor
            npv_costs_variance += year_variance / (discount_factor ** 2)

        lcoe = npv_costs / npv_generation
        lcoe_sd = np.sqrt(npv_costs_variance) / npv_generation

        data['gm_pathway_lcoe'][:, p_idx, 0] = lcoe
        data['gm_pathway_lcoe_sd'][:, p_idx, 0] = lcoe_sd

    return data
    