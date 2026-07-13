# -*- coding: utf-8 -*-
"""
============================================================
ftt_gmp_methane.py
============================================================
Functions for the calculation of methane costs.


Functions included:
    - get_methane_lc
        Returns current year levelised costs of methane production

"""

# global imports
import numpy as np

# local library imports

def get_methane_lc(data, year, mol_cost_titles, molecule_titles, 
                   co2_type = "DAC"):
    """
    Returns current year levelised costs of methane production in £/kwh.
    Modifies the input 'data' dictionary in-place.

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
    co2_type: str
        Type of CO2 source ("DAC" or "DOC" -- defaults to "DAC")

    Returns
    ----------
    data: Global model data dictionary, updated with new levelised costs of 
        methane production
    """
    
    molecule_costs = data['gm_costs_molecules'][0, :, :].copy()
    
    capacity_kw = 3000        # Placeholder
    capacity_factor = molecule_costs[molecule_titles['2 Synthetic methane'],
                                     mol_cost_titles['Capacity factor']]
    mwh_per_t = molecule_costs[molecule_titles['2 Synthetic methane'],
                               mol_cost_titles['Efficiency (MWh/t)']]
    # Electricity price in £/kWh -- Placeholder, hardcoded for now
    elec_price = 0.10
    elec_price_sd = 0.02
    kwh_per_t_ch4 = molecule_costs[molecule_titles['2 Synthetic methane'],
                                   mol_cost_titles['Energy content (kWh/t)']]
    
    h2_price = data['gm_lcoh'][0, 0, 0]                 # £/kWh
    h2_price_sd = data['gm_lcoh_sd'][0, 0, 0]  
    
    co2_type = co2_type.upper()
    if co2_type == "DOC":
        co2_price = data['gm_lcodoc'][0, 0, 0]          # £/tCO2
        co2_price_sd = data['gm_lcodoc_sd'][0, 0, 0]
    else:
        co2_price = data['gm_lcodac'][0, 0, 0]
        co2_price_sd = data['gm_lcodac_sd'][0, 0, 0] 

    # 1. Production Metrics
    # Total annual energy output of methane in kWh
    annual_ch4_output_kwh = capacity_kw * (capacity_factor * 8760)
    # Convert annual energy output to absolute tonnes of methane
    annual_capacity_tonnes = annual_ch4_output_kwh / kwh_per_t_ch4
    # Convert annual energy output to absolute electricity input in kWh
    annual_elec_input_kwh = annual_capacity_tonnes * mwh_per_t * 1000
    
    # 2. Feedstock Requirements
    h2_req_per_kwh = molecule_costs[molecule_titles['2 Synthetic methane'], 
                                    mol_cost_titles['H2 required (kwh/kwh)']]
    co2_req_per_kwh = molecule_costs[molecule_titles['2 Synthetic methane'], 
                                     mol_cost_titles['CO2 required (tco2/kwh)']]
    
    annual_h2_needed_kwh = annual_ch4_output_kwh * h2_req_per_kwh
    annual_co2_needed_tonnes = annual_ch4_output_kwh * co2_req_per_kwh

    # 3. Extract Base Costs & Variances
    raw_capex = molecule_costs[molecule_titles['2 Synthetic methane'], 
                               mol_cost_titles['Capex (GBP/t)']] * annual_capacity_tonnes
    raw_opex = molecule_costs[molecule_titles['2 Synthetic methane'],
                              mol_cost_titles['Opex (GBP/t)']] * annual_capacity_tonnes

    sd_capex = molecule_costs[molecule_titles['2 Synthetic methane'], 
                              mol_cost_titles['Capex SD']] * annual_capacity_tonnes
    sd_opex = molecule_costs[molecule_titles['2 Synthetic methane'], 
                             mol_cost_titles['Opex SD']] * annual_capacity_tonnes

    t_project = int(molecule_costs[molecule_titles['2 Synthetic methane'],
                                   mol_cost_titles['Lifetime']]) 
    dr = molecule_costs[molecule_titles['2 Synthetic methane'], 
                        mol_cost_titles['Discount rate']]
    
    # 4. Feedstock Annual Costs and Variances
    annual_h2_cost_gbp = annual_h2_needed_kwh * h2_price
    annual_co2_cost_gbp = annual_co2_needed_tonnes * co2_price
    annual_elec_cost_gbp = annual_elec_input_kwh * elec_price
    
    # Variance
    annual_h2_variance = (annual_h2_needed_kwh * h2_price_sd) ** 2
    annual_co2_variance = (annual_co2_needed_tonnes * co2_price_sd) ** 2
    annual_elec_variance = (annual_elec_input_kwh * elec_price_sd) ** 2


    # 5. Initialize Cash Flows
    npv_costs = raw_capex  # Year 0 upfront cost
    npv_generation = 0
    npv_costs_variance = sd_capex ** 2  # Year 0 variance

    # 6. Lifetime Loop
    for age in range(1, t_project + 1):
        # Sum total annual O&M and feedstock costs
        lifetime_year_costs = (raw_opex + annual_h2_cost_gbp + annual_co2_cost_gbp +
                               annual_elec_cost_gbp)
        
        # Total variance for this operating year (assumed independent inputs)
        year_variance = ((sd_opex ** 2) + annual_h2_variance + annual_co2_variance +
                         annual_elec_variance)
        
        discount_factor = (1 + dr) ** age
        
        npv_costs += lifetime_year_costs / discount_factor
        npv_generation += annual_ch4_output_kwh / discount_factor
        
        # Variance propagation using squared discount factors
        npv_costs_variance += year_variance / (discount_factor ** 2)

    # 7. Final Metrics
    lcom = npv_costs / npv_generation
    total_costs_sd = np.sqrt(npv_costs_variance)
    lcom_sd = total_costs_sd / npv_generation

    # Save to data dictionary
    data['gm_lcom'][:, 0, 0] = lcom
    data['gm_lcom_sd'][:, 0, 0] = lcom_sd

    return data