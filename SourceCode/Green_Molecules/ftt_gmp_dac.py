# -*- coding: utf-8 -*-
"""
============================================================
ftt_gmp_dac.py
============================================================
Functions for the calculation of DAC costs.


Functions included:
    - get_dac_lc
        Returns current year levelised costs of DAC production

"""

# global imports

# local library imports

def get_dac_lc(data, year, titles):
    """
    Returns current year levelised costs of DAC production in £/tCO2.
    Modifies the input 'data' dictionary in-place.

    Parameters
    -----------
    data: dictionary of NumPy arrays
        Model variables for given year of solution
    year: int
        Current year
    titles: dictionary of lists
        Dictionary containing all title classification

    Returns
    ----------
    data: Global model data dictionary, updated with new levelised costs of 
        DAC production
    """
    # Categories for the molecule cost matrix
    rem_cost_titles = {category: index for index, category 
                       in enumerate(titles['cost_titles_removal'])}
    removal_titles = {category: index for index, category 
                       in enumerate(titles['titles_removal'])}
    
    removal_costs = data['gm_costs_removal'][0, :, :].copy()

    # PLACEHOLDER VALUES FOR NOW
    capacity_tCO2 = 1000 
    capacity_factor = 0.90   # 90% utilization
    mwh_per_tCO2 = removal_costs[removal_titles['1 DAC'],
                                 rem_cost_titles['Efficiency (MWh/tCO2)']]
    
    # Electricity price in £/kWh -- hardcoded for now
    elec_price = 0.10
    
    # 1. Production and Resource Inputs
    # Annual drawdown (tCO2/year)
    annual_removal_tCO2 = capacity_tCO2 * capacity_factor
    # Annual electricity input (kWh/year)
    annual_elec_input_kwh = annual_removal_tCO2 * mwh_per_tCO2 * 1000
    
    # 2. Financial Metrics Extraction
    raw_capex = (removal_costs[removal_titles['1 DAC'], 
                                   rem_cost_titles['Capex (GBP/tCO2)']] * 
                     annual_removal_tCO2)
    raw_opex = (removal_costs[removal_titles['1 DAC'], 
                                  rem_cost_titles['Opex (GBP/tCO2)']] * 
                    annual_removal_tCO2)

    lt_project = int(removal_costs[removal_titles['1 DAC'], 
                                   rem_cost_titles['Lifetime']])
    dr = removal_costs[removal_titles['1 DAC'], 
                        rem_cost_titles['Discount rate']]

    # Absolute annual electricity cost (£)
    annual_elec_cost_gbp = annual_elec_input_kwh * elec_price

    # 3. Cash Flow & Discounting Setup
    npv_costs = (
        0.5 * raw_capex / (1 + dr) ** 0.5 + 
        0.5 * raw_capex / (1 + dr) ** 1
    ) # 2 year build time with costs split evenly between the 2 years (£)
    npv_generation = 0
    
    costs_no_dr = raw_capex
    generation_no_dr = 0

    # 4. Lifetime Loop
    for age in range(2, lt_project + 2):
        lifetime_year_costs = (raw_opex + 
                               annual_elec_cost_gbp)
        
        discount_factor = (1 + dr) ** age
        
        npv_costs += lifetime_year_costs / discount_factor
        npv_generation += annual_removal_tCO2 / discount_factor
        
        costs_no_dr += lifetime_year_costs  
        generation_no_dr += annual_removal_tCO2  

    # 5. LCOH Calculation & Assignment
    lcodac = npv_costs / npv_generation
    lcodac_no_dr = costs_no_dr / generation_no_dr
    
    data['gm_lcodac'][:, 0, 0] = lcodac
    data['gm_lcodac_no_dr'][:, 0, 0] = lcodac_no_dr

    return data