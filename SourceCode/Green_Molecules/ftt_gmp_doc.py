# -*- coding: utf-8 -*-
"""
============================================================
ftt_gmp_doc.py
============================================================
Functions for the calculation of DOC costs.


Functions included:
    - get_doc_lc
        Returns current year levelised costs of DOC production

"""

# global imports
import numpy as np

# local library imports

def get_doc_lc(data, year, rem_cost_titles, removal_titles):
    """
    Returns current year levelised costs of DOC production in £/tCO2.
    Modifies the input 'data' dictionary in-place.

    Parameters
    -----------
    data: dictionary of NumPy arrays
        Model variables for given year of solution
    year: int
        Current year
    rem_cost_titles: dictionary
        Dictionary containing the indices for the cost titles
    removal_titles: dictionary
        Dictionary containing the indices for the removal titles

    Returns
    ----------
    data: Global model data dictionary, updated with new levelised costs of 
        DOC production
    """
    
    removal_costs = data['gm_costs_removal'][0, :, :].copy()

    # PLACEHOLDER VALUES FOR NOW
    capacity_tCO2 = 1000
    capacity_factor = float(removal_costs[removal_titles['2 DOC'],
                                         rem_cost_titles['Capacity factor']])
    mwh_per_tCO2 = removal_costs[removal_titles['2 DOC'],
                                 rem_cost_titles['Elec efficiency (MWh/tCO2)']]
    
    # Electricity price in £/kWh
    elec_price = data['gm_elec_price'][0, 0, 0]
    elec_price_sd = data['gm_elec_price_sd'][0, 0, 0]
    
    # 1. Production and Resource Inputs
    # Annual drawdown (tCO2/year)
    annual_removal_tCO2 = capacity_tCO2 * capacity_factor
    # Annual electricity input (kWh/year)
    annual_elec_input_kwh = annual_removal_tCO2 * mwh_per_tCO2 * 1000
    
    # 2. Financial Metrics Extraction
    raw_capex = (removal_costs[removal_titles['2 DOC'], 
                                   rem_cost_titles['Capex (GBP/tCO2)']] * 
                     annual_removal_tCO2)
    raw_capex_sd = (removal_costs[removal_titles['2 DOC'], 
                                   rem_cost_titles['Capex SD']] *
                        annual_removal_tCO2)
    raw_opex = (removal_costs[removal_titles['2 DOC'], 
                                  rem_cost_titles['Opex (GBP/tCO2)']] * 
                    annual_removal_tCO2)
    raw_opex_sd = (removal_costs[removal_titles['2 DOC'], 
                                  rem_cost_titles['Opex SD']] *
                       annual_removal_tCO2)

    lt_project = int(removal_costs[removal_titles['2 DOC'], 
                                   rem_cost_titles['Lifetime']])
    dr = removal_costs[removal_titles['2 DOC'], 
                        rem_cost_titles['Discount rate']]

    # Absolute annual electricity cost (£)
    annual_elec_cost_gbp = annual_elec_input_kwh * elec_price
    annual_elec_variance = (annual_elec_input_kwh * elec_price_sd) ** 2

    # 3. Cash Flow & Discounting Setup
    npv_costs = (
        0.5 * raw_capex / (1 + dr) ** 0 + 
        0.5 * raw_capex / (1 + dr) ** 1
    ) # 2 year build time with costs split evenly between the 2 years (£)
    npv_generation = 0
    npv_cost_variance = (
        (0.5 * raw_capex_sd / (1 + dr) ** 0) ** 2 + 
        (0.5 * raw_capex_sd / (1 + dr) ** 1) ** 2
    ) 
    
    costs_no_dr = raw_capex
    generation_no_dr = 0
    costs_no_dr_variance = raw_capex_sd ** 2

    # 4. Lifetime Loop
    for age in range(2, lt_project + 2):
        lifetime_year_costs = (raw_opex + 
                               annual_elec_cost_gbp)
        year_variance = (raw_opex_sd ** 2) + annual_elec_variance
        
        discount_factor = (1 + dr) ** age
        
        npv_costs += lifetime_year_costs / discount_factor
        npv_generation += annual_removal_tCO2 / discount_factor
        npv_cost_variance += year_variance / (discount_factor ** 2)
        
        costs_no_dr += lifetime_year_costs  
        generation_no_dr += annual_removal_tCO2  
        costs_no_dr_variance += year_variance

    # 5. LCOH Calculation & Assignment
    lcodoc = npv_costs / npv_generation
    lcodoc_no_dr = costs_no_dr / generation_no_dr
    total_costs_sd = np.sqrt(npv_cost_variance)
    lcodoc_sd = total_costs_sd / npv_generation
    total_costs_no_dr_sd = np.sqrt(costs_no_dr_variance)
    lcodoc_no_dr_sd = total_costs_no_dr_sd / generation_no_dr
    
    data['gm_lcodoc'][:, 0, 0] = lcodoc
    data['gm_lcodoc_no_dr'][:, 0, 0] = lcodoc_no_dr
    data['gm_lcodoc_sd'][:, 0, 0] = lcodoc_sd
    data['gm_lcodoc_no_dr_sd'][:, 0, 0] = lcodoc_no_dr_sd

    return data