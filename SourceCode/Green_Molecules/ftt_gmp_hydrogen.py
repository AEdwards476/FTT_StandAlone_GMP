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

# local library imports

def get_hydrogen_lc(data, year, titles):
    """
    Returns current year levelised costs of hydrogen production in £/kwh.
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
        hydrogen production
    """
    # Categories for the molecule cost matrix
    mol_cost_titles = {category: index for index, category 
                       in enumerate(titles['cost_titles_molecules'])}
    molecule_titles = {category: index for index, category 
                       in enumerate(titles['titles_molecules'])}
    
    molecule_costs = data['gm_costs_molecules'][0, :, :].copy()
    
    # Constants
    KWH_PER_TONNE_H2 = 33330  # LHV of Hydrogen (~33.33 kWh/kg)

    # PLACEHOLDER VALUES FOR NOW
    capacity_kw = 6500 
    capacity_factor = 0.90   # 90% utilization
    mwh_per_t = molecule_costs[molecule_titles['3 Hydrogen BOP'],
                               mol_cost_titles['Efficiency (MWh/t)']]
    # Electricity price in £/kWh -- hardcoded for now
    elec_price = 0.10
    
    # 1. Production and Resource Inputs
    annual_elec_input_kwh = capacity_kw * (capacity_factor * 8760)
    # Convert MWh/t to kWh/t by multiplying by 1000
    annual_capacity_tonnes = annual_elec_input_kwh / (mwh_per_t * 1000)
    
    # Fix: Calculate H2 energy output using the physical energy density constant
    annual_h2_output_kwh = annual_capacity_tonnes * KWH_PER_TONNE_H2 
    
    # 2. Financial Metrics Extraction
    raw_capex_bop = (molecule_costs[molecule_titles['3 Hydrogen BOP'], 
                                   mol_cost_titles['Capex (GBP/t)']] * 
                     annual_capacity_tonnes)
    raw_capex_stack = (molecule_costs[molecule_titles['4 Hydrogen stack'],
                                     mol_cost_titles['Capex (GBP/t)']] * 
                       annual_capacity_tonnes)

    raw_opex_bop = (molecule_costs[molecule_titles['3 Hydrogen BOP'], 
                                  mol_cost_titles['Opex (GBP/t)']] * 
                    annual_capacity_tonnes)
    raw_opex_stack = (molecule_costs[molecule_titles['4 Hydrogen stack'], 
                                    mol_cost_titles['Opex (GBP/t)']] * 
                      annual_capacity_tonnes)

    t_project = int(molecule_costs[molecule_titles['3 Hydrogen BOP'],
                                   mol_cost_titles['Lifetime']]) 
    t_stack = int(molecule_costs[molecule_titles['4 Hydrogen stack'],
                                 mol_cost_titles['Lifetime']])
    dr = molecule_costs[molecule_titles['3 Hydrogen BOP'], 
                        mol_cost_titles['Discount rate']]

    # Absolute annual electricity cost (£)
    annual_elec_cost_gbp = annual_elec_input_kwh * elec_price

    # 3. Cash Flow & Discounting Setup
    npv_costs = raw_capex_bop + raw_capex_stack  # Year 0 Upfront Costs (£)
    npv_generation = 0
    
    costs_no_dr = raw_capex_bop + raw_capex_stack 
    generation_no_dr = 0

    # 4. Lifetime Loop
    for age in range(1, t_project + 1):
        lifetime_year_costs = (raw_opex_bop + raw_opex_stack + 
                               annual_elec_cost_gbp)
        
        # Stack replacement logic
        if age % t_stack == 0 and age < t_project:
            lifetime_year_costs += raw_capex_stack
            
        discount_factor = (1 + dr) ** age
        
        npv_costs += lifetime_year_costs / discount_factor
        npv_generation += annual_h2_output_kwh / discount_factor
        
        costs_no_dr += lifetime_year_costs  
        generation_no_dr += annual_h2_output_kwh  

    # 5. LCOH Calculation & Assignment
    lcoh = npv_costs / npv_generation
    lcoh_no_dr = costs_no_dr / generation_no_dr
    
    data['gm_lcoh'][:, 0, 0] = lcoh
    data['gm_lcoh_no_dr'][:, 0, 0] = lcoh_no_dr

    return data