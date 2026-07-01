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
    data: Global model data dictionary

    """
    # Categories for the molecule cost matrix
    mol_cost_titles = {category: index for index, category
                  in enumerate(titles['cost_titles_molecules'])}
    molecule_titles = {category: index for index, category
                  in enumerate(titles['titles_molecules'])}
    
    molecule_costs = data['gm_costs_molecules'][0, :, :].copy()
    
    # set conversion factor for t to kwh
    conversion_factor = 1/33330

    # PLACEHOLDER VALUES FOR NOW
    # To be replaced once we understand what we are doing with electricity prices
    capacity_kw = 6500       # Your debug message implies a ~6.5 MW system
    capacity_factor = 0.90   # 90% utilization
    efficiency = 0.65        # 65% LHV efficiency
    elec_price = 0.05
    
    # Extract raw costs from the matrix, which are £/t
    annual_capacity_tonnes = (capacity_kw * capacity_factor *
                              8760 * efficiency) / 33.33 / 1000
    
    raw_capex_bop = molecule_costs[molecule_titles['3 Hydrogen BOP'], 
                                   mol_cost_titles['Capex (GBP/t)']] * annual_capacity_tonnes
    raw_capex_stack = molecule_costs[molecule_titles['4 Hydrogen stack'],
                                     mol_cost_titles['Capex (GBP/t)']] * annual_capacity_tonnes
    
    raw_opex_bop = molecule_costs[molecule_titles['3 Hydrogen BOP'], 
                                  mol_cost_titles['Opex (GBP/t)']] * annual_capacity_tonnes
    raw_opex_stack = molecule_costs[molecule_titles['4 Hydrogen stack'], 
                                    mol_cost_titles['Opex (GBP/t)']] * annual_capacity_tonnes

    t_project = int(molecule_costs[molecule_titles['3 Hydrogen BOP'],
                                   mol_cost_titles['Lifetime']]) 
    t_stack = int(molecule_costs[molecule_titles['4 Hydrogen stack'],
                                 mol_cost_titles['Lifetime']])
    dr = molecule_costs[molecule_titles['3 Hydrogen BOP'],
                        mol_cost_titles['Discount rate']]

    # Calculate annual energy and feedstock costs
    annual_elec_input_kwh = capacity_kw * (capacity_factor * 8760)
    
    # Apply efficiency to get H2 energy output
    annual_h2_output_kwh = annual_elec_input_kwh * efficiency 
    
    # Absolute annual electricity cost in £
    annual_elec_cost_gbp = annual_elec_input_kwh * elec_price

    # Cash Flow Setup
    npv_costs = raw_capex_bop + raw_capex_stack  # Year 0 Upfront Costs (£)
    npv_generation = 0

    # 5. Lifetime Loop
    for age in range(1, t_project + 1):
        # Total absolute £ spent this year
        lifetime_year_costs = raw_opex_bop + raw_opex_stack + annual_elec_cost_gbp
        
        # Add stack replacement cost if applicable
        if age % t_stack == 0 and age < t_project:
            lifetime_year_costs += raw_capex_stack
            
        discount_factor = (1 + dr) ** age
        
        npv_costs += lifetime_year_costs / discount_factor
        npv_generation += annual_h2_output_kwh / discount_factor

    # Final division: Total Discounted £ / Total Discounted kWh
    lcoh = npv_costs / npv_generation
    data['gm_lcoh'][:,0,0] = lcoh
    
    return data