# -*- coding: utf-8 -*-
"""
============================================================
ftt_gmp_main.py
============================================================
Green Molecules Project FTT module.

This is the main file for FTT-Green Molecules.


Functions included:
    - solve
        Main solution function for the module

"""

# Third party imports
import numpy as np

# Local library imports
from SourceCode.Green_Molecules.ftt_gmp_dac import get_dac_lc
from SourceCode.Green_Molecules.ftt_gmp_doc import get_doc_lc
from SourceCode.Green_Molecules.ftt_gmp_hydrogen import get_hydrogen_lc
from SourceCode.Green_Molecules.ftt_gmp_methane import get_methane_lc
from SourceCode.Green_Molecules.ftt_gmp_lcoe import get_lcoe
from SourceCode.Green_Molecules.ftt_gmp_lbd import calc_lbd

from SourceCode.ftt_core.ftt_sales_or_investments import get_sales
from SourceCode.ftt_core.ftt_shares import shares_change
from SourceCode.ftt_core.ftt_mandate import implement_seeding, implement_mandate
from SourceCode.ftt_core.ftt_exogenous_sales import exogenous_sales
from SourceCode.ftt_core.ftt_exogenous_capacity import regulation_correction

from SourceCode.support.divide import divide
from SourceCode.support.check_market_shares import check_market_shares

# -----------------------------------------------------------------------------
# ----------------------------- Main ------------------------------------------
# -----------------------------------------------------------------------------
def solve(data, time_lag, titles, histend, year):
    """
    Main solution function for the GMP module.

    This function simulates investor decision making for the production of 
    green molecules and the subsequent production of electricity for grid 
    balancing. 
    
    Levelised costs (from the get_gm_lc function) are taken and market shares
    for each pathway are simulated to ensure demand (curtailment) is met.

    Parameters
    -----------
    data: dictionary of NumPy arrays
        Model variables for given year of solution
    time_lag: type
        Model variables from the previous year
    titles: dictionary of lists
        Dictionary containing all title classification
    histend: dict of integers
        Final year of histrorical data by variable
    year: int
        Current year

    Returns
    ----------
    data: dictionary of NumPy arrays
        Model variables for the given year of solution

    """
    # Initialise data_dt dictionary
    data_dt = {}
    # Save previous year data to data_dt
    for var in time_lag.keys():
        data_dt[var] = np.copy(time_lag[var])
        
    # Set number of iterations for the solution loop
    no_it = int(data['noit'][0, 0, 0])
    
    # MAIN IN-YEAR SOLUTION LOOP    
    for t in range(1, no_it + 1):

        # CO2 REMOVAL
        rem_cost_titles = {category: index for index, category 
                        in enumerate(titles['cost_titles_removal'])}
        removal_titles = {category: index for index, category 
                        in enumerate(titles['titles_removal'])}
        data = get_doc_lc(data, year, rem_cost_titles, removal_titles)
        data = get_dac_lc(data, year, rem_cost_titles, removal_titles)
        
        # MOLECULE PRODUCTION
        molecule_titles = {category: index for index, category 
                        in enumerate(titles['titles_molecules'])}
        mol_cost_titles = {category: index for index, category 
                        in enumerate(titles['cost_titles_molecules'])}
        data = get_hydrogen_lc(data, year, mol_cost_titles, molecule_titles)
        data = get_methane_lc(data, year, mol_cost_titles, molecule_titles,
                            co2_type = "DAC")
        data = get_methane_lc(data, year, mol_cost_titles, molecule_titles,
                            co2_type = "DOC")
        
        # COMBUSTION
        comb_cost_titles = {category: index for index, category
                            in enumerate(titles['cost_titles_combustion'])}
        combustion_titles = {category: index for index, category
                            in enumerate(titles['titles_combustion'])}
        pathway_titles = {category: index for index, category
                        in enumerate(titles['titles_gm_pathways'])}
        data = get_lcoe(data, year, mol_cost_titles, molecule_titles,
                        comb_cost_titles, combustion_titles, pathway_titles,
                        rem_cost_titles, removal_titles)
        
        if year > histend['gm_costs_molecules'] :
            calc_lbd(data, data_dt, time_lag, year, rem_cost_titles, 
                     mol_cost_titles, comb_cost_titles)
        
        # Update data_dt for the next timestep
        for var in time_lag.keys():
            data_dt[var] = np.copy(data[var])
                
    return data