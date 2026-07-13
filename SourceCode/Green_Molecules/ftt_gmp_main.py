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
def solve(data, time_lag, titles, histend, year, domain):
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