# -*- coding: utf-8 -*-
"""
============================================================
ftt_gmp_lbd.py
============================================================
Functions for the calculation of learning by doing.


Functions included:
    - calc_lbd
        Main function cost matrix values based on cumulative global deployment of
        each technology and the learning rate for each technology.
    - get_learning_factor
        Returns the learning factor for technologies based on cumulative deployment.
    
"""

# global imports
import numpy as np

# local imports

def calc_lbd(data, data_dt, time_lag, year, rem_cost_titles, mol_cost_titles, 
             comb_cost_titles):
    """
    Returns updated values for cost matrix from learning by doing effects
    based on global deployment.

    Parameters
    -----------
    data: dictionary of NumPy arrays
        Model variables for given year of solution
    time_lag: dictionary of NumPy arrays
        Model variables from the previous year
    year: int
        Current year
    rem_cost_titles: dictionary of lists
        Dictionary containing all title classification for removal costs
    mol_cost_titles: dictionary of lists
        Dictionary containing all title classification for molecule costs
    comb_cost_titles: dictionary of lists
        Dictionary containing all title classification for combustion costs
        

    Returns
    ----------
    data: Global model data dictionary, updated with new cost matrx values
    """
    # 1. Extract capex, opex, and efficiency values
    capex_removal = data_dt['gm_costs_removal'][:, :, rem_cost_titles['Capex (GBP/tCO2/year)']]
    opex_removal = data_dt['gm_costs_removal'][:, :, rem_cost_titles['Opex (GBP/tCO2/year)']]
    elec_eff_removal = data_dt['gm_costs_removal'][:, :, rem_cost_titles['Elec efficiency (MWh/tCO2)']]
    heat_eff_removal = data_dt['gm_costs_removal'][:, :, rem_cost_titles['Heat efficiency (MWh/tCO2)']]

    capex_molecule = data_dt['gm_costs_molecules'][:, :, mol_cost_titles['Capex (GBP/kW)']]
    opex_molecule = data_dt['gm_costs_molecules'][:, :, mol_cost_titles['Opex (GBP/kW)']]
    eff_molecule = data_dt['gm_costs_molecules'][:, :, mol_cost_titles['Efficiency (MWh/t)']]
    
    capex_combustion = data_dt['gm_costs_combustion'][:, :, comb_cost_titles['Capex (GBP/kW)']]
    opex_combustion = data_dt['gm_costs_combustion'][:, :, comb_cost_titles['Opex (GBP/kW)']]
    eff_combustion = data_dt['gm_costs_combustion'][:, :, comb_cost_titles['Efficiency (%)']]
    
    # 2. Extract learning rate values    
    capex_lr_removal = data['gm_costs_removal'][:, :, rem_cost_titles['Capex learning exp']]
    opex_lr_removal = data['gm_costs_removal'][:, :, rem_cost_titles['Opex learning exp']]
    eff_lr_removal = data['gm_costs_removal'][:, :, rem_cost_titles['Efficiency learning exp']]
    
    capex_lr_molecule = data['gm_costs_molecules'][:, :, mol_cost_titles['Capex learning exp']]
    opex_lr_molecule = data['gm_costs_molecules'][:, :, mol_cost_titles['Opex learning exp']]
    eff_lr_molecule = data['gm_costs_molecules'][:, :, mol_cost_titles['Efficiency learning exp']]
    
    capex_lr_combustion = data['gm_costs_combustion'][:, :, comb_cost_titles['Capex learning exp']]
    opex_lr_combustion = data['gm_costs_combustion'][:, :, comb_cost_titles['Opex learning exp']]
    eff_lr_combustion = data['gm_costs_combustion'][:, :, comb_cost_titles['Efficiency learning exp']]
    
    # 3. Extract cumulative deployment values
    capacity_removal = data["gm_gbl_cap_removal"][0, :, 0]
    capacity_dt_removal = data_dt["gm_gbl_cap_removal"][0, :, 0]
    capacity_molecule = data["gm_gbl_cap_molecules"][0, :, 0]
    capacity_dt_molecule = data_dt["gm_gbl_cap_molecules"][0, :, 0]
    capacity_combustion = data["gm_gbl_cap_combustion"][0, :, 0]
    capacity_dt_combustion = data_dt["gm_gbl_cap_combustion"][0, :, 0]
    
    # 4. Get learning factors
    learning_factor_capex_removal = get_learning_factor(capacity_removal, 
                                                        capacity_dt_removal, 
                                                        capex_lr_removal)
    learning_factor_opex_removal = get_learning_factor(capacity_removal, 
                                                        capacity_dt_removal,
                                                        opex_lr_removal)
    learning_factor_eff_removal = get_learning_factor(capacity_removal, 
                                                      capacity_dt_removal,
                                                      eff_lr_removal)
    
    learning_factor_capex_molecule = get_learning_factor(capacity_molecule,
                                                        capacity_dt_molecule,
                                                        capex_lr_molecule)
    
    learning_factor_opex_molecule = get_learning_factor(capacity_molecule,
                                                        capacity_dt_molecule,
                                                        opex_lr_molecule)
    learning_factor_eff_molecule = get_learning_factor(capacity_molecule,
                                                      capacity_dt_molecule,
                                                      eff_lr_molecule)

    learning_factor_capex_combustion = get_learning_factor(capacity_combustion,
                                                          capacity_dt_combustion,
                                                          capex_lr_combustion)
    learning_factor_opex_combustion = get_learning_factor(capacity_combustion,
                                                          capacity_dt_combustion,
                                                          opex_lr_combustion)
    learning_factor_eff_combustion = get_learning_factor(capacity_combustion,
                                                        capacity_dt_combustion,
                                                        eff_lr_combustion)
    
    # 5. Update cost matrix values
    data['gm_costs_removal'][:, :, rem_cost_titles['Capex (GBP/tCO2/year)']] = capex_removal * learning_factor_capex_removal
    data['gm_costs_removal'][:, :, rem_cost_titles['Opex (GBP/tCO2/year)']] = opex_removal * learning_factor_opex_removal
    data['gm_costs_removal'][:, :, rem_cost_titles['Elec efficiency (MWh/tCO2)']] = elec_eff_removal * learning_factor_eff_removal
    data['gm_costs_removal'][:, :, rem_cost_titles['Heat efficiency (MWh/tCO2)']] = heat_eff_removal * learning_factor_eff_removal
    
    data['gm_costs_molecules'][:, :, mol_cost_titles['Capex (GBP/kW)']] = capex_molecule * learning_factor_capex_molecule
    data['gm_costs_molecules'][:, :, mol_cost_titles['Opex (GBP/kW)']] = opex_molecule * learning_factor_opex_molecule
    data['gm_costs_molecules'][:, :, mol_cost_titles['Efficiency (MWh/t)']] = eff_molecule * learning_factor_eff_molecule
    
    data['gm_costs_combustion'][:, :, comb_cost_titles['Capex (GBP/kW)']] = capex_combustion * learning_factor_capex_combustion
    data['gm_costs_combustion'][:, :, comb_cost_titles['Opex (GBP/kW)']] = opex_combustion * learning_factor_opex_combustion
    
    # Efficency for combustion is a percentage, so need to ensure that the learning factor is applied correctly
    old_comb_loss = 1 - eff_combustion
    new_comb_loss = old_comb_loss * learning_factor_eff_combustion
    new_eff_combustion = 1 - new_comb_loss
    data['gm_costs_combustion'][:, :, comb_cost_titles['Efficiency (%)']] = new_eff_combustion
    
    return data


def get_learning_factor(capacity, capacity_dt, learning_exp):
    """
    Returns the learning factor for technologies based on cumulative deployment.

    Parameters
    -----------
    capacity: array
        Current capacity of technologies
    capacity_dt: array
        Capacity of technologies from the previous timestep
    learning_exp: array
        Learning exponents for the technologies

    Returns
    ----------
    learning_factor: array
        The calculated learning factor for the technologies
    """
    # Placeholder! First appoximate additions based on difference between capacity and capacity_dt
    additions = capacity - capacity_dt
    # Convert exponent to absolute value
    learning_exp = abs(learning_exp)
    
    # Safe division: only divide where capacity_dt and additions are greater than 0
    # Otherwise, default the ratio to 1.0 (which means learning_factor = 1.0)
    safe_ratio = np.where(
        (capacity_dt > 0) & (additions > 0), 
        capacity / np.where(capacity_dt > 0, capacity_dt, 1.0), 
        1.0
    )
    learning_factors = safe_ratio ** (-learning_exp)
    return learning_factors