python Utilities/monte_carlo.py generate --base-scenario S0,nat_scale,gas_shock_loc,gas_shock_nat --n 100
python Utilities/monte_carlo.py run --base-scenario S0,nat_scale,gas_shock_loc,gas_shock_nat
python figures/make_figures.py --base-scenario S0,nat_scale,gas_shock_loc,gas_shock_nat --format svg
python figures/make_figures.py --base-scenario S0,nat_scale,gas_shock_loc,gas_shock_nat --format png