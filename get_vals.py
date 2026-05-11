import sys, os
sys.path.insert(0, os.path.abspath('src'))
from Stirling import cycle_stirling

m_air = 29.11e-6
v_min_m3kg = (31.996 * 1e-6) / m_air
v_max_m3kg = (43.783 * 1e-6) / m_air

res_th = cycle_stirling(T_min=397.1, T_max=501.1, V_min=v_min_m3kg, V_max=v_max_m3kg, show_plot=False)
W_cycle_th = res_th['energetics']['W_cycle'] * m_air * 1000 # in mJ
print("W_net_th_mJ =", W_cycle_th)
print("eta_th =", res_th['energetics']['eta'])
