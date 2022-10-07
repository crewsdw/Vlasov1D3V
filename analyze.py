import numpy as np
import grid as g
import variables as var
import fields
import plotter as my_plt
import data
import cupy as cp

# Normalization parameters
om_pc, vt_c = 10.0, 0.3  # 0.1
charge_sign = -1.0

# elements and order
elements, order = [10, 12, 28, 28], 10

# Geometry
grid_fundamental = 0.41  # 0.1  # 0.1  # / om_pc
length = 2.0 * np.pi / grid_fundamental
lows = np.array([-0.5 * length, -8, -20, -20])
highs = np.array([0.5 * length, 8, 20, 20])
grid = g.PhaseSpace(lows=lows, highs=highs, elements=elements, order=order, charge_sign=charge_sign, om_pc=om_pc)

# Read data
data_file = data.Data(folder='data\\', filename='run_to_t6.0')
t_data, f_data, n_data, jv_data, jw_data, ex_data, ey_data, ez_data, by_data, bz_data = data_file.read_data()

# Look at final distribution
final_distribution = var.Distribution(resolutions=elements, order=order)
final_distribution.arr_nodal = cp.asarray(f_data[-1, :, :, :, :, :, :])
final_distribution.fourier_transform()

plotter3 = my_plt.Plotter3D(grid=grid)
plotter3.distribution_contours3d(distribution=final_distribution, spectral_idx=0, ctype='real')
