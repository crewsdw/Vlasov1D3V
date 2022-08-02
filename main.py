import numpy as np
import cupy as cp
import grid as g
import variables as var
import fields
import plotter as my_plt
import fluxes as fx
import data
import time as timer
import timestep as ts
from copy import deepcopy

# Normalization parameters
om_pc, vt_c = 10.0, 0.1
charge_sign = -1.0  # electron
eigenvalue = 0.4096275588561705 + 0.4652331645969896j

# elements and order
elements, order = [8, 20, 20, 20], 10

# Geometry
grid_fundamental = 0.1  # / om_pc
length = 2.0 * np.pi / grid_fundamental
lows = np.array([-0.5 * length, -10, -10, -10])
highs = np.array([0.5 * length, 10, 10, 10])
grid = g.PhaseSpace(lows=lows, highs=highs, elements=elements, order=order, charge_sign=charge_sign, om_pc=om_pc)

# build distribution
distribution = var.Distribution(resolutions=elements, order=order)
distribution.initialize(grid=grid, vt=1, alpha=1, ring_gamma=6, wavenumber=grid_fundamental, eigenvalue=eigenvalue)
distribution.compute_zero_moment(grid=grid)
distribution.compute_moment_1(grid=grid)
# distribution.fourier_transform()
# distribution.inverse_fourier_transform()

# static and dynamic fields
static_fields = fields.Static(resolution=elements[0])
static_fields.gauss(distribution=distribution, grid=grid)
dynamic_fields = fields.Dynamic(resolution=elements[0], vt_c=vt_c, om_pc=om_pc)
dynamic_fields.initialize(grid=grid, eigenvalue=eigenvalue)
print(dynamic_fields.magnetic_x.arr_spectral)

plotter = my_plt.Plotter(grid=grid)
# plotter.spatial_scalar_plot(scalar=distribution.moment0, y_axis='Zero moment', spectrum=False)
# plotter.spatial_scalar_plot(scalar=distribution.moment_v, y_axis='v moment', spectrum=False)
# plotter.spatial_scalar_plot(scalar=distribution.moment_w, y_axis='w moment', spectrum=False)
# plotter.show()

# plotter3 = my_plt.Plotter3D(grid=grid)
# plotter3.distribution_contours3d(distribution=distribution, spectral_idx=0, ctype='real')

# Set up fluxes
phase_space_flux = fx.PhaseSpaceFlux(resolutions=elements, x_modes=grid.x.modes,  # pad_width=grid.x.pad_width,
                                     order=order, charge_sign=charge_sign,
                         om_pc=om_pc, nu=0, plotter=plotter)
phase_space_flux.initialize_zero_pad(grid=grid)
space_flux = fx.SpaceFlux(resolution=elements[0], c=1/vt_c)

# Set up time-stepper
print('Lorentz force dt estimate:{:0.3e}'.format(1.0/(np.sqrt(3)*highs[1]/om_pc)))
print('Spatial flux dt estimate:{:0.3e}'.format(1.0/(np.sqrt(3)*highs[1]*grid.x.wavenumbers[-1])))
dt = 1e-5  # 1.025e-02 * 1.0
step = 1e-5  # 1.025e-02 * 1.0
final_time = 1.0e-3
steps = int(np.abs(final_time // dt))

datafile = data.Data(folder='data\\', filename='test_may16')
datafile.create_file(distribution=distribution.arr_nodal.get(), density=distribution.moment0.arr_nodal.get(),
                     current_v=distribution.moment_v.arr_nodal.get(), current_w=distribution.moment_w.arr_nodal.get(),
                     electric_x=static_fields.electric_x.arr_nodal.get(),
                     electric_y=dynamic_fields.electric_y.arr_nodal.get(),
                     electric_z=dynamic_fields.electric_z.arr_nodal.get(),
                     magnetic_y=dynamic_fields.magnetic_y.arr_nodal.get(),
                     magnetic_z=dynamic_fields.magnetic_z.arr_nodal.get())

stepper = ts.Stepper(dt=dt, resolutions=elements, order=order, steps=steps,  grid=grid,
                     phase_space_flux=phase_space_flux, space_flux=space_flux)
stepper.main_loop(distribution=distribution, static_field=static_fields, dynamic_field=dynamic_fields,
                  grid=grid, data_file=data)

plotter.time_series_plot(time_in=stepper.time_array, series_in=stepper.ex_energy,
                         y_axis='Electric x energy', log=True, give_rate=False)
plotter.time_series_plot(time_in=stepper.time_array, series_in=stepper.ey_energy,
                         y_axis='Electric y energy', log=True, give_rate=False)
plotter.time_series_plot(time_in=stepper.time_array, series_in=stepper.ez_energy,
                         y_axis='Electric z energy', log=True, give_rate=False)
plotter.time_series_plot(time_in=stepper.time_array, series_in=stepper.by_energy,
                         y_axis='Magnetic y energy', log=True, give_rate=False)
plotter.time_series_plot(time_in=stepper.time_array, series_in=stepper.bz_energy,
                         y_axis='Magnetic z energy', log=True, give_rate=False)
plotter.time_series_plot(time_in=stepper.time_array, series_in=stepper.thermal_energy,
                         y_axis='Thermal energy', log=False)
plotter.time_series_plot(time_in=stepper.time_array, series_in=stepper.density_array,
                         y_axis='Total density', log=False)
plotter.time_series_plot(time_in=stepper.time_array, series_in=(stepper.ex_energy + stepper.ey_energy +
                                                                stepper.ez_energy + stepper.by_energy +
                                                                stepper.bz_energy + stepper.thermal_energy),
                         y_axis='Total energy', log=False)
plotter.show()
