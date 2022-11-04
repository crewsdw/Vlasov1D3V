import numpy as np
# import cupy as cp
import grid as g
import variables as var
import fields
import plotter as my_plt
import fluxes as fx
import data
# import time as timer
import timestep as ts

# from copy import deepcopy

# Normalization parameters
# om_pc, vt_c = 10.0, 0.1  # 0.1
om_pc, vt_c = 10.0, 0.3  # 0.1
charge_sign = -1.0  # electron
# eigenvalue = 0.12721160224570427 + 0.7777815241590355j
# eigenvalue = -0.5813939194797665 + 0.47487085962291437j
# eigenvalue = 0.4096275588561705 + 0.4652331645969896j
# eigenvalue = 1.5867126026383591 + 0.47966536397472853j
# eigenvalue = 0.13525063576435203+2.269550074409259j
eigenvalue = 0.14473193221558822 + 1.473344510925545j

# elements and order
elements, order = [10, 12, 28, 28], 10

# Geometry
grid_fundamental = 0.41  # 0.1  # 0.1  # / om_pc
length = 2.0 * np.pi / grid_fundamental
lows = np.array([-0.5 * length, -8, -20, -20])
highs = np.array([0.5 * length, 8, 20, 20])
grid = g.PhaseSpace(lows=lows, highs=highs, elements=elements, order=order, charge_sign=charge_sign, om_pc=om_pc)

om = eigenvalue * grid_fundamental
# print(om)
# j_e_factor = 1j * om * (1 - 1 / (eigenvalue**2) / (vt_c**2))
# print(j_e_factor)

# static and dynamic fields
dynamic_fields = fields.Dynamic(resolution=elements[0], vt_c=vt_c, om_pc=om_pc)
dynamic_fields.initialize(grid=grid, eigenvalue=eigenvalue, wavenumber=grid_fundamental)
# print(dynamic_fields.magnetic_x.arr_spectral)
# jy_target = j_e_factor * dynamic_fields.electric_y.arr_spectral[1]
# jz_target = j_e_factor * dynamic_fields.electric_z.arr_spectral[1]
# print('\nTarget currents')
# print(jy_target)
# print(jz_target)

# build distribution
distribution = var.Distribution(resolutions=elements, order=order)
distribution.initialize(grid=grid, vt=0.5, alpha=2, ring_gamma=6, wavenumber=grid_fundamental, eigenvalue=eigenvalue,
                        dynamic_fields=dynamic_fields)  # maybe ?
distribution.compute_zero_moment(grid=grid)
distribution.compute_moment_1(grid=grid)
# print('\nActual currents')
# print(-1.0 * distribution.moment_v.arr_spectral[1])
# print(-1.0 * distribution.moment_w.arr_spectral[1])
# quit()

static_fields = fields.Static(resolution=elements[0])
static_fields.gauss(distribution=distribution, grid=grid)
# distribution.fourier_transform()
# distribution.inverse_fourier_transform()

plotter = my_plt.Plotter(grid=grid)
# plotter.spatial_scalar_plot(scalar=distribution.moment0, y_axis='Zero moment', spectrum=True)
plotter.spatial_scalar_plot(scalar=distribution.moment_v, y_axis='v moment', spectrum=True)
# plotter.spatial_scalar_plot(scalar=distribution.moment_w, y_axis='w moment', spectrum=False)
# plotter.spatial_scalar_plot(scalar=dynamic_fields.electric_y, y_axis='E_y', spectrum=False)
# plotter.spatial_scalar_plot(scalar=dynamic_fields.electric_z, y_axis='E_z', spectrum=False)
# plotter.spatial_scalar_plot(scalar=dynamic_fields.magnetic_y, y_axis='B_y', spectrum=False)
plotter.spatial_scalar_plot(scalar=dynamic_fields.magnetic_z, y_axis='B_z', spectrum=True)
plotter.show()
# plotter3 = my_plt.Plotter3D(grid=grid)
# ctype = 'absolute'
# plotter3.distribution_contours3d(distribution=distribution, spectral_idx=0, ctype=ctype)
# plotter3.distribution_contours3d(distribution=distribution, spectral_idx=1, ctype=ctype)
# plotter3.distribution_contours3d(distribution=distribution, spectral_idx=2, ctype=ctype)
# plotter3 = my_plt.Plotter3D(grid=grid)
# plotter3.distribution_contours3d(distribution=distribution, spectral_idx=1, ctype='real')

# Set up fluxes
phase_space_flux = fx.PhaseSpaceFlux(resolutions=elements, x_modes=grid.x.modes,  # pad_width=grid.x.pqad_width,
                                     order=order, charge_sign=charge_sign,
                                     om_pc=om_pc, nu=0, plotter=plotter)
phase_space_flux.initialize_zero_pad(grid=grid)
space_flux = fx.SpaceFlux(resolution=elements[0], c=1 / vt_c)

# Set up time-stepper
print('Lorentz force dt estimate:{:0.3e}'.format(1.0 / (np.sqrt(3) * highs[2] / om_pc)))
print('Spatial flux dt estimate:{:0.3e}'.format(1.0 / (np.sqrt(3) * np.sqrt(2) * highs[1] * grid.x.wavenumbers[-1])))
dt = 2.5e-4  # 1.025e-02 * 1.0
step = 2.5e-4  # 1.025e-02 * 1.0
final_time = 0.10  # 7.5  # 6

steps = int(np.abs(final_time // dt))

datafile = data.Data(folder='data/', filename='run_to_t' + str(final_time))
datafile.create_file(distribution=distribution.arr_nodal.get(), density=distribution.moment0.arr_nodal.get(),
                     current_v=distribution.moment_v.arr_nodal.get(), current_w=distribution.moment_w.arr_nodal.get(),
                     electric_x=static_fields.electric_x.arr_nodal.get(),
                     electric_y=dynamic_fields.electric_y.arr_nodal.get(),
                     electric_z=dynamic_fields.electric_z.arr_nodal.get(),
                     magnetic_y=dynamic_fields.magnetic_y.arr_nodal.get(),
                     magnetic_z=dynamic_fields.magnetic_z.arr_nodal.get())

stepper = ts.Stepper(dt=dt, resolutions=elements, order=order, steps=steps, grid=grid,
                     phase_space_flux=phase_space_flux, space_flux=space_flux)
stepper.main_loop(distribution=distribution, static_field=static_fields, dynamic_field=dynamic_fields,
                  grid=grid, data_file=datafile)

# Save energies
time_series = data.TimeSeries(folder='data/', filename='energies_run_to_t' + str(final_time))
time_series.create_file(time=stepper.time_array, e_ex=stepper.ex_energy,
                        e_ey=stepper.ey_energy, e_ez=stepper.ez_energy,
                        e_by=stepper.by_energy, e_bz=stepper.bz_energy,
                        e_th=stepper.thermal_energy,
                        e_tot=(stepper.ex_energy + stepper.ey_energy +
                               stepper.ez_energy + stepper.by_energy +
                               stepper.bz_energy + stepper.thermal_energy),
                        n_tot=stepper.density_array)

# plotter = my_plt.Plotter(grid=grid)
# plotter.spatial_scalar_plot(scalar=distribution.moment0, y_axis='Zero moment', spectrum=True)
# plotter.spatial_scalar_plot(scalar=distribution.moment_v, y_axis='v moment', spectrum=True)
# plotter.spatial_scalar_plot(scalar=distribution.moment_w, y_axis='w moment', spectrum=True)
# plotter.show()

# plotter3 = my_plt.Plotter3D(grid=grid)
# plotter3.distribution_contours3d(distribution=distribution, spectral_idx=0, ctype=ctype)
# plotter3.distribution_contours3d(distribution=distribution, spectral_idx=1, ctype='real')
# plotter3.distribution_contours3d(distribution=distribution, spectral_idx=2, ctype=ctype)

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
plotter.spatial_scalar_plot(scalar=distribution.moment_v, y_axis=r'velocity $v_y$', spectrum=True)
plotter.spatial_scalar_plot(scalar=dynamic_fields.electric_y, y_axis=r'field $E_y$', spectrum=True)
plotter.time_series_plot(time_in=stepper.time_array, series_in=(stepper.ex_energy + stepper.ey_energy +
                                                                stepper.ez_energy + stepper.by_energy +
                                                                stepper.bz_energy + stepper.thermal_energy),
                         y_axis='Total energy', log=False)
plotter.show()
