import numpy as np
import time as timer
import variables as var
import fields
import cupy as cp

nonlinear_ssp_rk_switch = {
    2: [[1 / 2, 1 / 2, 1 / 2]],
    3: [[3 / 4, 1 / 4, 1 / 4],
        [1 / 3, 2 / 3, 2 / 3]]
}


class Stepper:
    def __init__(self, dt, resolutions, order, steps, grid, phase_space_flux, space_flux):
        self.x_res, self.u_res, self.v_res, self.w_res = resolutions
        self.resolutions = resolutions
        self.order = order
        self.dt = dt
        self.steps = steps
        self.phase_space_flux = phase_space_flux
        self.space_flux = space_flux

        # RK coefficients
        self.rk_coefficients = np.array(nonlinear_ssp_rk_switch.get(3, "nothing"))

        # tracking arrays
        self.time = 0
        self.next_time = 0
        self.ex_energy = np.array([])
        self.ey_energy = np.array([])
        self.ez_energy = np.array([])
        self.by_energy = np.array([])
        self.bz_energy = np.array([])
        self.time_array = np.array([])
        self.thermal_energy = np.array([])
        self.density_array = np.array([])

        # semi-implicit advection matrix
        self.implicit_x_advection_matrix = None
        self.build_advection_matrix(grid=grid)

        # save-times
        self.save_times = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 0])

    def build_advection_matrix(self, grid):
        """ Construct the global backward advection matrix """
        backward_advection_operator = (cp.eye(grid.u.order)[None, None, :, :] -
                                       0.5 * self.dt * -1j * grid.x.device_wavenumbers[:, None, None, None] *
                                       grid.u.translation_matrix[None, :, :, :])
        # forward_advection_operator = (cp.eye(grid.u.order)[None, None, :, :] +
        #                               0.5 * self.dt * -1j * grid.x.device_wavenumbers[:, None, None, None] *
        #                               grid.u.translation_matrix[None, :, :, :])
        # inv_backward_advection = cp.linalg.inv(backward_advection_operator)
        # self.implicit_x_advection_matrix = cp.matmul(inv_backward_advection, forward_advection_operator)
        self.implicit_x_advection_matrix = cp.linalg.inv(backward_advection_operator)

    def main_loop(self, distribution, static_field, dynamic_field, grid, data_file):
        print('Beginning main loop')
        # Compute first two steps with ssp-rk3 and save fluxes
        # zero stage
        # distribution.compute_zero_moment()
        # distribution.compute_moment_1(grid=grid)
        static_field.gauss(distribution=distribution, grid=grid)
        ps_flux0 = self.phase_space_flux.semi_discrete_rhs_semi_implicit(distribution=distribution,
                                                                         static_field=static_field,
                                                                         dynamic_field=dynamic_field, grid=grid)
        ey_flux0, ez_flux0 = self.space_flux.ampere(distribution=distribution, dynamic_field=dynamic_field, grid=grid)
        by_flux0, bz_flux0 = self.space_flux.faraday(dynamic_field=dynamic_field, grid=grid)

        # first step
        self.ssp_rk3(distribution=distribution, static_field=static_field, dynamic_field=dynamic_field, grid=grid)
        self.time += self.dt

        # first stage
        static_field.gauss(distribution=distribution, grid=grid)
        ps_flux1 = self.phase_space_flux.semi_discrete_rhs_semi_implicit(distribution=distribution,
                                                                         static_field=static_field,
                                                                         dynamic_field=dynamic_field, grid=grid)
        ey_flux1, ez_flux1 = self.space_flux.ampere(distribution=distribution, dynamic_field=dynamic_field, grid=grid)
        by_flux1, bz_flux1 = self.space_flux.faraday(dynamic_field=dynamic_field, grid=grid)

        # second step
        self.ssp_rk3(distribution=distribution, static_field=static_field, dynamic_field=dynamic_field, grid=grid)
        self.time += self.dt

        # store first two fluxes
        previous_phase_space_fluxes = [ps_flux1, ps_flux0]
        previous_dynamic_fluxes = [[ey_flux1, ey_flux0],
                                   [ez_flux1, ez_flux0],
                                   [by_flux1, by_flux0],
                                   [bz_flux1, bz_flux0]]

        # Main loop
        t0, save_counter = timer.time(), 0
        for i in range(self.steps):
            previous_phase_space_fluxes, previous_dynamic_fluxes = self.adams_bashforth(
                distribution=distribution, static_field=static_field, dynamic_field=dynamic_field, grid=grid,
                previous_phase_space_fluxes=previous_phase_space_fluxes, previous_dynamic_fluxes=previous_dynamic_fluxes
            )
            self.time += self.dt

            if i % 20 == 0:
                self.time_array = np.append(self.time_array, self.time)
                static_field.gauss(distribution=distribution, grid=grid)
                self.ex_energy = np.append(self.ex_energy, static_field.compute_field_energy(grid=grid).get())
                self.ey_energy = np.append(self.ey_energy, dynamic_field.compute_electric_y_energy(grid=grid).get())
                self.ez_energy = np.append(self.ez_energy, dynamic_field.compute_electric_z_energy(grid=grid).get())
                self.by_energy = np.append(self.by_energy, dynamic_field.compute_magnetic_y_energy(grid=grid).get())
                self.bz_energy = np.append(self.bz_energy, dynamic_field.compute_magnetic_z_energy(grid=grid).get())
                self.thermal_energy = np.append(self.thermal_energy, distribution.total_thermal_energy(grid=grid).get())
                self.density_array = np.append(self.density_array, distribution.total_density(grid=grid).get())
                print('\nTook 20 steps, time is {:0.3e}'.format(self.time))
                print('Time since start is ' + str((timer.time() - t0) / 60.0) + ' minutes')
                # print(self.time_array)

            # if np.abs(self.time - self.save_times[save_counter]) < 5.0e-3:
            #     distribution.inverse_fourier_transform()
            #     data_file.save_data(distribution=distribution.arr_nodal.get(),
            #                         density=distribution.moment0.arr_nodal.get(),
            #                         current_y=distribution.moment_v.arr_nodal.get(),
            #                         current_z=distribution.moment_w.arr_nodal.get(),
            #                         electric_x=static_field.electric_x.arr_nodal.get(),
            #                         electric_y=dynamic_field.electric_y.arr_nodal.get(),
            #                         electric_z=dynamic_field.electric_z.arr_nodal.get(),
            #                         magnetic_y=dynamic_field.magnetic_y.arr_nodal.get(),
            #                         magnetic_z=dynamic_field.magnetic_z.arr_nodal.get(),
            #                         time=self.save_times[save_counter])
            #     save_counter += 1

        print('\nAll done at time is {:0.3e}'.format(self.time))
        print('Total steps were ' + str(self.steps))
        print('Time since start is {:0.3e}'.format((timer.time() - t0)))

        # Saving final data
        distribution.inverse_fourier_transform()
        data_file.save_data(distribution=distribution.arr_nodal.get(),
                            density=distribution.moment0.arr_nodal.get(),
                            current_v=distribution.moment_v.arr_nodal.get(),
                            current_w=distribution.moment_w.arr_nodal.get(),
                            electric_x=static_field.electric_x.arr_nodal.get(),
                            electric_y=dynamic_field.electric_y.arr_nodal.get(),
                            electric_z=dynamic_field.electric_z.arr_nodal.get(),
                            magnetic_y=dynamic_field.magnetic_y.arr_nodal.get(),
                            magnetic_z=dynamic_field.magnetic_z.arr_nodal.get(),
                            time=self.time)

    def ssp_rk3(self, distribution, static_field, dynamic_field, grid):
        # Cut-off (avoid CFL advection instability as this is fully explicit)
        # cutoff = 50

        # Stage set-up
        phase_space_stage0 = var.Distribution(resolutions=self.resolutions,
                                              order=self.order)
        dynamic_field_stage0 = fields.Dynamic(resolution=self.resolutions[0], vt_c=dynamic_field.vt_c, om_pc=grid.om_pc)
        phase_space_stage1 = var.Distribution(resolutions=self.resolutions,
                                              order=self.order)
        dynamic_field_stage1 = fields.Dynamic(resolution=self.resolutions[0], vt_c=dynamic_field.vt_c, om_pc=grid.om_pc)

        # zero stage
        static_field.gauss(distribution=distribution, grid=grid)
        phase_space_rhs = self.phase_space_flux.semi_discrete_rhs_fully_explicit(distribution=distribution,
                                                                                 static_field=static_field,
                                                                                 dynamic_field=dynamic_field, grid=grid)
        electric_y_rhs, electric_z_rhs = self.space_flux.ampere(distribution=distribution,
                                                                dynamic_field=dynamic_field, grid=grid)
        magnetic_y_rhs, magnetic_z_rhs = self.space_flux.faraday(dynamic_field=dynamic_field, grid=grid)

        # phase_space_rhs[grid.x.device_modes > cutoff, :, :, :, :] = 0
        # electric_y_rhs[grid.x.device_modes > cutoff] = 0
        # electric_z_rhs[grid.x.device_modes > cutoff] = 0
        # magnetic_y_rhs[grid.x.device_modes > cutoff] = 0
        # magnetic_z_rhs[grid.x.device_modes > cutoff] = 0
        #
        phase_space_stage0.arr_spectral = (distribution.arr_spectral + self.dt * phase_space_rhs)
        dynamic_field_stage0.electric_y.arr_spectral = (dynamic_field.electric_y.arr_spectral +
                                                        self.dt * electric_y_rhs)
        dynamic_field_stage0.electric_z.arr_spectral = (dynamic_field.electric_z.arr_spectral +
                                                        self.dt * electric_z_rhs)
        dynamic_field_stage0.magnetic_x.arr_spectral = dynamic_field.magnetic_x.arr_spectral
        dynamic_field_stage0.magnetic_y.arr_spectral = (dynamic_field.magnetic_y.arr_spectral +
                                                        self.dt * magnetic_y_rhs)
        dynamic_field_stage0.magnetic_z.arr_spectral = (dynamic_field.magnetic_z.arr_spectral +
                                                        self.dt * magnetic_z_rhs)

        # first stage
        static_field.gauss(distribution=phase_space_stage0, grid=grid)
        phase_space_rhs = self.phase_space_flux.semi_discrete_rhs_fully_explicit(distribution=phase_space_stage0,
                                                                                 static_field=static_field,
                                                                                 dynamic_field=dynamic_field_stage0,
                                                                                 grid=grid)
        electric_y_rhs, electric_z_rhs = self.space_flux.ampere(distribution=phase_space_stage0,
                                                                dynamic_field=dynamic_field_stage0, grid=grid)
        magnetic_y_rhs, magnetic_z_rhs = self.space_flux.faraday(dynamic_field=dynamic_field_stage0, grid=grid)
        # print('Here')
        # #
        # phase_space_rhs[grid.x.device_modes > cutoff, :, :, :, :] = 0
        # electric_y_rhs[grid.x.device_modes > cutoff] = 0
        # magnetic_z_rhs[grid.x.device_modes > cutoff] = 0
        #
        phase_space_stage1.arr_spectral = (
                self.rk_coefficients[0, 0] * distribution.arr_spectral +
                self.rk_coefficients[0, 1] * phase_space_stage0.arr_spectral +
                self.rk_coefficients[0, 2] * self.dt * phase_space_rhs
        )
        dynamic_field_stage1.electric_y.arr_spectral = (
                self.rk_coefficients[0, 0] * dynamic_field.electric_y.arr_spectral +
                self.rk_coefficients[0, 1] * dynamic_field_stage0.electric_y.arr_spectral +
                self.rk_coefficients[0, 2] * self.dt * electric_y_rhs
        )
        dynamic_field_stage1.electric_z.arr_spectral = (
                self.rk_coefficients[0, 0] * dynamic_field.electric_z.arr_spectral +
                self.rk_coefficients[0, 1] * dynamic_field_stage0.electric_z.arr_spectral +
                self.rk_coefficients[0, 2] * self.dt * electric_z_rhs
        )
        dynamic_field_stage1.magnetic_x.arr_spectral = dynamic_field.magnetic_x.arr_spectral
        dynamic_field_stage1.magnetic_y.arr_spectral = (
                self.rk_coefficients[0, 0] * dynamic_field.magnetic_y.arr_spectral +
                self.rk_coefficients[0, 1] * dynamic_field_stage0.magnetic_y.arr_spectral +
                self.rk_coefficients[0, 2] * self.dt * magnetic_y_rhs
        )
        dynamic_field_stage1.magnetic_z.arr_spectral = (
                self.rk_coefficients[0, 0] * dynamic_field.magnetic_z.arr_spectral +
                self.rk_coefficients[0, 1] * dynamic_field_stage0.magnetic_z.arr_spectral +
                self.rk_coefficients[0, 2] * self.dt * magnetic_z_rhs
        )
        # print(dynamic_field.magnetic_z.arr_spectral.shape)
        # print(dynamic_field_stage1.magnetic_z.arr_spectral.shape)

        # second stage
        static_field.gauss(distribution=phase_space_stage1, grid=grid)
        #
        phase_space_rhs = self.phase_space_flux.semi_discrete_rhs_fully_explicit(distribution=phase_space_stage1,
                                                                                 static_field=static_field,
                                                                                 dynamic_field=dynamic_field_stage1,
                                                                                 grid=grid)
        electric_y_rhs, electric_z_rhs = self.space_flux.ampere(distribution=phase_space_stage1,
                                                                dynamic_field=dynamic_field_stage1, grid=grid)
        magnetic_y_rhs, magnetic_z_rhs = self.space_flux.faraday(dynamic_field=dynamic_field_stage1, grid=grid)
        # print('and Here')
        #
        # phase_space_rhs[grid.x.device_modes > cutoff, :, :, :, :] = 0
        # electric_y_rhs[grid.x.device_modes > cutoff] = 0
        # magnetic_z_rhs[grid.x.device_modes > cutoff] = 0
        distribution.arr_spectral = (
                self.rk_coefficients[1, 0] * distribution.arr_spectral +
                self.rk_coefficients[1, 1] * phase_space_stage1.arr_spectral +
                self.rk_coefficients[1, 2] * self.dt * phase_space_rhs
        )
        dynamic_field.electric_y.arr_spectral = (
                self.rk_coefficients[1, 0] * dynamic_field.electric_y.arr_spectral +
                self.rk_coefficients[1, 1] * dynamic_field_stage1.electric_y.arr_spectral +
                self.rk_coefficients[1, 2] * self.dt * electric_y_rhs
        )
        dynamic_field.electric_z.arr_spectral = (
                self.rk_coefficients[1, 0] * dynamic_field.electric_z.arr_spectral +
                self.rk_coefficients[1, 1] * dynamic_field_stage1.electric_z.arr_spectral +
                self.rk_coefficients[1, 2] * self.dt * electric_z_rhs
        )
        dynamic_field.magnetic_y.arr_spectral = (
                self.rk_coefficients[1, 0] * dynamic_field.magnetic_y.arr_spectral +
                self.rk_coefficients[1, 1] * dynamic_field_stage1.magnetic_y.arr_spectral +
                self.rk_coefficients[1, 2] * self.dt * magnetic_y_rhs
        )
        dynamic_field.magnetic_z.arr_spectral = (
                self.rk_coefficients[1, 0] * dynamic_field.magnetic_z.arr_spectral +
                self.rk_coefficients[1, 1] * dynamic_field_stage1.magnetic_z.arr_spectral +
                self.rk_coefficients[1, 2] * self.dt * magnetic_z_rhs
        )

    def adams_bashforth(self, distribution, static_field, dynamic_field, grid,
                        previous_phase_space_fluxes, previous_dynamic_fluxes):
        # static field determination
        static_field.gauss(distribution=distribution, grid=grid)
        # semi-discrete evaluations
        phase_space_rhs = self.phase_space_flux.semi_discrete_rhs_semi_implicit(distribution=distribution,
                                                                                static_field=static_field,
                                                                                dynamic_field=dynamic_field, grid=grid)
        electric_y_rhs, electric_z_rhs = self.space_flux.ampere(distribution=distribution,
                                                                dynamic_field=dynamic_field, grid=grid)
        magnetic_y_rhs, magnetic_z_rhs = self.space_flux.faraday(dynamic_field=dynamic_field, grid=grid)
        # semi-implicit phase space advance
        distribution.arr_spectral += self.dt * (
                (23 / 12 * phase_space_rhs -
                 4 / 3 * previous_phase_space_fluxes[0] +
                 5 / 12 * previous_phase_space_fluxes[1]) +
                0.5 * self.phase_space_flux.spectral_advection(distribution=distribution, grid=grid))
        distribution.arr_spectral = cp.einsum('nmjk,nmkrspq->nmjrspq',
                                              self.implicit_x_advection_matrix, distribution.arr_spectral)
        # distribution.arr_spectral = cp.einsum('nmjk,nmkrspq->nmjrspq',
        #                                       self.implicit_x_advection_matrix, distribution.arr_spectral)
        # dynamic field advance
        dynamic_field.electric_y.arr_spectral += self.dt * (
            (23 / 12 * electric_y_rhs -
             4 / 3 * previous_dynamic_fluxes[0][0] +
             5 / 12 * previous_dynamic_fluxes[0][1])
        )
        dynamic_field.electric_z.arr_spectral += self.dt * (
            (23 / 12 * electric_z_rhs -
             4 / 3 * previous_dynamic_fluxes[1][0] +
             5 / 12 * previous_dynamic_fluxes[1][1])
        )
        dynamic_field.magnetic_y.arr_spectral += self.dt * (
            (23 / 12 * magnetic_y_rhs -
             4 / 3 * previous_dynamic_fluxes[2][0] +
             5 / 12 * previous_dynamic_fluxes[2][1])
        )
        dynamic_field.magnetic_z.arr_spectral += self.dt * (
            (23 / 12 * magnetic_z_rhs -
             4 / 3 * previous_dynamic_fluxes[3][0] +
             5 / 12 * previous_dynamic_fluxes[3][1])
        )
        # save fluxes
        previous_phase_space_fluxes = [phase_space_rhs, previous_phase_space_fluxes[0]]
        previous_dynamic_fluxes = [[electric_y_rhs, previous_dynamic_fluxes[0][0]],
                                   [electric_z_rhs, previous_dynamic_fluxes[1][0]],
                                   [magnetic_y_rhs, previous_dynamic_fluxes[2][0]],
                                   [magnetic_z_rhs, previous_dynamic_fluxes[3][0]]]
        return previous_phase_space_fluxes, previous_dynamic_fluxes
