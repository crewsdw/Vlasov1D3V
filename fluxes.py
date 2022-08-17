import cupy as cp
# import variables as var
import plotter as my_plt

# For testing
import matplotlib.pyplot as plt


def basis_product(flux, basis_arr, axis, permutation):
    return cp.transpose(cp.tensordot(flux, basis_arr,
                                     axes=([axis], [1])),
                        axes=permutation)


class PhaseSpaceFlux:
    def __init__(self, resolutions, x_modes, order, charge_sign, om_pc, nu, plotter):
        # grid.x.modes + grid.x.pad_width
        # Size
        # self.pad_field = cp.zeros(x_modes + pad_width) + 0j
        resolutions[0] = x_modes
        # resolutions[0] = cp.fft.irfft(self.pad_field, norm='forward', axis=0).shape[0]
        self.resolutions = resolutions
        self.order = order
        self.charge_sign = charge_sign
        self.om_pc = om_pc
        # hyperviscosity
        self.nu = nu

        self.plotter = plotter

        self.permutations = [(0, 1, 6, 2, 3, 4, 5),
                             (0, 1, 2, 3, 6, 4, 5),
                             (0, 1, 2, 3, 4, 5, 6)]

        # dimension-dependent lists of slices into the phase space
        self.boundary_slices = [[(slice(self.resolutions[0]),
                                  slice(self.resolutions[1]), 0,
                                  slice(self.resolutions[2]), slice(self.order),
                                  slice(self.resolutions[3]), slice(self.order)),

                                 (slice(self.resolutions[0]),
                                  slice(self.resolutions[1]), -1,
                                  slice(self.resolutions[2]), slice(self.order),
                                  slice(self.resolutions[3]), slice(self.order))],

                                [(slice(self.resolutions[0]),
                                  slice(self.resolutions[1]), slice(self.order),
                                  slice(self.resolutions[2]), 0,
                                  slice(self.resolutions[3]), slice(self.order)),

                                 (slice(self.resolutions[0]),
                                  slice(self.resolutions[1]), slice(self.order),
                                  slice(self.resolutions[2]), -1,
                                  slice(self.resolutions[3]), slice(self.order))],

                                [(slice(self.resolutions[0]),
                                  slice(self.resolutions[1]), slice(self.order),
                                  slice(self.resolutions[2]), slice(self.order),
                                  slice(self.resolutions[3]), 0),
                                 (slice(self.resolutions[0]),
                                  slice(self.resolutions[1]), slice(self.order),
                                  slice(self.resolutions[2]), slice(self.order),
                                  slice(self.resolutions[3]), -1)]]

        self.boundary_slices_pad = [[(slice(self.resolutions[0]),
                                      slice(self.resolutions[1] + 2), 0,
                                      slice(self.resolutions[2]), slice(self.order),
                                      slice(self.resolutions[3]), slice(self.order)),

                                     (slice(self.resolutions[0]),
                                      slice(self.resolutions[1] + 2), -1,
                                      slice(self.resolutions[2]), slice(self.order),
                                      slice(self.resolutions[3]), slice(self.order))],

                                    [(slice(self.resolutions[0]),
                                      slice(self.resolutions[1]), slice(self.order),
                                      slice(self.resolutions[2] + 2), 0,
                                      slice(self.resolutions[3]), slice(self.order)),

                                     (slice(self.resolutions[0]),
                                      slice(self.resolutions[1]), slice(self.order),
                                      slice(self.resolutions[2] + 2), -1,
                                      slice(self.resolutions[3]), slice(self.order))],

                                    [(slice(self.resolutions[0]),
                                      slice(self.resolutions[1]), slice(self.order),
                                      slice(self.resolutions[2]), slice(self.order),
                                      slice(self.resolutions[3] + 2), 0),

                                     (slice(self.resolutions[0]),
                                      slice(self.resolutions[1]), slice(self.order),
                                      slice(self.resolutions[2]), slice(self.order),
                                      slice(self.resolutions[3] + 2), -1)]]

        self.flux_input_slices = [(slice(self.resolutions[0]),
                                   slice(1, self.resolutions[1] + 1), slice(self.order),
                                   slice(self.resolutions[2]), slice(self.order),
                                   slice(self.resolutions[3]), slice(self.order)),

                                  (slice(self.resolutions[0]),
                                   slice(self.resolutions[1]), slice(self.order),
                                   slice(1, self.resolutions[2] + 1), slice(self.order),
                                   slice(self.resolutions[3]), slice(self.order)),

                                  (slice(self.resolutions[0]),
                                   slice(self.resolutions[1]), slice(self.order),
                                   slice(self.resolutions[2]), slice(self.order),
                                   slice(1, self.resolutions[3] + 1), slice(self.order))]

        self.pad_slices = [(slice(self.resolutions[0]),
                            slice(1, self.resolutions[1] + 1),
                            slice(self.resolutions[2]), slice(self.order),
                            slice(self.resolutions[3]), slice(self.order)),

                           (slice(self.resolutions[0]),
                            slice(self.resolutions[1]), slice(self.order),
                            slice(1, self.resolutions[2] + 1),
                            slice(self.resolutions[3]), slice(self.order)),

                           (slice(self.resolutions[0]),
                            slice(self.resolutions[1]), slice(self.order),
                            slice(self.resolutions[2]), slice(self.order),
                            slice(1, self.resolutions[3] + 1))]

        self.field_slices = [(slice(self.resolutions[0]),
                              )]

        # Array sizes for allocation
        self.num_flux_sizes = [(self.resolutions[0],
                                self.resolutions[1], 2,
                                self.resolutions[2], self.order,
                                self.resolutions[3], self.order),

                               (self.resolutions[0],
                                self.resolutions[1], self.order,
                                self.resolutions[2], 2,
                                self.resolutions[3], self.order),

                               (self.resolutions[0],
                                self.resolutions[1], self.order,
                                self.resolutions[2], self.order,
                                self.resolutions[3], 2)]

        self.padded_flux_sizes = [(self.resolutions[0],
                                   self.resolutions[1] + 2, self.order,
                                   self.resolutions[2], self.order,
                                   self.resolutions[3], self.order),

                                  (self.resolutions[0],
                                   self.resolutions[1], self.order,
                                   self.resolutions[2] + 2, self.order,
                                   self.resolutions[3], self.order),

                                  (self.resolutions[0],
                                   self.resolutions[1], self.order,
                                   self.resolutions[2], self.order,
                                   self.resolutions[3] + 2, self.order)]

        self.sub_elements = [2, 4, 6]
        self.directions = [1, 3, 5]

        # arrays
        # self.flux_ex = var.Distribution(resolutions=resolutions, order=order)
        # self.flux_ey = var.Distribution(resolutions=resolutions, order=order)
        # self.flux_ez = var.Distribution(resolutions=resolutions, order=order)
        # self.flux_by = var.Distribution(resolutions=resolutions, order=order)
        # self.flux_bz = var.Distribution(resolutions=resolutions, order=order)
        # total output
        # self.output = var.Distribution(resolutions=resolutions, order=order)
        # Initialize zero-pads
        self.pad_field = None
        self.pad_electric_field = None
        self.pad_magnetic_fields = None
        self.pad_spectrum = None

    def initialize_zero_pad(self, grid):
        self.pad_field = cp.zeros(grid.x.modes + grid.x.pad_width) + 0j
        self.pad_electric_field = cp.zeros(grid.x.modes + grid.x.pad_width) + 0j
        self.pad_magnetic_fields = cp.zeros((2, grid.x.modes + grid.x.pad_width)) + 0j
        self.pad_spectrum = cp.zeros((grid.x.modes + grid.x.pad_width,
                                      grid.u.elements, grid.u.order,
                                      grid.v.elements, grid.v.order,
                                      grid.w.elements, grid.w.order)) + 0j

    def compute_spectral_flux(self, distribution, field, grid):
        """ Compute the flux convolution(field, distribution) using pseudospectral method """
        # self.initialize_zero_pad(grid=grid)
        self.pad_field[:-grid.x.pad_width] = field.arr_spectral
        self.pad_spectrum[:-grid.x.pad_width, :, :, :, :, :, :] = distribution.arr_spectral
        # Pseudospectral product
        field_nodal = cp.fft.irfft(self.pad_field, norm='forward', axis=0)
        distr_nodal = cp.fft.irfft(self.pad_spectrum, norm='forward', axis=0)
        nodal_flux = cp.multiply(field_nodal[:, None, None, None, None, None, None], distr_nodal)
        return cp.fft.rfft(nodal_flux, axis=0, norm='forward')[:-grid.x.pad_width, :, :, :, :, :, :]

    def compute_internal_and_numerical_flux_u(self, distribution, electric_field_x,
                                              magnetic_field_y, magnetic_field_z, grid):
        """ Compute the flux convolution(field, distribution) using pseudospectral method
            and compute the numerical flux by upwinding """
        # Pad the fields
        self.pad_electric_field[:-grid.x.pad_width] = electric_field_x.arr_spectral
        self.pad_magnetic_fields[0, :-grid.x.pad_width] = magnetic_field_y.arr_spectral
        self.pad_magnetic_fields[1, :-grid.x.pad_width] = magnetic_field_z.arr_spectral
        self.pad_spectrum[:-grid.x.pad_width, :, :, :, :, :, :] = distribution.arr_spectral

        # Pseudospectral products
        e_x = cp.fft.irfft(self.pad_electric_field, norm='forward', axis=0)
        b_y = cp.fft.irfft(self.pad_magnetic_fields[0, :], norm='forward', axis=0)
        b_z = cp.fft.irfft(self.pad_magnetic_fields[1, :], norm='forward', axis=0)
        distribution_nodal = cp.fft.irfft(self.pad_spectrum, norm='forward', axis=0)

        # Compute nodal fluxes (component-wise Lorentz force)
        field = self.charge_sign * (e_x[:, None, None, None, None] + (
                (grid.v.device_arr[None, :, :, None, None] *
                 b_z[:, None, None, None, None]) -
                (grid.w.device_arr[None, None, None, :, :] *
                 b_y[:, None, None, None, None])
        ))
        nodal_flux = cp.multiply(field[:, None, None, :, :, :, :],
                                 distribution_nodal)

        # Compute upwind flux based on nodal flux
        nodal_num_flux = self.nodal_upwind_flux_u(flux=nodal_flux, field=field, dim=0)
        # full_num_flux = cp.zeros_like(nodal_flux)
        # full_num_flux[:, :, :, :, 0, :, :] = nodal_num_flux[:, :, :, :, 0, :, :]
        # full_num_flux[:, :, :, :, -1, :, :] = nodal_num_flux[:, :, :, :, 1, :, :]
        #
        # print('At plotter')
        # plotter = my_plt.Plotter(grid=grid)
        # plotter.velocity_contour_plot(scalar=nodal_flux, x_idx=1, vel=0, vel_idx=100)
        # plotter.velocity_contour_plot(scalar=full_num_flux, x_idx=1, vel=0, vel_idx=100)
        # plt.show()
        # plotter.show()

        # Return internal flux and numerical fluxes in spectral variables
        return (cp.fft.rfft(nodal_flux, axis=0, norm='forward')[:-grid.x.pad_width, :, :, :, :, :, :],
                cp.fft.rfft(nodal_num_flux, axis=0, norm='forward')[:-grid.x.pad_width, :, :, :, :, :, :])

    def compute_internal_and_numerical_flux_v(self, distribution, electric_field_y,
                                              magnetic_field_x, magnetic_field_z, grid):
        """ Compute the flux convolution(field, distribution) using pseudospectral method
            and compute the numerical flux by upwinding """
        # Pad the fields
        self.pad_electric_field[:-grid.x.pad_width] = electric_field_y.arr_spectral
        self.pad_magnetic_fields[0, :-grid.x.pad_width] = magnetic_field_x.arr_spectral
        self.pad_magnetic_fields[1, :-grid.x.pad_width] = magnetic_field_z.arr_spectral
        self.pad_spectrum[:-grid.x.pad_width, :, :, :, :, :, :] = distribution.arr_spectral

        # Pseudospectral products
        e_y = cp.fft.irfft(self.pad_electric_field, norm='forward', axis=0)
        b_x = cp.fft.irfft(self.pad_magnetic_fields[0, :], norm='forward', axis=0)
        b_z = cp.fft.irfft(self.pad_magnetic_fields[1, :], norm='forward', axis=0)
        distribution_nodal = cp.fft.irfft(self.pad_spectrum, norm='forward', axis=0)

        # Compute nodal fluxes (component-wise Lorentz force)
        field = self.charge_sign * (e_y[:, None, None, None, None] +
                -1.0 * (grid.u.device_arr[None, :, :, None, None] *
                 b_z[:, None, None, None, None]) -
                (grid.w.device_arr[None, None, None, :, :] *
                 b_x[:, None, None, None, None])
        )
        # field = field * cp.ones_like(distribution_nodal)
        nodal_flux = cp.multiply(field[:, :, :, None, None, :, :],
                                 distribution_nodal)

        # Compute upwind flux based on nodal flux
        nodal_num_flux = self.nodal_upwind_flux_v(flux=nodal_flux, field=field, dim=1)
        # full_num_flux = cp.zeros_like(nodal_flux)
        # full_num_flux[:, :, :, :, 0, :, :] = nodal_num_flux[:, :, :, :, 0, :, :]
        # full_num_flux[:, :, :, :, -1, :, :] = nodal_num_flux[:, :, :, :, 1, :, :]
        #
        # print('At plotter')
        # x_idx = 3
        # print(e_y[x_idx])
        # print(b_z[x_idx])
        # plotter = my_plt.Plotter(grid=grid)
        # plotter.velocity_contour_plot(scalar=nodal_flux, x_idx=x_idx, vel=2, vel_idx=100)
        # plotter.velocity_contour_plot(scalar=full_num_flux, x_idx=x_idx, vel=2, vel_idx=100)
        # plt.show()
        # plotter.show()

        # Return internal flux and numerical fluxes in spectral variables
        return (cp.fft.rfft(nodal_flux, axis=0, norm='forward')[:-grid.x.pad_width, :, :, :, :, :, :],
                cp.fft.rfft(nodal_num_flux, axis=0, norm='forward')[:-grid.x.pad_width, :, :, :, :, :, :])

    def compute_internal_and_numerical_flux_w(self, distribution, electric_field_z,
                                              magnetic_field_x, magnetic_field_y, grid):
        """ Compute the flux convolution(field, distribution) using pseudospectral method
            and compute the numerical flux by upwinding """
        # Pad the fields
        self.pad_electric_field[:-grid.x.pad_width] = electric_field_z.arr_spectral
        self.pad_magnetic_fields[0, :-grid.x.pad_width] = magnetic_field_x.arr_spectral
        self.pad_magnetic_fields[1, :-grid.x.pad_width] = magnetic_field_y.arr_spectral
        self.pad_spectrum[:-grid.x.pad_width, :, :, :, :, :, :] = distribution.arr_spectral

        # Pseudospectral products
        e_z = cp.fft.irfft(self.pad_electric_field, norm='forward', axis=0)
        b_x = cp.fft.irfft(self.pad_magnetic_fields[0, :], norm='forward', axis=0)
        b_y = cp.fft.irfft(self.pad_magnetic_fields[1, :], norm='forward', axis=0)
        distribution_nodal = cp.fft.irfft(self.pad_spectrum, norm='forward', axis=0)

        # Compute nodal fluxes (component-wise Lorentz force)
        field = self.charge_sign * (e_z[:, None, None, None, None] +
                (grid.u.device_arr[None, :, :, None, None] *
                 b_y[:, None, None, None, None]) -
                (grid.v.device_arr[None, None, None, :, :] *
                 b_x[:, None, None, None, None])
        )
        nodal_flux = cp.multiply(field[:, :, :, :, :, None, None],
                                 distribution_nodal)

        # Compute upwind flux based on nodal flux
        nodal_num_flux = self.nodal_upwind_flux_w(flux=nodal_flux, field=field, dim=2)

        # Return internal flux and numerical fluxes in spectral variables
        return (cp.fft.rfft(nodal_flux, axis=0, norm='forward')[:-grid.x.pad_width, :, :, :, :, :, :],
                cp.fft.rfft(nodal_num_flux, axis=0, norm='forward')[:-grid.x.pad_width, :, :, :, :, :, :])

    def semi_discrete_rhs_semi_implicit(self, distribution, static_field, dynamic_field, grid):
        """ Computes the semi-discrete equation for the transport equation """
        # Compute the distribution RHS
        return (grid.u.J[None, :, None, None, None, None, None] * self.u_flux(distribution=distribution, grid=grid,
                                                                              static_field=static_field,
                                                                              dynamic_field=dynamic_field) +
                grid.v.J[None, None, None, :, None, None, None] * self.v_flux(distribution=distribution, grid=grid,
                                                                              static_field=static_field,
                                                                              dynamic_field=dynamic_field) +
                grid.w.J[None, None, None, None, None, :, None] * self.w_flux(distribution=distribution, grid=grid,
                                                                              static_field=static_field,
                                                                              dynamic_field=dynamic_field) -
                (self.nu * grid.x.device_wavenumbers_fourth[:, None, None, None, None, None, None] *
                 distribution.arr_spectral))

    def semi_discrete_rhs_fully_explicit(self, distribution, static_field, dynamic_field, grid):
        """ Computes the semi-discrete equation for the transport equation """
        # Compute the three fluxes with zero-padded FFTs
        # Compute the distribution RHS
        return (grid.u.J[None, :, None, None, None, None, None] * self.u_flux(distribution=distribution, grid=grid,
                                                                              static_field=static_field,
                                                                              dynamic_field=dynamic_field) +
                grid.v.J[None, None, None, :, None, None, None] * self.v_flux(distribution=distribution, grid=grid,
                                                                              static_field=static_field,
                                                                              dynamic_field=dynamic_field) +
                grid.w.J[None, None, None, None, None, :, None] * self.w_flux(distribution=distribution, grid=grid,
                                                                              static_field=static_field,
                                                                              dynamic_field=dynamic_field) +
                self.spectral_advection(distribution=distribution, grid=grid) -
                self.nu * grid.x.device_wavenumbers_fourth[:, None, None, None, None, None, None] *
                distribution.arr_spectral)

    def u_flux(self, distribution, grid, static_field, dynamic_field):
        """ Compute the DG-projection of the u-directed flux divergence """
        # Pre-condition internal flux by the integration of the velocity coordinate
        # flux = self.charge_sign * (self.flux_ex.arr_spectral + cp.einsum('rps,mijrs->mijrp',
        #                                                                      grid.v.translation_matrix,
        #                                                                      self.flux_bz.arr_spectral))
        # flux = self.compute_spectral_flux(distribution=distribution,
        #                                   field=static_field.electric_x, grid=grid)
        # flux += (grid.v.device_arr[None, None, None, :, :, None, None] *
        #          self.compute_spectral_flux(distribution=distribution,
        #                                     field=dynamic_field.magnetic_z, grid=grid))
        # flux += -1.0 * (grid.w.device_arr[None, None, None, None, None, :, :] *
        #                 self.compute_spectral_flux(distribution=distribution,
        #                                            field=dynamic_field.magnetic_y, grid=grid))
        flux, num_flux = self.compute_internal_and_numerical_flux_u(distribution=distribution,
                                                                    electric_field_x=static_field.electric_x,
                                                                    magnetic_field_y=dynamic_field.magnetic_y,
                                                                    magnetic_field_z=dynamic_field.magnetic_z,
                                                                    grid=grid)
        return (basis_product(flux=flux, basis_arr=grid.u.local_basis.internal,
                              axis=self.sub_elements[0], permutation=self.permutations[0]) -
                basis_product(flux=num_flux, basis_arr=grid.u.local_basis.numerical,
                              axis=self.sub_elements[0], permutation=self.permutations[0]))
        # return (basis_product(flux=self.charge_sign * flux, basis_arr=grid.u.local_basis.internal,
        #                       axis=2, permutation=self.permutations[0]) -
        #         self.numerical_flux(distribution=distribution, flux=self.charge_sign * flux, grid=grid, dim=0))

    def v_flux(self, distribution, grid, static_field, dynamic_field):
        # flux = self.charge_sign * (self.flux_ey.arr_spectral - cp.einsum('ijk,mikrs->mijrs',
        #                                                                  grid.u.translation_matrix,
        #                                                                  self.flux_bz.arr_spectral))
        # flux = self.compute_spectral_flux(distribution=distribution,
        #                                   field=dynamic_field.electric_y, grid=grid)
        # flux += -1.0 * (grid.u.device_arr[None, :, :, None, None, None, None] *
        #                 self.compute_spectral_flux(distribution=distribution,
        #                                            field=dynamic_field.magnetic_z, grid=grid))
        # for constant x-field
        # flux += (grid.w.device_arr[None, None, None, None, None, :, :] * distribution.arr_spectral / self.om_pc)
        flux, num_flux = self.compute_internal_and_numerical_flux_v(distribution=distribution,
                                                                    electric_field_y=dynamic_field.electric_y,
                                                                    magnetic_field_x=dynamic_field.magnetic_x,
                                                                    magnetic_field_z=dynamic_field.magnetic_z,
                                                                    grid=grid)

        return (basis_product(flux=flux, basis_arr=grid.u.local_basis.internal,
                              axis=self.sub_elements[1], permutation=self.permutations[1]) -
                basis_product(flux=num_flux, basis_arr=grid.u.local_basis.numerical,
                              axis=self.sub_elements[1], permutation=self.permutations[1]))
        # return (basis_product(flux=self.charge_sign * flux, basis_arr=grid.v.local_basis.internal,
        #                       axis=4, permutation=self.permutations[1]) -
        #         self.numerical_flux(distribution=distribution, flux=self.charge_sign * flux, grid=grid, dim=1))

    def w_flux(self, distribution, grid, static_field, dynamic_field):
        # flux = self.charge_sign * (self.flux_ey.arr_spectral - cp.einsum('ijk,mikrs->mijrs',
        #                                                                  grid.u.translation_matrix,
        #                                                                  self.flux_bz.arr_spectral))
        # flux = self.compute_spectral_flux(distribution=distribution,
        #                                   field=dynamic_field.electric_z, grid=grid)
        # # flux of transverse b-field
        # flux += (grid.u.device_arr[None, :, :, None, None, None, None] *
        #          self.compute_spectral_flux(distribution=distribution,
        #                                     field=dynamic_field.magnetic_y, grid=grid))
        # # for constant x-field
        # flux += -1.0 * (grid.v.device_arr[None, None, None, :, :, None, None] *
        # # distribution.arr_spectral / self.om_pc)
        flux, num_flux = self.compute_internal_and_numerical_flux_w(distribution=distribution,
                                                                    electric_field_z=dynamic_field.electric_z,
                                                                    magnetic_field_x=dynamic_field.magnetic_x,
                                                                    magnetic_field_y=dynamic_field.magnetic_y,
                                                                    grid=grid)
        return (basis_product(flux=flux, basis_arr=grid.u.local_basis.internal,
                              axis=self.sub_elements[2], permutation=self.permutations[2]) -
                basis_product(flux=num_flux, basis_arr=grid.u.local_basis.numerical,
                              axis=self.sub_elements[2], permutation=self.permutations[2]))
        # return (basis_product(flux=flux, basis_arr=grid.w.local_basis.internal,
        #                       axis=6, permutation=self.permutations[2]) -
        #         self.numerical_flux(distribution=distribution, flux=self.charge_sign * flux, grid=grid, dim=2))

    def numerical_flux(self, distribution, flux, grid, dim):
        # Allocate
        num_flux = cp.zeros(self.num_flux_sizes[dim]) + 0j

        # Set padded flux
        padded_flux = cp.zeros(self.padded_flux_sizes[dim]) + 0j
        padded_flux[self.flux_input_slices[dim]] = flux

        # Lax-Friedrichs flux
        num_flux[self.boundary_slices[dim][0]] = -0.5 * (cp.roll(padded_flux[self.boundary_slices_pad[dim][1]],
                                                                 shift=+1, axis=self.directions[dim])[
                                                             self.pad_slices[dim]] +
                                                         flux[self.boundary_slices[dim][0]])
        num_flux[self.boundary_slices[dim][1]] = +0.5 * (cp.roll(padded_flux[self.boundary_slices_pad[dim][0]],
                                                                 shift=-1, axis=self.directions[dim])[
                                                             self.pad_slices[dim]] +
                                                         flux[self.boundary_slices[dim][1]])

        # # re-use padded_flux array for padded_distribution
        padded_flux[self.flux_input_slices[dim]] = distribution.arr_spectral
        constant = cp.amax(cp.absolute(flux), axis=self.sub_elements[dim])

        num_flux[self.boundary_slices[dim][0]] += -0.5 * (
            cp.multiply(constant,
                        (cp.roll(padded_flux[self.boundary_slices_pad[dim][1]],
                                 shift=+1, axis=self.directions[dim])[self.pad_slices[dim]] -
                         distribution.arr_spectral[self.boundary_slices[dim][0]]))
        )
        num_flux[self.boundary_slices[dim][1]] += -0.5 * (
            cp.multiply(constant,
                        (cp.roll(padded_flux[self.boundary_slices_pad[dim][0]],
                                 shift=-1, axis=self.directions[dim])[self.pad_slices[dim]] -
                         distribution.arr_spectral[self.boundary_slices[dim][1]]))
        )
        return basis_product(flux=num_flux, basis_arr=grid.u.local_basis.numerical,
                             axis=self.sub_elements[dim], permutation=self.permutations[dim])

    def nodal_upwind_flux_u(self, flux, field, dim):
        # print(flux.shape[0])
        self.num_flux_sizes[dim] = (flux.shape[0],) + self.num_flux_sizes[dim][1:]
        # Allocate
        num_flux = cp.zeros(self.num_flux_sizes[dim])  # + 0j

        # Alternative:
        one_negatives = cp.where(condition=field < 0, x=1, y=0)
        one_positives = cp.where(condition=field >= 0, x=1, y=0)
        # print(one_negatives.shape)

        # set padded flux
        self.padded_flux_sizes[dim] = (flux.shape[0],) + self.padded_flux_sizes[dim][1:]
        # print(self.padded_flux_sizes[dim])
        self.flux_input_slices[dim] = (slice(flux.shape[0]),) + self.flux_input_slices[dim][1:]

        padded_flux = cp.zeros(self.padded_flux_sizes[dim])  # + 0j
        padded_flux[self.flux_input_slices[dim]] = flux
        # padded_flux[:, 0, -1] = 0.0  # -self.flux.arr[:, 0, 0]
        # padded_flux[:, -1, 0] = 0.0  # -self.flux.arr[:, -1, 0]

        self.boundary_slices[dim][0] = (slice(num_flux.shape[0]),) + self.boundary_slices[dim][0][1:]
        self.boundary_slices[dim][1] = (slice(num_flux.shape[0]),) + self.boundary_slices[dim][1][1:]
        self.boundary_slices_pad[dim][0] = (slice(num_flux.shape[0]),) + self.boundary_slices_pad[dim][0][1:]
        self.boundary_slices_pad[dim][1] = (slice(num_flux.shape[0]),) + self.boundary_slices_pad[dim][1][1:]
        self.pad_slices[dim] = (slice(num_flux.shape[0]),) + self.pad_slices[dim][1:]

        # Upwind flux, left face
        num_flux[self.boundary_slices[dim][0]] = -1.0 * (
                cp.multiply(cp.roll(padded_flux[self.boundary_slices_pad[dim][1]],
                                    shift=+1,
                                    axis=self.directions[dim])[
                                self.pad_slices[dim]],
                            one_positives[:, None, :, :, :, :]) +
                cp.multiply(padded_flux[self.boundary_slices_pad[dim][0]][
                                self.pad_slices[dim]],
                            one_negatives[:, None, :, :, :, :]))
        # Upwind fluxes, right face
        num_flux[self.boundary_slices[dim][1]] = (
                cp.multiply(padded_flux[self.boundary_slices_pad[dim][1]][
                                self.pad_slices[dim]],
                            one_positives[:, None, :, :, :, :]) +
                cp.multiply(cp.roll(padded_flux[self.boundary_slices_pad[dim][0]],
                                    shift=-1,
                                    axis=self.directions[dim])[
                                self.pad_slices[dim]],
                            one_negatives[:, None, :, :, :, :])
        )

        return num_flux

    def nodal_upwind_flux_v(self, flux, field, dim):
        self.num_flux_sizes[dim] = (flux.shape[0],) + self.num_flux_sizes[dim][1:]
        # Allocate
        num_flux = cp.zeros(self.num_flux_sizes[dim])

        # Set sizes (lazy... fix)
        self.padded_flux_sizes[dim] = (flux.shape[0],) + self.padded_flux_sizes[dim][1:]
        self.flux_input_slices[dim] = (slice(flux.shape[0]),) + self.flux_input_slices[dim][1:]
        self.boundary_slices[dim][0] = (slice(num_flux.shape[0]),) + self.boundary_slices[dim][0][1:]
        self.boundary_slices[dim][1] = (slice(num_flux.shape[0]),) + self.boundary_slices[dim][1][1:]
        self.boundary_slices_pad[dim][0] = (slice(num_flux.shape[0]),) + self.boundary_slices_pad[dim][0][1:]
        self.boundary_slices_pad[dim][1] = (slice(num_flux.shape[0]),) + self.boundary_slices_pad[dim][1][1:]
        self.pad_slices[dim] = (slice(num_flux.shape[0]),) + self.pad_slices[dim][1:]

        # Alternative:
        one_negatives = cp.where(condition=field < 0, x=1, y=0)
        one_positives = cp.where(condition=field >= 0, x=1, y=0)

        # set padded flux
        padded_flux = cp.zeros(self.padded_flux_sizes[dim])  # + 0j
        # print(self.flux_input_slices[dim])
        # print(self.boundary_slices_pad[dim][1])
        # print(one_negatives.shape)
        # print(self.pad_slices[dim])
        # print(self.directions[dim])
        # print(padded_flux[self.boundary_slices_pad[dim][0]].shape)
        # quit()
        padded_flux[self.flux_input_slices[dim]] = flux
        # padded_flux[:, 0, -1] = 0.0  # -self.flux.arr[:, 0, 0]
        # padded_flux[:, -1, 0] = 0.0  # -self.flux.arr[:, -1, 0]

        # Upwind flux, left face
        num_flux[self.boundary_slices[dim][0]] = -1.0 * (
                cp.multiply(cp.roll(padded_flux[self.boundary_slices_pad[dim][1]],
                                    shift=+1,
                                    axis=self.directions[dim])[
                                self.pad_slices[dim]],
                            one_positives[:, :, :, None, :, :]) +
                cp.multiply(padded_flux[self.boundary_slices_pad[dim][0]][
                                self.pad_slices[dim]],
                            one_negatives[:, :, :, None, :, :])
        )
        # Upwind fluxes, right face
        num_flux[self.boundary_slices[dim][1]] = (
                cp.multiply(padded_flux[self.boundary_slices_pad[dim][1]][
                                self.pad_slices[dim]],
                            one_positives[:, :, :, None, :, :]) +
                cp.multiply(cp.roll(padded_flux[self.boundary_slices_pad[dim][0]],
                                    shift=-1,
                                    axis=self.directions[dim])[
                                self.pad_slices[dim]],
                            one_negatives[:, :, :, None, :, :])
        )

        return num_flux

    def nodal_upwind_flux_w(self, flux, field, dim):
        self.num_flux_sizes[dim] = (flux.shape[0],) + self.num_flux_sizes[dim][1:]
        # Allocate
        num_flux = cp.zeros(self.num_flux_sizes[dim])

        # Alternative:
        one_negatives = cp.where(condition=field < 0, x=1, y=0)
        one_positives = cp.where(condition=field >= 0, x=1, y=0)

        # Set sizes (lazy... fix)
        self.padded_flux_sizes[dim] = (flux.shape[0],) + self.padded_flux_sizes[dim][1:]
        self.flux_input_slices[dim] = (slice(flux.shape[0]),) + self.flux_input_slices[dim][1:]
        self.boundary_slices[dim][0] = (slice(num_flux.shape[0]),) + self.boundary_slices[dim][0][1:]
        self.boundary_slices[dim][1] = (slice(num_flux.shape[0]),) + self.boundary_slices[dim][1][1:]
        self.boundary_slices_pad[dim][0] = (slice(num_flux.shape[0]),) + self.boundary_slices_pad[dim][0][1:]
        self.boundary_slices_pad[dim][1] = (slice(num_flux.shape[0]),) + self.boundary_slices_pad[dim][1][1:]
        self.pad_slices[dim] = (slice(num_flux.shape[0]),) + self.pad_slices[dim][1:]

        # set padded flux
        padded_flux = cp.zeros(self.padded_flux_sizes[dim])  # + 0j
        padded_flux[self.flux_input_slices[dim]] = flux
        # padded_flux[:, 0, -1] = 0.0  # -self.flux.arr[:, 0, 0]
        # padded_flux[:, -1, 0] = 0.0  # -self.flux.arr[:, -1, 0]

        # Upwind flux, left face
        num_flux[self.boundary_slices[dim][0]] = -1.0 * (
                cp.multiply(cp.roll(padded_flux[self.boundary_slices_pad[dim][1]],
                                    shift=+1,
                                    axis=self.directions[dim])[
                                self.pad_slices[dim]],
                            one_positives[:, :, :, :, :, None]) +
                cp.multiply(padded_flux[self.boundary_slices_pad[dim][0]][
                                self.pad_slices[dim]],
                            one_negatives[:, :, :, :, :, None])
        )
        # Upwind fluxes, right face
        num_flux[self.boundary_slices[dim][1]] = (
                cp.multiply(padded_flux[self.boundary_slices_pad[dim][1]][
                                self.pad_slices[dim]],
                            one_positives[:, :, :, :, :, None]) +
                cp.multiply(cp.roll(padded_flux[self.boundary_slices_pad[dim][0]],
                                    shift=-1,
                                    axis=self.directions[dim])[
                                self.pad_slices[dim]],
                            one_negatives[:, :, :, :, :, None])
        )

        return num_flux

    def spectral_advection(self, distribution, grid):
        return -1j * cp.multiply(grid.x.device_wavenumbers[:, None, None, None, None, None, None],
                                 cp.einsum('ijk,mikrspq->mijrspq', grid.u.translation_matrix,
                                           distribution.arr_spectral))


class SpaceFlux:
    def __init__(self, resolution, c):
        self.resolution = resolution
        self.c = c

    def faraday(self, dynamic_field, grid):
        return cp.array([1j * grid.x.device_wavenumbers * dynamic_field.electric_z.arr_spectral,
                         -1j * grid.x.device_wavenumbers * dynamic_field.electric_y.arr_spectral])

    def ampere(self, distribution, dynamic_field, grid):
        distribution.compute_moment_1(grid=grid)
        return (
                cp.array([((self.c ** 2.0) * (-1j * grid.x.device_wavenumbers * dynamic_field.magnetic_z.arr_spectral) -
                           grid.charge_sign * distribution.moment_v.arr_spectral),
                          ((self.c ** 2.0) * (1j * grid.x.device_wavenumbers * dynamic_field.magnetic_y.arr_spectral) -
                           grid.charge_sign * distribution.moment_w.arr_spectral)])
        )
