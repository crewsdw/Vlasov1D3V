import cupy as cp
import variables as var


class Static:
    """ Class for static fields governed by Gauss's law, here E_x """
    def __init__(self, resolution):
        self.electric_x = var.SpaceScalar(resolution=resolution)

    def gauss(self, distribution, grid, invert=True):
        # Compute zeroth moment, integrate(c_n(v)dv)
        distribution.compute_zero_moment(grid=grid)

        # Adjust for charge neutrality
        distribution.moment0.arr_spectral[grid.x.zero_idx] -= 1.0

        # Compute field spectrum
        self.electric_x.arr_spectral = (-1j * grid.charge_sign *
                                        cp.nan_to_num(cp.divide(distribution.moment0.arr_spectral,
                                                                grid.x.device_wavenumbers)))

        if invert:
            self.electric_x.inverse_fourier_transform()

    def compute_field_energy(self, grid):
        self.electric_x.inverse_fourier_transform()
        return self.electric_x.integrate_energy(grid=grid)


class Dynamic:
    """ Class for dynamic fields described by Ampere/Faraday laws, here E_y and B_z """
    def __init__(self, resolution, vt_c, om_pc):
        self.electric_y = var.SpaceScalar(resolution=resolution)
        self.electric_z = var.SpaceScalar(resolution=resolution)

        self.magnetic_x = var.SpaceScalar(resolution=resolution)
        self.magnetic_y = var.SpaceScalar(resolution=resolution)
        self.magnetic_z = var.SpaceScalar(resolution=resolution)
        self.vt_c = vt_c
        self.om_pc = om_pc

        self.eig_y, self.eig_z = None, None

    def initialize(self, grid, eigenvalue, wavenumber):
        # Set eigenmode
        self.eigenmode(grid=grid, amplitude=1.0e-2, wavenumber=wavenumber, eigenvalue=eigenvalue)  # ? 1.54j, 0.736j

        # # For now: no initial fields
        # self.electric_y.arr_nodal, self.electric_z.arr_nodal = 0*self.electric_y.arr_nodal, 0*self.electric_z.arr_nodal
        # self.magnetic_y.arr_nodal, self.magnetic_z.arr_nodal = 0*self.magnetic_y.arr_nodal, 0*self.magnetic_z.arr_nodal

        # Initialize constant magnetic field
        self.magnetic_x.arr_nodal = cp.ones_like(self.magnetic_z.arr_nodal) / self.om_pc

        # Fourier-transform
        self.magnetic_x.fourier_transform()
        self.magnetic_y.fourier_transform()
        self.magnetic_z.fourier_transform()

        self.electric_y.fourier_transform()
        self.electric_z.fourier_transform()

    def eigenmode(self, grid, amplitude, wavenumber, eigenvalue):
        # Nodal values (need to think about this some more)
        # self.magnetic_x = self.om_pc  # cp.real()
        sq2 = cp.sqrt(2)
        self.eig_y = 1.0j / sq2 * amplitude
        self.eig_z = -1.0 / sq2 * amplitude
        self.electric_y.arr_nodal = cp.real(self.eig_y * cp.exp(1j * wavenumber * grid.x.device_arr))
        self.electric_z.arr_nodal = cp.real(self.eig_z * cp.exp(1j * wavenumber * grid.x.device_arr))
        
        self.magnetic_y.arr_nodal = cp.real(-self.eig_z / eigenvalue * cp.exp(1j * wavenumber * grid.x.device_arr))
        self.magnetic_z.arr_nodal = cp.real(self.eig_y / eigenvalue * cp.exp(1j * wavenumber * grid.x.device_arr))
        
        # wtf was I thinking?
        # self.magnetic_y.arr_nodal = cp.real(-1j * amplitude * cp.exp(1j * wavenumber * grid.x.device_arr) / eigenvalue) / sq2
        # self.magnetic_z.arr_nodal = cp.real(amplitude * cp.exp(1j * wavenumber * grid.x.device_arr) / eigenvalue) / sq2

    def compute_magnetic_y_energy(self, grid):
        self.magnetic_y.inverse_fourier_transform(), self.magnetic_z.inverse_fourier_transform()
        return ((self.magnetic_y.integrate_energy(grid=grid)) /
                (self.vt_c ** 2.0))

    def compute_magnetic_z_energy(self, grid):
        self.magnetic_z.inverse_fourier_transform()
        return ((self.magnetic_z.integrate_energy(grid=grid)) /
                (self.vt_c ** 2.0))

    def compute_electric_y_energy(self, grid):
        self.electric_y.inverse_fourier_transform()
        return self.electric_y.integrate_energy(grid=grid)

    def compute_electric_z_energy(self, grid):
        self.electric_z.inverse_fourier_transform()
        return self.electric_z.integrate_energy(grid=grid)
