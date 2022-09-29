import cupy as cp


class SpaceScalar:
    """ examples: fields, density, etc. """

    def __init__(self, resolution):
        self.res = resolution
        self.arr_nodal, self.arr_spectral = None, None

    def fourier_transform(self):
        self.arr_spectral = cp.fft.rfft(self.arr_nodal, norm='forward')

    def inverse_fourier_transform(self):
        self.arr_nodal = cp.fft.irfft(self.arr_spectral, norm='forward')

    def integrate(self, grid):
        arr_add = cp.append(self.arr_nodal, self.arr_nodal[0])
        return trapz(arr_add, grid.x.dx)

    def integrate_energy(self, grid):
        arr = 0.5 * self.arr_nodal ** 2.0
        arr_add = cp.append(arr, arr[0])
        return trapz(arr_add, grid.x.dx)


class Distribution:
    def __init__(self, resolutions, order):
        self.resolutions, self.order = resolutions, order

        self.arr_nodal, self.arr_spectral = None, None
        self.moment0, self.moment2 = SpaceScalar(resolution=resolutions[0]), SpaceScalar(resolution=resolutions[0])
        self.moment_v = SpaceScalar(resolution=resolutions[0])
        self.moment_w = SpaceScalar(resolution=resolutions[0])

    def fourier_transform(self):
        self.arr_spectral = cp.fft.rfft(self.arr_nodal, axis=0, norm='forward')

    def inverse_fourier_transform(self):
        self.arr_nodal = cp.fft.irfft(self.arr_spectral, axis=0, norm='forward')

    def compute_zero_moment(self, grid):
        self.moment0.arr_spectral = grid.moment0(variable=self.arr_spectral)
        self.moment0.inverse_fourier_transform()

    def compute_moment_1(self, grid):
        self.moment_v.arr_spectral = grid.moment_v(variable=self.arr_spectral)
        self.moment_w.arr_spectral = grid.moment_w(variable=self.arr_spectral)
        self.moment_v.inverse_fourier_transform(), self.moment_w.inverse_fourier_transform()

    def compute_second_moment(self, grid):
        self.moment2.arr_spectral = grid.second_moment(variable=self.arr_spectral)
        self.moment2.inverse_fourier_transform()

    def total_density(self, grid):
        self.compute_zero_moment(grid=grid)
        return self.moment0.integrate(grid=grid)

    def total_thermal_energy(self, grid):
        self.compute_second_moment(grid=grid)
        return 0.5 * self.moment2.integrate(grid=grid)

    def nodal_flatten(self):
        return self.arr_nodal.reshape(self.resolutions[0],
                                      self.resolutions[1] * self.order, self.resolutions[2] * self.order,
                                      self.resolutions[3] * self.order)

    def spectral_flatten(self):
        return self.arr_spectral.reshape(self.arr_spectral.shape[0],
                                         self.resolutions[1] * self.order, self.resolutions[2] * self.order,
                                         self.resolutions[3] * self.order)

    def initialize(self, grid, vt, alpha, ring_gamma, wavenumber, eigenvalue, dynamic_fields):
        ix, iu, iv = cp.ones_like(grid.x.device_arr), cp.ones_like(grid.u.device_arr), cp.ones_like(grid.v.device_arr)

        ring_distribution = cp.tensordot(ix, grid.ring_distribution(thermal_velocity=vt,
                                                                    alpha=alpha,
                                                                    ring_parameter=ring_gamma),
                                         axes=0)

        perturbation = grid.eigenfunction(thermal_velocity=vt, alpha=alpha, ring_parameter=ring_gamma,
                                          eigenvalue=eigenvalue, wavenumber=wavenumber, dynamic_fields=dynamic_fields)

        self.arr_nodal = cp.asarray(ring_distribution + perturbation)
        self.fourier_transform()


def trapz(y, dx):
    """ Custom trapz routine using cupy """
    return cp.sum(y[:-1] + y[1:]) * dx / 2.0
