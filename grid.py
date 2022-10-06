import numpy as np
import cupy as cp
import basis as b
import matplotlib.pyplot as plt
import scipy.special as sp


class SpaceGrid:
    """ In this scheme, the spatial grid is uniform and transforms are accomplished by DFT """
    def __init__(self, low, high, elements):
        # grid limits and elements
        self.low, self.high = low, high
        self.elements = elements

        # domain and element widths
        self.length = self.high - self.low
        self.dx = self.length / self.elements

        # element Jacobian
        self.J = 2.0 / self.dx

        # arrays
        self.arr, self.device_arr = None, None
        self.create_grid()

        # spectral properties
        self.modes = elements // 2 + 1  # Nyquist frequency
        self.fundamental = 2.0 * np.pi / self.length
        # self.wavenumbers = self.fundamental * np.arange(-self.modes, self.modes)
        self.wavenumbers = self.fundamental * np.arange(self.modes)
        # print(self.wavenumbers)
        self.device_modes = cp.arange(self.modes)
        self.device_wavenumbers = cp.array(self.wavenumbers)
        self.device_wavenumbers_fourth = self.device_wavenumbers ** 4.0
        self.device_wavenumbers_fourth[self.device_wavenumbers < 0.5] = 0  # 1
        self.zero_idx = 0  # int(self.modes)
        # self.two_thirds_low = int((1 * self.modes)//3 + 1)
        # self.two_thirds_high = self.wavenumbers.shape[0] - self.two_thirds_low
        self.pad_width = int((1 * self.modes)//3 + 1)
        # print(self.two_thirds_low)
        # print(self.two_thirds_high)
        print(self.length)
        print(self.fundamental)

    def create_grid(self):
        """ Build evenly spaced grid, assumed periodic """
        self.arr = np.linspace(self.low, self.high - self.dx, num=self.elements)
        self.device_arr = cp.asarray(self.arr)


class VelocityGrid:
    """ In this experiment, the velocity grid is an LGL quadrature grid """

    def __init__(self, low, high, elements, order):
        self.low, self.high = low, high
        self.elements, self.order = elements, order
        self.local_basis = b.LGLBasis1D(order=self.order)

        self.element_idxs = np.arange(self.elements)

        # domain and element widths
        self.length = self.high - self.low
        self.dx = self.length / self.elements

        # arrays
        self.arr, self.device_arr = None, None
        self.mid_points = None
        self.create_even_grid()

        # stretch / transform elements
        self.dx_grid = None
        # self.stretch_grid()
        # self.create_triple_grid(lows=np.array([self.low, -7, 7]),
        #                         highs=np.array([-7, 7, self.high]),
        #                         elements=np.array([4, 14, 4]))
        if self.dx_grid is None:
            self.dx_grid = self.dx * cp.ones(self.elements)

        # jacobian
        self.J = cp.asarray(2.0 / self.dx_grid)
        self.J_host = self.J.get()
        self.min_dv = cp.amin(self.dx_grid)
        # plt.figure()
        # for i in range(self.elements):
        #     plt.plot(np.zeros_like(self.arr[i, :]), self.arr[i, :], 'ko')
        # plt.show()

        # global quad weights
        self.global_quads = cp.tensordot(cp.ones(elements),
                                         cp.asarray(self.local_basis.weights), axes=0)

        # global translation matrix
        mid_identity = np.tensordot(self.mid_points, np.eye(self.local_basis.order), axes=0)
        self.translation_matrix = cp.asarray(mid_identity +
                                             self.local_basis.translation_matrix[None, :, :] /
                                             self.J[:, None, None].get())

        # create monotonic grid for plotting
        self.monogrid = None
        self.initialize_monogrid()

        # quad matrix
        # self.modes = 2.0 * np.pi / self.length * np.arange(int(2 * self.elements))  # only positive frequencies
        # self.fourier_quads = (self.local_basis.weights[None, None, :] *
        #                       np.exp(-1j * self.modes[:, None, None] * self.arr[None, :, :]) /
        #                       self.J[None, :, None].get()) / self.length
        # self.grid_phases = np.exp(1j * self.modes[None, None, :] * self.arr[:, :, None])

    def create_even_grid(self):
        """ Build global grid """
        # translate to [0, 1]
        nodes_iso = (np.array(self.local_basis.nodes) + 1) / 2
        # element left boundaries (including ghost elements)
        xl = np.linspace(self.low, self.high - self.dx, num=self.elements)
        # construct coordinates
        self.arr = np.zeros((self.elements, self.order))
        for i in range(self.elements):
            self.arr[i, :] = xl[i] + self.dx * np.array(nodes_iso)
        # send to device
        self.device_arr = cp.asarray(self.arr)
        self.mid_points = np.array([0.5 * (self.arr[i, -1] + self.arr[i, 0]) for i in range(self.elements)])

    def create_triple_grid(self, lows, highs, elements):
        """ Build a three-segment grid, each evenly-spaced """
        # translate to [0, 1]
        nodes_iso = (np.array(self.local_basis.nodes) + 1) / 2
        # element left boundaries (including ghost elements)
        dxs = (highs - lows) / elements
        xl0 = np.linspace(lows[0], highs[0] - dxs[0], num=elements[0])
        xl1 = np.linspace(lows[1], highs[1] - dxs[1], num=elements[1])
        xl2 = np.linspace(lows[2], highs[2] - dxs[2], num=elements[2])
        # construct coordinates
        self.arr = np.zeros((elements[0] + elements[1] + elements[2], self.order))
        for i in range(elements[0]):
            self.arr[i, :] = xl0[i] + dxs[0] * nodes_iso
        for i in range(elements[1]):
            self.arr[elements[0] + i, :] = xl1[i] + dxs[1] * nodes_iso
        for i in range(elements[2]):
            self.arr[elements[0] + elements[1] + i, :] = xl2[i] + dxs[2] * nodes_iso
        # send to device
        self.device_arr = cp.asarray(self.arr)
        self.mid_points = np.array([0.5 * (self.arr[i, -1] + self.arr[i, 0]) for i in range(self.elements)])
        self.dx_grid = self.device_arr[:, -1] - self.device_arr[:, 0]

    def zero_moment(self, function, idx):
        return cp.tensordot(self.global_quads / self.J[:, None], function, axes=([0, 1], idx))

    def second_moment(self, function, idx):
        return cp.tensordot(self.global_quads / self.J[:, None], cp.multiply(self.device_arr[None, :, :] ** 2.0,
                                                                             function),
                            axes=([0, 1], idx))

    def compute_maxwellian(self, thermal_velocity, drift_velocity):
        return cp.exp(-0.5 * ((self.device_arr - drift_velocity) /
                              thermal_velocity) ** 2.0) / (np.sqrt(2.0 * np.pi) * thermal_velocity)

    def compute_maxwellian_gradient(self, thermal_velocity, drift_velocity):
        return (-1.0 * ((self.device_arr - drift_velocity) / thermal_velocity ** 2.0) *
                self.compute_maxwellian(thermal_velocity=thermal_velocity, drift_velocity=drift_velocity))

    def initialize_monogrid(self):
        self.monogrid = np.zeros(self.elements * (self.order - 1) + 1)
        for i in range(self.elements):
            self.monogrid[i*(self.order-1):(i+1)*(self.order-1)] = self.arr[i, :-1]
        self.monogrid[-1] = self.arr[-1, -1]


class PhaseSpace:
    def __init__(self, lows, highs, elements, order, charge_sign, om_pc):
        self.x = SpaceGrid(low=lows[0], high=highs[0], elements=elements[0])
        self.u = VelocityGrid(low=lows[1], high=highs[1], elements=elements[1], order=order)
        self.v = VelocityGrid(low=lows[2], high=highs[2], elements=elements[2], order=order)
        self.w = VelocityGrid(low=lows[3], high=highs[3], elements=elements[3], order=order)

        # These probably belong in variables class
        self.v_mag_sq = (self.u.device_arr[:, :, None, None, None, None] ** 2.0 +
                         self.v.device_arr[None, None, :, :, None, None] ** 2.0 +
                         self.w.device_arr[None, None, None, None, :, :] ** 2.0)
        self.charge_sign = charge_sign
        self.om_pc = om_pc  # cyclotron freq. ratio

    def ring_distribution(self, thermal_velocity, alpha, ring_parameter):
        # Cylindrical coordinates grid set-up, using wave-number x.k1
        u = outer3(self.u.arr, np.ones_like(self.v.arr), np.ones_like(self.w.arr))
        v = outer3(np.ones_like(self.u.arr), self.v.arr, np.ones_like(self.w.arr))
        w = outer3(np.ones_like(self.u.arr), np.ones_like(self.v.arr), self.w.arr)
        r = np.sqrt(v ** 2.0 + w ** 2.0)

        # Set distribution
        x = 0.5 * (r / alpha) ** 2.0
        factor = 1 / (2.0 * np.pi * (alpha ** 2.0) * sp.gamma(ring_parameter + 1.0))
        ring = factor * np.multiply(x ** ring_parameter, np.exp(-x))
        maxwell = np.exp(-0.5*u**2/thermal_velocity**2) / np.sqrt(2 * np.pi * thermal_velocity**2)

        return cp.asarray(maxwell * ring)

    def eigenfunction(self, thermal_velocity, alpha, ring_parameter, eigenvalue, wavenumber, dynamic_fields):
        ''' Now stupid-proof '''
        # Cylindrical coordinates grid set-up, using wave-number wavenumber
        u = outer3(self.u.arr, np.ones_like(self.v.arr), np.ones_like(self.w.arr))
        v = outer3(np.ones_like(self.u.arr), self.v.arr, np.ones_like(self.w.arr))
        w = outer3(np.ones_like(self.u.arr), np.ones_like(self.v.arr), self.w.arr)
        r = np.sqrt(v ** 2.0 + w ** 2.0)
        phi = np.arctan2(w, v)
        # beta = - self.x.fundamental * r * self.om_pc
        # vt = alpha

        # radial gradient of distribution
        # x = 0.5 * (r / vt) ** 2.0
        # ring = 1 / (2.0 * np.pi * (vt ** 2.0) * sp.gamma(ring_parameter + 1.0)) * np.multiply(x ** ring_parameter,
        #                                                                                     np.exp(-x))
        x = (r / alpha) ** 2
        ring = 1 / (np.pi * alpha**2 * sp.gamma(ring_parameter + 1.0)) * np.multiply(x ** ring_parameter, np.exp(-x))
        ring_1 = 1 / (np.pi * alpha**2 * sp.gamma(ring_parameter)) * np.multiply(x ** (ring_parameter-1), np.exp(-x))
        maxwell = np.exp(-0.5 * u ** 2 / thermal_velocity ** 2) / np.sqrt(2 * np.pi * thermal_velocity ** 2)
        f = ring * maxwell
        # df_dv_perp = np.multiply(f, (ring_parameter / (x + 1.0e-16) - 1.0)) / (thermal_velocity ** 2.0)
        df_dv_perp = np.multiply(maxwell, r * (ring_1 - ring)/(2 * alpha**2))
        df_dv_para = np.multiply(f, u / thermal_velocity ** 2)

        # set up eigenmode
        zeta_cyclotron = -1 / self.om_pc / wavenumber
        zeta = eigenvalue
        denominator_p = zeta - u - zeta_cyclotron
        denominator_m = zeta - u + zeta_cyclotron
        # fac1 = np.exp(1j*phi) / denominator_p
        # fac2 = np.exp(-1j*phi) / denominator_m
        fac1 = np.exp(-1j * phi) / denominator_p
        fac2 = np.exp(1j * phi) / denominator_m

        v_cross_grad = r * df_dv_para - u * df_dv_perp
        A = df_dv_perp + v_cross_grad / zeta

        # sq2 = cp.sqrt(2)
        # amplitude = 1.0e-3
        # E_y = 1.0j / sq2 * amplitude  # * cp.exp(1j * wavenumber * self.x.device_arr)
        # E_z = 1.0 / sq2 * amplitude  # * cp.exp(1j * wavenumber * self.x.device_arr)
        # eig = A * np.exp(1j*phi) / denominator_p
        # eig = 1.0 / sq2.get() * amplitude * A * np.exp(1j * phi) / denominator_p
        # eig = 1.0e-3 * -1j * A * (1.0 * (fac1 + fac2) + 1j * (1j) * (fac1 - fac2)) / 2.0 / wavenumber

        # eig = 1j * A * 0.5 * (np.exp(1j * phi) / denominator_p + np.exp(-1j * phi) / denominator_m)

        ''' Testing: Pure transverse current mode '''
        # eig = 1.0e-3 * df_dv_perp * np.exp(1j * phi)

        ''' Kinetic eigenmode given electric field amplitudes '''
        # eig = -1j * A * (dynamic_fields.eig_y.get() * (fac1 + fac2) +
        #                  1j * dynamic_fields.eig_z.get() * (fac1 - fac2)) / 2.0 / wavenumber
        ex, ey = dynamic_fields.eig_y.get(), dynamic_fields.eig_z.get()
        eig = -1j * A * ((ex + 1j * ey) * fac1 + (ex - 1j * ey) * fac2) / 2.0 / wavenumber
        # E_perp = np.real(dynamic_fields.eig_y.get()) + 1j * np.real(dynamic_fields.eig_z.get())
        # eig = -1j * A * E_perp / 2.0 * (fac1 + fac2) / wavenumber

        return cp.asarray(np.real(np.tensordot(np.exp(1j * wavenumber * self.x.arr), eig, axes=0)))

    def moment0(self, variable):
        return self.u.zero_moment(
            function=self.v.zero_moment(
                function=self.w.zero_moment(
                    variable,
                    idx=[5, 6]
                ),
                idx=[3, 4]
            ),
            idx=[1, 2]
        )

    def second_moment(self, variable):
        integrand = self.v_mag_sq[None, :, :, :, :, :, :] * variable
        return self.u.zero_moment(
            function=self.v.zero_moment(
                function=self.w.zero_moment(
                    integrand,
                    idx=[5, 6]
                ),
                idx=[3, 4]
            ),
            idx=[1, 2]
        )

    def moment_u(self, variable):
        integrand = self.u.device_arr[None, :, :, None, None, None, None] * variable
        return self.u.zero_moment(
            function=self.v.zero_moment(
                function=self.w.zero_moment(
                    integrand,
                    idx=[5, 6]
                ),
                idx=[3, 4]
            ),
            idx=[1, 2]
        )

    def moment_v(self, variable):
        integrand = self.v.device_arr[None, None, None, :, :, None, None] * variable
        return self.u.zero_moment(
            function=self.v.zero_moment(
                function=self.w.zero_moment(
                    integrand,
                    idx=[5, 6]
                ),
                idx=[3, 4]
            ),
            idx=[1, 2]
        )

    def moment_w(self, variable):
        integrand = self.w.device_arr[None, None, None, None, None, :, :] * variable
        return self.u.zero_moment(
            function=self.v.zero_moment(
                function=self.w.zero_moment(
                    integrand,
                    idx=[5, 6]
                ),
                idx=[3, 4]
            ),
            idx=[1, 2]
        )


def outer3(a, b, c):
    """
    Compute outer tensor product of vectors a, b, and c
    :param a: vector a_i
    :param b: vector b_j
    :param c: vector c_k
    :return: tensor a_i b_j c_k as numpy array
    """
    return np.tensordot(a, np.tensordot(b, c, axes=0), axes=0)
