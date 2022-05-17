import cupy as cp
import numpy as np
import pyvista as pv
import matplotlib.pyplot as plt


class Plotter:
    def __init__(self, grid, colormap='RdPu'):
        self.colormap = colormap
        self.grid = grid
        # Build structured grid, nodal
        self.U, self.V = np.meshgrid(grid.u.arr.flatten(), grid.v.arr.flatten(), indexing='ij')
        self.x = grid.x.arr
        self.k = grid.x.wavenumbers / grid.x.fundamental
        self.length = grid.x.length

    def spatial_scalar_plot(self, scalar, y_axis, spectrum=True):
        if scalar.arr_nodal is None:
            scalar.inverse_fourier_transform()

        plt.figure()
        plt.plot(self.x.flatten(), scalar.arr_nodal.flatten().get(), 'o')
        plt.xlabel('x'), plt.ylabel(y_axis)
        plt.grid(True), plt.tight_layout()

        if spectrum:
            plt.figure()
            spectrum = scalar.arr_spectral.flatten().get()
            plt.plot(self.k.flatten(), np.real(spectrum), 'ro', label='real')
            plt.plot(self.k.flatten(), np.imag(spectrum), 'go', label='imaginary')
            plt.xlabel('Modes'), plt.ylabel(y_axis + ' spectrum')
            plt.grid(True), plt.legend(loc='best'), plt.tight_layout()

    def time_series_plot(self, time_in, series_in, y_axis, log=False, give_rate=False):
        time, series = time_in, series_in / self.length
        plt.figure()
        if log:
            plt.semilogy(time, series, 'o--')
        else:
            plt.plot(time, series, 'o--')
        plt.xlabel('Time')
        plt.ylabel(y_axis)
        plt.grid(True), plt.tight_layout()
        if give_rate:
            lin_fit = np.polyfit(time, np.log(series), 1)
            exact = 2 * 0.1 * 3.48694202e-01
            print('\nNumerical rate: {:0.10e}'.format(lin_fit[0]))
            # print('cf. exact rate: {:0.10e}'.format(2 * 2.409497728e-01))  #
            print('cf. exact rate: {:0.10e}'.format(exact))
            print('The difference is {:0.10e}'.format(lin_fit[0] - exact))

    def show(self):
        plt.show()


class Plotter3D:
    """
    Plots objects on 3D piecewise (as in DG) grid
    """

    def __init__(self, grid):
        # Build structured grid, full space
        # (ix, iu, iv) = (cp.ones(grid.x.elements+1),
        #                 cp.ones(grid.u.elements * grid.u.order),
        #                 cp.ones(grid.v.elements * grid.v.order))
        # modified_x = cp.append(grid.x.device_arr, grid.x.device_arr[-1] + grid.x.dx)
        # (x3, u3, v3) = (outer3(a=modified_x, b=iu, c=iv),
        #                 outer3(a=ix, b=grid.u.device_arr.flatten(), c=iv),
        #                 outer3(a=ix, b=iu, c=grid.v.device_arr.flatten()))
        # self.grid = pv.StructuredGrid(x3, u3, v3)

        # build velocity space grid
        iu, iv, iw = (cp.ones(grid.u.elements * grid.u.order),
                      cp.ones(grid.v.elements * grid.v.order),
                      cp.ones(grid.w.elements * grid.w.order))
        (u3, v3, w3) = (outer3(a=grid.u.device_arr.flatten(), b=iv, c=iw),
                        outer3(a=iu, b=grid.v.device_arr.flatten(), c=iw),
                        outer3(a=iu, b=iv, c=grid.w.device_arr.flatten()))
        self.grid = pv.StructuredGrid(u3, v3, w3)

    def distribution_contours3d(self, distribution, spectral_idx, ctype='real'):
        """
        plot contours of a scalar function f=f(idx, x,y,z) on Plotter3D's grid
        """
        # new_dist = np.zeros((distribution.u_res, distribution.order,
        #                      distribution.v_res, distribution.order,
        #                      distribution.w_res, distribution.order))
        new_dist = 0
        if ctype == 'real':
            new_dist = np.real(distribution.arr_spectral[spectral_idx, :, :, :, :, :, :].get())
        if ctype == 'imag':
            new_dist = np.imag(distribution.arr_spectral[spectral_idx, :, :, :, :, :, :].get())
        if ctype == 'absolute':
            new_dist = np.absolute(distribution.arr_spectral[spectral_idx, :, :, :, :, :, :].get())

        contours = np.array([np.amin(new_dist) / 7, np.amax(new_dist) / 7])
        # contours = np.linspace(np.amin(new_dist), np.amax(new_dist), num=12)
        self.grid['.'] = new_dist.reshape(new_dist.shape[0] * new_dist.shape[1],
                                          new_dist.shape[2] * new_dist.shape[3],
                                          new_dist.shape[4] * new_dist.shape[5]).transpose().flatten()

        plot_contours = self.grid.contour(contours)

        # Create plot
        p = pv.Plotter()
        p.add_mesh(plot_contours, cmap='summer', opacity=1, show_scalar_bar=False)
        p.show_grid()
        p.show()  # auto_close=False)


def outer3(a, b, c):
    """
    Compute outer tensor product of vectors a, b, and c
    :param a: vector a_i
    :param b: vector b_j
    :param c: vector c_k
    :return: tensor a_i b_j c_k as numpy array
    """
    return cp.tensordot(a, cp.tensordot(b, c, axes=0), axes=0).get()
