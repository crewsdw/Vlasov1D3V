import numpy as np
import scipy.special as sp
import matplotlib.pyplot as plt
import scipy.optimize as opt
import plasma_dispersion as pd

k = 0.01
vt_c = 0.1
sq2 = 2 ** 0.5
gamma = 6
alpha = 1
vt = 1
om_pc = 10.0


def dispersion_function_plus(k, z):
    """
    Computes plasma dispersion function epsilon_para(k, zeta) = 0 for + branch of whistlers
    """
    k_sq = k ** 2.0
    z_c = 1 / om_pc / k
    z_e = (z - z_c) / sq2

    return (1 - (z * vt_c) ** 2.0 +
            (1 + (gamma + 1) / 4 * (alpha ** 2) * pd.Zprime(z_e) + z_c * pd.Z(z_e)) / k_sq * (vt_c ** 2.0))


def dispersion_function_minus(k, z):
    """
    Computes plasma dispersion function epsilon_para(k, zeta) = 0 for - branch of whistlers
    """
    k_sq = k ** 2.0
    z_c = 1 / om_pc / k
    z_e = (z + z_c) / sq2

    return (1 - (z * vt_c) ** 2.0 +
            (1 + (gamma + 1) / 4 * (alpha ** 2) * pd.Zprime(z_e) - z_c * pd.Z(z_e)) / k_sq * (vt_c ** 2.0))


def analytic_jacobian_plus(k, z):
    k_sq = k ** 2.0
    z_c = 1 / om_pc / k
    z_e = (z - z_c) / sq2

    return (-2.0 * z + 0.25 * (gamma + 1) * (alpha ** 2) * pd.Zdoubleprime(z_e) / k_sq / sq2
            + z_c * pd.Zprime(z_e) / k_sq / sq2) * (vt_c ** 2.0)


def analytic_jacobian_minus(k, z):
    k_sq = k ** 2.0
    z_c = 1 / om_pc / k
    z_e = (z - z_c) / sq2

    return (-2.0 * z + 0.25 * (gamma + 1) * (alpha ** 2) * pd.Zdoubleprime(z_e) / k_sq / sq2
            + z_c * pd.Zprime(z_e) / k_sq / sq2) * (vt_c ** 2.0)


def dispersion_fsolve_plus(z, k):
    complex_z = z[0] + 1j * z[1]
    d = dispersion_function_plus(k, complex_z)
    return [np.real(d), np.imag(d)]


def jacobian_fsolve_plus(z, k):
    complex_z = z[0] + 1j * z[1]
    jac = analytic_jacobian_plus(k, complex_z)
    jr, ji = np.real(jac), np.imag(jac)
    return [[jr, -ji], [ji, jr]]


def dispersion_fsolve_minus(z, k):
    complex_z = z[0] + 1j * z[1]
    d = dispersion_function_minus(k, complex_z)
    return [np.real(d), np.imag(d)]


def jacobian_fsolve_minus(z, k):
    complex_z = z[0] + 1j * z[1]
    jac = analytic_jacobian_minus(k, complex_z)
    jr, ji = np.real(jac), np.imag(jac)
    return [[jr, -ji], [ji, jr]]


# Phase velocities
zr = np.linspace(-25, 25, num=200)
zi = np.linspace(-5, 5, num=200)
z = np.tensordot(zr, np.ones_like(zi), axes=0) + 1.0j * np.tensordot(np.ones_like(zr), zi, axes=0)

ZR, ZI = np.meshgrid(zr, zi, indexing='ij')

#
# solution = opt.root(dispersion_fsolve_plus, x0=np.array([0, 0.5]),
#                     args=(wavenumber, 1), jac=jacobian_fsolve, tol=1.0e-15)
# print(solution.x[1])

mu_p = dispersion_function_plus(k, z)
mu_m = dispersion_function_minus(k, z)

plt.figure()
plt.contour(ZR, ZI, np.real(mu_p), 0, colors='r', linewidths=3)
plt.contour(ZR, ZI, np.imag(mu_p), 0, colors='g', linewidths=3)
plt.xlabel('Real phase velocity'), plt.ylabel('Imaginary phase velocity')
plt.grid(True), plt.title(r'Dispersion function $\mathcal{D}_+$'), plt.tight_layout()

plt.figure()
plt.contour(ZR, ZI, np.real(mu_m), 0, colors='r', linewidths=3)
plt.contour(ZR, ZI, np.imag(mu_m), 0, colors='g', linewidths=3)
plt.xlabel('Real phase velocity'), plt.ylabel('Imaginary phase velocity')
plt.grid(True), plt.title(r'Dispersion function $\mathcal{D}_-$'), plt.tight_layout()

plt.show()

# Obtain some solutions
k = np.linspace(k, 0.2, num=1000)
sols_p = np.zeros_like(k) + 0j
guess_r, guess_i = 25, 0
for idx, wave in enumerate(k):
    guess_r += 2e-1
    solution = opt.root(dispersion_fsolve_plus, x0=np.array([guess_r, guess_i]),
                        args=wave, jac=jacobian_fsolve_plus, tol=1.0e-10)
    guess_r, guess_i = solution.x
    sols_p[idx] = (guess_r + 1j * guess_i)

sols_m = np.zeros_like(k) + 0j
guess_r, guess_i = -25, 0
for idx, wave in enumerate(k):
    guess_r += 2e-1
    solution = opt.root(dispersion_fsolve_minus, x0=np.array([guess_r, guess_i]),
                        args=wave, jac=jacobian_fsolve_minus, tol=1.0e-10)
    guess_r, guess_i = solution.x
    sols_m[idx] = (guess_r + 1j * guess_i)

plt.figure()
plt.plot(k, k * np.real(sols_p), 'r', linewidth=3, label='Real +')
plt.plot(k, k * np.imag(sols_p), 'g', linewidth=3, label='Imaginary +')
plt.plot(k, k * np.real(sols_m), 'r', linewidth=3, label='Real -')
plt.plot(k, k * np.imag(sols_m), 'g', linewidth=3, label='Imaginary -')
# plt.plot(k, k * np.ones_like(k) / vt_c, 'k--', label='Lightspeed', linewidth=3)
# plt.plot(k, k * -1 * np.ones_like(k) / vt_c, 'k--', label='Lightspeed', linewidth=3)
# plt.plot([skin_depth, skin_depth], [0, 10 / 3], 'b--', linewidth=1,
#          label=r'$0.3\times\sqrt{8}$')
plt.xlabel(r'Wavenumber $k\lambda_D$'), plt.ylabel(r'Frequency $\omega/\omega_p$')
# , plt.ylabel(r'Phase velocity $\zeta / v_t$')
plt.title(r'Anisotropic Maxwellian, $v_{tx}/c = 0.3$, Ring $\gamma = 6$')
plt.grid(True), plt.legend(loc='best'), plt.tight_layout()

# plt.figure()
# plt.plot(k, k * np.real(sols), 'r', label='real', linewidth=3)
# plt.plot(k, k * np.imag(sols), 'g', label='imag', linewidth=3)
# plt.xlabel(r'Wavenumber k'), plt.ylabel(r'Frequency $\omega_p / \omega_{pe}$')
# plt.grid(True), plt.legend(loc='best'), plt.tight_layout()
plt.show()


