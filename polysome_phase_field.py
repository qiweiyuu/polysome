# impose Dirichlet boundary condition to fix the bulk concentration
from jax import config

config.update("jax_enable_x64", True)

import numpy as onp
import matplotlib.pyplot as plt
import jax.numpy as np
from jax import lax
import os, argparse

linewidth = 2

parser = argparse.ArgumentParser(description="MavQ_4species_V1_1")
parser.add_argument("--L", type=int, default=1, help="rescaled length")
parser.add_argument("--N", type=int, default=128)

parser.add_argument("--k1", type=float, default=0.004, help="k1")
parser.add_argument("--kn1", type=float, default=0.003, help="kn1")

parser.add_argument("--chip", type=float, default=0.8)
parser.add_argument("--chin", type=float, default=1.2)
parser.add_argument("--chinp", type=float, default=4.0)

parser.add_argument(
    "--lmda", type=float, default=0.02, help="interface width, in micron"
)
parser.add_argument(
    "--Dn",
    type=float,
    default=0.01,
    help="diffusion coefficient of nucleoid, in micron^2/s",
)
parser.add_argument(
    "--Dp",
    type=float,
    default=0.05,
    help="diffusion coefficient of polysomes, in micron^2/s",
)
parser.add_argument("--Vn", type=float, default=10)
parser.add_argument("--Vp", type=float, default=5)
parser.add_argument(
    "--T", type=float, default=2400, help="total elongation time (in s)"
)
parser.add_argument("--dt", type=float, default=0.01, help="time step (in s)")
parser.add_argument(
    "--l-center", type=float, default=0.3, help="initial nucleoid length "
)
parser.add_argument(
    "--phi-center-n", type=float, default=0.5, help="initial nucleoid concentration"
)
parser.add_argument(
    "--phi-center-p", type=float, default=0.1, help="initial polysome concentration"
)
parser.add_argument(
    "--phi-boundary-n", type=float, default=0.1, help="boundary nucleoid concentration"
)
parser.add_argument(
    "--phi-boundary-p", type=float, default=0.4, help="boundary polysome concentration"
)
parser.add_argument(
    "--init-name",
    type=str,
    default=None,
    help="file name for the initial condition",
)
parser.add_argument("--l-init", type=float, default=2.0, help="initial cell length ")
parser.add_argument("--l-final", type=float, default=3.5, help="final cell length ")

parser.add_argument(
    "--l-decay", type=float, default=2.0 / 16, help="decay length of degradation rate"
)


parser.add_argument("--filename", type=str, default="data_v3_1/")
flags = parser.parse_args()


############################################################################################################
chip, chin, chinp = flags.chip, flags.chin, flags.chinp
lmda = flags.lmda
Dn, Dp = flags.Dn * flags.Vn, flags.Dp * flags.Vp
v_molecule = np.array([flags.Vp, flags.Vn])

# total cell length
l_init = flags.l_init
l_final = flags.l_final
T = flags.T
gamma = onp.log(l_final / l_init) / T
l_t_fun = lambda t: l_init * onp.exp(gamma * t)

############################################################################################################
# x_list = np.linspace(-flags.L, flags.L, flags.N)
dx = 2 * flags.L / flags.N
x_list = np.arange(-flags.L, flags.L, dx) + dx / 2
dx = x_list[1] - x_list[0]
kx_list = 2 * np.pi * np.fft.fftfreq(flags.N, d=dx)
kx = kx_list
kxj = kx_list * 1j
k2 = kx_list**2
k4 = kx_list**4


phitol = 1e-6


def calc_mu_fft(
    phi,
    phitfft,
    chip=flags.chip,
    chin=flags.chin,
    chinp=flags.chinp,
    lmda2=flags.lmda**2,
):
    mu_entropy = (
        np.log(np.minimum(np.maximum(phi, phitol), 1 - phitol))
        / v_molecule[:, np.newaxis]
    )
    mu_entropy_solvent = np.log(
        np.minimum(np.maximum(1 - phi.sum(axis=0), phitol), 1 - phitol)
    )
    mu_p_fft = (
        np.fft.fft(mu_entropy[0] - mu_entropy_solvent)
        + chip * (1 - k2 * lmda2) * (-2 * phitfft[0] - phitfft[1])
        + (chinp - chin) * (1 - k2 * lmda2) * phitfft[1]
    )
    mu_n_fft = (
        np.fft.fft(mu_entropy[1] - mu_entropy_solvent)
        + chin * (1 - k2 * lmda2) * (-2 * phitfft[1] - phitfft[0])
        + (chinp - chip) * (1 - k2 * lmda2) * phitfft[0]
    )
    return np.array([mu_p_fft, mu_n_fft])


# artificial term to stabilize the UV
Astable = kx**4 * lmda**2 / l_init**2 * max([chip, chin, chinp]) * max([Dp, Dn]) / 2


# semi-implicit + predictor-corrector iteration
def calc_step(
    phi,
    phitfft,
    dt,
    pcg_itr=4,
    Dp=flags.Dp,
    Dn=flags.Dn,
    k1=flags.k1,
    kn1=flags.kn1,
    lmda2=flags.lmda**2,
):
    mu_fft = calc_mu_fft(phi, phitfft, lmda2=lmda2)
    dphi_fft_diffusion = kxj * np.fft.fft(
        np.array([Dp, Dn])[:, np.newaxis] * phi * np.fft.ifft(kxj * mu_fft).real
    )
    dphi_fft_reaction = np.array(
        [k1 * phitfft[1] - np.fft.fft(kn1 * phi[0]), np.zeros_like(phitfft[0])]
    )
    dphi_predictor = dphi_fft_diffusion + dphi_fft_reaction + Astable * phitfft
    phitfft_predictor = (phitfft + dt * (dphi_predictor)) / (1 + dt * Astable)
    phit_predictor = np.fft.ifft(phitfft_predictor).real
    phit_predictor = phit_predictor.at[:, flags.N // 2 :].set(
        phit_predictor[:, flags.N // 2 - 1 :: -1]
    )
    for itr in range(pcg_itr):
        mu_fft_corrector = calc_mu_fft(phit_predictor, phitfft_predictor, lmda2=lmda2)
        dphi_fft_diffusion_corrector = kxj * np.fft.fft(
            np.array([Dp, Dn])[:, np.newaxis]
            * phit_predictor
            * np.fft.ifft(kxj * mu_fft_corrector).real
        )
        dphi_fft_reaction_corrector = np.array(
            [
                k1 * phitfft_predictor[1] - np.fft.fft(kn1 * phit_predictor[0]),
                np.zeros_like(phitfft[0]),
            ]
        )
        dphi_corrector = (
            dphi_fft_diffusion_corrector
            + dphi_fft_reaction_corrector
            + Astable * phitfft_predictor
        )
        phitfft_predictor = (phitfft + dt * (dphi_predictor + dphi_corrector) / 2) / (
            1 + dt * Astable
        )
        phit_predictor = np.fft.ifft(phitfft_predictor).real
        phit_predictor = phit_predictor.at[:, flags.N // 2 :].set(
            phit_predictor[:, flags.N // 2 - 1 :: -1]
        )

    phi = phit_predictor
    # volume cannot be negative
    phi = np.maximum(phi, phitol)
    return phi, np.fft.fft(phi)


def initialize_phi(phi_center, phi_boundary, center_length, interface_width):
    phi = onp.ones((2, flags.N)) * phi_boundary[:, np.newaxis]
    phi = (
        phi
        + (
            onp.tanh(
                (center_length - onp.abs((x_list + flags.L / 2))) / interface_width
            )
            + 1
        )
        / 2
        * (phi_center - phi_boundary)[:, np.newaxis]
    )
    phi[:, flags.N // 2 :] = phi[:, flags.N // 2 - 1 :: -1]
    return phi


############################################################################################################
## initialization
# use the profiles specified in flags.init_name if it is not None, otherwise compute the steady-state profile at the initial cell length

if flags.init_name is not None:
    phi_init_ss = np.load(flags.init_name)["phi_init"]
    phi_init_ss_fft = np.fft.fft(phi_init_ss)
else:
    l_center = flags.l_center
    phit_init = initialize_phi(
        onp.array([flags.phi_center_p, flags.phi_center_n]),
        onp.array([flags.phi_boundary_p, flags.phi_boundary_n]),
        l_center,
        flags.lmda,
    )

    dt = flags.dt
    N_steps = int(6000 / dt)

    def sub_run(vals, i):
        vals = calc_step(
            vals[0],
            vals[1],
            dt,
            Dp=Dp / l_init ** (2),
            Dn=Dn / l_init ** (2),
            k1=flags.k1,
            kn1=flags.kn1 + gamma,
            lmda2=flags.lmda ** (2) / l_init ** (2),
        )
        return vals, vals[0]

    (phi_init_ss, phi_init_ss_fft), phi_init_trace = lax.scan(
        sub_run, (phit_init.copy(), np.fft.fft(phit_init)), np.arange(N_steps), unroll=4
    )

############################################################################################################
## elongation

dt = flags.dt
N_steps = int(T / dt)

t_trace = np.arange(N_steps) * dt
l_trace = np.array(l_t_fun(t_trace))


def sub_run(vals, i):
    vals = calc_step(
        vals[0],
        vals[1],
        dt,
        Dp=Dp / l_trace[i] ** 2,
        Dn=Dn / l_trace[i] ** 2,
        k1=flags.k1,
        kn1=flags.kn1 + gamma,
        lmda2=flags.lmda**2 / l_trace[i] ** 2,
    )
    return vals, vals[0]


(phit, phit_fft), phi_trace = lax.scan(
    sub_run, (phi_init_ss, phi_init_ss_fft), np.arange(N_steps), unroll=4
)

############################################################################################################
# save the simulation results
steps_skip = min(200, N_steps // 200)
np.savez(
    flags.filename + f"phi_grow.npz",
    phi_trace=phi_trace[::steps_skip],
    t_trace=t_trace[::steps_skip],
    l_trace=l_trace[::steps_skip],
)