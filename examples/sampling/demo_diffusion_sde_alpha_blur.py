r"""
Comparing VE diffusion sampling across alpha values and blur severities
=======================================================================

This demo shows two complementary effects when using
:class:`deepinv.sampling.VarianceExplodingDiffusion` (VE-SDE):

* how the parameter :math:`\alpha` changes unconditional sampling, and
* how progressively stronger low-pass filters affect posterior sampling.

As in :ref:`sphx_glr_auto_examples_sampling_demo_diffusion_sde.py`, we use a
pretrained :class:`deepinv.models.NCSNpp` denoiser together with
:class:`deepinv.sampling.PosteriorDiffusion`.

For unconditional sampling, we repeat the experiment for several values of
:math:`\alpha`:

* :math:`\alpha = 0` corresponds to deterministic ODE sampling.
* Larger :math:`\alpha` values inject more stochasticity during sampling.

For posterior sampling, we keep :math:`\alpha` fixed and vary an ideal
low-pass filter in the Fourier domain. Smaller cutoff radii keep fewer Fourier
coefficients, so more high-frequency content is removed from the measurements.

.. note::

    We keep the number of diffusion steps small for the sake of speed. In
    practice, using more steps usually improves the sample quality.
"""

# %%
import torch
from pathlib import Path
import deepinv as dinv
from deepinv.models import NCSNpp
from deepinv.optim import ZeroFidelity
from deepinv.sampling import (
    DPSDataFidelity,
    PiGDMDataFidelity,
    MomentMatchingDataFidelity,
    EulerSolver,
    PosteriorDiffusion,
    VarianceExplodingDiffusion,
)


device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
dtype = torch.float64 if torch.cuda.is_available() else torch.float32
figsize = 2.5
num_samples = 5
sample_shape = (num_samples, 3, 64, 64)
cutoff_radii = [0.2]

# We compare a deterministic reverse process with increasingly stochastic ones.
alpha_values = [0.0, 0.1, 0.2, 0.25, 0.3, 0.4]

# We use the pretrained FFHQ-64 model from the EDM framework.
denoiser = NCSNpp(pretrained="download").to(device).eval()

num_steps = 300
rng = torch.Generator(device).manual_seed(42)
timesteps = torch.linspace(1.0, 0.001, num_steps)
solver = EulerSolver(timesteps=timesteps, rng=rng)

sigma_min = 0.001
sigma_max = 100.0

def build_sampler(alpha, data_fidelity):
    sde = VarianceExplodingDiffusion(
        sigma_min=sigma_min,
        sigma_max=sigma_max,
        alpha=alpha,
        device=device,
        dtype=dtype,
    )
    return PosteriorDiffusion(
        data_fidelity=data_fidelity,
        sde=sde,
        denoiser=denoiser,
        solver=solver,
        dtype=dtype,
        device=device,
        verbose=True,
    )


def build_lowpass_physics(img_size, cutoff_radius):
    _, height, width = img_size
    fy = torch.fft.fftfreq(height, device=device)
    fx = torch.fft.fftfreq(width, device=device)
    grid_y, grid_x = torch.meshgrid(fy, fx, indexing="ij")
    radial_frequency = torch.sqrt(grid_x.square() + grid_y.square())
    lowpass_mask = (
        (radial_frequency <= cutoff_radius)
        .to(torch.float32)
        .unsqueeze(0)
        .unsqueeze(0)
    )

    def apply_lowpass(z):
        z_f = torch.fft.fft2(z, norm="ortho")
        return torch.fft.ifft2(z_f * lowpass_mask, norm="ortho").real

    return dinv.physics.LinearPhysics(
        A=apply_lowpass,
        A_adjoint=apply_lowpass,
        img_size=img_size,
    )


# %% Posterior sampling for progressively stronger frequency filtering
# --------------------------------------------------------------------
#
# We now solve the same inverse problem with progressively smaller low-pass
# cutoffs. Each step keeps fewer Fourier coefficients than the previous one.

ffhq_image = '00001.png'
if Path(ffhq_image).exists():
    x = dinv.utils.load_image(ffhq_image, img_size=64, resize_mode="resize").to(device)
else:
    x = dinv.utils.load_example(
        "celeba_example.jpg",
        img_size=64,
        resize_mode="resize",
    ).to(device)
dinv.utils.plot(
    [x],
    titles=["original"],
    save_fn="test.png",
    save_dir="test",
)
x_plot = x.repeat(num_samples, 1, 1, 1)

posterior_samples = []
case_titles = [f"cutoff={cutoff_radius:.2f}" for cutoff_radius in cutoff_radii]
sampler = build_sampler(
    alpha=posterior_alpha,
    data_fidelity=MomentMatchingDataFidelity(denoiser=denoiser, weight=1.0),
)
for cutoff_radius in cutoff_radii:
    # physics = build_lowpass_physics(x.shape[1:], cutoff_radius)
    physics = dinv.physics.BlurFFT(
        x.shape[1:],
        filter=dinv.physics.blur.gaussian_blur((5, 5)),
        noise_model=dinv.physics.GaussianNoise(
            sigma=0.1, rng=torch.Generator(device=x.device).manual_seed(123)
        ),
        device=device,
    )
    y = physics(x)
    x_hat = sampler(
        y=y,
        physics=physics,
        x_init=sample_shape,
        seed=11,
    )
    
    posterior_samples.append(x_hat)
    case_titles.append(f'alpha={alpha}')
 

dinv.utils.plot(
    [x_plot, y.repeat(num_samples, 1, 1, 1)] + posterior_samples,
    titles=["original"] + ["measurement"] + case_titles,
    suptitle=f"Posterior samples for different alphas",
    figsize=(figsize * (len(alpha_values) + 1), figsize * num_samples),
    max_imgs=num_samples,
    rescale_mode="clip",
    save_fn="ve_progressive_alpha.png",
    save_dir="ve_progressive_alpha",
)



# %% Unconditional VE sampling
# ----------------------------
#
# We first generate multiple unconditional samples for each value of
# :math:`\alpha`.

unconditional_samples = []
unconditional_titles = [f"alpha={alpha}" for alpha in alpha_values]
for alpha in alpha_values:
    sampler = build_sampler(alpha=alpha, data_fidelity=ZeroFidelity())
    sample = sampler(
        y=None,
        physics=None,
        x_init=sample_shape,
        seed=1,
    )
    unconditional_samples.append(sample)

dinv.utils.plot(
    unconditional_samples,
    titles=unconditional_titles,
    suptitle="Unconditional sampling with VE-SDE",
    figsize=(figsize * len(alpha_values), figsize * num_samples),
    max_imgs=num_samples,
    rescale_mode="clip",
    save_fn="ve_alpha_unconditional.png",
    save_dir="ve_alpha_unconditional_samples",
)
