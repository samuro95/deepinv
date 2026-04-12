r"""
Comparing VE diffusion sampling for different alpha values
==========================================================

This demo shows how the parameter :math:`\alpha` changes both unconditional
sampling and posterior sampling when using the
:class:`deepinv.sampling.VarianceExplodingDiffusion` (VE-SDE).

As in :ref:`sphx_glr_auto_examples_sampling_demo_diffusion_sde.py`, we use a
pretrained :class:`deepinv.models.NCSNpp` denoiser together with
:class:`deepinv.sampling.PosteriorDiffusion`. The only difference is that we
repeat the experiment for several values of :math:`\alpha`:

* :math:`\alpha = 0` corresponds to deterministic ODE sampling.
* Larger :math:`\alpha` values inject more stochasticity during sampling.

We compare:

* unconditional VE sampling, and
* posterior sampling for a Gaussian deblurring problem.

.. note::

    We keep the number of diffusion steps small for the sake of speed. In
    practice, using more steps usually improves the sample quality.
"""

# %%
import torch
import deepinv as dinv
from deepinv.models import NCSNpp
from deepinv.optim import ZeroFidelity
from deepinv.sampling import (
    DPSDataFidelity,
    EulerSolver,
    PosteriorDiffusion,
    VarianceExplodingDiffusion,
)


device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float64
figsize = 2.5

# We compare a deterministic reverse process with increasingly stochastic ones.
alpha_values = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]

# We use the pretrained FFHQ-64 model from the EDM framework.
denoiser = NCSNpp(pretrained="download").to(device).eval()

num_steps = 30
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
        verbose=False,
    )


# %% Unconditional VE sampling
# ----------------------------
#
# We first generate one unconditional sample for each value of :math:`\alpha`.

unconditional_samples = []
unconditional_titles = [f"alpha={alpha}" for alpha in alpha_values]
for alpha in alpha_values:
    sampler = build_sampler(alpha=alpha, data_fidelity=ZeroFidelity())
    sample = sampler(
        y=None,
        physics=None,
        x_init=(1, 3, 64, 64),
        seed=1,
    )
    unconditional_samples.append(sample)

dinv.utils.plot(
    unconditional_samples,
    titles=unconditional_titles,
    suptitle="Unconditional sampling with VE-SDE",
    figsize=(figsize * len(alpha_values), figsize),
    rescale_mode="clip",
    save_fn="ve_alpha_unconditional.png",
    save_dir="ve_alpha_unconditional_samples",
)


# %% Posterior sampling for Gaussian deblurring
# ---------------------------------------------
#
# We now solve the same Gaussian deblurring problem for the same values of
# :math:`\alpha`.

x = dinv.utils.load_example(
    "celeba_example.jpg",
    img_size=64,
    resize_mode="resize",
).to(device)

filter_blur = dinv.physics.blur.gaussian_blur(sigma=(3.0, 3.0))
physics = dinv.physics.BlurFFT(
    img_size=x.shape[1:],
    filter=filter_blur,
    device=device,
)
y = physics(x)

posterior_samples = []
posterior_titles = [f"posterior_alpha={alpha}" for alpha in alpha_values]
for alpha in alpha_values:
    sampler = build_sampler(
        alpha=alpha,
        data_fidelity=DPSDataFidelity(denoiser=denoiser, weight=1.0),
    )
    x_hat = sampler(
        y=y,
        physics=physics,
        seed=11,
    )
    posterior_samples.append(x_hat)

dinv.utils.plot(
    [x, y] + posterior_samples,
    titles=["original", "blurred_measurement"] + posterior_titles,
    figsize=(figsize * (len(alpha_values) + 2), figsize),
    rescale_mode="clip",
    save_fn="ve_alpha_posterior_blur.png",
    save_dir="ve_alpha_posterior_blur_samples",
)
