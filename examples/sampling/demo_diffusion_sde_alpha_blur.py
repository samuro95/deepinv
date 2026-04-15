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
num_samples = 5
sample_shape = (num_samples, 3, 64, 64)

# We compare a deterministic reverse process with increasingly stochastic ones.
alpha_values = [0.0, 0.1, 0.15, 0.2, 0.25, 0.3, 0.5]

# We use the pretrained FFHQ-64 model from the EDM framework.
denoiser = NCSNpp(pretrained="download").to(device).eval()

num_steps = 50
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


# %% Posterior sampling for Gaussian deblurring
# ---------------------------------------------
#
# We now solve the same Gaussian deblurring problem for the same values of
# :math:`\alpha`, drawing multiple posterior samples for each case.

ffhq_image = '00001.png'

x = dinv.utils.load_image(ffhq_image, img_size = 64, resize_mode='resize').to(device)
dinv.utils.plot(
    [x],
    titles=["original"],
    save_fn="test.png",
    save_dir="test",
)
filter_blur = dinv.physics.blur.gaussian_blur(sigma=(1.0, 1.0))
physics = dinv.physics.BlurFFT(
    img_size=x.shape[1:],
    filter=filter_blur,
    device=device,
)
y = physics(x)
x_plot = x.repeat(num_samples, 1, 1, 1)
y_plot = y.repeat(num_samples, 1, 1, 1)

posterior_samples = []
posterior_titles = [f"alpha={alpha}" for alpha in alpha_values]
for alpha in alpha_values:
    sampler = build_sampler(
        alpha=alpha,
        data_fidelity=DPSDataFidelity(denoiser=denoiser, weight=1.0),
    )
    x_hat = sampler(
        y=y,
        physics=physics,
        x_init=sample_shape,
        seed=11,
    )
    posterior_samples.append(x_hat)

dinv.utils.plot(
    [x_plot, y_plot] + posterior_samples,
    titles=["original", "blurred_measurement"] + posterior_titles,
    figsize=(figsize * (len(alpha_values) + 2), figsize * num_samples),
    max_imgs=num_samples,
    rescale_mode="clip",
    save_fn="ve_alpha_posterior_blur.png",
    save_dir="ve_alpha_posterior_blur_samples",
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
