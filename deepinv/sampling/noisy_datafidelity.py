from __future__ import annotations
from typing import Callable
import torch
from deepinv.optim import DataFidelity, Distance
from deepinv.optim.utils import bicgstab, conjugate_gradient
import deepinv as dinv
from deepinv.physics import Physics
from deepinv.models import Denoiser


class NoisyDataFidelity(DataFidelity):
    r"""
    Preconditioned data fidelity term for noisy data :math:`- \log p(y|x + \sigma(t) \omega)`
    with :math:`\omega\sim\mathcal{N}(0,\mathrm{I})`.

    This is a base class for the conditional classes for approximating :math:`\log p_t(y|x_t)` used in diffusion
    algorithms for inverse problems, in :class:`deepinv.sampling.PosteriorDiffusion`.

    It comes with a `.grad` method computing the score :math:`\nabla_{x_t} \log p_t(y|x_t)`.

    By default we have

    .. math::

        \begin{equation*}
            \nabla_{x_t} \log p(y|x + \sigma(t) \omega) = P(\forw{x_t'}-y),
        \end{equation*}


    where :math:`P` is a preconditioner and :math:`x_t'` is an estimation of the image :math:`x`.
    By default, :math:`P` is defined as :math:`A^\top`, :math:`x_t' = x_t` and this class matches the
    :class:`deepinv.optim.DataFidelity` class.

    :param deepinv.optim.Distance d: Distance metric to use for the data fidelity term. Default to :class:`deepinv.optim.L2Distance`.
    :param float weight: Weighting factor for the data fidelity term. Default to 1.
    """

    def __init__(self, d: Distance = None, weight=1.0, *args, **kwargs):
        super().__init__()
        if d is not None:
            self.d = Distance(d)
        else:
            self.d = dinv.optim.L2Distance()
        self.weight = weight

    def precond(
        self, u: torch.Tensor, physics: Physics, *args, **kwargs
    ) -> torch.Tensor:
        r"""
        The preconditioner :math:`P` for the data fidelity term. Default to :math:`A^{\top}`.

        :param torch.Tensor u: input tensor.
        :param deepinv.physics.Physics physics: physics model.

        :return: (torch.Tensor) preconditionned tensor :math:`P(u)`.
        """
        return (
            physics.A_adjoint(u)
            if isinstance(physics, dinv.physics.LinearPhysics)
            else physics.A_dagger(u)
        )

    def diff(
        self, x: torch.Tensor, y: torch.Tensor, physics: Physics, *args, **kwargs
    ) -> torch.Tensor:
        r"""
        Computes the difference :math:`A(x) - y` between the forward operator applied to the current iterate and the input data.


        :param torch.Tensor x: Current iterate.
        :param torch.Tensor y: Input data.
        :return: (torch.Tensor) difference between the forward operator applied to the current iterate and the input data.
        """
        return physics.A(x) - y

    def grad(
        self, x: torch.Tensor, y: torch.Tensor, physics: Physics, *args, **kwargs
    ) -> torch.Tensor:
        r"""
        Computes the gradient of the data-fidelity term.

        :param torch.Tensor x: Current iterate.
        :param torch.Tensor y: Input data.
        :param deepinv.physics.Physics physics: physics model
        :return: (torch.Tensor) data-fidelity term.
        """
        return self.precond(self.diff(x, y, physics), physics=physics)

    def forward(
        self, x: torch.Tensor, y: torch.Tensor, physics: Physics, *args, **kwargs
    ) -> torch.Tensor:
        r"""
        Computes the data-fidelity term.

        :param torch.Tensor x: input image
        :param torch.Tensor y: measurements
        :param deepinv.physics.Physics physics: forward operator
        :return: (torch.Tensor) loss term.
        """
        return self.d(physics.A(x), y) * self.weight


class DPSDataFidelity(NoisyDataFidelity):
    r"""
    Diffusion posterior sampling data-fidelity term.

    This corresponds to the :math:`p(y|x_t)` approximation proposed in `Diffusion Posterior Sampling for General Noisy Inverse Problems <https://arxiv.org/abs/2209.14687>`_.

    .. math::
            \begin{aligned}
            \nabla_x \log p_t(y|x) &= \nabla_x \frac{\lambda}{2\sqrt{m}} \| \forw{\denoiser{x}{\sigma}} - y \|
            \end{aligned}

    where :math:`\sigma = \sigma(t)` is the noise level, :math:`m` is the number of measurements (size of :math:`y`),
    and :math:`\lambda` controls the strength of the approximation.

    .. seealso::
        This class can be used for building custom DPS-based diffusion models.
        A self-contained implementation of the original DPS algorithm can be find in :class:`deepinv.sampling.DPS`.

    :param deepinv.models.Denoiser denoiser: Denoiser network
    :param float weight: Weighting factor for the data fidelity term. Default to 100.
    :param tuple[float] clip: If not `None`, clip the denoised output into `[clip[0], clip[1]]` interval. Default to `None`.
    """

    def __init__(
        self,
        denoiser: Denoiser = None,
        weight=1.0,
        clip: tuple = None,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.d = dinv.optim.L2Distance()
        self.denoiser = denoiser
        if clip is not None:
            assert len(clip) == 2
            clip = sorted(clip)
        self.clip = clip
        self.weight = weight

    def precond(
        self, x: torch.Tensor, physics: Physics, *args, **kwargs
    ) -> torch.Tensor:
        raise NotImplementedError

    def grad(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        physics: Physics,
        sigma,
        *args,
        get_model_outputs=False,
        **kwargs,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        r"""
        :param torch.Tensor x: Current iterate.
        :param torch.Tensor y: Input data.
        :param deepinv.physics.Physics physics: physics model
        :param float sigma: Standard deviation of the noise.
        :param bool get_model_outputs: If `True`, also return the denoised output along with the score. Default to `False`.

        :return: (:class:`torch.Tensor` or tuple of :class:`torch.Tensor`) score term (and denoised output if `get_model_outputs` is `True`).
        """
        with torch.enable_grad():
            x.requires_grad_(True)
            out = self.forward(
                x,
                y,
                physics,
                sigma,
                *args,
                get_model_outputs=get_model_outputs,
                **kwargs,
            )
            # In case we also want the denoised output
            if get_model_outputs:
                l2_loss = out[0]
            else:
                l2_loss = out

            grad_outputs = torch.ones_like(l2_loss)
        norm_grad = torch.autograd.grad(
            outputs=l2_loss, inputs=x, grad_outputs=grad_outputs
        )[0]
        if get_model_outputs:
            return norm_grad, out[1]
        else:
            return norm_grad

    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        physics: Physics,
        sigma,
        *args,
        get_model_outputs=False,
        **kwargs,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        r"""
        Returns the loss term :math:`\frac{\lambda}{2\sqrt{m}} \| \forw{\denoiser{x}{\sigma}} - y \|`.

        :param torch.Tensor x: input image
        :param torch.Tensor y: measurements
        :param deepinv.physics.Physics physics: forward operator
        :param float sigma: standard deviation of the noise.
        :param bool get_model_outputs: If `True`, also return the denoised output along with the loss. Default to `False`.

        :return: (:class:`torch.Tensor` or tuple of :class:`torch.Tensor`) loss term (and denoised output if `get_model_outputs` is `True`).
        """

        if isinstance(sigma, torch.Tensor):
            sigma = sigma.to(torch.float32)

        x0_t = self.denoiser(x.to(torch.float32), sigma, *args, **kwargs)

        if self.clip is not None:
            x0_t = torch.clip(x0_t, self.clip[0], self.clip[1])  # optional

        out = (self.d(physics.A(x0_t), y) * y.numel() / y.size(0)).sqrt() * self.weight

        if get_model_outputs:
            return out, x0_t
        else:
            return out


def _init_gaussian_posterior_guidance(
    module: NoisyDataFidelity,
    denoiser: Denoiser | None,
    cov_y: torch.Tensor | float | Callable | None,
    weight: float,
    clip: tuple | None,
    solver: str,
    max_iter: int,
    tol: float,
    verbose: bool,
):
    module.d = dinv.optim.L2Distance()
    module.denoiser = denoiser
    module.cov_y = cov_y
    if clip is not None:
        assert len(clip) == 2
        clip = tuple(sorted(clip))
    module.clip = clip
    module.weight = weight
    module.solver = solver
    module.max_iter = max_iter
    module.tol = tol
    module.verbose = verbose


def _prepare_gaussian_posterior_sigma(
    sigma: torch.Tensor | float, x: torch.Tensor
) -> torch.Tensor:
    if isinstance(sigma, torch.Tensor):
        return sigma.to(device=x.device, dtype=torch.float32)
    return torch.tensor(sigma, device=x.device, dtype=torch.float32)


def _reshape_gaussian_posterior_param(
    param: torch.Tensor | float, ref: torch.Tensor
) -> torch.Tensor:
    dtype = ref.real.dtype if ref.is_complex() else ref.dtype
    if isinstance(param, torch.Tensor):
        param = param.to(device=ref.device, dtype=dtype)
    else:
        param = torch.tensor(param, device=ref.device, dtype=dtype)

    if param.ndim == 0 or param.shape == ref.shape:
        return param
    if param.ndim == 1 and param.size(0) == ref.size(0):
        return param.view((param.size(0),) + (1,) * (ref.ndim - 1))
    return param


def _gaussian_observation_covariance(
    cov_y: torch.Tensor | float | Callable | None,
    ref: torch.Tensor,
    physics: Physics,
    method_name: str,
):
    if callable(cov_y):
        return cov_y

    if cov_y is not None:
        cov_y = _reshape_gaussian_posterior_param(cov_y, ref)
        return lambda v: cov_y * v

    noise_model = getattr(physics, "noise_model", None)
    if isinstance(noise_model, dinv.physics.ZeroNoise):
        return lambda v: torch.zeros_like(v)
    if isinstance(noise_model, dinv.physics.GaussianNoise):
        cov_y = _reshape_gaussian_posterior_param(noise_model.sigma**2, ref)
        return lambda v: cov_y * v

    raise ValueError(
        f"{method_name} assumes Gaussian observations. Provide `cov_y` explicitly "
        "or use a physics object with `GaussianNoise`/`ZeroNoise`."
    )


def _gaussian_posterior_denoise(
    denoiser: Denoiser | None,
    clip: tuple | None,
    x: torch.Tensor,
    sigma,
    *args,
    **kwargs,
) -> torch.Tensor:
    if denoiser is None:
        raise ValueError(
            "A denoiser must be provided either when constructing the data fidelity "
            "or through `PosteriorDiffusion`."
        )

    x0_t = denoiser(x.to(torch.float32), sigma, *args, **kwargs)
    if clip is not None:
        x0_t = torch.clip(x0_t, clip[0], clip[1])
    return x0_t


def _gaussian_posterior_measurement_jvp(
    physics: Physics, x: torch.Tensor, v: torch.Tensor
) -> torch.Tensor:
    if isinstance(physics, dinv.physics.LinearPhysics):
        return physics.A(v)

    if hasattr(torch.func, "jvp"):
        return torch.func.jvp(lambda z: physics.A(z), (x,), (v,))[1]

    return torch.autograd.functional.jvp(
        lambda z: physics.A(z), x, v, create_graph=False
    )[1]


def _solve_gaussian_posterior_system(
    solver: str,
    max_iter: int,
    tol: float,
    verbose: bool,
    A,
    b: torch.Tensor,
) -> torch.Tensor:
    solver = solver.lower()
    if solver == "cg":
        return conjugate_gradient(
            A=A,
            b=b,
            max_iter=max_iter,
            tol=tol,
            parallel_dim=0,
            verbose=verbose,
        )
    if solver == "bicgstab":
        return bicgstab(
            A=A,
            b=b,
            max_iter=max_iter,
            tol=tol,
            parallel_dim=0,
            verbose=verbose,
        )
    raise ValueError("Unsupported solver. Choose between 'CG' and 'BiCGStab'.")


def _gaussian_posterior_batch_inner(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return (x.conj() * y).reshape(x.size(0), -1).sum(dim=1).real


def _gaussian_posterior_score(
    module: NoisyDataFidelity,
    method_name: str,
    measurement_covariance,
    x: torch.Tensor,
    y: torch.Tensor,
    physics: Physics,
    sigma,
    *args,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    with torch.enable_grad():
        input_dtype = x.dtype
        x = x.detach().to(torch.float32)
        sigma = _prepare_gaussian_posterior_sigma(sigma, x)
        x0_t, denoiser_vjp = torch.func.vjp(
            lambda z: _gaussian_posterior_denoise(
                module.denoiser, module.clip, z, sigma, *args, **kwargs
            ),
            x,
        )
        Ax0_t = physics.A(x0_t)
        residual = y.to(device=Ax0_t.device, dtype=Ax0_t.dtype) - Ax0_t
        cov_y = _gaussian_observation_covariance(
            module.cov_y, residual, physics, method_name
        )

        def A_vjp(v):
            return physics.A_vjp(x0_t, v)

        def A_jvp(v):
            return _gaussian_posterior_measurement_jvp(physics, x0_t, v)

        def cov_y_xt(v):
            return cov_y(v) + measurement_covariance(
                module,
                v,
                sigma=sigma,
                denoiser_vjp=denoiser_vjp,
                A_vjp=A_vjp,
                A_jvp=A_jvp,
            )

        v = _solve_gaussian_posterior_system(
            module.solver, module.max_iter, module.tol, module.verbose, cov_y_xt, residual
        )
        score = denoiser_vjp(A_vjp(v))[0]

    return score.to(input_dtype), x0_t, residual, v


class PiGDMDataFidelity(NoisyDataFidelity):
    r"""
    Pseudoinverse-guided diffusion model (PiGDM) data-fidelity term.

    For Gaussian observations :math:`p(y|x)=\mathcal{N}(y|A(x), \Sigma_y)` and a denoiser
    :math:`\hat{x}(x_t,\sigma_t)`, PiGDM uses the isotropic approximation
    :math:`p(x|x_t)\approx\mathcal{N}(\hat{x}(x_t,\sigma_t), \sigma_t^2 I)`. This yields

    .. math::

        \nabla_{x_t}\log p_t(y|x_t)
        \approx J_{\hat{x}}(x_t)^\top J_A(\hat{x})^\top
        \left(\Sigma_y + \sigma_t^2 J_A(\hat{x}) J_A(\hat{x})^\top\right)^{-1}
        (y - A(\hat{x})).

    For linear operators this reduces to the pseudoinverse-guided update from
    `Pseudoinverse-Guided Diffusion Models for Inverse Problems <https://openreview.net/forum?id=9_gsMA8MRKQ>`_.

    :param deepinv.models.Denoiser denoiser: Denoiser network.
    :param torch.Tensor, float, callable, None cov_y: observation covariance operator. If `None`,
        it is inferred from `physics.noise_model` when it is Gaussian or zero.
    :param float weight: Weighting factor applied to the guidance term.
    :param tuple[float], None clip: Optional interval used to clip the denoiser output.
    :param str solver: Linear solver used in measurement space. Choose between `'CG'` and `'BiCGStab'`.
    :param int max_iter: Number of linear-solver iterations.
    :param float tol: Relative solver tolerance.
    :param bool verbose: If `True`, print solver convergence information.
    """

    def __init__(
        self,
        denoiser: Denoiser = None,
        cov_y: torch.Tensor | float | Callable | None = None,
        weight: float = 1.0,
        clip: tuple | None = None,
        solver: str = "CG",
        max_iter: int = 1,
        tol: float = 1e-3,
        verbose: bool = False,
        *args,
        **kwargs,
    ):
        super().__init__()
        _init_gaussian_posterior_guidance(
            self, denoiser, cov_y, weight, clip, solver, max_iter, tol, verbose
        )

    def precond(
        self, x: torch.Tensor, physics: Physics, *args, **kwargs
    ) -> torch.Tensor:
        raise NotImplementedError

    def grad(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        physics: Physics,
        sigma,
        *args,
        get_model_outputs: bool = False,
        **kwargs,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        r"""
        Computes the PiGDM approximation of :math:`-\nabla_{x_t}\log p_t(y|x_t)`.

        :param torch.Tensor x: Current iterate.
        :param torch.Tensor y: Input data.
        :param deepinv.physics.Physics physics: physics model.
        :param float, torch.Tensor sigma: Standard deviation of the diffusion noise.
        :param bool get_model_outputs: If `True`, also return the denoised output along with the score.

        :return: (:class:`torch.Tensor` or tuple of :class:`torch.Tensor`) score term
            (and denoised output if `get_model_outputs` is `True`).
        """
        score, x0_t, _, _ = _gaussian_posterior_score(
            self,
            "PiGDM",
            PiGDMDataFidelity._measurement_covariance,
            x,
            y,
            physics,
            sigma,
            *args,
            **kwargs,
        )
        grad = -self.weight * score
        if get_model_outputs:
            return grad, x0_t
        return grad

    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        physics: Physics,
        sigma,
        *args,
        get_model_outputs: bool = False,
        **kwargs,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        r"""
        Returns the PiGDM surrogate data-fidelity term
        :math:`\frac{\lambda}{2}(y - A(\hat{x}))^\top (\Sigma_y + \Sigma_{y|x_t})^{-1}(y - A(\hat{x}))`.

        :param torch.Tensor x: Input image.
        :param torch.Tensor y: Measurements.
        :param deepinv.physics.Physics physics: forward operator.
        :param float, torch.Tensor sigma: Standard deviation of the diffusion noise.
        :param bool get_model_outputs: If `True`, also return the denoised output along with the loss.

        :return: (:class:`torch.Tensor` or tuple of :class:`torch.Tensor`) loss term
            (and denoised output if `get_model_outputs` is `True`).
        """
        _, x0_t, residual, v = _gaussian_posterior_score(
            self,
            "PiGDM",
            PiGDMDataFidelity._measurement_covariance,
            x,
            y,
            physics,
            sigma,
            *args,
            **kwargs,
        )
        out = 0.5 * self.weight * _gaussian_posterior_batch_inner(residual, v)
        if get_model_outputs:
            return out, x0_t
        return out

    def _measurement_covariance(
        self,
        v: torch.Tensor,
        sigma: torch.Tensor,
        A_vjp,
        A_jvp,
    ) -> torch.Tensor:
        sigma2 = _reshape_gaussian_posterior_param(sigma.square(), v)
        return sigma2 * A_jvp(A_vjp(v))


class MomentMatchingDataFidelity(NoisyDataFidelity):
    r"""
    Moment matching data-fidelity term for Gaussian observations.

    Moment matching refines PiGDM by matching the posterior covariance of the denoiser:

    .. math::

        p(x|x_t)\approx\mathcal{N}(\hat{x}(x_t,\sigma_t), \sigma_t^2 J_{\hat{x}}(x_t)).

    This yields the approximation

    .. math::

        \nabla_{x_t}\log p_t(y|x_t)
        \approx J_{\hat{x}}(x_t)^\top J_A(\hat{x})^\top
        \left(\Sigma_y + \sigma_t^2 J_A(\hat{x}) J_{\hat{x}}(x_t)^\top J_A(\hat{x})^\top\right)^{-1}
        (y - A(\hat{x})).

    This implementation follows the moment-matching posterior denoiser used by the authors of
    `Learning Diffusion Priors from Observations by Expectation Maximization <https://arxiv.org/abs/2405.13712>`_,
    using automatic differentiation to evaluate the required Jacobian-vector products.

    :param deepinv.models.Denoiser denoiser: Denoiser network.
    :param torch.Tensor, float, callable, None cov_y: observation covariance operator. If `None`,
        it is inferred from `physics.noise_model` when it is Gaussian or zero.
    :param float weight: Weighting factor applied to the guidance term.
    :param tuple[float], None clip: Optional interval used to clip the denoiser output.
    :param str solver: Linear solver used in measurement space. Choose between `'CG'` and `'BiCGStab'`.
    :param int max_iter: Number of linear-solver iterations.
    :param float tol: Relative solver tolerance.
    :param bool verbose: If `True`, print solver convergence information.
    """

    def __init__(
        self,
        denoiser: Denoiser = None,
        cov_y: torch.Tensor | float | Callable | None = None,
        weight: float = 1.0,
        clip: tuple | None = None,
        solver: str = "CG",
        max_iter: int = 1,
        tol: float = 1e-3,
        verbose: bool = False,
        *args,
        **kwargs,
    ):
        super().__init__()
        _init_gaussian_posterior_guidance(
            self, denoiser, cov_y, weight, clip, solver, max_iter, tol, verbose
        )

    def precond(
        self, x: torch.Tensor, physics: Physics, *args, **kwargs
    ) -> torch.Tensor:
        raise NotImplementedError

    def grad(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        physics: Physics,
        sigma,
        *args,
        get_model_outputs: bool = False,
        **kwargs,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        r"""
        Computes the moment-matching approximation of :math:`-\nabla_{x_t}\log p_t(y|x_t)`.

        :param torch.Tensor x: Current iterate.
        :param torch.Tensor y: Input data.
        :param deepinv.physics.Physics physics: physics model.
        :param float, torch.Tensor sigma: Standard deviation of the diffusion noise.
        :param bool get_model_outputs: If `True`, also return the denoised output along with the score.

        :return: (:class:`torch.Tensor` or tuple of :class:`torch.Tensor`) score term
            (and denoised output if `get_model_outputs` is `True`).
        """
        score, x0_t, _, _ = _gaussian_posterior_score(
            self,
            "Moment matching",
            MomentMatchingDataFidelity._measurement_covariance,
            x,
            y,
            physics,
            sigma,
            *args,
            **kwargs,
        )
        grad = -self.weight * score
        if get_model_outputs:
            return grad, x0_t
        return grad

    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        physics: Physics,
        sigma,
        *args,
        get_model_outputs: bool = False,
        **kwargs,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        r"""
        Returns the moment-matching surrogate data-fidelity term
        :math:`\frac{\lambda}{2}(y - A(\hat{x}))^\top (\Sigma_y + \Sigma_{y|x_t})^{-1}(y - A(\hat{x}))`.

        :param torch.Tensor x: Input image.
        :param torch.Tensor y: Measurements.
        :param deepinv.physics.Physics physics: forward operator.
        :param float, torch.Tensor sigma: Standard deviation of the diffusion noise.
        :param bool get_model_outputs: If `True`, also return the denoised output along with the loss.

        :return: (:class:`torch.Tensor` or tuple of :class:`torch.Tensor`) loss term
            (and denoised output if `get_model_outputs` is `True`).
        """
        _, x0_t, residual, v = _gaussian_posterior_score(
            self,
            "Moment matching",
            MomentMatchingDataFidelity._measurement_covariance,
            x,
            y,
            physics,
            sigma,
            *args,
            **kwargs,
        )
        out = 0.5 * self.weight * _gaussian_posterior_batch_inner(residual, v)
        if get_model_outputs:
            return out, x0_t
        return out

    def _measurement_covariance(
        self,
        v: torch.Tensor,
        sigma: torch.Tensor,
        denoiser_vjp,
        A_vjp,
        A_jvp,
    ) -> torch.Tensor:
        sigma2 = _reshape_gaussian_posterior_param(sigma.square(), v)
        return sigma2 * A_jvp(denoiser_vjp(A_vjp(v))[0])
