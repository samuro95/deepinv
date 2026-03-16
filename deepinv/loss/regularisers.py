import torch
from deepinv.loss.loss import Loss


class JacobianSpectralNorm(Loss):
    def __init__(
        self,
        max_iter: int = 10,
        tol: float = 1e-3,
        eval_mode: bool = False,
        verbose: bool = False,
        reduction: str = "max",
        reduced_batchsize: int = None,
        eps: float = 1e-10,
    ):
        super().__init__()
        self.name = "jsn"
        self.max_iter = max_iter
        self.tol = tol
        self.eval = eval_mode
        self.verbose = verbose
        self.reduced_batchsize = reduced_batchsize
        self.eps = eps

        self.reduction = lambda x: x
        if reduction is not None:
            if not isinstance(reduction, str):
                raise ValueError("Reduction should be a string or None.")
            elif reduction.lower() == "mean":
                self.reduction = lambda x: torch.mean(x)
            elif reduction.lower() == "sum":
                self.reduction = lambda x: torch.sum(x)
            elif reduction.lower() == "max":
                self.reduction = lambda x: torch.max(x)
            elif reduction.lower() == "none":
                pass
            else:
                raise ValueError(
                    'Reduction should be "mean", "sum", "max", "none" or None.'
                )

    def _safe_norm(self, t):
        return torch.linalg.vector_norm(
            t, dim=tuple(range(1, t.dim())), keepdim=True
        ).clamp_min(self.eps)

    def _reduce_batch(self, x, y):
        if self.reduced_batchsize is not None:
            x = x[: self.reduced_batchsize]
            y = y[: self.reduced_batchsize]
        return x, y

    def forward(self, y, x, **kwargs):
        x, y = self._reduce_batch(x, y)

        assert x.shape[0] == y.shape[0], (
            f"x and y should have the same number of instances. "
            f"Got {x.shape[0]} vs. {y.shape[0]}"
        )
        u = torch.randn_like(x)
        u = u / self._safe_norm(u)

        zold = None

        for it in range(self.max_iter):
            w = torch.ones_like(y, requires_grad=True)

            jvp = torch.autograd.grad(
                y, x, w, create_graph=True, retain_graph=True
            )[0]

            v = torch.autograd.grad(
                jvp, w, u, create_graph=not self.eval, retain_graph=True
            )[0]  # J u

            (jtj_u,) = torch.autograd.grad(
                y, x, v, retain_graph=True, create_graph=True
            )  # J^T J u

            u_flat = u.flatten(1)
            v_flat = jtj_u.flatten(1)

            denom = (u_flat.pow(2).sum(dim=-1)).clamp_min(self.eps)
            z = (u_flat * v_flat).sum(dim=-1) / denom

            # Prevent tiny negative values from roundoff
            z = z.clamp_min(0.0)

            if zold is not None:
                rel_var = torch.linalg.vector_norm(z - zold)
                if rel_var < self.tol:
                    if self.verbose:
                        print(
                            f"Power iteration converged at iter {it}, "
                            f"val={z.sqrt().tolist()}, relvar={rel_var.item()}"
                        )
                    break

            zold = z.detach().clone()

            u = jtj_u / self._safe_norm(jtj_u)

            if self.eval:
                w = w.detach()
                v = v.detach()
                u = u.detach()

        out = torch.sqrt(z.clamp_min(self.eps))
        out = torch.nan_to_num(out, nan=float("inf"), posinf=float("inf"), neginf=0.0)
        return self.reduction(out.view(-1))

class FNEJacobianSpectralNorm(Loss):
    r"""
    Computes the Firm-Nonexpansiveness Jacobian spectral norm.

    Given a function :math:`f:\mathbb{R}^n\to\mathbb{R}^n`, this module computes the spectral
    norm of the Jacobian of :math:`2f-\operatorname{Id}` (where :math:`\operatorname{Id}` denotes the
    identity) in :math:`x`, i.e.

    .. math::

        \|\frac{d(2f-\operatorname{Id})}{du}(x)\|_2,

    as proposed by :footcite:t:`pesquet2021learning`.
    This spectral norm is computed with the :class:`deepinv.loss.JacobianSpectralNorm` class.

    .. note::

        This implementation assumes that the input :math:`x` is batched with shape `(B, ...)`, where B is the batch size.

    :param int max_iter: maximum numer of iteration of the power method.
    :param float tol: tolerance for the convergence of the power method.
    :param bool eval_mode: set to ``False`` if one does not want to backpropagate through the spectral norm (default), set to ``True`` otherwise.
    :param bool verbose: whether to print computation details or not.
    :param str reduction: reduction in batch dimension. One of ["mean", "sum", "max"], operation to be performed after all spectral norms have been computed. If ``None``, a vector of length ``batch_size`` will be returned. Defaults to "max".
    :param int reduced_batchsize: if not `None`, the batch size will be reduced to this value for the computation of the spectral norm. Can be useful to reduce memory usage and computation time when the batch size is large.

    |sep|

    :Examples:

    .. doctest::

        >>> import torch
        >>> from deepinv.loss.regularisers import FNEJacobianSpectralNorm
        >>> _ = torch.manual_seed(0)
        >>>
        >>> reg_fne = FNEJacobianSpectralNorm(max_iter=100, tol=1e-5, eval_mode=False, verbose=True)
        >>> A = torch.diag(torch.Tensor(range(1, 51))).unsqueeze(0)  # creates a diagonal matrix with largest eigenvalue = 50
        >>>
        >>> def model_base(x):
        ...     return x @ A
        >>>
        >>> def FNE_model(x):
        ...     A_bis = torch.linalg.inv((A + torch.eye(A.shape[1])))  # Creates the resolvent of A, which is firmly nonexpansive
        ...     return x @ A_bis
        >>>
        >>> x = torch.randn((1, A.shape[1])).unsqueeze(0)
        >>>
        >>> out = model_base(x)
        >>> regval = reg_fne(out, x, model_base)
        >>> print(regval) # returns approx 99 (model is expansive, with Lipschitz constant 50)
        tensor(98.9999)
        >>> out = FNE_model(x)
        >>> regval = reg_fne(out, x, FNE_model)
        >>> print(regval) # returns a value smaller than 1 (model is firmly nonexpansive)
        tensor(0.9595)
    """

    def __init__(
        self,
        max_iter: int = 10,
        tol: float = 1e-3,
        eval_mode: bool = False,
        verbose: bool = False,
        reduction: str = "max",
        reduced_batchsize: int = None,
    ):
        super(FNEJacobianSpectralNorm, self).__init__()

        self.reduced_batchsize = reduced_batchsize

        self.spectral_norm_module = JacobianSpectralNorm(
            max_iter=max_iter,
            tol=tol,
            verbose=verbose,
            eval_mode=eval_mode,
            reduction=reduction,
            reduced_batchsize=reduced_batchsize,
        )

    def _reduce_batch(self, x, y):
        """
        Reduces the batch dimension of the input tensors x and y.
        """
        if self.reduced_batchsize is not None:
            x = x[: self.reduced_batchsize]
            y = y[: self.reduced_batchsize]
        return x, y

    def forward(
        self, y_in, x_in, model, *args_model, interpolation=False, **kwargs_model
    ):
        r"""
        Computes the Firm-Nonexpansiveness (FNE) Jacobian spectral norm of a model.

        :param torch.Tensor y_in: input of the model (by default), of dimension `(B, ...)`.
        :param torch.Tensor x_in: an additional point of the model (by default), of dimension `(B, ...)`.
        :param torch.nn.Module model: neural network, or function, of which we want to compute the FNE Jacobian spectral norm.
        :param `*args_model`: additional arguments of the model.
        :param bool interpolation: whether to input to model an interpolation between y_in and x_in instead of y_in (default is `False`).
        :param `**kargs_model`: additional keyword arguments of the model.
        """

        y_in, x_in = self._reduce_batch(y_in, x_in)

        if interpolation:
            eta = torch.rand(
                (y_in.size(0),) + (1,) * (y_in.dim() - 1), requires_grad=True
            ).to(y_in.device)
            x = eta * y_in.detach() + (1 - eta) * x_in.detach()
        else:
            x = y_in

        x.requires_grad_()
        x_out = model(x, *args_model, **kwargs_model)

        y = 2.0 * x_out - x

        return self.spectral_norm_module(y, x)
