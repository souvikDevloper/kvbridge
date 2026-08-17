"""Numerically stable, mergeable sufficient statistics for closed-form ridge."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True, slots=True)
class RidgeSolution:
    weight: Tensor
    bias: Tensor
    r2: float
    observations: int


class RidgeAccumulator:
    """Accumulate centered covariance statistics without retaining tokens.

    Statistics are mergeable, which permits sharded calibration and a final
    all-reduce in a distributed job. Batchwise Chan updates avoid the
    catastrophic cancellation caused by ``X.T @ X - n * outer(mean, mean)``.
    Float64 remains the safe default for especially ill-conditioned solves.
    """

    def __init__(
        self,
        x_features: int,
        y_features: int,
        *,
        dtype: torch.dtype = torch.float64,
        device: str | torch.device = "cpu",
    ):
        if x_features <= 0 or y_features <= 0:
            raise ValueError("feature dimensions must be positive")
        resolved_device = torch.device(device)
        if resolved_device.type not in {"cpu", "cuda"}:
            raise ValueError("ridge accumulation device must be cpu or cuda")
        if resolved_device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA ridge accumulation requested but CUDA is unavailable")
        self.x_features = x_features
        self.y_features = y_features
        self.dtype = dtype
        self.device = resolved_device
        self.count = 0
        self.mean_x = torch.zeros(x_features, dtype=dtype, device=resolved_device)
        self.mean_y = torch.zeros(y_features, dtype=dtype, device=resolved_device)
        self.centered_xtx = torch.zeros(
            (x_features, x_features), dtype=dtype, device=resolved_device
        )
        self.centered_xty = torch.zeros(
            (x_features, y_features), dtype=dtype, device=resolved_device
        )
        self.centered_yty = torch.zeros((), dtype=dtype, device=resolved_device)

    def _merge_statistics(
        self,
        count: int,
        mean_x: Tensor,
        mean_y: Tensor,
        centered_xtx: Tensor,
        centered_xty: Tensor,
        centered_yty: Tensor,
    ) -> None:
        if count <= 0:
            return
        if self.count == 0:
            self.count = count
            self.mean_x.copy_(mean_x)
            self.mean_y.copy_(mean_y)
            self.centered_xtx.copy_(centered_xtx)
            self.centered_xty.copy_(centered_xty)
            self.centered_yty.copy_(centered_yty)
            return

        previous = self.count
        total = previous + count
        delta_x = mean_x - self.mean_x
        delta_y = mean_y - self.mean_y
        correction = previous * count / total
        self.centered_xtx += centered_xtx + correction * torch.outer(delta_x, delta_x)
        self.centered_xty += centered_xty + correction * torch.outer(delta_x, delta_y)
        self.centered_yty += centered_yty + correction * torch.sum(delta_y * delta_y)
        self.mean_x += delta_x * (count / total)
        self.mean_y += delta_y * (count / total)
        self.count = total

    def update(self, x: Tensor, y: Tensor) -> None:
        if x.ndim != 2 or y.ndim != 2 or x.shape[0] != y.shape[0]:
            raise ValueError("x and y must be rank-2 with equal observation counts")
        if x.shape[1] != self.x_features or y.shape[1] != self.y_features:
            raise ValueError("feature count differs from accumulator configuration")
        x_acc = x.detach().to(device=self.device, dtype=self.dtype)
        y_acc = y.detach().to(device=self.device, dtype=self.dtype)
        count = x_acc.shape[0]
        if count == 0:
            return
        mean_x = x_acc.mean(dim=0)
        mean_y = y_acc.mean(dim=0)
        centered_x = x_acc - mean_x
        centered_y = y_acc - mean_y
        self._merge_statistics(
            count,
            mean_x,
            mean_y,
            centered_x.T @ centered_x,
            centered_x.T @ centered_y,
            (centered_y * centered_y).sum(),
        )

    def merge(self, other: RidgeAccumulator) -> RidgeAccumulator:
        if (self.x_features, self.y_features, self.dtype, self.device) != (
            other.x_features,
            other.y_features,
            other.dtype,
            other.device,
        ):
            raise ValueError("only like-shaped accumulators can be merged")
        if other is self:
            self._merge_statistics(
                other.count,
                other.mean_x.clone(),
                other.mean_y.clone(),
                other.centered_xtx.clone(),
                other.centered_xty.clone(),
                other.centered_yty.clone(),
            )
            return self
        self._merge_statistics(
            other.count,
            other.mean_x,
            other.mean_y,
            other.centered_xtx,
            other.centered_xty,
            other.centered_yty,
        )
        return self

    def state_dict(self) -> dict[str, Tensor | int]:
        """Return mergeable state suitable for checkpointing or process exchange."""
        return {
            "schema_version": 2,
            "count": self.count,
            "mean_x": self.mean_x.clone(),
            "mean_y": self.mean_y.clone(),
            "centered_xtx": self.centered_xtx.clone(),
            "centered_xty": self.centered_xty.clone(),
            "centered_yty": self.centered_yty.clone(),
        }

    def all_reduce_(self) -> RidgeAccumulator:
        """Sum sufficient statistics across an initialized torch.distributed group."""
        if not torch.distributed.is_available() or not torch.distributed.is_initialized():
            raise RuntimeError("torch.distributed must be initialized before all_reduce_")
        local_count = self.count
        count = torch.tensor(local_count, dtype=torch.int64, device=self.device)
        weighted_mean_x = self.mean_x * local_count
        weighted_mean_y = self.mean_y * local_count
        torch.distributed.all_reduce(count)
        torch.distributed.all_reduce(weighted_mean_x)
        torch.distributed.all_reduce(weighted_mean_y)
        global_count = int(count.item())
        if global_count == 0:
            return self
        global_mean_x = weighted_mean_x / global_count
        global_mean_y = weighted_mean_y / global_count

        # Parallel Chan merge: correct each rank's centered moments to the
        # global mean, then sum fixed-size tensors. This avoids all-gathering
        # an O(features^2) covariance matrix from every rank.
        delta_x = self.mean_x - global_mean_x
        delta_y = self.mean_y - global_mean_y
        global_xtx = self.centered_xtx + local_count * torch.outer(delta_x, delta_x)
        global_xty = self.centered_xty + local_count * torch.outer(delta_x, delta_y)
        global_yty = self.centered_yty + local_count * torch.sum(delta_y * delta_y)
        for tensor in (global_xtx, global_xty, global_yty):
            torch.distributed.all_reduce(tensor)
        self.count = global_count
        self.mean_x.copy_(global_mean_x)
        self.mean_y.copy_(global_mean_y)
        self.centered_xtx.copy_(global_xtx)
        self.centered_xty.copy_(global_xty)
        self.centered_yty.copy_(global_yty)
        return self

    @staticmethod
    def _equilibrated_solve(system: Tensor, right_hand_side: Tensor) -> Tensor:
        """Solve a ridge system after exact symmetric diagonal equilibration.

        KV features can differ by several orders of magnitude across heads and
        layers.  Solving the raw normal equations in FP32 can therefore overflow
        even when both sufficient statistics are finite.  If ``D`` contains the
        square root of the system diagonal, this routine solves

        ``(D^-1 A D^-1) Z = D^-1 B`` and returns ``D^-1 Z``.

        The transformation is algebraically equivalent to ``A W = B``; it does
        not change the configured ridge penalty.
        """
        diagonal = torch.diagonal(system)
        if not bool(torch.isfinite(diagonal).all().item()):
            raise FloatingPointError("ridge system diagonal is non-finite")
        if bool((diagonal <= 0).any().item()):
            raise FloatingPointError("ridge system is not positive on its diagonal")
        scale = torch.sqrt(diagonal)
        equilibrated = system / scale[:, None] / scale[None, :]
        equilibrated = (equilibrated + equilibrated.T) * 0.5
        scaled_rhs = right_hand_side / scale[:, None]
        factor, info = torch.linalg.cholesky_ex(equilibrated)
        if int(info.max().item()) == 0:
            scaled_solution = torch.cholesky_solve(scaled_rhs, factor)
        else:
            scaled_solution = torch.linalg.lstsq(equilibrated, scaled_rhs).solution
        return scaled_solution / scale[:, None]

    @classmethod
    def _solve_with_recovery(cls, system: Tensor, right_hand_side: Tensor) -> Tensor:
        """Solve on the configured device, recovering in CPU FP64 if needed."""
        weight: Tensor | None
        try:
            weight = cls._equilibrated_solve(system, right_hand_side)
        except RuntimeError:
            weight = None
        if weight is not None and bool(torch.isfinite(weight).all().item()):
            return weight

        # A CUDA FP32 factorization may complete without an error code yet
        # still overflow on a severely conditioned calibration system.  The
        # recovery is intentionally rare and bounded to one target layer.  It
        # preserves the same already-accumulated system and configured alpha.
        recovered = cls._equilibrated_solve(
            system.detach().to(device="cpu", dtype=torch.float64),
            right_hand_side.detach().to(device="cpu", dtype=torch.float64),
        )
        recovered = recovered.to(device=system.device, dtype=system.dtype)
        if not bool(torch.isfinite(recovered).all().item()):
            raise FloatingPointError(
                "ridge solve remained non-finite after equilibrated CPU FP64 recovery"
            )
        return recovered

    def solve(self, alpha: float) -> RidgeSolution:
        if self.count < 2:
            raise ValueError("at least two observations are required")
        if alpha < 0:
            raise ValueError("ridge penalty cannot be negative")
        centered_xtx = (self.centered_xtx + self.centered_xtx.T) * 0.5
        centered_xty = self.centered_xty
        for name, tensor in (
            ("centered XTX", centered_xtx),
            ("centered XTY", centered_xty),
            ("centered YTY", self.centered_yty),
            ("mean X", self.mean_x),
            ("mean Y", self.mean_y),
        ):
            if not bool(torch.isfinite(tensor).all().item()):
                raise FloatingPointError(f"ridge {name} statistics are non-finite")
        system = centered_xtx + alpha * torch.eye(
            self.x_features, dtype=self.dtype, device=self.device
        )
        weight = self._solve_with_recovery(system, centered_xty)
        bias = self.mean_y - self.mean_x @ weight
        if not bool(torch.isfinite(bias).all().item()):
            raise FloatingPointError("ridge bias is non-finite")

        centered_yty = self.centered_yty
        residual = (
            centered_yty
            - 2 * torch.sum(weight * centered_xty)
            + torch.sum(weight * (centered_xtx @ weight))
        )
        if centered_yty <= torch.finfo(self.dtype).eps:
            r2 = 1.0 if residual <= torch.finfo(self.dtype).eps else 0.0
        else:
            r2 = float((1.0 - residual / centered_yty).item())
        if not torch.isfinite(torch.tensor(r2)):
            raise FloatingPointError("ridge R2 is non-finite")
        return RidgeSolution(weight, bias, r2, self.count)
