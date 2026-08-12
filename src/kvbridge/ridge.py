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

    def solve(self, alpha: float) -> RidgeSolution:
        if self.count < 2:
            raise ValueError("at least two observations are required")
        if alpha < 0:
            raise ValueError("ridge penalty cannot be negative")
        centered_xtx = (self.centered_xtx + self.centered_xtx.T) * 0.5
        centered_xty = self.centered_xty
        system = centered_xtx + alpha * torch.eye(
            self.x_features, dtype=self.dtype, device=self.device
        )
        factor, info = torch.linalg.cholesky_ex(system)
        if int(info.max().item()) == 0:
            weight = torch.cholesky_solve(centered_xty, factor)
        else:
            weight = torch.linalg.lstsq(system, centered_xty).solution
        bias = self.mean_y - self.mean_x @ weight

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
        return RidgeSolution(weight, bias, r2, self.count)
