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
    """Accumulate X'X/X'Y without retaining calibration tokens.

    Statistics are mergeable, which permits sharded calibration and a final
    all-reduce in a distributed job. Float64 is the safe default for the solve.
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
        self.sum_x = torch.zeros(x_features, dtype=dtype, device=resolved_device)
        self.sum_y = torch.zeros(y_features, dtype=dtype, device=resolved_device)
        self.xtx = torch.zeros((x_features, x_features), dtype=dtype, device=resolved_device)
        self.xty = torch.zeros((x_features, y_features), dtype=dtype, device=resolved_device)
        self.yty = torch.zeros((), dtype=dtype, device=resolved_device)

    def update(self, x: Tensor, y: Tensor) -> None:
        if x.ndim != 2 or y.ndim != 2 or x.shape[0] != y.shape[0]:
            raise ValueError("x and y must be rank-2 with equal observation counts")
        if x.shape[1] != self.x_features or y.shape[1] != self.y_features:
            raise ValueError("feature count differs from accumulator configuration")
        x_acc = x.detach().to(device=self.device, dtype=self.dtype)
        y_acc = y.detach().to(device=self.device, dtype=self.dtype)
        self.count += x_acc.shape[0]
        self.sum_x += x_acc.sum(dim=0)
        self.sum_y += y_acc.sum(dim=0)
        self.xtx += x_acc.T @ x_acc
        self.xty += x_acc.T @ y_acc
        self.yty += (y_acc * y_acc).sum()

    def merge(self, other: RidgeAccumulator) -> RidgeAccumulator:
        if (self.x_features, self.y_features, self.dtype, self.device) != (
            other.x_features,
            other.y_features,
            other.dtype,
            other.device,
        ):
            raise ValueError("only like-shaped accumulators can be merged")
        self.count += other.count
        self.sum_x += other.sum_x
        self.sum_y += other.sum_y
        self.xtx += other.xtx
        self.xty += other.xty
        self.yty += other.yty
        return self

    def state_dict(self) -> dict[str, Tensor | int]:
        """Return mergeable state suitable for checkpointing or process exchange."""
        return {
            "count": self.count,
            "sum_x": self.sum_x.clone(),
            "sum_y": self.sum_y.clone(),
            "xtx": self.xtx.clone(),
            "xty": self.xty.clone(),
            "yty": self.yty.clone(),
        }

    def all_reduce_(self) -> RidgeAccumulator:
        """Sum sufficient statistics across an initialized torch.distributed group."""
        if not torch.distributed.is_available() or not torch.distributed.is_initialized():
            raise RuntimeError("torch.distributed must be initialized before all_reduce_")
        count = torch.tensor(self.count, dtype=torch.int64, device=self.device)
        torch.distributed.all_reduce(count)
        for tensor in (self.sum_x, self.sum_y, self.xtx, self.xty, self.yty):
            torch.distributed.all_reduce(tensor)
        self.count = int(count.item())
        return self

    def solve(self, alpha: float) -> RidgeSolution:
        if self.count < 2:
            raise ValueError("at least two observations are required")
        mean_x, mean_y = self.sum_x / self.count, self.sum_y / self.count
        centered_xtx = self.xtx - self.count * torch.outer(mean_x, mean_x)
        centered_xty = self.xty - self.count * torch.outer(mean_x, mean_y)
        system = centered_xtx + alpha * torch.eye(
            self.x_features, dtype=self.dtype, device=self.device
        )
        try:
            weight = torch.linalg.solve(system, centered_xty)
        except torch.linalg.LinAlgError:  # type: ignore[attr-defined]
            weight = torch.linalg.lstsq(system, centered_xty).solution
        bias = mean_y - mean_x @ weight

        centered_yty = self.yty - self.count * (mean_y * mean_y).sum()
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
