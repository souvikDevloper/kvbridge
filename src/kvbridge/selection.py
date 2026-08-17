"""Vectorized, mergeable sufficient statistics for paper-scale layer selection.

The original implementation creates one Python ``RidgeAccumulator`` per
target-layer/source-layer/head/K-or-V tuple.  At paper geometry that means
thousands of small accumulators and repeated host-to-device transfers.  This
module stores the same centered statistics in batched tensors and updates an
entire target-layer block in one operation.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from kvbridge.ridge import RidgeAccumulator


@dataclass(frozen=True, slots=True)
class SelectionResult:
    """Single-source regression scores for a target-layer block."""

    r2: Tensor
    observations: int

    def topk(self, count: int) -> tuple[Tensor, Tensor]:
        """Return descending scores and source indices, averaged over heads."""
        if count <= 0:
            raise ValueError("top-k count must be positive")
        source_layers = self.r2.shape[1]
        count = min(count, source_layers)
        layer_scores = self.r2.mean(dim=-1)
        return torch.topk(layer_scores, k=count, dim=1, largest=True, sorted=True)


class BatchedSelectionAccumulator:
    """Centered ridge statistics for ``[target, source, head]`` regressions.

    Input tensors use ``[layer, head, observation, feature]`` layout.  Source
    covariance is shared across every target layer, and target variance is
    shared across every candidate source layer.  That removes substantial
    duplicated state while remaining algebraically identical to independent
    per-head ridge regressions.
    """

    def __init__(
        self,
        source_layers: int,
        target_layers: int,
        heads: int,
        x_features: int,
        y_features: int,
        *,
        dtype: torch.dtype = torch.float32,
        device: str | torch.device = "cpu",
    ) -> None:
        dimensions = (source_layers, target_layers, heads, x_features, y_features)
        if min(dimensions) <= 0:
            raise ValueError("selection dimensions must be positive")
        self.source_layers = source_layers
        self.target_layers = target_layers
        self.heads = heads
        self.x_features = x_features
        self.y_features = y_features
        self.dtype = dtype
        self.device = torch.device(device)
        if self.device.type not in {"cpu", "cuda", "xla"}:
            raise ValueError("selection accumulation device must be cpu, cuda, or xla")
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA selection accumulation requested but CUDA is unavailable")
        if self.device.type == "xla":
            try:
                import torch_xla  # type: ignore[import-not-found, unused-ignore]  # noqa: F401
            except ImportError as error:
                raise RuntimeError(
                    "XLA selection accumulation requested but torch_xla is unavailable"
                ) from error

        self.count = 0
        self.mean_x = torch.zeros(
            (source_layers, heads, x_features), dtype=dtype, device=self.device
        )
        self.mean_y = torch.zeros(
            (target_layers, heads, y_features), dtype=dtype, device=self.device
        )
        self.centered_xtx = torch.zeros(
            (source_layers, heads, x_features, x_features), dtype=dtype, device=self.device
        )
        self.centered_xty = torch.zeros(
            (target_layers, source_layers, heads, x_features, y_features),
            dtype=dtype,
            device=self.device,
        )
        self.centered_yty = torch.zeros(
            (target_layers, heads), dtype=dtype, device=self.device
        )

    def update(self, source: Tensor, target: Tensor) -> None:
        """Merge an aligned observation batch without retaining token rows."""
        expected_source = (self.source_layers, self.heads, self.x_features)
        expected_target = (self.target_layers, self.heads, self.y_features)
        if source.ndim != 4 or (
            source.shape[0], source.shape[1], source.shape[3]
        ) != expected_source:
            raise ValueError(
                "source must have shape [source_layers, heads, observations, x_features]"
            )
        if target.ndim != 4 or (
            target.shape[0], target.shape[1], target.shape[3]
        ) != expected_target:
            raise ValueError(
                "target must have shape [target_layers, heads, observations, y_features]"
            )
        if source.shape[2] != target.shape[2]:
            raise ValueError("source and target observation counts must match")
        count = int(source.shape[2])
        if count == 0:
            return

        x = source.detach().to(device=self.device, dtype=self.dtype)
        y = target.detach().to(device=self.device, dtype=self.dtype)
        mean_x = x.mean(dim=2)
        mean_y = y.mean(dim=2)
        centered_x = x - mean_x.unsqueeze(2)
        centered_y = y - mean_y.unsqueeze(2)
        batch_xtx = torch.einsum("shnd,shne->shde", centered_x, centered_x)
        batch_xty = torch.einsum("shnd,thne->tshde", centered_x, centered_y)
        batch_yty = torch.sum(centered_y * centered_y, dim=(2, 3))

        if self.count == 0:
            self.count = count
            self.mean_x.copy_(mean_x)
            self.mean_y.copy_(mean_y)
            self.centered_xtx.copy_(batch_xtx)
            self.centered_xty.copy_(batch_xty)
            self.centered_yty.copy_(batch_yty)
            return

        previous = self.count
        total = previous + count
        correction = previous * count / total
        delta_x = mean_x - self.mean_x
        delta_y = mean_y - self.mean_y
        self.centered_xtx += batch_xtx + correction * torch.einsum(
            "shd,she->shde", delta_x, delta_x
        )
        self.centered_xty += batch_xty + correction * torch.einsum(
            "shd,the->tshde", delta_x, delta_y
        )
        self.centered_yty += batch_yty + correction * torch.sum(
            delta_y * delta_y, dim=-1
        )
        fraction = count / total
        self.mean_x += delta_x * fraction
        self.mean_y += delta_y * fraction
        self.count = total

    def state_dict(self) -> dict[str, Tensor]:
        """Return SafeTensors-compatible mergeable checkpoint state."""
        return {
            "count": torch.tensor(self.count, dtype=torch.int64, device=self.device),
            "mean_x": self.mean_x.clone(),
            "mean_y": self.mean_y.clone(),
            "centered_xtx": self.centered_xtx.clone(),
            "centered_xty": self.centered_xty.clone(),
            "centered_yty": self.centered_yty.clone(),
        }

    def load_state_dict(self, state: dict[str, Tensor]) -> None:
        """Restore a shape-checked state without accepting arbitrary pickles."""
        expected = {
            "mean_x": self.mean_x,
            "mean_y": self.mean_y,
            "centered_xtx": self.centered_xtx,
            "centered_xty": self.centered_xty,
            "centered_yty": self.centered_yty,
        }
        missing = ({"count"} | set(expected)) - set(state)
        if missing:
            raise ValueError(f"selection checkpoint is missing tensors: {sorted(missing)}")
        count_tensor = state["count"]
        if count_tensor.numel() != 1:
            raise ValueError("selection checkpoint count must be scalar")
        count = int(count_tensor.item())
        if count < 0:
            raise ValueError("selection checkpoint count cannot be negative")
        for name, destination in expected.items():
            value = state[name]
            if value.shape != destination.shape:
                raise ValueError(f"selection checkpoint {name} has the wrong shape")
            destination.copy_(value.to(device=self.device, dtype=self.dtype))
        self.count = count

    def solve(self, alpha: float) -> SelectionResult:
        """Solve every candidate regression and return ``[target, source, head]`` R²."""
        if self.count < 2:
            raise ValueError("at least two observations are required")
        if alpha < 0:
            raise ValueError("ridge penalty cannot be negative")
        for name, value in self.state_dict().items():
            if name != "count" and not bool(torch.isfinite(value).all().item()):
                raise FloatingPointError(f"selection {name} statistics are non-finite")

        identity = torch.eye(self.x_features, dtype=self.dtype, device=self.device)
        covariance = (self.centered_xtx + self.centered_xtx.transpose(-1, -2)) * 0.5
        system = covariance + alpha * identity
        # [target, source, head, x, y] -> [source, head, x, target*y].
        # A source/head system is shared by every target layer, so solve one
        # multi-RHS factorization rather than repeating it per target.
        right_hand_side = (
            self.centered_xty.permute(1, 2, 3, 0, 4)
            .reshape(
                self.source_layers,
                self.heads,
                self.x_features,
                self.target_layers * self.y_features,
            )
            .contiguous()
        )
        diagonal = torch.diagonal(system, dim1=-2, dim2=-1)
        if bool((diagonal <= 0).any().item()):
            raise FloatingPointError("selection ridge system has a non-positive diagonal")
        scale = torch.sqrt(diagonal)
        equilibrated = system / scale.unsqueeze(-1) / scale.unsqueeze(-2)
        equilibrated = (equilibrated + equilibrated.transpose(-1, -2)) * 0.5
        scaled_rhs = right_hand_side / scale.unsqueeze(-1)
        factor, info = torch.linalg.cholesky_ex(equilibrated)
        if int(info.max().item()) == 0:
            weights = torch.cholesky_solve(scaled_rhs, factor) / scale.unsqueeze(-1)
        else:
            # Ridge should normally keep every system positive definite.  If a
            # backend factorization rejects one, recover the bounded selection
            # block on CPU FP64 without changing its sufficient statistics.
            recovered = []
            for source_layer in range(self.source_layers):
                head_solutions = []
                for head in range(self.heads):
                    head_solutions.append(
                        RidgeAccumulator._equilibrated_solve(
                            system[source_layer, head].detach().to("cpu", torch.float64),
                            right_hand_side[source_layer, head]
                            .detach()
                            .to("cpu", torch.float64),
                        )
                    )
                recovered.append(torch.stack(head_solutions))
            weights = torch.stack(recovered).to(device=self.device, dtype=self.dtype)
        if not bool(torch.isfinite(weights).all().item()):
            raise FloatingPointError("selection ridge weights are non-finite")
        weights = weights.reshape(
            self.source_layers,
            self.heads,
            self.x_features,
            self.target_layers,
            self.y_features,
        ).permute(0, 1, 3, 2, 4)
        cross = self.centered_xty.permute(1, 2, 0, 3, 4)
        predicted_quadratic = torch.matmul(covariance.unsqueeze(2), weights)
        residual = (
            self.centered_yty.T.unsqueeze(0)
            - 2 * torch.sum(weights * cross, dim=(-2, -1))
            + torch.sum(weights * predicted_quadratic, dim=(-2, -1))
        )
        eps = torch.finfo(self.dtype).eps
        total = self.centered_yty.T.unsqueeze(0)
        scores = torch.where(
            total <= eps,
            torch.where(residual <= eps, 1.0, 0.0),
            1.0 - residual / total,
        ).permute(2, 0, 1)
        if not bool(torch.isfinite(scores).all().item()):
            raise FloatingPointError("selection R2 is non-finite")
        return SelectionResult(scores, self.count)
