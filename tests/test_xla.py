from __future__ import annotations

import torch

from kvbridge.xla import _largest_axis_partition_spec


def test_largest_axis_partition_spec() -> None:
    assert _largest_axis_partition_spec(torch.empty(())) == ()
    assert _largest_axis_partition_spec(torch.empty((3, 7))) == (None, "fsdp")
    assert _largest_axis_partition_spec(torch.empty((8, 8))) == ("fsdp", None)
