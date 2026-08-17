from typing import Any, cast

from experiments.run_tpu_scale import _logical_to_host


class _RecordingTensor:
    def __init__(self, events: list[str], *, rows: int = 8) -> None:
        self.events = events
        self.shape = (rows, 4)

    def detach(self) -> "_RecordingTensor":
        self.events.append("detach")
        return self

    def to(self, device: str) -> "_RecordingTensor":
        self.events.append(f"to:{device}")
        return self

    def __getitem__(self, item: object) -> "_RecordingTensor":
        self.events.append(f"slice:{item!r}")
        return _RecordingTensor(self.events, rows=4)

    def clone(self) -> "_RecordingTensor":
        self.events.append("clone")
        return self


def test_logical_to_host_trims_only_after_transfer() -> None:
    events: list[str] = []

    result = _logical_to_host(cast(Any, _RecordingTensor(events)), 4)

    assert result.shape[0] == 4
    assert events == ["detach", "to:cpu", "slice:slice(None, 4, None)", "clone"]


def test_logical_to_host_avoids_copy_for_full_batch() -> None:
    events: list[str] = []
    tensor = _RecordingTensor(events)

    result = _logical_to_host(cast(Any, tensor), 8)

    assert result is tensor
    assert events == ["detach", "to:cpu"]
