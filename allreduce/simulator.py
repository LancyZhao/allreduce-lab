"""Communication accounting shared by all algorithms."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class CommStats:
    algo: str
    world_size: int
    numel: int
    elem_bytes: int = 4
    steps: int = 0                                  # latency hops on the critical path
    sent_bytes: list[int] = field(default_factory=list)   # bytes SENT by each rank

    def rank_send(self, rank: int, nbytes: int) -> None:
        if not self.sent_bytes:
            self.sent_bytes = [0] * self.world_size
        self.sent_bytes[rank] += nbytes

    @property
    def total_bytes(self) -> int:
        return sum(self.sent_bytes)

    @property
    def peak_rank_bytes(self) -> int:
        """Bytes sent by the busiest rank — captures a root hotspot."""
        return max(self.sent_bytes) if self.sent_bytes else 0

    def as_dict(self) -> dict[str, object]:
        return {
            "algo": self.algo, "world_size": self.world_size, "numel": self.numel,
            "steps": self.steps, "total_bytes": self.total_bytes,
            "peak_rank_bytes": self.peak_rank_bytes,
        }
