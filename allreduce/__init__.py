"""allreduce-lab: a CPU, dependency-free lab for collective-communication
algorithms (ring / tree / naive all-reduce).

The simulator models the *data movement and step count* of each algorithm and
verifies both correctness (every rank ends with the elementwise sum) and the
textbook communication-volume formulas. Real-hardware bandwidth/latency comes
from NVIDIA `nccl-tests` (see the `nccl/` directory); this package explains and
reproduces the *why* behind those numbers.
"""

__version__ = "0.1.0"
