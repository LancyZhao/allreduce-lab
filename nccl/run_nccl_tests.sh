#!/usr/bin/env bash
# Run NVIDIA nccl-tests all_reduce_perf and save output for parse_nccl.py.
# Requires a GPU box with NCCL + nccl-tests built (https://github.com/NVIDIA/nccl-tests).
#
#   ./nccl/run_nccl_tests.sh 2 > nccl/my_output.txt
#   python nccl/parse_nccl.py nccl/my_output.txt
set -euo pipefail

NGPUS="${1:-2}"
PERF_BIN="${NCCL_PERF_BIN:-./build/all_reduce_perf}"   # path to nccl-tests binary

# -b min bytes, -e max bytes, -f step factor, -g #GPUs, -c check correctness
exec "$PERF_BIN" -b 1K -e 256M -f 2 -g "$NGPUS" -c 1
