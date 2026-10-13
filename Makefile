.PHONY: dev test lint cli regime nccl

dev:            ## install dev extras (pytest + ruff)
	python -m pip install -e ".[dev]"

test:           ## run the test suite (no GPU needed)
	pytest -q

lint:           ## static checks
	ruff check allreduce nccl tests

cli:            ## compare algorithms (8 ranks, 1M elems)
	python -m allreduce.cli --world-size 8 --numel 1048576

regime:         ## latency-vs-bandwidth sweep across message sizes
	python -m allreduce.cli --world-size 16 --regime

nccl:           ## parse the committed nccl-tests sample
	python nccl/parse_nccl.py nccl/sample_output.txt
