# HdcMamba

HdcMamba is a high-performance, memory-efficient implementation of the Mamba architecture using custom Triton kernels.

## Key Features
- **Parallel Chunked Scan**: Fast execution on GPUs with high occupancy.
- **Extreme Memory Optimization**: Zero intermediate buffer strategy for $O(D^2)$ memory usage instead of $O(L)$.
- **Full Triton Implementation**: Minimal Python overhead and no dependence on slow PyTorch compile.
- **Fused Kernels**: LayerNorm + Conv1d, SSM, Slot, and Output gating are all fused into efficient Triton kernels.

## Performance
HdcMamba-9v3 achieves significantly higher throughput than Transformer models with FlashAttention-2, especially for long sequences (e.g., 3.37x faster at L=32768).

## Installation
```bash
pip install -e .
```

## Benchmarking
To run the standard benchmarks:
```bash
python benchmark.py
```

## Training
Basic training example:
```bash
python train.py
```
