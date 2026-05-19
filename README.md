# A DSL for Optimizing Elementwise Computation in PyTorch

## Project Overview

This project implements a lightweight embedded DSL for explicitly defining fusion regions for elementwise tensor operations in PyTorch. The DSL integrates directly with torch.compile and allows users to annotate unary and binary elementwise operation chains for fused execution.

## Requirements

- Python 3
- PyTorch (with CUDA support)
- NVIDIA GPU
- Google Colab / Jupyter Notebook
- matplotlib
- numpy

## Files

**final_project.ipynb**: main notebook containing DSL implementation, benchmarks, experiments, and plots

**presentation.pdf**: final presentation slides

**figures/**: results folder

## Running the Project

1. Open the notebook in Google Colab or Jupyter
2. Enable GPU acceleration
3. Run all notebook cells in order
3. Runtime, memory, and ablation-study figures will be generated automatically

## Expected Results

Substantial runtime and memory improvements relative to eager execution, particularly for large tensors and longer operation chains:
- Up to 5x runtime speedup
- Significant reductions in peak GPU memory usage
- Increasing fusion benefits as operation chain length grows