# PyTorch Fundamentals

A structured introduction to PyTorch deep learning, built as preparation for the Energy Anomaly Detector project and MSc in Data Science.

## What this is
Three progressive notebooks covering PyTorch from tensors to a working anomaly detection system using autoencoders.

## Notebooks

| Notebook | Description |
|----------|-------------|
| `00_pytorch_basics.ipynb` | Tensors, operations, autograd, NumPy conversion |
| `01_pytorch_fundamental.ipynb` | Neural network architecture, loss functions, optimisers, training loop |
| `02_autoencoder.ipynb` | Autoencoder architecture, reconstruction error, anomaly detection on synthetic sine wave data |

## Key Concepts Covered
- PyTorch tensors vs NumPy arrays
- Automatic differentiation with autograd
- Building neural networks with `nn.Module`
- Loss functions — MSE, BCE, CrossEntropy
- Optimisers — Adam, SGD
- The 5-step training loop
- Encoder/decoder architecture
- Reconstruction error as anomaly signal
- Sliding window technique for time series

## Results
Autoencoder successfully detected all 4 injected anomalies in synthetic energy data using reconstruction error thresholding (mean + 3σ).

## Stack
- Python, PyTorch 2.2.2
- NumPy, Matplotlib
- Jupyter Notebooks

## Next
These fundamentals feed directly into the `energy-anomaly-detector` project — applying autoencoders to real energy consumption data.
