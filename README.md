# Quantum Contrastive Learning for Quark-Gluon Jet Classification

Hybrid classical-quantum pipeline that trains a variational quantum circuit to produce useful jet representations via contrastive learning (SimCLR). The quantum model matches the classical baseline in discrimination (AUC 0.858 vs 0.862) with ~450x fewer bottleneck parameters (54 vs ~24,000).

GSoC 2026 application to ML4SCI / QMLHEP.

## Pipeline

PointVector encoder (point cloud with directional attention) -> linear projection to 64-dim -> 6-qubit VQC (amplitude embedding, ZXZ rotations, CNOT ring + skip entanglement) -> 64-dim probability output -> NT-Xent contrastive loss

Evaluation: freeze encoder, fit logistic regression on embeddings, report AUC.

## Results

| Config | Samples | AUC | Bottleneck Params |
|---|---|---|---|
| Classical | 80k | 0.862 | ~24,000 |
| 6q / 2 layers | 10k | 0.838 | 54 |
| 6q / 3 layers | 40k | 0.855 | 54 |
| 6q / 3 layers | 80k | 0.858 | 54 |

## Running

Runs on Google Colab with GPU. Open the notebook and run all cells.

```
pip install torch pennylane energyflow scikit-learn matplotlib
```

## Dataset

Pythia8 Quark-Gluon jets (energyflow). 100k jets, 64 particles each, features: pT, eta, phi, log_pT.
