# ML_Materials_Science

This repository provides an exemplar implementation of machine learning (ML) potentials to compute the energies and forces associated with crystal structures.

In this project, we utilize a Python notebook to demonstrate the application of machine learning interatomic potentials, utilizing the MAML Python package. The ML models implemented here include:

1. Gaussian Approximation Potentials (GAP)
2. Spectral Neighbor Analysis Potential (SNAP)
3. Quantized SNAP (qSNAP)
4. Neural Network Potentials (NNP)
5. Moment Tensor Potentials (MTP)

## Requirements

To effectively run the code in this repository, you will need to have several libraries and software installed on your system, which include:

- LAMMPS (with QUIP, GAP, n2p2, and MLIP extensions)
- CMake and the Python CMake package
- N2P2
- MAML
- MLIP

## Use Case

For demonstration purposes, this implementation focuses on the Si-O system. The dataset used for training is generated randomly through the USPEX code and subsequently optimized with VASP.

Test structures for validating the trained models are sourced from the DAICS database, accessible at [DAICS](https://daics.net).
