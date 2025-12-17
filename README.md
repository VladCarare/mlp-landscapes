
<div align="center">  
  
# Global properties of the energy landscape: a testing and training arena for machine learned potentials

[![Paper](https://img.shields.io/badge/Paper-PDF-red)](https://doi.org/10.1038/s41524-025-01878-x)
[![Dataset](https://img.shields.io/badge/Dataset-Landscape17-blue)](https://doi.org/10.6084/m9.figshare.29949230)
</div>

## Abstract

Machine learning interatomic potentials (MLIPs) have achieved remarkable accuracy on standard benchmarks, yet their ability to reproduce molecular kinetics --critical for reaction rate calculations -- remains largely unexplored. We introduce Landscape17, a dataset of complete kinetic transition networks (KTNs) for the molecules of MD17, computed using hybrid-level density functional theory. Each KTN contains minima, transition states, and approximate steepest-descent paths, along with energies, forces, and Hessian eigenspectra at stationary points. We develop a comprehensive test suite to evaluate the MLIP ability to reproduce these reference landscapes and apply it to a number of state-of-the-art architectures. Our results reveal limitations in current MLIPs: all the models considered miss over half of the DFT transition states and generate stable unphysical structures throughout the potential energy surface. Data augmentation with pathway configurations improves reproduction of DFT potential energy surfaces, resulting in significant improvement in the global kinetics. However, these models still produce many spurious stable structures, indicating that current MLIP architectures face underlying challenges in capturing the topology of molecular potential energy surfaces. The Landscape17 benchmark provides a straightforward but demanding test of MLIPs for kinetic applications, requiring only up to a few hours of compute time. We propose this test for validation of next-generation MLIPs targeting reaction discovery and rate prediction.


## Overview

A computational workflow for reproducing energy landscape analysis of atomistic models, as presented in the accompanying publication.

This repository provides tools to analyze machine learning potentials through energy landscape exploration and kinetic transition network generation. The example demonstrates the methodology using ANI2x on salicylic acid, optimized for reasonable computation time on CPU.

## Key Features

- Energy landscape exploration using basin hopping
- Transition state identification and validation
- Kinetic transition network construction
- Automated analysis and visualization

## Requirements

- Python 3.11 (other version have not been tested)
- Conda package manager (this can be modified in RUNME.sh)
- 1 CPU (CPU works well for Aimnet2, ANI2x and GFN2-xTB. Other models may require a GPU)

## Installation

This package uses submodules. Clone the repository with:

```bash
git clone --recursive https://github.com/VladCarare/mlp-landscapes.git
```

> **Important:** Do not run additional installation commands at this step. The example script will handle all dependencies.

## Quick Start

### Running the Example

Execute the following command in the repository's root directory:

```bash
source RUNME.sh
```

This will:
1. Set up the conda environment with required dependencies
2. Run the salicylic acid example using ANI2x
3. Generate analysis plots and save results

### Output

All results and visualizations will be saved in:
```
examples/salicylic_acid_ani2x/landscape_runs/
```

## Configuration and Reproducibility

The example is configured for quick demonstration:
- **Initial seeds:** 3 (vs. 20 in the paper)
- **Basin hopping steps:** 5 per seed (vs. 50 in the paper)
- **Model:** ANI2x (vs. GFN2-xTB, Aimnet2, MACE, NequIP, Allegro, SO3LR, MACE-MP-0b3 in the paper)

The MLP landscapes which were generated and analyzed in the corresponding publication are available in `examples/production_landscapes/`.

To reproduce these results, modify the parameters inside `examples/salicylic_acid_ani2x/run_landscape_runs.py` and run for every model and every molecule.

To add models which are not present in the list above, one needs to add support for them in `external/topsearch-mlp_run/src/topsearch/potentials/ml_potentials.py`. This works best if the models already have an ASE interface.

## Notes

- **Package Manager:** The example uses conda by default. To use a different package manager, modify the `RUNME.sh` script accordingly.

- **MFPT Plots:** Mean first passage time plots may appear empty for small KTNs. To generate meaningful MFPT data, increase the number of basin hopping steps in the configuration file.

## Citation

If you use this workflow in your research, please cite:
> Cărare, V., Thiemann, F.L., Morrow, J.D., Wales, D.J., Pyzer-Knapp, E.O., Dicks, L. Global properties of the energy landscape: a testing and training arena for machine learned potentials. npj Comput Mater (2025). https://doi.org/10.1038/s41524-025-01878-x
 
