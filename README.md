# MLP Landscapes

A computational workflow for reproducing energy landscape analysis of atomistic models, as presented in the accompanying publication.

## Overview

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

## Configuration

The example is configured for quick demonstration:
- **Initial seeds:** 3 (vs. 20 in the paper)
- **Basin hopping steps:** 5 per seed (vs. 50 in the paper)
- **Model:** ANI2x (vs. GFN2-xTB, Aimnet2, MACE, NequIP, Allegro, SO3LR, MACE-MP-0b3 in the paper)

To reproduce full paper results, modify these parameters in:
```
examples/salicylic_acid_ani2x/run_landscape_runs.py
```

To add models which are not present in the list above, one needs to add support for them in `external/topsearch-mlp_run/src/topsearch/potentials/ml_potentials.py`. This works best if the models already have an ASE interface.

## Notes

- **Package Manager:** The example uses conda by default. To use a different package manager, modify the `RUNME.sh` script accordingly.

- **MFPT Plots:** Mean first passage time plots may appear empty for small KTNs. To generate meaningful MFPT data, increase the number of basin hopping steps in the configuration file.

## Citation

If you use this workflow in your research, please cite:
```
[Citation information to be added]
```

