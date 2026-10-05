# Salicylic acid with AIMNet2

A landscape generated with AIMNet2 at the settings used in the paper, 20
seeds and 50 basin hopping steps, to compare against the published
`examples/production_landscapes/aimnet2_non-altitude/salicylic`.

## Reproducing

From the repository root:

```bash
bash examples/salicylic_acid_aimnet2/reproduce.sh
```

`JOBS` controls how many seeds run at once and `SEEDS` which ones, so a
shorter check is possible without editing anything:

```bash
SEEDS=0,1 MLP_LANDSCAPES_BH_STEPS=5 bash examples/salicylic_acid_aimnet2/reproduce.sh
```

The script re-runs only seeds that have no `min.data`, so an interrupted
run can be restarted.

## Notes

This example has no `data/` of its own. The reference structures and the
DFT network describe the molecule, not the model, so they are read from
`examples/salicylic_acid_ani2x/data/`.

AIMNet2 is not installed by `RUNME.sh`. It comes from
`github.com/zubatyuk/aimnet2calc`, which is not on PyPI and has not been
updated since July 2024; the maintained successor is
`github.com/isayevlab/aimnetcentral`, which renames the package to
`aimnet` and so does not satisfy the `from aimnet2calc import AIMNet2ASE`
that `ml_potentials.py` does. Installing it needs torch present first:

```bash
pip install torch
pip install --no-build-isolation git+https://github.com/zubatyuk/aimnet2calc.git
```

AIMNet2 returns forces at float32 precision even though the calculator is
constructed under `torch.set_default_dtype(torch.float64)`, so the surface
carries float32 noise. The eigenvector search probes with a displacement of
1e-3, which is large enough not to amplify it.
