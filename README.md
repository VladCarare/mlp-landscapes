# mlp_landscapes

A workflow to reproduce the energy landscapes analysis of atomistic models, as displayed in the associated publication.

To keep the compute time manageable, we illustrate the case of the salicylic acid using ANI2x on CPU. We restrict the calculations to 3 initial seeds of 5 basin hopping steps each, as opposed to 20 initial seeds of 50 basin hopping steps that we used in the paper. 

# Installation

The package is using submodules, so the correct cloning command is:

`git clone --recursive https://github.com/you/your-repo.git`

Note: Do not run further installation commands at this step.

# Running the example

To set up the environment and run the example, run `source RUNME.sh` in the top level directory of this package.

All results and plots will be saved in `examples/salicylic_acid_ani2x/landscape_runs`. 

Note: The example is configured to run using conda to install the subpackages. Please feel free to modify the bash script if willing to use a different package manager.

Note2: The mean first passage time (MFPT) plot may be empty. It can be because the KTN is too small. If you want to obtain MPFTs increase the number of basin hopping steps in `examples/salicylic_acid_ani2x/run_landscape_runs.py`.