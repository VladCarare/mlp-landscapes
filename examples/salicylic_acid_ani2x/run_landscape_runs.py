# IMPORTS
import os
from pathlib import Path
import ase.io
from topsearch.data.coordinates import MolecularCoordinates
from sys import argv
from topsearch.similarity.molecular_similarity import MolecularSimilarity
from topsearch.data.kinetic_transition_network import KineticTransitionNetwork
from topsearch.global_optimisation.perturbations import MolecularPerturbation
from topsearch.global_optimisation.basin_hopping import BasinHopping
from topsearch.sampling.exploration import NetworkSampling
# from topsearch.plotting.disconnectivity import plot_disconnectivity_graph
from topsearch.transition_states.hybrid_eigenvector_following import HybridEigenvectorFollowing
from topsearch.transition_states.nudged_elastic_band import NudgedElasticBand
from topsearch.potentials.ml_potentials import MachineLearningPotential
from topsearch.potentials.force_fields import MMFF94

# Which example this run operates on. Override to point the same analysis
# scripts at a different model or molecule without copying them.
EXAMPLE_DIR = os.environ.get('MLP_LANDSCAPES_EXAMPLE',
                             'examples/salicylic_acid_ani2x')
# Reference structures and the DFT network. Separate from EXAMPLE_DIR so a
# new run can reuse the data of an existing one for the same molecule.
DATA_DIR = os.environ.get('MLP_LANDSCAPES_DATA', f'{EXAMPLE_DIR}/data')
RUNS_DIR = f'{EXAMPLE_DIR}/landscape_runs'


repo_root = Path(__file__).parent


# Which model to drive the search with, and how hard to search. The
# defaults are the quick demo; the published runs used 20 seeds and 50
# basin hopping steps. Seeds are independent, so they can be split across
# processes by giving each one a different MLP_LANDSCAPES_SEEDS.
MODEL = os.environ.get('MLP_LANDSCAPES_MODEL', 'torchani')
seeds = [int(s) for s in
         os.environ.get('MLP_LANDSCAPES_SEEDS', '0,1,2').split(',')]
N_BH_STEPS = int(os.environ.get('MLP_LANDSCAPES_BH_STEPS', '5'))
# How many times a pair of minima may be attempted. Since a pair that
# already has a transition state is no longer skipped, every pair now uses
# its whole budget, which is the dominant cost of a run. Set to 1 to spend
# one attempt per pair.
MAX_ATTEMPTS = int(os.environ.get('MLP_LANDSCAPES_MAX_ATTEMPTS', '3'))
atfile = f'{DATA_DIR}/salicylic_acid_3_structures.xyz'
fffile = f'{DATA_DIR}/salicylic_acid_for_force_field.xyz'
parent_run_dir = f'{RUNS_DIR}/'

molecule = 'salicylic'

for seed in seeds:

    # INITIALISATION
    atoms = ase.io.read(atfile,seed)
    species = atoms.get_chemical_symbols()
    position = atoms.get_positions().flatten()
    coords = MolecularCoordinates(species, position)

    
    ff = MMFF94(fffile)

    # Defaults to ANI2x. Other supported models are: aimnet2, mace,
    # mace-mp-0b3, nequip, so3lr
    # Please see external/topsearch/src/topsearch/potentials/ml_potentials.py for details on how to specify the potentials
    mlp = MachineLearningPotential(species, MODEL, 'default', "cpu", ff=ff)

    # Alignment distances are RMSD, so this is 0.3 Angstrom per atom and
    # means the same thing whatever the molecule size. It used to be 1.0 as
    # a root-sum-squared deviation, which for the 16 atoms of salicylic acid
    # worked out at 0.25 here, and which is why small molecules previously
    # needed the criterion lowered by hand. Matches combine_results.py.
    comparer = MolecularSimilarity(distance_criterion=0.3,
                                energy_criterion=5e-3,
                                weighted=False)
    ktn = KineticTransitionNetwork()
    step_taking = MolecularPerturbation(max_displacement=180.0,
                                        max_bonds=2)
    optimiser = BasinHopping(ktn=ktn, potential=mlp, similarity=comparer,
                            step_taking=step_taking,ignore_relreduc=False, opt_method='ase')
    hef = HybridEigenvectorFollowing(potential=mlp,
                                    ts_conv_crit=1e-2,
                                    ts_steps=100,
                                    pushoff=0.8,
                                    max_uphill_step_size=0.3,
                                    positive_eigenvalue_step=0.1,
                                    steepest_descent_conv_crit=1e-3,
                                    eigenvalue_conv_crit=5e-2)
    neb = NudgedElasticBand(potential=mlp,
                            force_constant=50.0,
                            image_density=15.0,
                            max_images=20,
                            neb_conv_crit=1e-2)
    explorer = NetworkSampling(ktn=ktn,
                            coords=coords,
                            global_optimiser=optimiser,
                            single_ended_search=hef,
                            double_ended_search=neb,
                            similarity=comparer,
                            max_connection_attempts_per_pair=MAX_ATTEMPTS)
    

    # BEGIN CALCULATIONS
    explorer.get_minima(coords=coords,
                        n_steps=N_BH_STEPS,
                        conv_crit=1e-3,
                        temperature=100.0,
                        test_valid=True)
    explorer.get_transition_states(method='ClosestEnumeration',
                                cycles=2,
                                remove_bounds_minima=False)
    

    new_folder = f'{parent_run_dir}/seed{seed}/'
    os.makedirs(new_folder)
    ktn.dump_network(text_path=new_folder)