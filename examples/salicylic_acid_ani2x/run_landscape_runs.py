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

repo_root = Path(__file__).parent


seeds = [0,1,2]
atfile = 'examples/salicylic_acid_ani2x/data/salicylic_acid_3_structures.xyz'
fffile = 'examples/salicylic_acid_ani2x/data/salicylic_acid_for_force_field.xyz'
parent_run_dir = 'examples/salicylic_acid_ani2x/landscape_runs/'

molecule = 'salicylic'

for seed in seeds:

    # INITIALISATION
    atoms = ase.io.read(atfile,seed)
    species = atoms.get_chemical_symbols()
    position = atoms.get_positions().flatten()
    coords = MolecularCoordinates(species, position)

    
    ff = MMFF94(fffile)

    # USING ANI2x FOR THIS EXAMPLE. Other supported models are :aimnet2, dftb, mace, allegro, nequip, mace-mp-0b3
    # Please see the mlp_run branch of topsearch, src/potentials/ml_potentials.py for details on how to specify the potentials
    mlp = MachineLearningPotential(species, 'torchani', 'default', "cpu",ff=ff)

    comparer = MolecularSimilarity(distance_criterion=1.0,
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
                            similarity=comparer)
    

    # BEGIN CALCULATIONS
    explorer.get_minima(coords=coords,
                        n_steps=5, # 50 USUALLY, BUT KEEPING IT LOW FOR EXAMPLE'S SAKE
                        conv_crit=1e-3,
                        temperature=100.0,
                        test_valid=True)
    explorer.get_transition_states(method='ClosestEnumeration',
                                cycles=2,
                                remove_bounds_minima=False)
    

    new_folder = f'{parent_run_dir}/seed{seed}/'
    os.makedirs(new_folder)
    ktn.dump_network(text_path=new_folder)