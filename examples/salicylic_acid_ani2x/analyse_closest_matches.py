"""
Kinetic Transition Network (KTN) Comparison Tool

This script compares energy landscapes represented as Kinetic Transition Networks (KTNs)
between high-level quantum chemistry calculations (DFT) and machine learning potentials (MLP).
It quantifies differences in structure and energy between closest minima and transition states pairs.

"""

import matplotlib.pyplot as plt
from ase.io import read
import os
import sys
import networkx as nx
import pickle
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdDetermineBonds

from topsearch.data.kinetic_transition_network import KineticTransitionNetwork
from topsearch.plotting.network import barrier_reweighting
from topsearch.similarity.molecular_similarity import MolecularSimilarity
from topsearch.data.coordinates import MolecularCoordinates


class KTNComparison:
    """Class for comparing two Kinetic Transition Networks"""
    
    def __init__(self, reference_ktn, ml_ktn, reference_atoms, distance_threshold=1.0):
        """
        Initialize the KTN comparison object.
        
        Parameters
        ----------
        reference_ktn : KineticTransitionNetwork
            Reference (usually DFT) kinetic transition network
        ml_ktn : KineticTransitionNetwork
            Machine learning potential kinetic transition network to compare
        reference_atoms : ase.Atoms
            Reference atoms object with canonical atom ordering
        distance_threshold : float, optional
            Distance threshold for considering structures as matching, by default 1.0
        """
        self.reference_ktn = reference_ktn
        self.ml_ktn = ml_ktn
        self.atoms = reference_atoms
        self.distance_threshold = distance_threshold
        self.minima_correspondences = None
        self.ts_distances = None
        self.comparer = MolecularSimilarity(
            distance_criterion=distance_threshold,  
            energy_criterion=5e-3,  
            weighted=False,
            allow_inversion=True
        )
        
    def find_minima_correspondences(self):
        """
        Find corresponding minima between reference and ML networks.
        
        Returns
        -------
        list
            List of tuples (ref_idx, ml_idx, distance) for corresponding minima
        """
        correspondences = []
        
        for ref_idx in range(self.reference_ktn.n_minima):
            print(f"Finding correspondence for reference minimum {ref_idx}")
            # print('Coords of first atom: ',self.reference_ktn.get_minimum_coords(ref_idx).reshape(-1,3)[0])
            ref_coords = MolecularCoordinates(
                self.atoms.get_chemical_symbols(),
                self.reference_ktn.get_minimum_coords(ref_idx)
            )
            
            # Find closest ML minimum
            best_distance = float('inf')
            best_ml_idx = -1
            
            for ml_idx in range(self.ml_ktn.n_minima):
                distance = self.comparer.closest_distance(
                    ref_coords,
                    self.ml_ktn.get_minimum_coords(ml_idx)
                )
                
                if distance < best_distance:
                    print(f'  Found closer match: ML minimum {ml_idx} at distance {distance:.3f}')
                    # print('Coords of first atom: ',self.ml_ktn.get_minimum_coords(ml_idx).reshape(-1,3)[0])
                    best_distance = distance
                    best_ml_idx = ml_idx
            
            # Only store correspondences below threshold
            if best_distance < self.distance_threshold:
                print(f'Tentatively matching {ref_idx} with {best_ml_idx}.')
                correspondences.append((ref_idx, best_ml_idx, best_distance))
        
        matched_ml_minima = set([i[1] for i in correspondences])
        filtered_minima_correspondences=[]
        for matched_ml_min in matched_ml_minima:
            matches = [i for i in correspondences if i[1]==matched_ml_min]
            lowest_rmsd_match = sorted(matches,key=lambda i: i[2])[0]
            filtered_minima_correspondences.append(lowest_rmsd_match)
            print(f'Selected best minima match: {lowest_rmsd_match}')

        self.minima_correspondences = filtered_minima_correspondences
        
        # Define global minimum indices and energies for both networks
        self._define_global_minima()
        
        return self.minima_correspondences
    
    
    def find_closest_minima_correspondences_despite_threshold(self):
        """
        Find corresponding minima between reference and ML networks despite threshold.
        
        Returns
        -------
        list
            List of tuples (ref_idx, ml_idx, distance) for corresponding minima
        """

        import numpy as np
        all_correspondences = np.zeros((self.reference_ktn.n_minima,self.ml_ktn.n_minima))

    
        for ref_idx in range(self.reference_ktn.n_minima):
            print(f"Finding correspondence for reference minimum {ref_idx}")
            # print('Coords of first atom: ',self.reference_ktn.get_minimum_coords(ref_idx).reshape(-1,3)[0])
            ref_coords = MolecularCoordinates(
                self.atoms.get_chemical_symbols(),
                self.reference_ktn.get_minimum_coords(ref_idx)
            )            
            
            for ml_idx in range(self.ml_ktn.n_minima):
                distance = self.comparer.closest_distance(
                    ref_coords,
                    self.ml_ktn.get_minimum_coords(ml_idx)
                )
                all_correspondences[ref_idx, ml_idx] = distance             
                
        filtered_minima_correspondences=[]
        max_iterations = np.min(all_correspondences.shape)
        for _ in range(max_iterations):
            best_idx_flattened = np.argmin(all_correspondences)
            ref_idx = int(best_idx_flattened/len(all_correspondences[0]))
            ml_idx = int(best_idx_flattened%len(all_correspondences[0]))
            filtered_minima_correspondences.append([ref_idx,ml_idx,all_correspondences[ref_idx,ml_idx]])
            all_correspondences[ref_idx] = np.ones(all_correspondences.shape[1])*np.inf
            all_correspondences[:,ml_idx] = np.ones(all_correspondences.shape[0])*np.inf
                
        self.minima_correspondences = filtered_minima_correspondences
        
        # Define global minimum indices and energies for both networks
        self._define_global_minima()
        
        return self.minima_correspondences
    
    def _define_global_minima(self):
        """
        Define the global minimum for each network based on energy.
        Also check if ML global minimum corresponds to DFT global minimum.
        """
        if self.minima_correspondences:
            # Find global minimum in reference network
            ref_minima_indices = [pair[0] for pair in self.minima_correspondences]
            ref_energies = [self.reference_ktn.G.nodes[i]["energy"] for i in ref_minima_indices]
            ref_global_min_idx = ref_minima_indices[np.argmin(ref_energies)]
            ref_global_min_energy = self.reference_ktn.G.nodes[ref_global_min_idx]["energy"]
            
            # Find global minimum in ML network
            ml_minima_indices = [pair[1] for pair in self.minima_correspondences]
            ml_energies = [self.ml_ktn.G.nodes[i]["energy"] for i in ml_minima_indices]
            ml_global_min_idx = ml_minima_indices[np.argmin(ml_energies)]
            ml_global_min_energy = self.ml_ktn.G.nodes[ml_global_min_idx]["energy"]
            
            # Store as properties
            self.reference_ktn.min_i = ref_global_min_idx
            self.reference_ktn.min_e = ref_global_min_energy
            self.ml_ktn.min_i = ml_global_min_idx
            self.ml_ktn.min_e = ml_global_min_energy
            
            # Check if global minima correspond structurally
            ref_global_min_correspondence = next(
                (pair for pair in self.minima_correspondences if pair[0] == ref_global_min_idx), 
                None
            )
            
            if ref_global_min_correspondence and ref_global_min_correspondence[1] != ml_global_min_idx:
                print('WARNING: THE STRUCTURE OF THE ML GLOBAL MINIMUM IS DIFFERENT THAN THE STRUCTURE OF THE DFT GLOBAL MINIMUM.')
                print(f'DFT global minimum (idx {ref_global_min_idx}) corresponds to ML structure {ref_global_min_correspondence[1]}')
                print(f'ML global minimum is structure {ml_global_min_idx}')
        else:
            print('WARNING: NO MATCHES')
            # Store as properties
            self.reference_ktn.min_i = np.nan
            self.reference_ktn.min_e = np.nan
            self.ml_ktn.min_i = np.nan
            self.ml_ktn.min_e = np.nan
    
    def find_ts_correspondences(self):
        """
        Find corresponding transition states between reference and ML networks.
        
        Returns
        -------
        tuple
            (ts_distances, relative_energy_diffs, absolute_energy_diffs)
        """
        if self.minima_correspondences is None:
            raise ValueError("Must call find_minima_correspondences before find_ts_correspondences")
        
        # Create mapping from reference to ML minima indices
        mapping = {ref_idx: ml_idx for ref_idx, ml_idx, _ in self.minima_correspondences}
        
        ts_distances = []
        ts_relative_energy_diffs = []
        ts_absolute_energy_diffs = []
        
        # Iterate through all transition states in reference network
        for ref_min1, ref_min2, edge_idx in self.reference_ktn.G.edges:
            print(f"Analyzing transition state between {ref_min1} and {ref_min2}")
            
            # Get transition state coordinates from reference network
            ts_coords = MolecularCoordinates(
                self.atoms.get_chemical_symbols(),
                self.reference_ktn.get_ts_coords(ref_min1, ref_min2, edge_idx)
            )
            
            # Check if corresponding minima exist in ML network
            if ref_min1 not in mapping or ref_min2 not in mapping:
                print(f"  No corresponding minimum for {ref_min1} or {ref_min2} in ML network")
                continue
                
            ml_min1, ml_min2 = mapping[ref_min1], mapping[ref_min2]
            print(f"  Found mapping: DFT minima ({ref_min1}, {ref_min2}) → ML minima ({ml_min1}, {ml_min2})")
            
            # Find closest transition state in ML network
            best_distance = float('inf')
            best_edge_idx = 0
            
            for ml_edge_idx in range(self.ml_ktn.G.number_of_edges(ml_min1, ml_min2)):
                distance = self.comparer.closest_distance(
                    ts_coords,
                    self.ml_ktn.get_ts_coords(ml_min1, ml_min2, ml_edge_idx)
                )
                
                if distance < best_distance:
                    print(f'  Found closer match: TS with edge index {ml_edge_idx} at distance {distance:.3f}')
                    best_distance = distance
                    best_edge_idx = ml_edge_idx
            
            # Only store correspondences below threshold
            if best_distance < self.distance_threshold:
                print(f'Matching {ref_min1, ref_min2, edge_idx} with {ml_min1, ml_min2, best_edge_idx}.')
                ts_distances.append(best_distance)
                
                # Calculate energy differences
                ref_ts_energy = self.reference_ktn.get_ts_energy(ref_min1, ref_min2, edge_idx)
                ml_ts_energy = self.ml_ktn.get_ts_energy(ml_min1, ml_min2, best_edge_idx)
                
                # Relative energy (with respect to global minimum)
                relative_energy_diff = (ref_ts_energy - self.reference_ktn.min_e) - \
                                      (ml_ts_energy - self.ml_ktn.min_e)
                ts_relative_energy_diffs.append(relative_energy_diff)
                
                # Absolute energy (direct difference)
                absolute_energy_diff = ref_ts_energy - ml_ts_energy
                ts_absolute_energy_diffs.append(absolute_energy_diff)
        
        self.ts_distances = ts_distances
        self.ts_relative_energy_diffs = ts_relative_energy_diffs
        self.ts_absolute_energy_diffs = ts_absolute_energy_diffs
        
        return ts_distances, ts_relative_energy_diffs, ts_absolute_energy_diffs

    def find_closest_ts_correspondences_despite_threshold(self):
        """
        Find corresponding transition states between reference and ML networks.
        
        Returns
        -------
        tuple
            (ts_distances, relative_energy_diffs, absolute_energy_diffs)
        """



        import numpy as np

        ref_ktn_energies = [self.reference_ktn.get_ts_energy(*ref_edge) for ref_edge in self.reference_ktn.G.edges]
        ml_ktn_energies = [self.ml_ktn.get_ts_energy(*ml_edge) for ml_edge in self.ml_ktn.G.edges]

        ts_distances = []
        ts_relative_energy_diffs = []
        ts_absolute_energy_diffs = []
        
        all_correspondences = np.zeros((self.reference_ktn.n_ts,self.ml_ktn.n_ts))
        # Iterate through all transition states in reference network
        for index_dft, (ref_min1, ref_min2, _) in enumerate(self.reference_ktn.G.edges):
            print(f"Analyzing transition state between {ref_min1} and {ref_min2}")

            # Get transition state coordinates from reference network
            ref_ts_coords = MolecularCoordinates(
                self.atoms.get_chemical_symbols(),
                self.reference_ktn.get_ts_coords(ref_min1, ref_min2, _)
            )

            
            for index_ml, (ml_min1, ml_min2, _) in enumerate(self.ml_ktn.G.edges):
                    distance = self.comparer.closest_distance(
                        ref_ts_coords,
                        self.ml_ktn.get_ts_coords(ml_min1,ml_min2, _)
                    )
                    all_correspondences[index_dft, index_ml] = distance  


        filtered_ts_correspondences=[]
        max_iterations = np.min(all_correspondences.shape)
        for _ in range(max_iterations):
            best_idx_flattened = np.argmin(all_correspondences)
            ref_idx = int(best_idx_flattened/len(all_correspondences[0]))
            ml_idx = int(best_idx_flattened%len(all_correspondences[0]))
            filtered_ts_correspondences.append([ref_idx,ml_idx,all_correspondences[ref_idx,ml_idx]])
            all_correspondences[ref_idx] = np.ones(all_correspondences.shape[1])*np.inf
            all_correspondences[:,ml_idx] = np.ones(all_correspondences.shape[0])*np.inf
                

        for ref_edge,ml_edge,best_distance in filtered_ts_correspondences:
            print(f'Matching {ref_edge} with {ml_edge} - best distance {best_distance}.')
            ts_distances.append(best_distance)
            
            # Calculate energy differences
            ref_ts_energy = ref_ktn_energies[ref_edge]
            ml_ts_energy = ml_ktn_energies[ml_edge]
            
            if self.minima_correspondences:
                # Relative energy (with respect to global minimum)
                rel_diff = (ref_ts_energy - self.reference_ktn.min_e) - \
                        (ml_ts_energy - self.ml_ktn.min_e)
            else:
                rel_diff=None
            ts_relative_energy_diffs.append(rel_diff)

            # Absolute energy (direct difference)
            absolute_energy_diff = ref_ts_energy - ml_ts_energy
            ts_absolute_energy_diffs.append(absolute_energy_diff)
    
        self.ts_distances = ts_distances
        self.ts_relative_energy_diffs = ts_relative_energy_diffs
        self.ts_absolute_energy_diffs = ts_absolute_energy_diffs
        
        return ts_distances, ts_relative_energy_diffs, ts_absolute_energy_diffs

    def calculate_statistics(self):
        """
        Calculate statistical measures for comparison between networks.
        
        Returns
        -------
        dict
            Dictionary containing all calculated statistics
        """
        if self.minima_correspondences is None:
            raise ValueError("Must call find_minima_correspondences first")
            
        if self.ts_distances is None:
            self.find_ts_correspondences()
        
        # Structure distances
        minima_distances = [d for _, _, d in self.minima_correspondences]
        
        # Energy differences for minima
        minima_relative_energy_diffs = []
        minima_absolute_energy_diffs = []
        
        for ref_idx, ml_idx, _ in self.minima_correspondences:
            ref_energy = self.reference_ktn.get_minimum_energy(ref_idx)
            ml_energy = self.ml_ktn.get_minimum_energy(ml_idx)
            
            # Relative energy (with respect to global minimum)
            rel_diff = (ref_energy - self.reference_ktn.min_e) - \
                      (ml_energy - self.ml_ktn.min_e)
            minima_relative_energy_diffs.append(rel_diff)
            
            # Absolute energy (direct difference)
            abs_diff = ref_energy - ml_energy
            minima_absolute_energy_diffs.append(abs_diff)
        
        # Count missing/extra features
        missing_minima = self.reference_ktn.n_minima - len(self.minima_correspondences)
        extra_minima = self.ml_ktn.n_minima - len(self.minima_correspondences)
        missing_ts = self.reference_ktn.n_ts - len(self.ts_distances)
        extra_ts = self.ml_ktn.n_ts - len(self.ts_distances)
        
        # Calculate summary statistics
        stats = {
            # Structure distance statistics
            'minima_distances': minima_distances,
            'ts_distances': self.ts_distances,
            
            # Energy difference statistics (with sign preserved)
            'minima_relative_energy_diffs': minima_relative_energy_diffs,
            'ts_relative_energy_diffs': self.ts_relative_energy_diffs,
            
            # Absolute energy differences
            'minima_absolute_energy_diffs': np.abs(minima_absolute_energy_diffs),
            'ts_absolute_energy_diffs': np.abs(self.ts_absolute_energy_diffs),
            
            # Missing/extra features
            'missing_minima': missing_minima,
            'extra_minima': extra_minima,
            'missing_ts': missing_ts,
            'extra_ts': extra_ts
        }
        
        return stats
    
    def remove_inversions(self, ktn, reference_atoms):
        """
        Remove structures that are related by inversion symmetry.
        
        Parameters
        ----------
        ktn : KineticTransitionNetwork
            KTN object to clean
        reference_atoms : ase.Atoms
            Reference atoms object
        """
        to_delete_1 = []  # First of the pair (usually kept)
        to_delete_2 = []  # Second of the pair (usually deleted)

        # Find all inversion-related pairs
        for i in range(ktn.n_minima):
            if i in to_delete_2:
                continue
                
            coords_i = MolecularCoordinates(
                reference_atoms.get_chemical_symbols(),
                ktn.get_minimum_coords(i)
            )
            
            for j in range(i+1, ktn.n_minima):
                # Quick energy check to avoid unnecessary comparisons
                if not np.isclose(
                    ktn.get_minimum_energy(i),
                    ktn.get_minimum_energy(j),
                    atol=5e-3
                ):
                    continue
                    
                # Check if structures are the same under inversion
                if self.comparer.test_same(
                    coords_i,
                    ktn.get_minimum_coords(j),
                    ktn.get_minimum_energy(i),
                    ktn.get_minimum_energy(j)
                ):
                    to_delete_1.append(i)
                    to_delete_2.append(j)
        
        # Decide which minima to delete based on connectivity
        edges = list(ktn.G.edges())
        to_delete_final = []
        
        for i in range(len(to_delete_1)):
            # Count edges connected to each minimum
            nedges_1 = sum(1 for edge in edges if to_delete_1[i] in edge)
            nedges_2 = sum(1 for edge in edges if to_delete_2[i] in edge)
            
            # Keep the one with more connections
            if nedges_2 > nedges_1:
                to_delete_final.append(to_delete_1[i])
                print(f'Removing minimum {to_delete_1[i]} (fewer connections)')
            else:
                to_delete_final.append(to_delete_2[i])
                print(f'Removing minimum {to_delete_2[i]} (fewer connections)')
        
        # Delete from highest to lowest index to avoid shifting issues
        for i in sorted(list(set(to_delete_final)), reverse=True):
            ktn.remove_minimum(i)


def canonicalize_atoms(filename, ktn):
    """
    Establish canonical atom ordering for consistent comparison.
    
    Parameters
    ----------
    filename : str
        Path to XYZ file
    ktn : KineticTransitionNetwork
        KTN object to update with canonical ordering
        
    Returns
    -------
    ase.Atoms
        Atoms object with canonical ordering
    """
    # Read molecule and determine connectivity
    mol = Chem.MolFromXYZFile(filename)
    rdDetermineBonds.DetermineConnectivity(mol)
    
    # Get canonical ordering
    canon_indices = Chem.CanonicalRankAtoms(mol)
    reordering = np.array(tuple(zip(
        *sorted([(new_idx, old_idx) for old_idx, new_idx in enumerate(canon_indices)])
    )))[1]
    
    # Reorder atoms
    atoms = read(filename)
    atoms = atoms[reordering]
    
    # Update coordinates in KTN
    for i in range(ktn.n_minima):
        coords = ktn.G.nodes[i]["coords"].reshape(-1, 3)
        ktn.G.nodes[i]["coords"] = coords[reordering].flatten()
        
    # Update transition state coordinates
    for min1, min2, edge_idx in ktn.G.edges:
        coords = ktn.G[min1][min2][edge_idx]["coords"].reshape(-1, 3)
        ktn.G[min1][min2][edge_idx]["coords"] = coords[reordering].flatten()
    
    return atoms


def remove_different_bonding_frameworks(ktn, reference_atoms):
    """
    Remove structures with different bonding patterns from a KTN.
    
    Parameters
    ----------
    ktn : KineticTransitionNetwork
        KTN object to clean
    reference_atoms : ase.Atoms
        Reference atoms object with correct bonding
    """
    elements = reference_atoms.get_chemical_symbols()
    reference_coords = reference_atoms.get_positions()
    
    # Create reference mol and get reference SMILES
    ref_mol = create_mol_from_coordinates(elements, reference_coords)
    rdDetermineBonds.DetermineConnectivity(ref_mol)
    ref_smiles = Chem.MolToSmiles(ref_mol,allHsExplicit=True,allBondsExplicit=True)
    
    # Check all minima
    nodes_to_delete = []
    for i in range(ktn.n_minima):
        coords = ktn.G.nodes[i]["coords"].reshape(-1, 3)
        test_mol = create_mol_from_coordinates(elements, coords)
        rdDetermineBonds.DetermineConnectivity(test_mol)
        test_smiles = Chem.MolToSmiles(test_mol,allHsExplicit=True,allBondsExplicit=True)
        
        if test_smiles != ref_smiles:# or not validate_hydrogen_bonding(test_mol):
            nodes_to_delete.append(i)
        
    
    # Delete from highest to lowest index to avoid shifting issues
    for i in sorted(nodes_to_delete, reverse=True):
        ktn.remove_minimum(i)
        
    # Check all transition states
    for min1, min2, edge_idx in list(ktn.G.edges):
        coords = ktn.G[min1][min2][edge_idx]["coords"].reshape(-1, 3)
        test_mol = create_mol_from_coordinates(elements, coords)
        rdDetermineBonds.DetermineConnectivity(test_mol)
        test_smiles = Chem.MolToSmiles(test_mol,allHsExplicit=True,allBondsExplicit=True)
        
        if test_smiles != ref_smiles:# or not validate_hydrogen_bonding(test_mol):
            ktn.remove_ts(min1, min2, edge_idx)

def remove_structures_with_multiply_bonded_hydrogen(ktn,reference_atoms):

    elements = reference_atoms.get_chemical_symbols()
    reference_coords = reference_atoms.get_positions().flatten()
    
    dummy_coords = MolecularCoordinates(
        reference_atoms.get_chemical_symbols(),
        reference_coords
    )
    # Check all minima
    nodes_to_delete = []
    for i in range(ktn.n_minima):
        coords = ktn.G.nodes[i]["coords"]
        dummy_coords.position = coords
        bond_labels = dummy_coords.get_connected_atoms()
        for atom_idx, neighbours in enumerate(bond_labels):
            if elements[atom_idx]=='H' and len(neighbours)>1:
                nodes_to_delete.append(i)
                break
    
    # Delete from highest to lowest index to avoid shifting issues
    for i in sorted(nodes_to_delete, reverse=True):
        ktn.remove_minimum(i)
        
    # Check all transition states
    for min1, min2, edge_idx in list(ktn.G.edges):
        coords = ktn.G[min1][min2][edge_idx]["coords"]
        dummy_coords.position = coords
        bond_labels = dummy_coords.get_connected_atoms()
        for atom_idx, neighbours in enumerate(bond_labels):
            if elements[atom_idx]=='H' and len(neighbours)>1:
                ktn.remove_ts(min1, min2, edge_idx)
                break
    

def create_mol_from_coordinates(elements, coordinates):
    """
    Create an RDKit molecule from elements and coordinates.
    
    Parameters
    ----------
    elements : list
        List of element symbols
    coordinates : numpy.ndarray
        Array of 3D coordinates
        
    Returns
    -------
    rdkit.Chem.rdchem.Mol
        RDKit molecule object
    """
    mol = Chem.RWMol()
    
    # Add atoms to the molecule
    for element in elements:
        atom = Chem.Atom(element)
        mol.AddAtom(atom)
    
    # Add a conformer and set 3D coordinates
    conf = Chem.Conformer(len(elements))
    for i, coord in enumerate(coordinates):
        conf.SetAtomPosition(i, coord)
    mol.AddConformer(conf)
    
    return mol

def find_missing_ml_minima(correspondences, ktn_ml):
    """
    Find ML minima that don't correspond to any reference minimum.
    
    Parameters
    ----------
    correspondences : list
        List of (ref_idx, ml_idx, distance) tuples
    ktn_ml : KineticTransitionNetwork
        ML potential KTN
        
    Returns
    -------
    list
        Indices of ML minima without corresponding reference minimum
    """
    mapped_ml_indices = [ml_idx for _, ml_idx, _ in correspondences]
    missing_ml_indices = [i for i in range(ktn_ml.n_minima) if i not in mapped_ml_indices]
    return missing_ml_indices


import numpy as np
from topsearch.data.kinetic_transition_network import KineticTransitionNetwork
from topsearch.similarity.molecular_similarity import MolecularSimilarity
from topsearch.data.coordinates import MolecularCoordinates
from ase.io import read 
from rdkit import Chem
from rdkit.Chem import rdDetermineBonds
import networkx as nx 

if __name__=='__main__':
    extra_tag = 'closest-matches'
            
    ml_landscape_path = f"examples/salicylic_acid_ani2x/landscape_runs/"
    dft_landscape_path = f"examples/salicylic_acid_ani2x/data/dft_ktn/"

    print('-'*50)
    # Load and prepare DFT network
    print("Loading DFT network...")
    dft_ktn = KineticTransitionNetwork()
    dft_ktn.read_network(dft_landscape_path)
    reference_atoms = canonicalize_atoms(f"examples/salicylic_acid_ani2x/data/dft_ktn/salicylic.xyz", dft_ktn)
    print(f'DFT network: {dft_ktn.n_minima} minima, {dft_ktn.n_ts} transition states')

    # Define paths (these should be configuration options in a real application)
    # Load and prepare ML network
    print("Loading ML network...")
    ml_ktn = KineticTransitionNetwork()
    ml_ktn.read_network(ml_landscape_path)
    canonicalize_atoms(f'examples/salicylic_acid_ani2x/data/salicylic_acid_ground_state_canon_perm.xyz', ml_ktn)

    # Clean ML network
    print("Cleaning ML network...")
    print(f"ML network before cleaning: {ml_ktn.n_minima} minima, {ml_ktn.n_ts} transition states")

    print("Removing structures with incorrect bonding...")
    remove_different_bonding_frameworks(ml_ktn, reference_atoms)
    print(f"After removing incorrect bonding: {ml_ktn.n_minima} minima, {ml_ktn.n_ts} transition states")

    print("Removing structures with multiply bonded hydrogens...")
    remove_structures_with_multiply_bonded_hydrogen(ml_ktn, reference_atoms)
    print(f"After removing incorrect bonded hydrogens: {ml_ktn.n_minima} minima, {ml_ktn.n_ts} transition states")

    # Initialize KTN comparison
    print("Comparing networks...")
    comparison = KTNComparison(dft_ktn, ml_ktn, reference_atoms, distance_threshold=0.3) # need it at 0.3 for inversion comparison

    print("Removing inversion-related structures...")
    comparison.remove_inversions(ml_ktn, reference_atoms)
    print(f"After removing inversions: {ml_ktn.n_minima} minima, {ml_ktn.n_ts} transition states")

    # Find corresponding minima
    print("Finding corresponding minima...")
    minima_correspondences = comparison.find_closest_minima_correspondences_despite_threshold()
    print(f"Found {len(minima_correspondences)} corresponding minima pairs")

    # Find extra ML minima
    missing_ml_minima = find_missing_ml_minima(minima_correspondences, ml_ktn)
    print(f"Extra ML minima without DFT correspondence: {missing_ml_minima}")

    comparison.distance_threshold = 1e8 # revert to 1e8 for TS comparison

    # Calculate TS correspondences 
    comparison.find_closest_ts_correspondences_despite_threshold()

    # Calculate statistics
    print("Calculating comparison statistics...")
    statistics = comparison.calculate_statistics()


    print("Saving results...")
    with open(f"{ml_landscape_path}/analysis_{extra_tag}.pkl", 'wb') as f:
        pickle.dump([
            dft_ktn, 
            ml_ktn, 
            minima_correspondences, 
            missing_ml_minima, 
            statistics
        ], f)
    print("Analysis complete!")

    print('\n'*5)