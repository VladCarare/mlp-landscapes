from topsearch.data.kinetic_transition_network import KineticTransitionNetwork
from topsearch.data.coordinates import MolecularCoordinates
from topsearch.similarity.molecular_similarity import MolecularSimilarity
from topsearch.data.kinetic_transition_network import KineticTransitionNetwork
from ase.io import read
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdDetermineBonds
from topsearch.data.kinetic_transition_network import KineticTransitionNetwork
from topsearch.similarity.molecular_similarity import MolecularSimilarity
from topsearch.data.coordinates import MolecularCoordinates

from networkx.exception import NetworkXError



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
    print(filename)
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

from ase import Atoms
from ase.io import write 

def remove_structures_with_multiply_bonded_hydrogen(ktn,reference_atoms):

    elements = reference_atoms.get_chemical_symbols()
    reference_coords = reference_atoms.get_positions().flatten()
    traj = []
    dummy_coords = MolecularCoordinates(
        reference_atoms.get_chemical_symbols(),
        reference_coords
    )
    # Check all minima
    nodes_to_delete = []
    for i in range(ktn.n_minima):
        coords = ktn.G.nodes[i]["coords"]
        dummy_coords.position = coords
        try: 
            bond_labels = dummy_coords.get_connected_atoms()
            for atom_idx, neighbours in enumerate(bond_labels):
                if elements[atom_idx]=='H' and len(neighbours)>1:
                    nodes_to_delete.append(i)
                    atoms = Atoms(elements,coords.reshape(-1,3))
                    traj.append(atoms)
                    break
        except NetworkXError:
            nodes_to_delete.append(i)
            atoms = Atoms(elements,coords.reshape(-1,3))
            traj.append(atoms)

    
    # Delete from highest to lowest index to avoid shifting issues
    for i in sorted(nodes_to_delete, reverse=True):
        ktn.remove_minimum(i)
        
    # Check all transition states
    for min1, min2, edge_idx in list(ktn.G.edges):
        coords = ktn.G[min1][min2][edge_idx]["coords"]
        dummy_coords.position = coords
        try:
            bond_labels = dummy_coords.get_connected_atoms()
            for atom_idx, neighbours in enumerate(bond_labels):
                if elements[atom_idx]=='H' and len(neighbours)>1:
                    ktn.remove_ts(min1, min2, edge_idx)
                    atoms = Atoms(elements,coords.reshape(-1,3))
                    traj.append(atoms)
                    break
        except NetworkXError:
            ktn.remove_ts(min1, min2, edge_idx)
            atoms = Atoms(elements,coords.reshape(-1,3))
            traj.append(atoms)
    return traj

def remove_dissociated_structures(ktn,reference_atoms):

    elements = reference_atoms.get_chemical_symbols()
    reference_coords = reference_atoms.get_positions().flatten()
    traj = []
    dummy_coords = MolecularCoordinates(
        reference_atoms.get_chemical_symbols(),
        reference_coords
    )
    # Check all minima
    nodes_to_delete = []
    for i in range(ktn.n_minima):
        coords = ktn.G.nodes[i]["coords"]
        dummy_coords.position = coords
        try: 
            bond_labels = dummy_coords.get_connected_atoms()
        except NetworkXError:
            nodes_to_delete.append(i)
            atoms = Atoms(elements,coords.reshape(-1,3))
            traj.append(atoms)

    
    # Delete from highest to lowest index to avoid shifting issues
    for i in sorted(nodes_to_delete, reverse=True):
        ktn.remove_minimum(i)
        
    # Check all transition states
    for min1, min2, edge_idx in list(ktn.G.edges):
        coords = ktn.G[min1][min2][edge_idx]["coords"]
        dummy_coords.position = coords
        try:
            bond_labels = dummy_coords.get_connected_atoms()
        except NetworkXError:
            ktn.remove_ts(min1, min2, edge_idx)
            atoms = Atoms(elements,coords.reshape(-1,3))
            traj.append(atoms)
    return traj

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



base_folder = f'examples/salicylic_acid_ani2x/landscape_runs/'
seeds = [0,1,2]
folders = [f'{base_folder}/seed{seed}' for seed in seeds]
combined_ktn = KineticTransitionNetwork()
n_ts = []
n_min = []
n_removed_ts = []
n_removed_min = []

dump_folder = f'{base_folder}/'

for idx, folder in enumerate(folders):
    try: 
        current_ktn = KineticTransitionNetwork()
        current_ktn.read_network(text_path=f'{folder}/')
        if idx == 0:
            atfile = f'examples/salicylic_acid_ani2x/data/salicylic_acid_ground_state.xyz'
            
            atoms = canonicalize_atoms(atfile, current_ktn)
            species = atoms.get_chemical_symbols()
            position = atoms.get_positions().flatten()
            coords = MolecularCoordinates(species, position)
            comparer = MolecularSimilarity(distance_criterion=0.3,
                                        energy_criterion=1e8,
                                        weighted=False,
                                        allow_inversion=True)
        else:
            canonicalize_atoms(atfile, current_ktn)
        # CLEAN                  
        removed_ts = current_ktn.n_ts
        removed_min = current_ktn.n_minima
        print(folder)
        print('n_ts_current',removed_ts,flush=True)
        print('n_min_current',removed_min,flush=True)
        
        removed_ts -= current_ktn.n_ts
        removed_min -= current_ktn.n_minima      

        if idx == 0:
            n_removed_ts.append(removed_ts)
            n_removed_min.append(removed_min)
        else:
            n_removed_ts.append(removed_ts+n_removed_ts[-1])
            n_removed_min.append(removed_min+n_removed_min[-1])

        combined_ktn.add_network(current_ktn,similarity=comparer,coords=coords)
        n_min.append(combined_ktn.n_minima)
        n_ts.append(combined_ktn.n_ts)
        print('n_ts:',n_ts,flush=True)
        print('n_ts_removed:',n_removed_ts,flush=True)
        print('n_min:',n_min,flush=True)
        print('n_min_removed:',n_removed_min,flush=True)
        print('\n----------------------------------------\n')
    except FileNotFoundError:
        print('n_ts_current: NONE, SKIPPING',flush=True)
        print('n_min_current: NONE, SKIPPING',flush=True)
        print('n_ts: NONE, SKIPPING',flush=True)
        print('n_ts_removed: NONE, SKIPPING',flush=True)
        print('n_min: NONE, SKIPPING',flush=True)
        print('n_min_removed: NONE, SKIPPING',flush=True)
        print('\n----------------------------------------\n')


combined_ktn.dump_network(text_path=dump_folder)