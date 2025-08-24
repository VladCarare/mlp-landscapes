from ase.io import read
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdDetermineBonds

from topsearch.data.kinetic_transition_network import KineticTransitionNetwork
from topsearch.data.coordinates import MolecularCoordinates


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
    ref_smiles = Chem.MolToSmiles(ref_mol,allHsExplicit=True, allBondsExplicit=True)
    
    # Check all minima
    nodes_to_delete = []
    for i in range(ktn.n_minima):
        coords = ktn.G.nodes[i]["coords"].reshape(-1, 3)
        test_mol = create_mol_from_coordinates(elements, coords)
        rdDetermineBonds.DetermineConnectivity(test_mol)
        test_smiles = Chem.MolToSmiles(test_mol,allHsExplicit=True, allBondsExplicit=True)
        
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
        test_smiles = Chem.MolToSmiles(test_mol,allHsExplicit=True, allBondsExplicit=True)
        
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




def canonicalize_atoms2(filename):
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
    
    return atoms


if __name__=='__main__':
    all_results = []

    ml_landscape_path = f"examples/salicylic_acid_ani2x/landscape_runs/"

    ml_ktn = KineticTransitionNetwork()
    ml_ktn.read_network(ml_landscape_path)


    reference_atoms = canonicalize_atoms2(f'examples/salicylic_acid_ani2x/data/salicylic_acid_ground_state.xyz')
    initial_n_min = ml_ktn.n_minima
    initial_n_ts = ml_ktn.n_ts

    remove_different_bonding_frameworks(ml_ktn, reference_atoms)

    remove_structures_with_multiply_bonded_hydrogen(ml_ktn, reference_atoms)

    n_physical_min = ml_ktn.n_minima
    n_physical_ts = ml_ktn.n_ts
    entry={'Molecule':'salicylic','Model':'ani2x','Type':'non_altitude',
        'initial_n_min':initial_n_min,'n_physical_min':n_physical_min,
            'initial_n_ts':initial_n_ts,'n_physical_ts':n_physical_ts}
    print(initial_n_min,initial_n_ts,'->',n_physical_min,n_physical_ts)
    all_results.append(entry)


    import pickle 
    print("Saving results...")
    with open(f"examples/salicylic_acid_ani2x/landscape_runs/count_unphysical_stationary_points.pkl", 'wb') as f:
        pickle.dump(all_results
        , f)
    print("Analysis complete!")