from ase import Atoms
from ase.io import read,write
import sys 
import os
import numpy as np 

reference_atoms = read(sys.argv[1])

if os.path.getsize('min.coords') == 0:
    print('No minima in min.coords, writing an empty minima.xyz',
          file=sys.stderr)
    write('minima.xyz',[])
    sys.exit(0)

# ndmin=2 keeps a single stationary point two-dimensional, so that the loop
# below sees one row rather than a bare coordinate vector.
minima = np.loadtxt('min.coords', ndmin=2)
energies = np.loadtxt('min.data', ndmin=2)
traj = []
for minimum,energy in zip(minima,energies):
    positions = minimum.reshape(-1,3)
    atoms = Atoms(reference_atoms)
    atoms.set_positions(positions)
    atoms.info['energy']=energy[-1]
    traj.append(atoms)

write('minima.xyz',traj)
