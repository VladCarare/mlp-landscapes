from ase import Atoms
from ase.io import read,write
import sys 
import os
import numpy as np 

reference_atoms = read(sys.argv[1])

# A short run can easily find no transition states at all, which leaves these
# files empty.
if os.path.getsize('ts.coords') == 0:
    print('No transition states in ts.coords, writing an empty ts.xyz',
          file=sys.stderr)
    write('ts.xyz',[])
    sys.exit(0)

# ndmin=2 keeps a single stationary point two-dimensional, so that the loop
# below sees one row rather than a bare coordinate vector.
transition_states = np.loadtxt('ts.coords', ndmin=2)
energies = np.loadtxt('ts.data', ndmin=2)
traj = []
for transition_state,energy in zip(transition_states,energies):
    positions = transition_state.reshape(-1,3)
    atoms = Atoms(reference_atoms)
    atoms.set_positions(positions)
    atoms.info['energy']=energy[-1]
    traj.append(atoms)

write('ts.xyz',traj)
