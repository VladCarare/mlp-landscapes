from ase import Atoms
from ase.io import read,write
import sys 
import glob
import numpy as np 
import os

reference_atoms = read(sys.argv[1])

minima = np.loadtxt('min.coords')
energies = np.loadtxt('min.data')
if minima.ndim==1:
    minima = [minima]
if energies.ndim==1:
    energies = [energies]
traj = []
for minimum,energy in zip(minima,energies):
    positions = minimum.reshape(-1,3)
    atoms = Atoms(reference_atoms)
    atoms.set_positions(positions)
    atoms.info['energy']=energy[-1]
    traj.append(atoms)

write('minima.xyz',traj)