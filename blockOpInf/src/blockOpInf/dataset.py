"""Container for snapshot matrix and associated data."""

from dataclasses import dataclass
import h5py as hf
import numpy as np

@dataclass
class Dataset:
    """
    Contains snapshot matrix, coordinate matrix, time vector, and associated
    metadata.
    """
    Q: np.ndarray
    X: np.ndarray
    t: np.ndarray
    states: list[str]
    coords: list[str]

    def __repr__(self):
        rep = []
        rep.append(f"t.shape\t= (k,)\t\t= {self.t.shape}")
        rep.append(f"Q.shape\t= (ng*ns, k)\t= {self.Q.shape}")
        rep.append(f"X.shape\t= (ng*nc, k)\t= {self.X.shape}")
        rep.append(f"states\t= (ns,)\t\t= {self.states}")
        rep.append(f"coords\t= (nc,)\t\t= {self.coords}")
        return "\n".join(rep)


def write_dataset(filename, dataset):
    """Write hdf5 (.h5) file containing snapshot data."""

    print(f'Writing snapshot data to\n\t{filename:s}...')

    # Create hdf5 file
    f = hf.File(filename, 'w')

    # Write data
    f.create_dataset('Q', data=dataset.Q)
    if dataset.X is not None:
        f.create_dataset('X', data=dataset.X)
    f.create_dataset('t', data=dataset.t)
    f.attrs['states'] = dataset.states
    f.attrs['coords'] = dataset.coords

    # Close hdf5 file
    f.close()

    print(f'Wrote snapshot data to\n\t{filename:s}')


def read_dataset(filename):
    """Load hdf5 (.h5) file containing snapshot data."""

    print(f'Loading dataset from \t{filename:s}...')

    # Load .hdf5 file
    f = hf.File(filename, 'r')

    # Read data
    Q = f['Q'][...]
    if 'X' in f.keys():
        X = f['X'][...]
    else:
        X = None
    t = f['t'][...]
    states = f.attrs['states']
    coords = f.attrs['coords']

    # Close hdf5 file
    f.close()

    print(f'Loaded dataset from \t{filename:s}')

    return Dataset(Q, X, t, states, coords)
