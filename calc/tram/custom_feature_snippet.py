import numpy as np

import pyemma.coordinates as coor
from pyemma.coordinates.data.featurization.misc import CustomFeature

def tanh_contact(traj, pairs, r0, widths):
    r = md.compute_distances(traj, pairs)
    return 0.5*(np.tanh((r0 - r)/widths) + 1)

if __name__ == "__main__":
    ...
    feat = coor.featurizer(topfile)
    feat.add_custom_feature(CustomFeature(tanh_contact, pair_idxs, r0, widths, dim=len(pair_idxs)))
