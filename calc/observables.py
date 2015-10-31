import os
import numpy as np

import mdtraj

# Should model code after pyemma and mdtraj packages.


def get_observable_function(args):
    supported_contact_functions = {"step":step_contact,"tanh":tanh_contact,"w_tanh":weighted_tanh_contact}

class ContactObservable(object):
    def __init__(self, top, pairs, r0):
        self.prefix_label = "CONTACT"
        self.top = top
        self.pairs = pairs
        self.r0 = r0

    def describe(self):
        labels = ["%s %s - %s" % (self.prefix_label,
                                  _describe_atom(self.top, pair[0]),
                                  _describe_atom(self.top, pair[1]))
                  for pair in self.distance_indexes]
        return labels

class TanhContactObservable(ContactObservable):
    """Smoothly increasing tanh contact function"""

    def __init__(self, top, pairs, *args, periodic=False):
        ContactObservable.__init__(top, pairs, args[0])
        self.prefix_label = "TANHCONTACT"
        self.widths = self.args[1]
        self.periodic = periodic

    def map(self, traj):
        r = mdtraj.compute_distances(traj, self.pairs, periodic=self.periodic)
        return 0.5*(np.tanh(2.*(self.r0 - r)/self.widths) + 1)

class WeightedTanhContactObservable(ContactObservable):
    def weighted_tanh_contact(self,r,r0,widths,weights):
        """Weighted smoothly increasing tanh contact function"""
        return weights*tanh_contact(r,r0,widths)

class StepContactObservable(ContactObservable):
    def step_contact(self,r,r0):
        """Step function indicator contact function"""
        return (r <= r0).astype(int)

class Energy(object):
    def __init__(self):
        pass

    def parameterize():
