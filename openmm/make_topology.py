
import os
import numpy as np

def periodic_distance(pos1, pos2, box_ang):
    dx = pos2 - pos1
    dx[0] -= box_ang*np.floor((dx[0]/box_ang) + 0.5)
    dx[1] -= box_ang*np.floor((dx[1]/box_ang) + 0.5)
    dx[2] -= box_ang*np.floor((dx[2]/box_ang) + 0.5)
    return np.linalg.norm(dx)

def write_solvent_pdb(sigma_slv, packing_fraction, box_edge, starting_pdb,
        dummypdb="dum.pdb"):
    """Create solvent in box"""

    # generate solvent
    n_slv = int(round(packing_fraction*((box_edge/sigma_slv)**3)))
    xyz_slv = np.random.uniform(high=box_edge/unit.angstroms, size=[n_slv,3])
    overlap_radius = sigma_slv/unit.angstrom

    box_ang = box_edge/unit.angstrom

    new_xyz_slv = []
    for i in range(len(xyz_slv)):
        not_overlapping = True
        if i < (len(xyz_slv) - 1):
            for n in range(i + 1, len(xyz_slv)):
                sep = periodic_distance(xyz_slv[i], xyz_slv[n], box_ang) 
                if sep < overlap_radius:
                    # remove solvent that is overlapping itself
                    not_overlapping = False
                    break
        if not_overlapping:
            new_xyz_slv.append(xyz_slv[i])
    xyz_slv = np.array(new_xyz_slv)
    n_slv = len(xyz_slv)

    # create topology
    topology = app.Topology()
    chain = topology.addChain()
    for i in range(n_slv):
        res = topology.addResidue("SLV", chain)
        topology.addAtom("SV", app.element.get_by_symbol("Sv"), res)

    positions = unit.Quantity(xyz_slv, unit.angstrom)

    topology.setUnitCellDimensions((box_edge/unit.nanometer)*omm.Vec3(1, 1, 1))

    pdb = app.PDBFile(dummypdb)
    with open(starting_pdb, "w") as fout:
        pdb.writeFile(topology, positions, file=fout)

def write_top_pdb(n_beads, sigma_slv, packing_fraction, sigma_ply, r0,
        box_edge, starting_pdb, poly_conf=None, dummypdb="dum.pdb"):
    """Create polymer in extended conformation sorrounded by solvent"""

    if poly_conf is None:
        # extended polymer has linear coordinates
        xyz_ply = np.vstack([(r0/unit.angstrom)*np.arange(n_beads), np.zeros(n_beads), np.zeros(n_beads)]).T
    else:
        poly_pdb = app.PDBFile(poly_conf)
        xyz_ply = np.array(poly_pdb.positions/unit.angstrom)

    # generate solvent
    n_slv = int(round(packing_fraction*((box_edge/sigma_slv)**3)))
    xyz_slv = np.random.uniform(high=box_edge/unit.angstroms, size=[n_slv,3])
    overlap_radius1 = 0.5*(sigma_slv + sigma_ply)/unit.angstrom
    overlap_radius2 = sigma_slv/unit.angstrom

    box_ang = box_edge/unit.angstrom

    new_xyz_slv = []
    for i in range(len(xyz_slv)):
        not_overlapping = True
        for j in range(len(xyz_ply)):
            sep = periodic_distance(xyz_slv[i], xyz_ply[j], box_ang) 
            if sep < overlap_radius1:
                # remove solvent that is too close to polymer
                not_overlapping = False
                break
        if i < (len(xyz_slv) - 1):
            for n in range(i + 1, len(xyz_slv)):
                sep = periodic_distance(xyz_slv[i], xyz_slv[n], box_ang) 
                if sep < overlap_radius2:
                    # remove solvent that is overlapping itself
                    not_overlapping = False
                    break
        if not_overlapping:
            new_xyz_slv.append(xyz_slv[i])
    xyz_slv = np.array(new_xyz_slv)
    n_slv = len(xyz_slv)

    # create topology
    topology = app.Topology()
    chain = topology.addChain()
    for i in range(n_beads):
        res = topology.addResidue("PLY", chain)
        atm = topology.addAtom("PL", app.element.get_by_symbol("Pl"), res)
        if i == 0:
            prev_atm = atm
        else:
            topology.addBond(prev_atm, atm)
            prev_atm = atm

    chain = topology.addChain()
    for i in range(n_slv):
        res = topology.addResidue("SLV", chain)
        topology.addAtom("SV", app.element.get_by_symbol("Sv"), res)

    xyz = np.vstack((xyz_ply, xyz_slv))
    positions = unit.Quantity(xyz, unit.angstrom)

    topology.setUnitCellDimensions(box_edge.value_in_unit(unit.nanometer)*omm.Vec3(1, 1, 1))

    pdb = app.PDBFile(dummypdb)
    with open(starting_pdb, "w") as fout:
        pdb.writeFile(topology, positions, file=fout)
