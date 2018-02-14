import numpy as np
import xml.etree.ElementTree as ET

import simtk.unit as unit
import simtk.openmm as omm
import simtk.openmm.app as app

import tabulated

def indent(elem, level=0):
    i = "\n" + level*"  "
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = i + "  "
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
        for elem in elem:
            indent(elem, level+1)
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
    else:
        if level and (not elem.tail or not elem.tail.strip()):
            elem.tail = i

def add_element_traits(elem, traits):
    for key, value in traits.items():
        elem.set(key, value)

def write_ff_file(n_beads, eps_slv, sigma_slv, mass_slv, eps_ply, sigma_ply, 
        mass_ply, bonded_params, cutoff, saveas="ff_cgs.xml"):
    """Build xml forcefield file for toy polymer
    
    Parameters
    ----------
    n_beads : int
        Number of monomers.
    eps_slv : float
    
    """
    rmin = 0.6*sigma_slv
    rmax = 1.5*cutoff

    # forcefield xml tags
    ff = ET.Element("ForceField")
    atm_types = ET.SubElement(ff, "AtomTypes")
    res_types = ET.SubElement(ff, "Residues")

    # solvent parameters
    add_element_traits(ET.SubElement(atm_types, "Type"),
        {"name":"Solv", "class": "SV", "element":"Sv", "mass":str(mass_slv/unit.amu)})
    add_element_traits(ET.SubElement(ET.SubElement(res_types, "Residue",
        attrib={"name":"SLV"}), "Atom"), {"name":"SV", "type":"Solv"})

    # add custom nonbonded interactions that are:
    #  1) Lennard-Jones (LJ) attractive between solvent atoms
    #  2) Weeks-Chandler-Andersen (WCA) repulsive for monomer-monomer and
    #  monomer-solvent
    cust_nb_f = ET.SubElement(ff, "CustomNonbondedForce",
            {"energy":"areslv*LJsw(r) + (1 - areslv)*WCA(r); areslv=step((flavor1 + flavor2) - 1.5)", 
                "bondCutoff":"3"})

    # tabulated WCA potential
    wca_f = ET.SubElement(cust_nb_f, "Function")
    add_element_traits(wca_f, {"name":"WCA", "type":"Continuous1D",
        "min":str(rmin/unit.nanometer), "max":str(rmax/unit.nanometer)})

    wca_tab = tabulated.wca_table(sigma_ply, eps_ply, rmin, rmax)
    wca_f.text = wca_tab

    # tabulated LJ potential that switches (exactly) to 0 at r_cut
    #ljsw_f = ET.SubElement(cust_nb_f, "Function")
    #add_element_traits(ljsw_f, {"name":"LJsw", "type":"Continuous1D",
    #    "min":str(rmin/unit.nanometer), "max":str(rmax/unit.nanometer)})

    #r_switch = cutoff - 0.11*unit.nanometer
    #r_cut = cutoff - 0.01*unit.nanometer
    #LJtab = tabulated.LJswitch_table(eps_slv, sigma_slv, rmin, rmax, r_switch, r_cut)
    #ljsw_f.text = LJtab

    # tabulated core-softened potential
    cs_f = ET.SubElement(cust_nb_f, "Function")
    add_element_traits(cs_f, {"name":"CS", "type":"Continuous1D",
        "min":str(rmin/unit.nanometer), "max":str(rmax/unit.nanometer)})

    cs_tab = tabulated.cs_table(sigma_ply, eps_ply, rmin, rmax)
    cs_f.text = cs_tab

    eps_ww = 20.38
    sigma_ww = 0.2429
    B = 23.35
    r0 = 0.2465 
    Delta = 0.1090
    y = tabulated.Chaimovich_table(eps_ww, sigma_ww, B, r0, Delta, rmin, rmax)


    # each particle class (solvent=SV, polymer=PL) has a 'flavor'. Flavors
    # determine the interaction
    add_element_traits(ET.SubElement(cust_nb_f, "PerParticleParameter"), {"name":"flavor"})
    add_element_traits(ET.SubElement(cust_nb_f, "Atom"), {"class":"SV", "flavor":"1"})
    add_element_traits(ET.SubElement(cust_nb_f, "Atom"), {"class":"PL", "flavor":"0"})

    for n in range(n_beads):
        # residues have the same set of atoms but we give them unique names in
        # order to get them bonded properly
        add_element_traits(ET.SubElement(atm_types, "Type"), 
            {"name":"Poly" + str(n + 1), "class": "PL", "element":"Pl", "mass":"37"})

        poly_res = ET.SubElement(res_types, "Residue", attrib={"name":"PL" + str(n + 1)})
        add_element_traits(ET.SubElement(poly_res, "Atom"), {"name":"PL" + str(n + 1), "type":"Poly" + str(n + 1)})
        add_element_traits(ET.SubElement(poly_res, "ExternalBond"), {"atomName":"PL" + str(n + 1)})

    # bonded interactions
    bond_f = ET.SubElement(ff, "HarmonicBondForce")
    kb_units = unit.kilojoule_per_mole/(unit.nanometer**2)
    ka_units = unit.kilojoule_per_mole/(unit.radian**2)
    r0, kb, theta0, ka = bonded_params
    for n in range(n_beads):
        if n > 0:
            # add harmonic bonds
            add_element_traits(ET.SubElement(bond_f, "Bond"),
                {"type1":"Poly" + str(n) , "type2":"Poly" + str(n + 1), 
                    "length":str(r0/unit.nanometer),"k":str(kb/kb_units)})

    # add harmonic angles
    angle_f = ET.SubElement(ff, "HarmonicAngleForce")
    for n in range(n_beads):
        if (n > 1):
            add_element_traits(ET.SubElement(angle_f, "Angle"),
                {"type1":"Poly" + str(n - 1) , "type2":"Poly" + str(n), "type3":"Poly" + str(n + 1), 
                    "angle":str(theta0/unit.radians),"k":str(ka/ka_units)})

    indent(ff)
    with open(saveas, "w") as fout:
        fout.write(ET.tostring(ff))


