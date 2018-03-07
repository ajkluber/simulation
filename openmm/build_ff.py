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

def toy_polymer_params(soft_bonds=False):
    # parameters for coarse-grain polymer are taken from:
    # Anthawale 2007
    sigma_ply = 0.373*unit.nanometer
    eps_ply = 0.58517*unit.kilojoule_per_mole
    mass_ply = 37.*unit.amu
    r0 = 0.153*unit.nanometer 
    theta0 = 111*unit.degree
    if soft_bonds:
        kb = 1000.*unit.kilojoule_per_mole/(unit.nanometer**2)
        ka = 500*unit.kilojoule_per_mole/(unit.radian**2)
    else:
        kb = 334720.*unit.kilojoule_per_mole/(unit.nanometer**2)
        ka = 462.*unit.kilojoule_per_mole/(unit.radian**2)
    bonded_params = [r0, kb, theta0, ka]

    return sigma_ply, eps_ply, mass_ply, bonded_params

def CS_water_params():
    eps_ww = 20.38*unit.kilojoule_per_mole
    sigma_ww = 0.2429*unit.nanometer
    B = 23.35*unit.kilojoule_per_mole
    r0 = 0.2465*unit.nanometer
    Delta = 0.1090*unit.nanometer
    mass_slv = 18.*unit.amu
    return eps_ww, sigma_ww, B, r0, Delta, mass_slv

def LJ_water_params():
    # LJ solvent parameters.
    sigma_slv = 0.3151*unit.nanometer
    eps_slv = 1*unit.kilojoule_per_mole
    mass_slv = 18.*unit.amu
    return sigma_slv, eps_slv, mass_slv

def add_toy_polymer_ff_items(n_beads, ff, atm_types, res_types, soft_bonds=False):

    sigma_ply, eps_ply, mass_ply, bonded_params = toy_polymer_params(soft_bonds=soft_bonds)

    for n in range(n_beads):
        # residues have the same set of atoms but we give them unique names in
        # order to get them bonded properly
        add_element_traits(ET.SubElement(atm_types, "Type"), 
            {"name":"Poly" + str(n + 1), "class": "PL", "element":"Pl", "mass":str(mass_ply/unit.amu)})

        poly_res = ET.SubElement(res_types, "Residue", attrib={"name":"PL" + str(n + 1)})
        add_element_traits(ET.SubElement(poly_res, "Atom"), {"name":"PL" + str(n + 1), "type":"Poly" + str(n + 1)})
        add_element_traits(ET.SubElement(poly_res, "ExternalBond"), {"atomName":"PL" + str(n + 1)})

    # bonded interactions
    bond_f = ET.SubElement(ff, "HarmonicBondForce")
    kb_units = unit.kilojoule_per_mole/(unit.nanometer**2)
    r0, kb, theta0, ka = bonded_params
    for n in range(n_beads):
        if n > 0:
            # add harmonic bonds
            add_element_traits(ET.SubElement(bond_f, "Bond"),
                {"type1":"Poly" + str(n) , "type2":"Poly" + str(n + 1), 
                    "length":str(r0/unit.nanometer),"k":str(kb/kb_units)})

    # add harmonic angles
    angle_f = ET.SubElement(ff, "HarmonicAngleForce")
    ka_units = unit.kilojoule_per_mole/(unit.radian**2)
    for n in range(n_beads):
        if (n > 1):
            add_element_traits(ET.SubElement(angle_f, "Angle"),
                {"type1":"Poly" + str(n - 1) , "type2":"Poly" + str(n), "type3":"Poly" + str(n + 1), 
                    "angle":str(theta0/unit.radians),"k":str(ka/ka_units)})


def toy_polymer_LJ_water(n_beads, cutoff, saveas="ff_cgs.xml", soft_bonds=False):
    """Build xml forcefield file for toy polymer
    
    Parameters
    ----------
    n_beads : int
        Number of monomers.
    cutoff : float
        Cutoff radius for long range interactions.
    
    """

    sigma_slv, eps_slv, mass_slv = LJ_water_params()
    sigma_ply, eps_ply, mass_ply, bonded_params = toy_polymer_params(soft_bonds=soft_bonds)

    rmin = 0.6*sigma_slv
    rmax = 1.5*cutoff

    #app.element.solvent = app.element.Element(201, "Solvent", "Sv", mass_slv)
    #app.element.polymer = app.element.Element(200, "Polymer", "Pl", mass_ply)

    # forcefield xml tags
    ff = ET.Element("ForceField")
    atm_types = ET.SubElement(ff, "AtomTypes")
    res_types = ET.SubElement(ff, "Residues")

    # solvent parameters
    add_element_traits(ET.SubElement(atm_types, "Type"),
        {"name":"Solv", "class": "LJ", "element":"Sv", "mass":str(mass_slv/unit.amu)})
    add_element_traits(ET.SubElement(ET.SubElement(res_types, "Residue",
        attrib={"name":"SLV"}), "Atom"), {"name":"LJ", "type":"Solv"})

    # add custom nonbonded interactions that are:
    #  1) Lennard-Jones (LJ) attractive between solvent atoms
    #  2) Weeks-Chandler-Andersen (WCA) repulsive for monomer-monomer and
    #  monomer-solvent
    cust_nb_f = ET.SubElement(ff, "CustomNonbondedForce",
            {"energy":"areslv*LJsw(r) + (1 - areslv)*WCA(r); areslv=step((flavor1 + flavor2) - 1.5)", 
                "bondCutoff":"3"})

    # each particle class (solvent=LJ, polymer=PL) has a 'flavor'. Flavors
    # determine the interaction
    add_element_traits(ET.SubElement(cust_nb_f, "PerParticleParameter"), {"name":"flavor"})
    add_element_traits(ET.SubElement(cust_nb_f, "Atom"), {"class":"LJ", "flavor":"1"})
    add_element_traits(ET.SubElement(cust_nb_f, "Atom"), {"class":"PL", "flavor":"0"})

    # tabulated WCA potential
    wca_f = ET.SubElement(cust_nb_f, "Function")
    add_element_traits(wca_f, {"name":"WCA", "type":"Continuous1D",
        "min":str(rmin/unit.nanometer), "max":str(rmax/unit.nanometer)})

    wca_tab = tabulated.wca_table(sigma_ply, eps_ply, rmin, rmax)
    wca_f.text = wca_tab

    # tabulated LJ potential that switches (exactly) to 0 at r_cut
    ljsw_f = ET.SubElement(cust_nb_f, "Function")
    add_element_traits(ljsw_f, {"name":"LJsw", "type":"Continuous1D",
        "min":str(rmin/unit.nanometer), "max":str(rmax/unit.nanometer)})

    r_switch = cutoff - 0.11*unit.nanometer
    r_cut = cutoff - 0.01*unit.nanometer
    LJtab = tabulated.LJ_table(eps_slv, sigma_slv, rmin, rmax, r_switch, r_cut, switch=True)
    ljsw_f.text = LJtab

    # add polymer only items
    add_toy_polymer_ff_items(n_beads, ff, atm_types, res_types, soft_bonds=soft_bonds)

    indent(ff)
    with open(saveas, "w") as fout:
        fout.write(ET.tostring(ff))

def toy_polymer_CS_water(n_beads, cutoff, saveas="ff_cgs.xml", soft_bonds=False):
    """Build xml forcefield file for toy polymer
    
    Parameters
    ----------
    n_beads : int
        Number of monomers.
    
    """

    eps_ww, sigma_ww, B, r0, Delta, mass_slv = CS_water_params()
    sigma_ply, eps_ply, mass_ply, bonded_params = toy_polymer_params(soft_bonds=soft_bonds)

    rmin = 0.6*sigma_ww
    rmax = 1.5*cutoff

    # forcefield xml tags
    ff = ET.Element("ForceField")
    atm_types = ET.SubElement(ff, "AtomTypes")
    res_types = ET.SubElement(ff, "Residues")

    # solvent parameters
    add_element_traits(ET.SubElement(atm_types, "Type"),
        {"name":"Solv", "class": "CS", "element":"Sv", "mass":str(mass_slv/unit.amu)})
    add_element_traits(ET.SubElement(ET.SubElement(res_types, "Residue",
        attrib={"name":"SLV"}), "Atom"), {"name":"CS", "type":"Solv"})

    # add custom nonbonded interactions that are:
    #  1) Core-softened (CS) potential between solvent atoms
    #  2) Weeks-Chandler-Andersen (WCA) repulsive for monomer-monomer and
    #  monomer-solvent
    cust_nb_f = ET.SubElement(ff, "CustomNonbondedForce",
            {"energy":"areslv*CS(r) + (1 - areslv)*WCA(r); areslv=step((flavor1 + flavor2) - 1.5)", 
                "bondCutoff":"3"})

    # each particle class (solvent=CS, polymer=PL) has a 'flavor'. Flavors
    # determine the interaction
    add_element_traits(ET.SubElement(cust_nb_f, "PerParticleParameter"), {"name":"flavor"})
    add_element_traits(ET.SubElement(cust_nb_f, "Atom"), {"class":"CS", "flavor":"1"})
    add_element_traits(ET.SubElement(cust_nb_f, "Atom"), {"class":"PL", "flavor":"0"})

    # tabulated WCA potential
    wca_f = ET.SubElement(cust_nb_f, "Function")
    add_element_traits(wca_f, {"name":"WCA", "type":"Continuous1D",
        "min":str(rmin/unit.nanometer), "max":str(rmax/unit.nanometer)})

    wca_tab = tabulated.wca_table(sigma_ply, eps_ply, rmin, rmax)
    wca_f.text = wca_tab

    # tabulated core-softened potential
    cs_f = ET.SubElement(cust_nb_f, "Function")
    add_element_traits(cs_f, {"name":"CS", "type":"Continuous1D",
        "min":str(rmin/unit.nanometer), "max":str(rmax/unit.nanometer)})

    r_switch = cutoff - 0.11*unit.nanometer
    r_cut = cutoff - 0.01*unit.nanometer
    cs_tab = tabulated.Chaimovich_table(eps_ww, sigma_ww, B, r0, Delta, rmin, rmax, r_switch, r_cut, switch=True)
    cs_f.text = cs_tab

    # add polymer only items
    add_toy_polymer_ff_items(n_beads, ff, atm_types, res_types, soft_bonds=soft_bonds)

    indent(ff)
    with open(saveas, "w") as fout:
        fout.write(ET.tostring(ff))

