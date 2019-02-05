from __future__ import absolute_import
import sys
import numpy as np
import xml.etree.ElementTree as ET

import simtk.unit as unit
import simtk.openmm as omm
import simtk.openmm.app as app

import simulation.openmm.tabulated as tabulated

############################################
# parameters
############################################
def toy_polymer_params_soft_bonds():
    # SOFTER parameters for coarse-grain polymer are taken from:
    sigma_ply = 0.373*unit.nanometer
    eps_ply = 1*unit.kilojoule_per_mole
    mass_ply = 37.*unit.amu
    r0 = 0.153*unit.nanometer 
    theta0 = 111*unit.degree
    kb = 100.*unit.kilojoule_per_mole/(unit.nanometer**2)
    ka = 20.*unit.kilojoule_per_mole/(unit.radian**2)
    bonded_params = [r0, kb, theta0, ka]

    return sigma_ply, eps_ply, mass_ply, bonded_params

def toy_polymer_params():
    # parameters for coarse-grain polymer are taken from:
    # Anthawale 2007
    sigma_ply = 0.373*unit.nanometer
    eps_ply = 0.58517*unit.kilojoule_per_mole
    mass_ply = 37.*unit.amu
    bonded_params = toy_polymer_bonded_params()

    return sigma_ply, eps_ply, mass_ply, bonded_params

def toy_polymer_bonded_params():
    # parameters for coarse-grain polymer are taken from:
    # Anthawale 2007
    r0 = 0.153*unit.nanometer 
    theta0 = 111*unit.degree
    kb = 334720.*unit.kilojoule_per_mole/(unit.nanometer**2)
    ka = 462.*unit.kilojoule_per_mole/(unit.radian**2)
    bonded_params = [r0, kb, theta0, ka]

    return bonded_params

def LJ_polymer_params():
    # parameters for coarse-grain polymer are taken from:
    # Anthawale 2007
    sigma_ply = 0.373*unit.nanometer
    eps_ply = 0.1*unit.kilojoule_per_mole
    mass_ply = 37.*unit.amu
    r0 = 0.153*unit.nanometer 
    theta0 = 111*unit.degree
    kb = 334720.*unit.kilojoule_per_mole/(unit.nanometer**2)
    ka = 462.*unit.kilojoule_per_mole/(unit.radian**2)
    bonded_params = [r0, kb, theta0, ka]

    return sigma_ply, eps_ply, mass_ply, bonded_params

def CS_water_table(T):
    #TODO
    T_tab = np.array([280., 290., 300., 310., 320.])
    eps_tab = np.array([22.41, 21.39, 20.38, 20.05, 19.73])
    sigma_tab = np.array([0.2420, 0.2425, 0.2429, 0.2430, 0.2431])
    B_tab = np.array([25.77, 24.56, 23.35, 23.15, 22.95])
    r0_tab = np.array([0.2446, 0.2455, 0.2465, 0.2451, 0.2437])

    table = [np.array([0.1099, 0.1095, 0.1090, 0.1098, 0.1106]),
            np.array([280., 290., 300., 310., 320.]),
            np.array([22.41, 21.39, 20.38, 20.05, 19.73]),
            np.array([0.2420, 0.2425, 0.2429, 0.2430, 0.2431]),
            np.array([25.77, 24.56, 23.35, 23.15, 22.95]),
            np.array([0.2446, 0.2455, 0.2465, 0.2451, 0.2437]),
            np.array([0.1099, 0.1095, 0.1090, 0.1098, 0.1106])]

    for i in range(1, len(table)):
        pass

    units_to_use = [unit.kiljoule_per_mole, unit.nanometer,
        unit.kiljoule_per_mole, unit.nanometer, unit.nanometer] 

    import scipy.interpolate
    scipy.interpolate.interp1d(T, sigma) 

    eps_ww = 20.38*unit.kilojoule_per_mole
    sigma_ww = 0.2429*unit.nanometer
    B = 23.35*unit.kilojoule_per_mole
    r0 = 0.2465*unit.nanometer
    Delta = 0.1090*unit.nanometer
    mass_slv = 18.*unit.amu

def CS_water_params():
    # parameters at 300K
    eps_ww = 20.38*unit.kilojoule_per_mole
    sigma_ww = 0.2429*unit.nanometer
    B = 23.35*unit.kilojoule_per_mole
    r0 = 0.2465*unit.nanometer
    Delta = 0.1090*unit.nanometer
    mass_slv = 18.*unit.amu
    return eps_ww, sigma_ww, B, r0, Delta, mass_slv

def LJ_water_params():
    # LJ solvent parameters.
    eps_slv = 1.*unit.kilojoule_per_mole
    sigma_slv = 0.3151*unit.nanometer
    mass_slv = 18.*unit.amu
    return eps_slv, sigma_slv, mass_slv

#########################################################
# constructing XML files
#########################################################
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

def add_atom_and_res_types(n_beads, atm_types, res_types, mass_ply):
    for n in range(n_beads):
        # residues have the same set of atoms but we give them unique names in
        # order to get them bonded properly
        add_element_traits(ET.SubElement(atm_types, "Type"), 
            {"name":"Poly" + str(n + 1), "class": "PL", "element":"Pl", "mass":str(mass_ply/unit.amu)})

        poly_res = ET.SubElement(res_types, "Residue", attrib={"name":"PL" + str(n + 1)})
        add_element_traits(ET.SubElement(poly_res, "Atom"), {"name":"PL" + str(n + 1), "type":"Poly" + str(n + 1)})
        add_element_traits(ET.SubElement(poly_res, "ExternalBond"), {"atomName":"PL" + str(n + 1)})

def add_bonds_angles(n_beads, ff, r0, kb, theta0, ka):

    # bonded interactions
    bond_f = ET.SubElement(ff, "HarmonicBondForce")
    kb_units = unit.kilojoule_per_mole/(unit.nanometer**2)
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

def add_bonded_term_items(n_beads, ff, atm_types, res_types, soft_bonds=False):

    if soft_bonds:
        sigma_ply, eps_ply, mass_ply, bonded_params = toy_polymer_params_soft_bonds()
    else:
        sigma_ply, eps_ply, mass_ply, bonded_params = toy_polymer_params()
    r0, kb, theta0, ka = toy_polymer_bonded_params()

    add_atom_and_res_types(n_beads, atm_types, res_types, mass_ply)
    add_bonds_angles(n_beads, ff, r0, kb, theta0, ka)

def polymer_in_solvent(n_beads, ply_potential, slv_potential, saveas="ff_cgs.xml", **kwargs):
    """Build xml forcefield file for polymer in coarse-grain solvent
    
    Parameters
    ----------
    n_beads : int
        Number of monomers.
    
    ply_potential : str
        String indicates type of interactions for polymer-polymer.

    slv_potential : str
        String indicates type of interactions for solvent-solvent.

    saveas : str
        Filename to save forcefield as.

    kwargs : dict
        Dictionary of additional parameters needed depending on chosen
        interactions.
    """

    # Custom nonbonded interactions for LJ12-10, WCA, and CS interactions

    # polymer-polymer and polymer-solvent interactions are the same: either LJ or WCA
    # solvent-solvent interactions are either LJ or CS
    assert ply_potential in ["LJ", "WCA", "LJ6", "r12"]
    assert slv_potential in ["LJ", "CS", "SPC", "NONE"]

    rmin = 0.1*unit.nanometers
    rmax = 1.5*unit.nanometers

    # forcefield xml tags
    ff = ET.Element("ForceField")
    atm_types = ET.SubElement(ff, "AtomTypes")
    res_types = ET.SubElement(ff, "Residues")

    if ply_potential == "LJ6" and slv_potential == "SPC":
        # polymer is SPC water
        eps_ply = kwargs["eps_ply"]
        sigma_ply = kwargs["sigma_ply"]
        mass_ply = kwargs["mass_ply"]
        str_eps_ply = str(eps_ply/unit.kilojoule_per_mole)
        str_sig_ply = str(sigma_ply/unit.nanometer)
        nonbond_f = ET.SubElement(ff, "NonbondedForce",
            attrib={"coulomb14scale":"0.833333","lj14scale":"0.5"})

        add_element_traits(ET.SubElement(nonbond_f, "Atom"),
            {"class":"PL", "charge":"0.0", "sigma":str_sig_ply, "epsilon":str_eps_ply})

        # add polymer only items
        add_bonded_term_items(n_beads, ff, atm_types, res_types, mass_ply)
    else:
        eng_str = ""
        if slv_potential == "NONE":
            if ply_potential == "LJ":
                eps_ply = kwargs["eps_ply"]
                sigma_ply = kwargs["sigma_ply"]
                mass_ply = kwargs["mass_ply"]
                eng_str += "eps_ply*(5*((rmin_ply/r)^12) - 6*((rmin_ply/r)^10))"
            elif ply_potential == "r12":
                eps_ply = kwargs["eps_ply"]
                sigma_ply = kwargs["sigma_ply"]
                mass_ply = kwargs["mass_ply"]
                eng_str += "eps_ply*((rmin_ply/r)^12)"
            elif ply_potential == "WCA":
                eng_str += "WCA(r)"
                sigma_ply, eps_ply, mass_ply, bonded_params = toy_polymer_params()

            eng_str += ";"
        else:
            # compound expression if polymer in solvent
            if slv_potential == "LJ":
                eps_slv = kwargs["eps_slv"]
                sigma_slv = kwargs["sigma_slv"]
                mass_slv = kwargs["mass_slv"]
                eng_str += "areslv*eps_slv*(5*((rmin_slv/r)^12) - 6*((rmin_slv/r)^10)) + "
            elif slv_potential == "CS":
                eng_str += "areslv*CS(r) + "
                eps_ww, sigma_ww, B, r0, Delta, mass_slv = CS_water_params()

            if ply_potential == "LJ":
                eps_ply = kwargs["eps_ply"]
                sigma_ply = kwargs["sigma_ply"]
                mass_ply = kwargs["mass_ply"]
                eng_str += "(1 - areslv)*eps_ply*(5*((rmin_ply/r)^12) - 6*((rmin_ply/r)^10))"
            elif ply_potential == "r12":
                eps_ply = kwargs["eps_ply"]
                sigma_ply = kwargs["sigma_ply"]
                mass_ply = kwargs["mass_ply"]
                eng_str += "(1 - areslv)*eps_ply*((rmin_ply/r)^12)"
            elif ply_potential == "WCA":
                eng_str += "(1 - areslv)*WCA(r)"
                sigma_ply, eps_ply, mass_ply, bonded_params = toy_polymer_params()

            eng_str += "; areslv=step((flavor1 + flavor2) - 1.5)"


        # each particle class (solvent=CS, water=OW, polymer=PL) has a 'flavor'. Flavors
        # determine the interaction
        cust_nb_f = ET.SubElement(ff, "CustomNonbondedForce", {"energy":eng_str, "bondCutoff":"3"})
        add_element_traits(ET.SubElement(cust_nb_f, "PerParticleParameter"), {"name":"flavor"})
        add_element_traits(ET.SubElement(cust_nb_f, "Atom"), {"class":"PL", "flavor":"0"})

        if ply_potential in ["r12", "LJ"]:
            str_eps_ply = str(eps_ply/unit.kilojoule_per_mole)
            str_rmin_ply = str(sigma_ply/unit.nanometer)
            add_element_traits(ET.SubElement(cust_nb_f, "GlobalParameter"), {"name":"eps_ply", "defaultValue":str_eps_ply})
            add_element_traits(ET.SubElement(cust_nb_f, "GlobalParameter"), {"name":"rmin_ply", "defaultValue":str_rmin_ply})
        elif ply_potential == "WCA":
            # tabulated WCA potential
            wca_f = ET.SubElement(cust_nb_f, "Function")
            add_element_traits(wca_f, {"name":"WCA", "type":"Continuous1D",
                "min":str(rmin/unit.nanometer), "max":str(rmax/unit.nanometer)})

            wca_tab = tabulated.wca_table(sigma_ply, eps_ply, rmin, rmax)
            wca_f.text = wca_tab

        if slv_potential != "NONE":
            # solvent parameters
            add_element_traits(ET.SubElement(atm_types, "Type"),
                {"name":"Solv", "class": "CS", "element":"Sv", "mass":str(mass_slv/unit.amu)})
            add_element_traits(ET.SubElement(ET.SubElement(res_types, "Residue",
                attrib={"name":"SLV"}), "Atom"), {"name":"CS", "type":"Solv"})

            add_element_traits(ET.SubElement(cust_nb_f, "Atom"), {"class":"CS", "flavor":"1"})

            if slv_potential == "LJ":
                str_eps_slv = str(eps_slv/unit.kilojoule_per_mole)
                str_rmin_slv = str(sigma_slv/unit.nanometer)
                add_element_traits(ET.SubElement(cust_nb_f, "GlobalParameter"), {"name":"eps_slv", "defaultValue":str_eps_slv})
                add_element_traits(ET.SubElement(cust_nb_f, "GlobalParameter"), {"name":"rmin_slv", "defaultValue":str_rmin_slv})
            elif slv_potential == "CS":
                # tabulated core-softened potential
                cs_f = ET.SubElement(cust_nb_f, "Function")
                add_element_traits(cs_f, {"name":"CS", "type":"Continuous1D",
                    "min":str(rmin/unit.nanometer), "max":str(rmax/unit.nanometer)})

                cs_tab = tabulated.Chaimovich_table(eps_ww, sigma_ww, B, r0, Delta, rmin, rmax, 0, 0, switch=False)
                cs_f.text = cs_tab

        # add bonded terms 
        add_bonded_term_items(n_beads, ff, atm_types, res_types, mass_ply)

    indent(ff)
    out_str = ET.tostring(ff).decode(sys.stdout.encoding)
    with open(saveas, "w") as fout:
        fout.write(out_str)

#def cv_force_xml(n_beads, cv_expr, cv_grid, Ucv_table):
#
#    ff = ET.Element("ForceField")
#    cust_cv_f = ET.SubElement(ff, "CustomCVForce", {"energy":"Table(Q)", })
#    Ucv_item = ET.SubElement(cust_cv_f, "Function", 
#            {"name":"Table", "type":"Continuous1D", 
#                "min":"{:.8}".format(cv_grid[0]),"max":"{:.8}".format(cv_grid[-1])})
#
#    table_str = ""
#    for i in range(len(Ucv_ext)):
#        table_str += "{:.8f}".format(Ucv_ext[i])
#        if (i % 10) == 0:
#            table_str += "\n"
#        else:
#            table_str += " "
#
#    Ucv_item.text = table_str[:-1]
#
#    cust_cv
#    cust_cv_f = ET.SubElement(ff, "CustomManyParticleForce", {"particlesPerSet":n_beads, "energy":cv_expr, "nh"})
#
#
#    indent(ff)
#    with open("test.xml", "w") as fout:
#        fout.write(ET.tostring(ff))



