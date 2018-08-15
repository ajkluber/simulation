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

def add_toy_polymer_ff_items(n_beads, ff, atm_types, res_types, soft_bonds=False):

    if soft_bonds:
        sigma_ply, eps_ply, mass_ply, bonded_params = toy_polymer_params_soft_bonds()
    else:
        sigma_ply, eps_ply, mass_ply, bonded_params = toy_polymer_params()

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
    r0, kb, theta0, ka = toy_polymer_bonded_params()
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

    assert ply_potential in ["LJ", "WCA"]
    assert slv_potential in ["LJ", "CS"]

    # polymer-polymer and polymer-solvent interactions are the same: either LJ or WCA
    # solvent-solvent interactions are either LJ or CS
    eng_str = ""
    if slv_potential == "LJ":
        eps_slv = kwargs["eps_slv"]
        sigma_slv = kwargs["sigma_slv"]
        mass_slv = kwargs["mass_slv"]
        eng_str += "areslv*eps_slv*(5*((rmin_slv/r)^12) - 6*((rmin_slv/r)^10)) + "
    else:
        eng_str += "areslv*CS(r) + "
        eps_ww, sigma_ww, B, r0, Delta, mass_slv = CS_water_params()

    if ply_potential == "LJ":
        eps_ply = kwargs["eps_ply"]
        sigma_ply = kwargs["sigma_ply"]
        mass_ply = kwargs["mass_ply"]
        eng_str += "(1 - areslv)*eps_ply*(5*((rmin_ply/r)^12) - 6*((rmin_ply/r)^10))"
    else:
        eng_str += "(1 - areslv)*WCA(r)"
        sigma_ply, eps_ply, mass_ply, bonded_params = toy_polymer_params()

    eng_str += "; areslv=step((flavor1 + flavor2) - 1.5)"

    rmin = 0.1*unit.nanometers
    rmax = 1.5*unit.nanometers

    # forcefield xml tags
    ff = ET.Element("ForceField")
    atm_types = ET.SubElement(ff, "AtomTypes")
    res_types = ET.SubElement(ff, "Residues")

    # solvent parameters
    add_element_traits(ET.SubElement(atm_types, "Type"),
        {"name":"Solv", "class": "CS", "element":"Sv", "mass":str(mass_slv/unit.amu)})
    add_element_traits(ET.SubElement(ET.SubElement(res_types, "Residue",
        attrib={"name":"SLV"}), "Atom"), {"name":"CS", "type":"Solv"})

    cust_nb_f = ET.SubElement(ff, "CustomNonbondedForce", {"energy":eng_str,"bondCutoff":"3"})

    # each particle class (solvent=CS, polymer=PL) has a 'flavor'. Flavors
    # determine the interaction
    add_element_traits(ET.SubElement(cust_nb_f, "PerParticleParameter"), {"name":"flavor"})
    add_element_traits(ET.SubElement(cust_nb_f, "Atom"), {"class":"CS", "flavor":"1"})
    add_element_traits(ET.SubElement(cust_nb_f, "Atom"), {"class":"PL", "flavor":"0"})

    if ply_potential == "LJ":
        str_eps_ply = str(eps_ply/unit.kilojoule_per_mole)
        str_rmin_ply = str(sigma_ply/unit.nanometer)
        add_element_traits(ET.SubElement(cust_nb_f, "GlobalParameter"), {"name":"eps_ply", "defaultValue":str_eps_ply})
        add_element_traits(ET.SubElement(cust_nb_f, "GlobalParameter"), {"name":"rmin_ply", "defaultValue":str_rmin_ply})
    else:
        # tabulated WCA potential
        wca_f = ET.SubElement(cust_nb_f, "Function")
        add_element_traits(wca_f, {"name":"WCA", "type":"Continuous1D",
            "min":str(rmin/unit.nanometer), "max":str(rmax/unit.nanometer)})

        wca_tab = tabulated.wca_table(sigma_ply, eps_ply, rmin, rmax)
        wca_f.text = wca_tab

    if slv_potential == "LJ":
        str_eps_slv = str(eps_slv/unit.kilojoule_per_mole)
        str_rmin_slv = str(sigma_slv/unit.nanometer)
        add_element_traits(ET.SubElement(cust_nb_f, "GlobalParameter"), {"name":"eps_slv", "defaultValue":str_eps_slv})
        add_element_traits(ET.SubElement(cust_nb_f, "GlobalParameter"), {"name":"rmin_slv", "defaultValue":str_rmin_slv})
    else:
        # tabulated core-softened potential
        cs_f = ET.SubElement(cust_nb_f, "Function")
        add_element_traits(cs_f, {"name":"CS", "type":"Continuous1D",
            "min":str(rmin/unit.nanometer), "max":str(rmax/unit.nanometer)})

        cs_tab = tabulated.Chaimovich_table(eps_ww, sigma_ww, B, r0, Delta, rmin, rmax, 0, 0, switch=False)
        cs_f.text = cs_tab

    # add polymer only items
    add_toy_polymer_ff_items(n_beads, ff, atm_types, res_types, mass_ply)

    indent(ff)
    with open(saveas, "w") as fout:
        fout.write(ET.tostring(ff))

def LJ_toy_polymer_LJ_water(n_beads, cutoff, solvent_params, saveas="ff_cgs.xml", soft_bonds=False):
    """Build xml forcefield file for toy polymer
    
    Parameters
    ----------
    n_beads : int
        Number of monomers.
    cutoff : float
        Cutoff radius for long range interactions.
    
    """
    ## DEPRECATED

    #sigma_slv, eps_slv, mass_slv = LJ_water_params()
    sigma_slv, eps_slv, mass_slv = solvent_params
    sigma_ply, eps_ply, mass_ply, bonded_params = LJ_polymer_params()

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
    #  1) Strong Lennard-Jones (LJ) interactions between solvent atoms
    #  2) Weak LJ for polymer-polymer and polymer-solvent interactions
    cust_nb_f = ET.SubElement(ff, "CustomNonbondedForce",
            {"energy":"areslv*eps_slv*(5*((rmin_slv/r)^12) - 6*((rmin_slv/r)^10)) + (1 - areslv)*eps_ply*(5*((rmin_ply/r)^12) - 6*((rmin_ply/r)^10)); areslv=step((flavor1 + flavor2) - 1.5);", 
                "bondCutoff":"3"})
            #{"energy":"areslv*epsslv*((sigmaslv/r)^12 - (sigmaslv/r)^6) + (1 - areslv)*4*epsply*((sigmaply/r)^12 - (sigmaply/r)^6); areslv=step((flavor1 + flavor2) - 1.5); ", 

    # LJ parameters are global
    str_eps_slv = str(eps_slv/unit.kilojoule_per_mole)
    str_rmin_slv = str(sigma_slv/unit.nanometer)

    str_eps_ply = str(eps_ply/unit.kilojoule_per_mole)
    str_rmin_ply = str(sigma_ply/unit.nanometer)

    add_element_traits(ET.SubElement(cust_nb_f, "GlobalParameter"), {"name":"eps_slv", "defaultValue":str_eps_slv})
    add_element_traits(ET.SubElement(cust_nb_f, "GlobalParameter"), {"name":"rmin_slv", "defaultValue":str_rmin_slv})
    add_element_traits(ET.SubElement(cust_nb_f, "GlobalParameter"), {"name":"eps_ply", "defaultValue":str_eps_ply})
    add_element_traits(ET.SubElement(cust_nb_f, "GlobalParameter"), {"name":"rmin_ply", "defaultValue":str_rmin_ply})

    # each particle class (solvent=LJ, polymer=PL) has a 'flavor'. Flavors
    # determine the interaction
    add_element_traits(ET.SubElement(cust_nb_f, "PerParticleParameter"), {"name":"flavor"})
    add_element_traits(ET.SubElement(cust_nb_f, "Atom"), {"class":"LJ", "flavor":"1"})
    add_element_traits(ET.SubElement(cust_nb_f, "Atom"), {"class":"PL", "flavor":"0"})

    # OpenMM switches interaction to be exactly zero at cutoff distance
    r_switch = cutoff - 0.11*unit.nanometer
    r_cut = cutoff - 0.01*unit.nanometer

    # add polymer only items
    add_toy_polymer_ff_items(n_beads, ff, atm_types, res_types, soft_bonds=soft_bonds)

    indent(ff)
    with open(saveas, "w") as fout:
        fout.write(ET.tostring(ff))

def toy_polymer_LJ_water(n_beads, cutoff, saveas="ff_cgs.xml", soft_bonds=False):
    """Build xml forcefield file for toy polymer
    
    Parameters
    ----------
    n_beads : int
        Number of monomers.
    cutoff : float
        Cutoff radius for long range interactions.
    
    """
    ## DEPRECATED

    sigma_slv, eps_slv, mass_slv = LJ_water_params()
    if soft_bonds:
        sigma_ply, eps_ply, mass_ply, bonded_params = toy_polymer_params_soft_bonds()
    else:
        sigma_ply, eps_ply, mass_ply, bonded_params = toy_polymer_params()

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
            #{"energy":"areslv*4*epsslv*((sigmaslv/r)^12 - (sigmaslv/r)^6) + (1 - areslv)*4*epsply*((sigmaply/r)^12 - (sigmaply/r)^6); areslv=step((flavor1 + flavor2) - 1.5); ", 

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

    ## DEPRECATED

    eps_ww, sigma_ww, B, r0, Delta, mass_slv = CS_water_params()
    if soft_bonds:
        sigma_ply, eps_ply, mass_ply, bonded_params = toy_polymer_params_soft_bonds()
    else:
        sigma_ply, eps_ply, mass_ply, bonded_params = toy_polymer_params()

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

def many_body_toy_system(n_beads, cutoff, saveas="ff_ex.xml"):
    """Build xml forcefield file for toy system
    
    Parameters
    ----------
    n_beads : int
        Number of monomers.
    
    """

    #TODO

    sigma = 0.3*unit.nanometer
    eps = 0.5*unit.kilojoule_per_mole
    mass = 37.*unit.amu
    r0 = 0.153*unit.nanometer 

    rmin = 0.2*unit.nanometer
    rmax = 1.5*cutoff

    # forcefield xml tags
    ff = ET.Element("ForceField")
    atm_types = ET.SubElement(ff, "AtomTypes")
    res_types = ET.SubElement(ff, "Residues")

    # solvent parameters
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

def read_polymer_params(n_beads, s_frames, bonds, angles, non_bond_wca, non_bond_gaussian, method):

    savedir = "coeff_vs_s"
    if bonds:
        savedir += "_bond"
    if angles:
        savedir += "_angle"
    if non_bond_wca:
        savedir += "_wca"
    if non_bond_gaussians:
        savedir += "_gauss"
    savedir += "_" + method

    n_params = np.sum([ int(x == True) for x in [bonds, angles, non_bond_wca, non_bond_gaussians]])

    #for n in range(n_params):
    #np.save("{}/coeff_{}_s_{}.npy".format(savedir, n+1, s_frames), temp_coeff)

