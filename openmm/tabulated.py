import numpy as np

import simtk.unit as unit

################### UTILITIES ###########################
def create_table_string(y):
    tab_string = ""
    temp_str = ""
    for i in range(len(y)):
        temp_str += " {:.12f}".format(y[i])
        if len(temp_str) >= 80:
            tab_string += temp_str + "\n"
            temp_str = ""
    return tab_string

def apply_switch(y, r, r_switch, r_cut):
    """Multiple potential by switching function to force it to zero at r_cut"""
    y_sw = unit.Quantity(np.zeros(len(r)), unit.kilojoule_per_mole)
    y_sw[r < r_switch] = y[r < r_switch]
    sw_region = (r >= r_switch) & (r <= r_cut)
    y_sw[sw_region] = y[sw_region]*switch(r[sw_region], r_switch, r_cut)
    return y_sw

def switch(r, r_switch, r_cut):
    """Switch between 1 (r = r_switch) and 0 (r = r_cut)"""
    x = (r - r_switch)/(r_cut - r_switch)
    return 1 - 6*(x**5) + 15*(x**4) - 10*(x**3)

def wca_table(sigma, eps, rmin, rmax, n_points=1000):
    """Generate tabulated values of Weeks-Chandler-Andersen (WCA) potential"""

    r = unit.Quantity(np.linspace(rmin/unit.nanometer, rmax/unit.nanometer, n_points), unit.nanometer)
    y = WCA(r, sigma, eps)
    return create_table_string(y/unit.kilojoule_per_mole) 

def LJ_table(eps, sigma, rmin, rmax, r_switch, r_cut, n_points=1000, switch=False):
    """Generate tabulated values of a Lennard-Jones (LJ) potential that switches to zero"""
    r = unit.Quantity(np.linspace(rmin/unit.nanometer, rmax/unit.nanometer, n_points), unit.nanometer)

    y = LJ(r, eps, sigma)
    if switch:
        y = apply_switch(y, r, r_switch, r_cut)
    return create_table_string(y/unit.kilojoule_per_mole)

def Chaimovich_table(eps_ww, sigma_ww, B, r0, Delta, rmin, rmax, r_switch, r_cut, n_points=1000, switch=False):
    """Isotropic 'core-softened' potential"""
    r = unit.Quantity(np.linspace(rmin/unit.nanometer, rmax/unit.nanometer, n_points), unit.nanometer)

    y = Chaimovich_CS(r, eps_ww, sigma_ww, B, r0, Delta)
    if switch:
        y = apply_switch(y, r, r_switch, r_cut)
    return create_table_string(y/unit.kilojoule_per_mole)

################### INTERACTION FUNCTIONS #################
def WCA(r, sigma, eps):
    val = np.zeros(len(r), float)
    r0 = sigma*(2**(1./6))
    val[r < r0] = 4.*((sigma/r[r < r0])**12 - (sigma/r[r < r0])**6) + 1.
    return eps*val

def LJ(r, eps, sigma):
    return 4*eps*((sigma/r)**12 - (sigma/r)**6)

def Gaussian(r, B, r0, Delta):
    return B*np.exp(-((r - r0)/Delta)**2)

def Chaimovich_CS(r, eps_ww, sigma_ww, B, r0, Delta):
    """Chaimovich and Shell isotropic water model"""
    return LJ(r, eps_ww, sigma_ww) + Gaussian(r, B, r0, Delta)

if __name__ == "__main__":
    # I love Ellen!!!!
    sigma = 0.373*unit.nanometer
    eps = 0.13986*unit.kilocalorie_per_mole

    #rmin = 0.05*unit.nanometer
    rmin = 0.75*sigma
    rmax = 1.1*unit.nanometer
    r_cut = 0.9*unit.nanometer
    r_switch = r_cut - 0.1*unit.nanometer

    y = LJswitch_table(eps, sigma, rmin, rmax, r_switch, r_cut)

    #y_kJ = y.value_in_unit(unit.kilojoule_per_mole)
    #r_nm = r.value_in_unit(unit.nanometer)

    raise SystemExit
    import matplotlib.pyplot as plt
    plt.plot(r/unit.nanometer, y/unit.kilocalorie_per_mole, label="WCA")
    plt.plot(r/unit.nanometer, y2/unit.kilocalorie_per_mole, label="LJ12")
    plt.ylim(-1, 2)
    plt.legend(loc=1)
    plt.axvline(sigma/unit.nanometer, ls='--', color='k', label=r"$\sigma$")
    plt.axvline(0.75*sigma/unit.nanometer, ls='-.', color='r', label=r"$0.75\sigma$")
    plt.show()
