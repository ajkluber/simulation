import os
import glob
import numpy as np
import subprocess as sb

import heat_capacity_vs_T

if __name__ == "__main__":
    if not os.path.exists("temp_dirs"):
        tdirs = glob.glob("T_*_0")                    
        tdirs.sort()
        tdir_string = "\n".join(tdirs)
        with open("temp_dirs", "w") as fout:
            fout.write(tdir_string)
    else:
        with open("temp_dirs", "r") as fin:
            tdirs = fin.read().split()

    print "calculating energy"
    for i in range(len(tdirs)):
        if not os.path.exists("{}/Etot.dat".format(tdirs[i])):
            os.chdir(tdirs[i])
            #calculate total potential energy
            sb.call("g_energy_sbm -f ener.edr -o Etot -xvg none << HERE \n Potential \n HERE", shell=True)
            Etot = np.loadtxt("Etot.xvg",usecols=(1,))
            np.save("Etot.npy", Etot)
            os.chdir("..")

    heat_capacity_vs_T.calculate_Cv("temp_dirs", engfile="Etot.npy")

