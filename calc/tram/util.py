import os

def get_organized_temps(tempfile):
    """Get directory names by temperature"""
    with open(tempfile, "r") as fin:
        temperature_dirs = fin.read().split()

    organized_temps = {}
    for i in range(len(temperature_dirs)):
        temp_dir = temperature_dirs[i]
        temp_T = float((temp_dir.split("/")[0]).split("_")[1])
        if not temp_T in organized_temps.keys():
            organized_temps[temp_T] = [temp_dir]
        else:
            organized_temps[temp_T].append(temp_dir)

    return organized_temps
