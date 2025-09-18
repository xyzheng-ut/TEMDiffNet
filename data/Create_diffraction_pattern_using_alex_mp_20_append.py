import numpy as np
import matplotlib.pyplot as plt

from matplotlib.colors import LinearSegmentedColormap
import os
from pymatgen.analysis.diffraction.tem import TEMCalculator
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
import pandas as pd
from pymatgen.core import Structure
from monty.serialization import loadfn

SYMM_DATA = loadfn(os.path.join(os.path.dirname(__file__), "symm_data.json"))

def plt_tem_style(x,y, intensities,index,j):

    colors = ['white', 'white']
    custom_cmap = LinearSegmentedColormap.from_list("custom_cmap", colors, N=2)
    fig, ax = plt.subplots(figsize=(5.12, 5.12))
    ax.scatter(x, y,s=intensities*350, c=intensities,alpha=intensities,cmap=custom_cmap, marker='o') #,

    plt.xlim([-4, 4])
    plt.ylim([-4, 4])
    plt.grid(False)
    plt.gca().set_facecolor('black')
    plt.gca().set_aspect('equal', adjustable='box')

    plt.xticks([])
    plt.yticks([])
    plt.subplots_adjust(left=0, bottom=0, right=1, top=1, hspace=0, wspace=0)
    plt.savefig(rf'/home/rc/PythonProjects_tem2/alex_mp_20/diffraction_patterns/train/{index}-{j}.png', dpi=50)
    plt.close()


def get_bravais_lattices(num):
    if num in [1,2]:
        return "aP"
    elif num in [3,4,6,7,10,11,13,14]:
        return "mP"
    elif num in [5,8,9,12,15]:
        return "mS"
    elif num in np.linspace(16,19,4).astype(int) or num in np.linspace(25,34,10).astype(int) or num in np.linspace(47,62,16).astype(int):
        return "oP"
    elif num in [20,21,35,36,37,63,64,65,66,67,68]:
        return "oS-c"
    elif num in [38,39,40,41]:
        return "oS-a"
    elif num in [23,24,44,45,46,71,72,73,74]:
        return "oI"
    elif num in [22,42,43,69,70]:
        return "oF"
    elif num in [75,76,77,78,81,83,84,85,86,89,90,91,92,93,94,95,96,99,100,101,102,103,104,105,106,111,112,113,114,115,116,117,118] or num in np.linspace(123,138,16).astype(int):
        return "tP"
    elif num in [79,80,82,87,88,97,98,107,108,109,110,119,120,121,122,139,140,141,142]:
        return "tI"
    elif num in np.linspace(143,167,25).astype(int):
        return "hR"
    elif num in np.linspace(168,193,26).astype(int):
        return "hP"
    elif num in [195,198,200,201,205,207,208,212,213,215,218,221,222,223,224]:
        return "cP"
    elif num in [197,199,204,206,211,214,217,220,229,230]:
        return "cI"
    elif num in [196,202,203,209,210,216,219,225,226,227,228]:
        return "cF"
    elif num==194:
        return "hcp"
    else:
        print("error, symmetry number is invalid")


def sg2intlnum(sg):
    abbreviated_spacegroup_symbols = SYMM_DATA["abbreviated_spacegroup_symbols"]
    if sg in abbreviated_spacegroup_symbols:
        sg = abbreviated_spacegroup_symbols[sg]
    for n, v in SYMM_DATA["space_group_encoding"].items():
        if n == sg:
            intlnum = v["int_number"]

    return intlnum


if os.path.exists("label.txt"):
  os.remove("label_train.txt")
with open('label_train.txt', 'w') as txt:
    txt.write("index;formula;lattice_configuration;symmetry_number;material_id;a;b;c;alpha;beta;gamma\n")

path = r"/home/rc/PythonProjects_tem2/alex_mp_20/train.csv"  # change it
data = pd.read_csv(path)
cif = data["cif"]
index = data.values[:,0]
space_group = data["space_group"]
chemical_system = data["chemical_system"]
material_id = data["material_id"]
zone_axis = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [1, 0, 1], [0, 1, 1], [1, 1, 1], [1, 1, 2],
             [1, 2, 1], [2, 1, 1], [0, 1, 2], [0, 1, 3], [0, 2, 3], [0, 1, 4], [1, 1, 3], [1, 1, 4], [1, 2, 2], [1, 2, 3], [1, 3, 3], [2, 2, 3],
                    [2, 3, 3]]

for i in range(len(index)):
    path = r"/home/rc/PythonProjects_tem2/alex_mp_20/cif/train/%d.cif" % i
    structure = Structure.from_file(path)
    intl = sg2intlnum(space_group[i])
    lattice_conf = get_bravais_lattices(intl)
    sga = SpacegroupAnalyzer(structure)
    structure = sga.get_refined_structure()
    [a,b,c]=structure.lattice.abc
    [alpha, beta, gamma] = structure.lattice.angles

    for j in range(len(zone_axis_append)):
        # conventional_structure = sga.get_conventional_standard_structure()
        calculator = TEMCalculator(beam_direction=zone_axis_append[j])
        points = calculator.generate_points(-10, 11)
        tem_dots = calculator.tem_dots(structure, points)
        xs = [0]
        ys = [0]
        # hkls = []
        intensities = [1]
        for dot in tem_dots:
            xs.append(dot.position[0])
            ys.append(dot.position[1])
            # hkls.append(str(dot.hkl))
            intensities.append(dot.intensity)
        plt_tem_style(xs,ys,np.array(intensities), i, j+len(zone_axis))
