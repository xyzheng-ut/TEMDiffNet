import pandas as pd

path = r"/home/rc/PythonProjects_tem2/alex_mp_20/train.csv"  # change it
data = pd.read_csv(path)
cif = data["cif"]
index = data.values[:,0]

for i in range(len(index)):
    file = r"/home/rc/PythonProjects_tem2/alex_mp_20/cif/train/%d.cif" % i
    with open(file, mode="w") as f:
        f.write(cif[i])