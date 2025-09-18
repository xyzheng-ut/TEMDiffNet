import numpy as np
import pandas as pd
import keras
import os
import cv2


def load_image(path, index, dim, j):
    dim = dim
    filename = str(index)+"-"+str(j) + '.png'
    img = cv2.imread(os.path.join(path, filename), 0)
    img = cv2.resize(img, (dim, dim), interpolation=cv2.INTER_AREA)
    img = np.expand_dims(img, axis=-1)
    return img / 255.


learning_rate = 0.0005
weight_decay = 0.0001
num_epochs = 200
batch_size = 64
data_num = 100000
dim = 64
dic_lattice = {"aP": 0, "cF": 1, "cI": 2, "cP": 3, "hP": 4, "hR": 5, "hcp": 6, "mP": 7, "mS": 8, "oF": 9, "oI": 10,
               "oP": 11, "oS": 12, "tI": 13, "tP": 14}
dic_zone_axis = {"100": 0, "010": 1, "001": 2, "110": 3, "101": 4, "011": 5, "111": 6, "112": 7, "121": 8, "211": 9,
                "012": 10, "013":11, "023":12, "014": 13, "113": 14, "114":15, "122": 16, "123": 17, "133": 18, "223": 19, "233": 20}


data = pd.read_csv("/home/rc/PythonProjects_tem2/alex_mp_20/label_train.txt", delimiter=";")
# index lattice_configuration  zone_axis  symmetry_number material_id
lattice_configuration = data["lattice_configuration"].values
lattice_configuration[lattice_configuration == "oS-c"] = "oS"
lattice_configuration[lattice_configuration == "oS-a"] = "oS"
a = data["a"].values
b = data["b"].values
c = data["c"].values
alpha = data["alpha"].values
beta = data["beta"].values
gamma = data["gamma"].values
material_id = data["material_id"]
index = data["index"]
print("max data length: ", len(index))

abc = np.array([a, b, c])
abg = np.array([alpha, beta, gamma])
y_para = np.concatenate((abc, abg), axis=0).T

img_path = r"/home/rc/PythonProjects_tem2/alex_mp_20/diffraction_patterns/train"

max_num = 10240

for i in range(len(index)//max_num):
    images = np.zeros([max_num * len(dic_zone_axis), dim, dim, 1])
    y_lattice = []
    y_zone = []
    y_abcabg=[]
    for j in range(max_num):
        for k in range(len(dic_zone_axis)):
            img = load_image(img_path, index[i*max_num+j], dim, k)
            images[ j *len(dic_zone_axis ) +k] = img
            y_lattice.append(dic_lattice[lattice_configuration[i*max_num+j]])
            y_zone.append(k)
            y_abcabg.append(y_para[i * max_num + j])

    np.save(f"dataset/imgs_10240_{i}.npy", images)
    np.save(f"dataset/y_lattice_10240_{i}.npy", y_lattice)
    np.save(f"dataset/y_zone_10240_{i}.npy", y_zone)
    np.save(f"dataset/y_abcabg_10240_{i}.npy", np.array(y_abcabg))
