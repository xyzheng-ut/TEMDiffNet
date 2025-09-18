import numpy as np
import matplotlib.pyplot as plt
from keras.src.ops import dtype
from skimage import filters
import cv2
import pandas as pd
import keras
import sklearn
import tensorflow as tf
import os
from keras import layers
import seaborn as sns
from sklearn.metrics import confusion_matrix
import time
import datetime
import re


tf.get_logger().setLevel("ERROR")
print(tf.__version__)
print(tf.config.list_physical_devices())
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
print("tensorflow is running on GPU:", tf.config.list_physical_devices('GPU'))
gpus = tf.config.list_physical_devices('GPU')

dic_lattice = {"aP": 0, "cF": 1, "cI": 2, "cP": 3, "hP": 4, "hR": 5, "hcp": 6, "mP": 7, "mS": 8, "oF": 9, "oI": 10,
               "oP": 11, "oS": 12, "tI": 13, "tP": 14}
dic_zone_axis = {"100": 0, "010": 1, "001": 2, "110": 3, "101": 4, "011": 5, "111": 6, "112": 7, "121": 8, "211": 9,
                "012": 10, "013":11, "023":12, "014": 13, "113": 14, "114":15, "122": 16, "123": 17, "133": 18, "223": 19, "233": 20}


for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.8)
config = tf.compat.v1.ConfigProto(gpu_options=gpu_options)
session = tf.compat.v1.Session(config=config)


def load_dataset_from_npy(path, index, batch_size, img_num=7):
    imgs = np.load(os.path.join(path, f"imgs_10240_{index}.npy"))
    y_lattice = np.load(os.path.join(path, f"y_lattice_10240_{index}.npy"))

    y_lattice = keras.utils.to_categorical(y_lattice, len(dic_lattice))


    new_imgs = np.zeros((len(imgs)//len(dic_zone_axis)*(len(dic_zone_axis)//img_num), 64,64,img_num))
    new_y_lattice = []
    new_y_abcabg = []
    for i in range(len(imgs)//len(dic_zone_axis)):
        indices = np.random.permutation(len(dic_zone_axis))
        for j in range(len(dic_zone_axis)//img_num):
            for k in range(img_num):
                new_imgs[i*(len(dic_zone_axis)//img_num)+j,:,:,k] = np.squeeze(imgs[i*(len(dic_zone_axis)) + indices[k+j*img_num]])

            new_y_lattice.append(y_lattice[i*len(dic_zone_axis)])

    x, y_lattice = sklearn.utils.shuffle( new_imgs, np.array(new_y_lattice),random_state=22)
    train_num = round(0.9 * len(x))
    train_dataset = tf.data.Dataset.from_tensor_slices((x[:train_num], y_lattice[:train_num])).map(
        preprocess).shuffle(buffer_size=1024).batch(batch_size)
    val_dataset = tf.data.Dataset.from_tensor_slices((x[train_num:], y_lattice[train_num:])).map(
        preprocess).batch(batch_size)
    return train_dataset, val_dataset


def load_image(path, index, dim):
    dim = dim
    filename = str(index) + '.png'
    img = cv2.imread(os.path.join(path, filename), 0)
    img = cv2.resize(img, (dim, dim), interpolation=cv2.INTER_AREA)
    # img = np.expand_dims(img, axis=-1)
    return img / 255.


def preprocess(x, y1):
    x = tf.cast(x, dtype=tf.float32)
    y1 = tf.cast(y1, dtype=tf.float32)

    return x, y1


def activation_block(x):
    x = layers.Activation("gelu")(x)
    return layers.BatchNormalization()(x)


def conv_stem(x, filters: int, patch_size: int):
    x = layers.Conv2D(filters, kernel_size=patch_size, strides=patch_size)(x)
    return activation_block(x)


def conv_mixer_block(x, filters: int, kernel_size: int):
    # Depthwise convolution.
    x0 = x
    x = layers.DepthwiseConv2D(kernel_size=kernel_size, padding="same")(x)
    x = layers.Add()([activation_block(x), x0])  # Residual.

    # Pointwise convolution.
    x = layers.Conv2D(filters, kernel_size=1)(x)
    x = activation_block(x)

    return x


class EmbeddedLayer(keras.Layer):
    def __init__(self, i=0):
        super().__init__()
        self.i = i
    def call(self, x):
        return tf.expand_dims(x[:,:,:,self.i],axis=-1)


class ConcateLayer(keras.layers.Layer):  # Correct inheritance
    def call(self, inputs):  # Use a single 'inputs' argument
        return tf.concat(inputs, axis=-1)  # Use a list, not a tuple


def get_conv_mixer_256_8(
    image_size=(128,128,1), filters=128, depth=4, kernel_size=3, patch_size=4, num_classes=15,num_classes2=13
):
    """ConvMixer-256/8: https://openreview.net/pdf?id=TVHS5Y4dNvM.
    The hyperparameter values are taken from the paper.get_conv_mixer_256_8
    """
    inputs = keras.Input(image_size)
    # x0 = EmbeddedLayer()(inputs)

    # Extract patch embeddings.
    x0 = conv_stem(inputs, filters, patch_size)


    # ConvMixer blocks.
    for _ in range(depth):
        x0 = conv_mixer_block(x0, filters, kernel_size)

    # Classification block.
    x0 = layers.GlobalAvgPool2D()(x0)

    output1 = layers.Dense(num_classes, activation="softmax")(x0)
    output2 = layers.Dense(num_classes2, activation="softmax")(x0)
    return keras.Model(inputs, [x0,output1, output2])


def MLP(pretrained_model, image_size, num_classes=15):
    inputs = keras.Input(image_size)
    out=[]
    for i in range(image_size[-1]):
        xi = EmbeddedLayer(i=i)(inputs)
        xi,_,_ = pretrained_model(xi)
        out.append(xi)
    out=ConcateLayer()(out)
    # out = layers.Dense(128, activation="relu")(out)
    output1 = layers.Dense(num_classes, activation="softmax")(out)

    return keras.Model(inputs, output1)

def main():
    start_time = time.time()

    os.makedirs("./results", exist_ok=True)
    os.makedirs("./ckpt", exist_ok=True)
    os.makedirs("./figure/lattice", exist_ok=True)
    os.makedirs("./figure/lattice_para", exist_ok=True)
    with open('./results/loss.txt', 'w') as txt:
        txt.write("epoch;loss;accuracy_lattice\n")

    learning_rate = 0.0001
    weight_decay = 0.0001
    num_epochs = 200
    batch_size = 64
    num = 6
    dim = 64
    data_path = "/home/rc/PythonProjects_tem2/alex_mp_20/dataset"
    optimizer = keras.optimizers.AdamW(
        learning_rate=learning_rate, weight_decay=weight_decay
    )
    model_pretrained = get_conv_mixer_256_8(image_size=[dim,dim,1], num_classes=len(dic_lattice), num_classes2=len(dic_zone_axis))
    model_pretrained.load_weights("/home/rc/PythonProjects_tem2/convmixer/0730pretrained/ckpt/74.weights.h5")
    # model_pretrained.trainable = False
    model_pretrained.summary()
    model = MLP(model_pretrained, image_size=[dim,dim,num], num_classes=15)
    model.summary()
    print("started training...")
    for epoch in range(num_epochs):
        loss_total = 0
        for data_index in range(27):  # 27 datasets
            train_dataset, _ = load_dataset_from_npy(data_path, data_index, batch_size,num)
            for step, (x, y1) in enumerate(train_dataset):
                with tf.GradientTape() as tape:
                    logits1 = model(x)
                    loss = tf.reduce_mean(tf.losses.categorical_crossentropy(y1, logits1))
                    loss_total+=loss.numpy()

                grads = tape.gradient(loss, model.trainable_variables)
                optimizer.apply_gradients(zip(grads, model.trainable_variables))

        loss = loss_total/step/27
        print(epoch, "loss: ", loss)

        for data_index in range(27):

            _, val_dataset = load_dataset_from_npy(data_path, data_index, batch_size, num)

            # plot test
            total_sum = 0
            total_accuracy_lattice = 0
            lattice_pre = []
            lattice_y = []

            for x, y1 in val_dataset:

                pred1  = model(x)
                accuracy_lattice = tf.reduce_mean(tf.metrics.CategoricalAccuracy()(y1, pred1))

                total_sum += 1

                total_accuracy_lattice += accuracy_lattice
                pred1 = tf.math.argmax(pred1, axis=-1)
                y1 = tf.math.argmax(y1, axis=-1)
                lattice_pre = tf.concat([lattice_pre, pred1], axis=0)
                lattice_y = tf.concat([lattice_y, y1], axis=0)

            total_accuracy_lattice = total_accuracy_lattice / total_sum


        print(epoch, "accuracy_lattice: ", total_accuracy_lattice.numpy())


        with open('./results/loss.txt', 'a') as f:
            f.write(str(epoch)+";"+str(loss) + ";" + str(total_accuracy_lattice.numpy()) +"\n")

        # plot

        cm = confusion_matrix(lattice_y.numpy(), lattice_pre.numpy())
        unique, counts = np.unique(lattice_y, return_counts=True)
        # figure=sns.heatmap(cm / counts, annot=True,
        #             fmt='.2%', cmap='Blues')
        figure=sns.heatmap(cm/counts[:, np.newaxis], cmap='Blues')
        figure.set_xticklabels(list(dic_lattice.keys()))
        figure.set_yticklabels(list(dic_lattice.keys()), rotation=90)
        figure.set_xlabel("True Bravais lattice")
        figure.set_ylabel("Predicted Bravais lattice")
        plt.tight_layout()
        figure = figure.get_figure()
        figure.savefig('./figure/lattice/cm_%d.png' % epoch)
        plt.close()

        model.save_weights('ckpt/%d.weights.h5' % epoch)

    with open('./results/summary.txt', 'w') as f:
        f.write(str(datetime.datetime.now()) + "\n")
        f.write(f"Tensorflow version: " + tf.__version__ + "\n")
        f.write("Training time: %s seconds" % (time.time() - start_time))


if __name__ == "__main__":
    main()