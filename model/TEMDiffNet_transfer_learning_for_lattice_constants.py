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
from sklearn import metrics


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


def load_dataset_from_npy(path, index, batch_size, img_num=5):
    imgs = np.load(os.path.join(path, f"imgs_10240_{index}.npy"))
    y_abcabg = np.load(os.path.join(path, f"y_abcabg_10240_{index}.npy"))
    y_abcabg[:,-3:] = y_abcabg[:,-3:]/150
    y_abcabg[:, :3] = y_abcabg[:, :3]/50
    y_abcabg = y_abcabg[:,-3:]
    y_lattice = np.load(os.path.join(path, f"y_lattice_10240_{index}.npy"))

    mask = np.where((y_lattice == 0) | (y_lattice == 7) | (y_lattice == 8))[0]
    y_abcabg = y_abcabg[mask]
    imgs = imgs[mask]

    new_imgs = np.zeros((len(imgs)//len(dic_zone_axis)*(len(dic_zone_axis)//img_num), 64,64,img_num))
    new_y_abcabg = []
    for i in range(len(imgs)//len(dic_zone_axis)):
        indices = np.random.permutation(len(dic_zone_axis))
        for j in range(len(dic_zone_axis)//img_num):
            for k in range(img_num):
                new_imgs[i*(len(dic_zone_axis)//img_num)+j,:,:,k] = np.squeeze(imgs[i*(len(dic_zone_axis)) + indices[k+j*img_num]])

            new_y_abcabg.append(y_abcabg[i*len(dic_zone_axis)])

    x, y_abcabg = sklearn.utils.shuffle( new_imgs, np.array(new_y_abcabg),random_state=22)
    train_num = round(0.9 * len(x))
    train_dataset = tf.data.Dataset.from_tensor_slices((x[:train_num], y_abcabg[:train_num])).map(
        preprocess).shuffle(buffer_size=1024).batch(batch_size)
    val_dataset = tf.data.Dataset.from_tensor_slices((x[train_num:], y_abcabg[train_num:])).map(
        preprocess).batch(batch_size)
    return train_dataset, val_dataset


def para_filter(lattice_pre, a_pred, b_pred, c_pred, alpha_pred, beta_pred, gamma_pred):
    lattices = list(dic_lattice.keys())
    a_out = np.zeros(len(a_pred))
    b_out = np.zeros(len(a_pred))
    c_out = np.zeros(len(a_pred))
    alpha_out = np.zeros(len(a_pred))
    beta_out = np.zeros(len(a_pred))
    gamma_out = np.zeros(len(a_pred))
    for i in range(len(lattice_pre)):
        a = a_pred[i]
        b = b_pred[i]
        c = c_pred[i]
        alpha = alpha_pred[i]
        beta = beta_pred[i]
        gamma = gamma_pred[i]
        ind = int(lattice_pre[i])
        lattice_conf = lattices[ind]
        if lattice_conf in ["mP","mS"]:
            alpha = gamma = 90
        elif lattice_conf in ["oP","oI","oF","oS"]:
            alpha = beta = gamma = 90
        elif lattice_conf in ["tP", "tI"]:
            alpha = beta = gamma = 90
            a = b = (a+b)/2
        elif lattice_conf in ["hR"]:
            a =b =c = (a+b+c)/3
            alpha = beta = gamma = (alpha+beta+gamma)/3
        elif lattice_conf in ["hP", "hcp"]:
            a = b = (a+b)/2
            alpha = beta  = 90
            gamma = 120
        elif lattice_conf in ["cP", "cI", "cF"]:
            a=b=c=(a+b+c)/3
            alpha = beta = gamma = 90

        a_out[i] = a
        b_out[i] = b
        c_out[i] = c
        alpha_out[i] = alpha
        beta_out[i] = beta
        gamma_out[i] = gamma
    return [a_out,b_out,c_out,alpha_out,beta_out,gamma_out]


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
    The hyperparameter values are taken from the paper.
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
    out = layers.Dense(128, activation="relu")(out)
    output = layers.Dense(3, activation="softplus")(out)
    return keras.Model(inputs, output)


def cal_r2(y, y_pre):

    if ~np.isnan(y).all():
        mask = ~tf.math.is_nan(y)
        masked_y = tf.boolean_mask(y, mask).numpy()
        masked_y_pre = tf.boolean_mask(y_pre, mask).numpy()
        r2 = metrics.r2_score(masked_y, masked_y_pre)
    else:
        r2 = 0.

    return r2


def main():
    start_time = time.time()

    os.makedirs("./results", exist_ok=True)
    os.makedirs("./ckpt", exist_ok=True)
    os.makedirs("./figure/lattice", exist_ok=True)
    os.makedirs("./figure/lattice_para", exist_ok=True)
    with open('./results/loss.txt', 'w') as txt:
        txt.write("epoch;loss;mse\n")

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
    model_pretrained.load_weights("/home/rc/PythonProjects_tem2/convmixer/0716pretrained/ckpt/19.weights.h5")
    # model_pretrained.trainable = False
    model_pretrained.summary()
    model = MLP(model_pretrained, image_size=[dim,dim,num], num_classes=15)
    model.summary()
    print("started training...")
    for epoch in range(num_epochs):
        loss_total = 0
        for data_index in range(27):  # 27 datasets
            train_dataset, _ = load_dataset_from_npy(data_path, data_index, batch_size,num)
            for step, (x, y2) in enumerate(train_dataset):
                with tf.GradientTape() as tape:
                    logits2 = model(x)
                    loss2 = tf.losses.MSE(y2, logits2)
                    loss = tf.reduce_mean(loss2)
                    loss_total+=loss.numpy()

                grads = tape.gradient(loss, model.trainable_variables)
                optimizer.apply_gradients(zip(grads, model.trainable_variables))

        loss = loss_total/step/27
        print(epoch, "loss: ", loss)

        for data_index in range(27):

            _, val_dataset = load_dataset_from_npy(data_path, data_index, batch_size, num)

            # plot test
            total_sum = 0

            # lattice parameters
            total_mse =0
            a_real = []
            a_pred = []
            b_real = []
            b_pred = []
            c_real = []
            c_pred = []


            for x, y2 in val_dataset:

                pred2  = model(x)

                total_sum += 1
                # lattice parameters
                mse_lattice_para = tf.reduce_mean(tf.losses.MSE(y2, pred2))
                total_mse += tf.reduce_sum(mse_lattice_para)

                a_real = np.append(a_real, y2[:, 0].numpy() *150)
                b_real = np.append(b_real, y2[:, 1].numpy() * 150)
                c_real = np.append(c_real, y2[:, 2].numpy() * 150)

                a_pred = np.append(a_pred, pred2[:, 0].numpy() * 150)
                b_pred = np.append(b_pred, pred2[:, 1].numpy() * 150)
                c_pred = np.append(c_pred, pred2[:, 2].numpy() * 150)

            # a_pred, b_pred, c_pred, alpha_pred, beta_pred, gamma_pred = para_filter(lattice_pre.numpy(),
            #                                                                         a_pred, b_pred, c_pred, alpha_pred,
            #                                                                         beta_pred, gamma_pred)
            total_mse = total_mse/ total_sum


        print(epoch, "total_mse: ", total_mse.numpy())

        with open('./results/loss.txt', 'a') as f:
            f.write(str(epoch)+";"+str(loss) + ";" + str(total_mse.numpy()) +"\n")

        # plot
        r2 = cal_r2(a_real/150, a_pred/150)
        g = sns.jointplot(x=a_real, y=a_pred,
                          xlim=[0, 150], ylim=[0, 150],
                          color="C0", height=3.5)
        g.set_axis_labels(r"True $\alpha$ [°]", r"Predicted $\alpha$ [°]")
        plt.title(f'Epoch: {epoch}, r2: {r2:.4f}', size=6)
        plt.tight_layout()
        plt.savefig('figure/lattice_para/alpha_%d.png' % epoch)
        plt.close()

        r2 = cal_r2(b_real/150, b_pred/150)
        g = sns.jointplot(x=b_real, y=b_pred,
                        xlim=[0, 150], ylim=[0, 150],
                          color="C1", height=3.5)
        g.set_axis_labels(r"True $\beta$ [°]", r"Predicted $\beta$ [°]")
        plt.title(f'Epoch: {epoch}, r2: {r2:.4f}', size=6)
        plt.tight_layout()
        plt.savefig('figure/lattice_para/beta_%d.png' % epoch)
        plt.close()

        r2 = cal_r2(c_real/150, c_pred/150)
        g = sns.jointplot(x=c_real, y=c_pred,
                          xlim=[0, 150], ylim=[0, 150],
                          color="C2", height=3.5)
        g.set_axis_labels(r"True $\gamma$ [°]", r"Predicted $\gamma$ [°]")
        plt.title(f'Epoch: {epoch}, r2: {r2:.4f}', size=6)
        plt.tight_layout()
        plt.savefig('figure/lattice_para/gamma_%d.png' % epoch)
        plt.close()

        model.save_weights('ckpt/%d.weights.h5' % epoch)

    with open('./results/summary.txt', 'w') as f:
        f.write(str(datetime.datetime.now()) + "\n")
        f.write(f"Tensorflow version: " + tf.__version__ + "\n")
        f.write("Training time: %s seconds" % (time.time() - start_time))


if __name__ == "__main__":
    main()