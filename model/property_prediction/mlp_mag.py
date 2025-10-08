import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from tensorflow.keras import regularizers
from tensorflow.keras.models import Sequential
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
import pandas as pd
import collections
import sklearn
import seaborn as sns
from sklearn import metrics
from sklearn.model_selection import train_test_split
from ChemicalSystemMultiHotEmbedding import ChemicalSystemMultiHotEmbedding
import shap
from pymatgen.core import Element
all_elements = [el.symbol for el in Element]
all_elements= all_elements[:101]
element_keys = np.linspace(0,len(all_elements)-1,len(all_elements)).astype(int)
element_mapping = dict(zip(all_elements, element_keys))

dic_lattice = {"aP": 0, "cF": 1, "cI": 2, "cP": 3, "hP": 4, "hR": 5, "hcp": 6, "mP": 7, "mS": 8, "oF": 9, "oI": 10,
               "oP": 11, "oS": 12, "tI": 13, "tP": 14}
lattice_name = list(dic_lattice.keys())

abcabg = [r"$a$", r"$b$", r"$c$", r"$\alpha$", r"$\beta$", r"$\gamma$"]
feature_name = all_elements + lattice_name + abcabg
print(feature_name)

# sns.set(color_codes=True)
# sns.set_theme(style="ticks")

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
print(tf.__version__)
print(tf.config.list_physical_devices())
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

print("tensorflow is running on GPU:", tf.config.list_physical_devices('GPU'))
gpus = tf.config.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
logical_gpus = tf.config.list_logical_devices('GPU')

colors = sns.color_palette("tab10")


def to_multi_hot(subsets, mapping):
    out = []
    for i in range(len(subsets)):
        subset = subsets[i]
        indices = [mapping[s] for s in subset]
        multi_hot = np.zeros(len(mapping), dtype=int)
        multi_hot[indices] = 1
        out.append(multi_hot)
    return np.array(out)


def normolize(x):
    mean = np.mean(x, axis=0)  # shape: (3,)
    std = np.std(x, axis=0)  # shape: (3,)
    x_normalized = (x - mean) / std
    return np.expand_dims(x_normalized, axis=-1), mean, std


def lattice2onehot(lattice_configuration):
    dic_lattice = {"aP": 0, "cF": 1, "cI": 2, "cP": 3, "hP": 4, "hR": 5, "hcp": 6, "mP": 7, "mS": 8, "oF": 9, "oI": 10,
                   "oP": 11, "oS": 12, "tI": 13, "tP": 14}
    lattice_onehot = []
    for i in range(len(lattice_configuration)):
        lattice_onehot.append(dic_lattice[lattice_configuration[i]])
    lattice_onehot = keras.utils.to_categorical(lattice_onehot, len(dic_lattice))

    return np.array(lattice_onehot)


def preprocess(x,y):
    x = tf.cast(x, dtype=tf.float32)
    y = tf.cast(y, dtype=tf.float32)
    y = tf.expand_dims(y, axis=-1)

    return x,y


def cal_r2(y, y_pre):

    if ~np.isnan(y).all():
        mask = ~tf.math.is_nan(y)
        masked_y = tf.boolean_mask(y, mask).numpy()
        masked_y_pre = tf.boolean_mask(y_pre, mask).numpy()
        r2 = metrics.r2_score(masked_y, masked_y_pre)
    else:
        r2 = 0.

    return r2


def load_dataset(path_train, path_val, batch_size=128):
    ds_train = pd.read_csv(path_train, delimiter=';')
    ds_val = pd.read_csv(path_val, delimiter=';')
    ds_train=pd.concat([ds_train, ds_val], ignore_index=True, sort=False)

    # input
    formula = ds_train["formula"].values
    lattice_configuration = ds_train["lattice_configuration"].values
    lattice_configuration[lattice_configuration == "oS-c"] = "oS"
    lattice_configuration[lattice_configuration == "oS-a"] = "oS"
    a = ds_train["a"].values
    b = ds_train["b"].values
    c = ds_train["c"].values
    alpha = ds_train["alpha"].values
    beta = ds_train["beta"].values
    gamma = ds_train["gamma"].values
    formula_multihot = to_multi_hot(ChemicalSystemMultiHotEmbedding.convert_to_list_of_str(formula), element_mapping)
    # formula_multihot = ChemicalSystemMultiHotEmbedding.sequences_to_multi_hot(ChemicalSystemMultiHotEmbedding.convert_to_list_of_str(formula))
    lattice_onehot = lattice2onehot(lattice_configuration)
    a_normalized, _, _ = normolize(a)
    b_normalized, _, _ = normolize(b)
    c_normalized, _, _ = normolize(c)
    alpha_normalized, _, _ = normolize(alpha)
    beta_normalized, _, _ = normolize(beta)
    gamma_normalized, _, _ = normolize(gamma)

    # output
    band_gap = ds_train["band_gap"].values/8
    bulk_modulus = ds_train["bulk_modulus"].values/500
    mag_density = ds_train["mag_density"].values/0.25

    band_gap_normalized, band_gap_mean, band_gap_std = normolize(band_gap)
    bulk_modulus_normalized, bulk_modulus_mean, bulk_modulus_std = normolize(bulk_modulus)
    mag_density_normalized, mag_density_mean, mag_density_std = normolize(mag_density)

    plt.hist(mag_density, density=True, bins=30)
    plt.show()

    x = np.concatenate((formula_multihot, lattice_onehot, a_normalized, b_normalized, c_normalized, alpha_normalized, beta_normalized, gamma_normalized), axis=-1)
    # y = np.concatenate((band_gap_normalized, bulk_modulus_normalized, mag_density_normalized), axis=-1)

    # do mask
    mask = ~np.isnan(mag_density).ravel()
    x = x[mask]
    mag_density = mag_density[mask]
    mag_density = np.abs(mag_density)

    # mag_density[mag_density < 0] = 0

    x, y = sklearn.utils.shuffle(x, mag_density,random_state=222)
    print("x shape: ", x.shape)
    print("y shape: ", y.shape)
    train_num = round(0.95 * len(x))
    train_dataset = tf.data.Dataset.from_tensor_slices((x[:train_num], y[:train_num])).map(
        preprocess).shuffle(buffer_size=1024).batch(batch_size)
    val_dataset = tf.data.Dataset.from_tensor_slices((x[train_num:], y[train_num:])).map(
        preprocess).batch(batch_size)

    return train_dataset, val_dataset, np.array([[band_gap_mean, band_gap_std], [bulk_modulus_mean, bulk_modulus_std],[mag_density_mean,mag_density_std]])


def load_dataset_nonzero(path_train, path_val, batch_size=128):
    ds_train = pd.read_csv(path_train, delimiter=';')
    ds_val = pd.read_csv(path_val, delimiter=';')
    ds_train=pd.concat([ds_train, ds_val], ignore_index=True, sort=False)

    # input
    formula = ds_train["formula"].values
    lattice_configuration = ds_train["lattice_configuration"].values
    lattice_configuration[lattice_configuration == "oS-c"] = "oS"
    lattice_configuration[lattice_configuration == "oS-a"] = "oS"
    a = ds_train["a"].values
    b = ds_train["b"].values
    c = ds_train["c"].values
    alpha = ds_train["alpha"].values
    beta = ds_train["beta"].values
    gamma = ds_train["gamma"].values

    formula_multihot = ChemicalSystemMultiHotEmbedding.sequences_to_multi_hot(ChemicalSystemMultiHotEmbedding.convert_to_list_of_str(formula))
    lattice_onehot = lattice2onehot(lattice_configuration)
    a_normalized, _, _ = normolize(a)
    b_normalized, _, _ = normolize(b)
    c_normalized, _, _ = normolize(c)
    alpha_normalized, _, _ = normolize(alpha)
    beta_normalized, _, _ = normolize(beta)
    gamma_normalized, _, _ = normolize(gamma)

    # output
    band_gap = ds_train["band_gap"].values/8
    bulk_modulus = ds_train["bulk_modulus"].values/500
    mag_density = ds_train["mag_density"].values/0.25

    band_gap_normalized, band_gap_mean, band_gap_std = normolize(band_gap)
    bulk_modulus_normalized, bulk_modulus_mean, bulk_modulus_std = normolize(bulk_modulus)
    mag_density_normalized, mag_density_mean, mag_density_std = normolize(mag_density)

    plt.hist(mag_density, density=True, bins=30)
    plt.show()

    x = np.concatenate((formula_multihot, lattice_onehot, a_normalized, b_normalized, c_normalized, alpha_normalized, beta_normalized, gamma_normalized), axis=-1)
    # y = np.concatenate((band_gap_normalized, bulk_modulus_normalized, mag_density_normalized), axis=-1)

    # do mask
    mask = ~np.isnan(mag_density).ravel()
    x = x[mask]
    mag_density = mag_density[mask]

    mask_nonzero = mag_density>0.
    x = x[mask_nonzero]
    mag_density = mag_density[mask_nonzero]

    x, y = sklearn.utils.shuffle(x, mag_density,random_state=222)
    print("x shape: ", x.shape)
    print("y shape: ", y.shape)
    train_num = round(0.95 * len(x))
    train_dataset = tf.data.Dataset.from_tensor_slices((x[:train_num], y[:train_num])).map(
        preprocess).shuffle(buffer_size=1024).batch(batch_size)
    val_dataset = tf.data.Dataset.from_tensor_slices((x[train_num:], y[train_num:])).map(
        preprocess).batch(batch_size)

    return train_dataset, val_dataset, np.array([[band_gap_mean, band_gap_std], [bulk_modulus_mean, bulk_modulus_std],[mag_density_mean,mag_density_std]])


def create_dir(path: str):
    """
    Create a directory of it does not exist
    :param path: Path to directory
    :return: None
    """
    if not os.path.exists(path):
        os.makedirs(path)


class PerformancePlotCallback(keras.callbacks.Callback):
    def __init__(self, data, mean_std):
        self.data = data
        self.mean_std = mean_std

    def on_epoch_end(self, epoch, logs={}):
        val_ds = self.data
        y_pred = self.model.predict(val_ds)
        y_val = [y for _, y in val_ds]
        y = np.concatenate(y_val, axis=0)
        y_pred = np.squeeze(y_pred)
        mean_std = self.mean_std
        r2 = cal_r2(y, y_pred)
        # plot
        g = sns.jointplot(x=y * 0.25,
                          y=y_pred * 0.25,
                          xlim=[-0.01, 0.26], ylim=[-0.01, 0.26],
                          color=colors[2],
                          # s=8, alpha=0.4,
                          height=4)

        g.set_axis_labels("Reference magnetic density [Å$^{-3}$]", "Predicted magnetic density [Å$^{-3}$]")
        plt.tight_layout()
        # plt.title(f'Epoch: {epoch}, r2: {r2:.4f}', size=6)
        plt.savefig(f'training_results/mlp/mag_density_{epoch}_{r2:.4f}.svg')
        plt.close()


class Classifer(keras.Model):
    def __init__(self):
        super(Classifer, self).__init__()
        self.fc1a = layers.Dense(256, activation="relu")
        self.fc1b = layers.Dense(256, activation="relu")
        self.fc1c = layers.Dense(256, activation="relu")
        self.fc2 = layers.Dense(1024, activation="relu")
        self.fc3 = layers.Dense(512, activation="relu")
        self.fc4 = layers.Dense(256, activation="relu")
        self.fc5 = layers.Dense(1, activation='sigmoid')
        self.loss_tracker = keras.metrics.Mean(name="loss")
        self.mse_metric = keras.metrics.BinaryAccuracy(name="binary_accuracy", dtype=None, threshold=0.5)

    def compile(self, optimizer):
        super().compile()
        self.optimizer = optimizer

    def call(self, x, training=None):
        x = tf.cast(x, dtype=tf.float32)
        x1,x2,x3 = x[:,:101], x[:,101:-6], x[:,-6:]
        x1 = self.fc1a(x1)
        x2 = self.fc1b(x2)
        x3 = self.fc1c(x3)
        x = tf.concat([x1, x2, x3], axis=-1)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        x = self.fc5(x)
        # x = self.noise(x)

        return x

    def train_step(self, data):
        # Unpack the data.
        x, y = data

        with tf.GradientTape() as tape:
            mask = ~tf.math.is_nan(y)
            y_pred = self(x, training=True)
            y = tf.cast(y != 0, tf.int32)
            y = tf.expand_dims(y, axis=-1)
            y = tf.cast(y, tf.int32)
            masked_y = tf.boolean_mask(y, mask)
            masked_y_pred = tf.boolean_mask(y_pred, mask)
            # Compute loss without reduction
            loss = tf.reduce_mean(keras.losses.binary_crossentropy(masked_y, masked_y_pred))

        grads = tape.gradient(loss, self.trainable_weights)
        self.optimizer.apply_gradients(
            zip(grads, self.trainable_weights)
        )

        # Monitor loss.
        self.loss_tracker.update_state(loss)
        self.mse_metric.update_state(masked_y, masked_y_pred)
        return {"loss": self.loss_tracker.result(), "binary_accuracy": self.mse_metric.result()}

    def test_step(self, data):
        # Unpack the data
        x, y = data
        mask = ~tf.math.is_nan(y)
        y = tf.cast(y != 0, tf.int32)
        y = tf.expand_dims(y, axis=-1)
        y = tf.cast(y, tf.int32)
        y_pred = self(x, training=False)
        masked_y = tf.boolean_mask(y, mask)
        masked_y_pred = tf.boolean_mask(y_pred, mask)
        # Compute predictions
        # Update the metrics.
        self.mse_metric.update_state(masked_y, masked_y_pred)
        # Return a dict mapping metric names to current value.
        # Note that it will include the loss (tracked in self.metrics).
        return {"binary_accuracy": self.mse_metric.result()}

    @property
    def metrics(self):
        # We list our `Metric` objects here so that `reset_states()` can be
        # called automatically at the start of each epoch
        # or at the start of `evaluate()`.
        # If you don't implement this property, you have to call
        # `reset_states()` yourself at the time of your choosing.
        return [self.loss_tracker, self.mse_metric]


class MLP(keras.Model):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1a = layers.Dense(256, activation="relu")
        self.fc1b = layers.Dense(256, activation="relu")
        self.fc1c = layers.Dense(256, activation="relu")
        self.fc2 = layers.Dense(1024, activation="relu")
        self.fc3 = layers.Dense(512, activation="relu")
        self.fc4 = layers.Dense(256, activation="relu")
        self.fc5 = layers.Dense(1, activation='softplus')
        self.loss_tracker = keras.metrics.Mean(name="loss")
        self.mse_metric = keras.metrics.MeanSquaredError(name="mse")

    def compile(self, optimizer):
        super().compile()
        self.optimizer = optimizer

    def call(self, x, training=None):
        x = tf.cast(x, dtype=tf.float32)
        x1,x2,x3 = x[:,:101], x[:,101:-6], x[:,-6:]
        x1 = self.fc1a(x1)
        x2 = self.fc1b(x2)
        x3 = self.fc1c(x3)
        x = tf.concat([x1, x2, x3], axis=-1)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        x = self.fc5(x)
        # x = self.noise(x)

        return x

    def train_step(self, data):
        # Unpack the data.
        x, y = data

        with tf.GradientTape() as tape:
            mask = ~tf.math.is_nan(y)
            y_pred = self(x, training=True)
            masked_y = tf.boolean_mask(y, mask)
            masked_y_pred = tf.boolean_mask(y_pred, mask)
            # Compute loss without reduction
            loss = tf.reduce_mean(keras.losses.mse(masked_y_pred, masked_y))

        grads = tape.gradient(loss, self.trainable_weights)
        self.optimizer.apply_gradients(
            zip(grads, self.trainable_weights)
        )

        # Monitor loss.
        self.loss_tracker.update_state(loss)
        self.mse_metric.update_state(masked_y_pred, masked_y)
        return {"loss": self.loss_tracker.result(), "mse": self.mse_metric.result()}

    def test_step(self, data):
        # Unpack the data
        x, y = data
        mask = ~tf.math.is_nan(y)
        y_pred = self(x, training=False)
        masked_y = tf.boolean_mask(y, mask)
        masked_y_pred = tf.boolean_mask(y_pred, mask)
        # Compute predictions
        # Update the metrics.
        self.mse_metric.update_state(masked_y, masked_y_pred)
        # Return a dict mapping metric names to current value.
        # Note that it will include the loss (tracked in self.metrics).
        return {"mse": self.mse_metric.result()}

    @property
    def metrics(self):
        # We list our `Metric` objects here so that `reset_states()` can be
        # called automatically at the start of each epoch
        # or at the start of `evaluate()`.
        # If you don't implement this property, you have to call
        # `reset_states()` yourself at the time of your choosing.
        return [self.loss_tracker, self.mse_metric]


def preprocess(x,y):
    x = tf.cast(x, dtype=tf.float32)
    y = tf.cast(y, dtype=tf.float32)

    return x,y


def parsing(mode="args"):
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, default=r"/datasets", help='Dataset path')
    parser.add_argument('--batch_size', type=int, default=128, help="batch size")
    parser.add_argument('--epochs', type=int, default=200, help="training epoch")
    parser.add_argument('--latent_dim', type=int, default=256, help="latent dim")
    args = parser.parse_args()
    return args


def main():
    args = parsing()
    create_dir("training_results/mlp")
    create_dir("training_results/mlp/ckpt")
    create_dir("training_results/classifier")
    create_dir("training_results/classifier/ckpt")

    # path_train = '/Users/user/Documents/papers/202503TEM_mattergen/dataset/mp/mp_data_exp.txt'
    # path_val = "/Users/user/Documents/papers/202503TEM_mattergen/dataset/mp/mp_data_theory.txt"
    path_train = "/Users/user/Documents/papers/202503TEM_mattergen/dataset/alex_mp_20/label_train.txt"
    path_val = "/Users/user/Documents/papers/202503TEM_mattergen/dataset/alex_mp_20/label_val.txt"
    train_ds,val_ds,mean_std = load_dataset(path_train, path_val)
    train_ds_nonzero,val_ds_nonzero,mean_std_nonzero = load_dataset_nonzero(path_train, path_val)

    early_stop = keras.callbacks.EarlyStopping(monitor='val_binary_accuracy', mode='max', patience=10)

    classifier = Classifer()
    classifier.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.0001),
    )
    history_classifier = classifier.fit(train_ds, validation_data=val_ds, epochs=args.epochs,
                                        callbacks=[early_stop]
                      )
    classifier.save_weights('training_results/classifier/ckpt/classifier.weights.h5')
    hist_df_classifier = pd.DataFrame(history_classifier.history)
    # Save to CSV
    hist_df_classifier.to_csv("training_results/classifier/training_history.csv", index=True)

    early_stop = keras.callbacks.EarlyStopping(monitor='loss', mode='min', patience=10)
    mlp = MLP()
    mlp.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.0001),
    )
    performance_simple = PerformancePlotCallback(val_ds_nonzero, mean_std_nonzero)
    history = mlp.fit(train_ds_nonzero, validation_data=val_ds_nonzero, epochs=args.epochs,
                      callbacks=[performance_simple, early_stop]
                      )
    mlp.save_weights('training_results/mlp/ckpt/mlp.weights.h5')
    # Convert to DataFrame
    hist_df = pd.DataFrame(history.history)

    # Save to CSV
    hist_df.to_csv("training_results/mlp/training_history.csv", index=True)

    y_val = [y for _, y in val_ds]
    y_val = np.concatenate(y_val, axis=0)

    y_pre = []
    for step, (x,y) in enumerate(val_ds):
        is_magnetic = classifier.predict(x) > 0.5
        # Initialize all outputs as 0
        magnetic_density_pred = np.zeros(len(x))
        # Predict values only for magnetic samples
        if np.any(is_magnetic):
            magnetic_density_pred[is_magnetic[:, 0]] = mlp.predict(x[is_magnetic[:, 0]]).flatten()
        y_pre.append(magnetic_density_pred)
    y_pre = np.concatenate(y_pre, axis=0)

    r2 = cal_r2(y_pre, y_val)
    # plot
    g = sns.jointplot(x=y_val * 0.25,
                      y=y_pre * 0.25,
                      xlim=[-0.01, 0.26], ylim=[-0.01, 0.26],
                      color=colors[2],
                      s=8, alpha=0.4,
                      height=4)

    g.set_axis_labels("Reference magnetic density [Å$^{-3}$]", "Predicted magnetic density [Å$^{-3}$]")
    plt.tight_layout()
    plt.title(f'r2: {r2:.4f}', size=6)
    plt.savefig(f'training_results/mag_density_test.svg')
    plt.close()

    # shap
    X_np, y_np = [], []
    for x_batch, y_batch in train_ds:
        X_np.extend(x_batch.numpy())
        y_np.extend(y_batch.numpy())

    X_np = np.array(X_np)
    print(X_np.shape)
    y_np = np.array(y_np)

    explainer = shap.Explainer(mlp, X_np, feature_names=feature_name)
    shap_values = explainer(X_np)
    shap.plots.beeswarm(shap_values, show=False, max_display=15, color_bar_label="Value of magnetic density", plot_size=(6, 4))

    plt.savefig(f"training_results/mlp/shap_summary_plot.svg", bbox_inches='tight')  # dpi=300,
    plt.close()



if __name__ == '__main__':
    main()