# Import des librairies nécessaires
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

# Vérification que l'on utilise bien TensorFLow 2.0
assert hasattr(tf, "function")


# On utilise le mnist de keras de reconnaissance de chiffres manuscrits
mnist = tf.keras.datasets.mnist
(images, targets), (images_test, targets_test) = mnist.load_data()

# On charge seulement une partie des données
images = images[:10000]
targets = targets [:10000]

# On convertit la forme du dataset en float
images = images.reshape(-1, 784)
images = images.astype(float)
images_test = images_test.reshape(-1, 784)
images_test = images_test.astype(float)
scaler = StandardScaler()
images = scaler.fit_transform(images)
images_test = scaler.transform(images_test)
print(images.shape)
print(targets.shape)


# On affiche la première image pour vérifier à quoi ressemble la première image (test seulement)
# plt.imshow(np.reshape(images[0], (28, 28)), cmap="binary")
# plt.show()

# On crée notre réseau de neurones.
# Ici on aura une première couche cachée de 256 valeurs puis une seconde de 128.
# La couche de sortie aura 10 noeuds pour les 10 valeurs de chiffre possibles
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(256, activation="relu"))
model.add(tf.keras.layers.Dense(128, activation="relu"))
model.add(tf.keras.layers.Dense(10, activation="softmax"))
model_output = model.predict(images[0:1])
# print(model_output, targets[0:1])

# Résumé du réseau de neurones, optionnel, pour test notamment
# model.summary()

#Compilation du réseau
model.compile(
    loss="sparse_categorical_crossentropy",
    optimizer="sgd",
    metrics=["accuracy"]
)

# Lancement de l'entraînement du réseau
history = model.fit(images, targets, epochs=10, validation_split=0.2)

#Mise en forme des graphiques de perte et de précision
loss_curve = history.history["loss"]
acc_curve = history.history["accuracy"]
loss_val_curve = history.history["val_loss"]
acc_val_curve = history.history["val_accuracy"]

plt.plot(loss_curve, label="Train")
plt.plot(loss_val_curve, label="Val")
plt.legend(loc='upper left')
plt.title("Loss")
plt.show()
plt.plot(acc_curve, label="Train")
plt.plot(acc_val_curve, label="Val")
plt.legend(loc='upper left')
plt.title("Accuracy")
plt.show()

#Derniers tests
loss, acc = model.evaluate(images_test, targets_test)
print("Test Loss", loss)
print("Test Accuracy", acc)

#Lucien BOUYEURE et Guillaume ROUX, Devoir 3 IA