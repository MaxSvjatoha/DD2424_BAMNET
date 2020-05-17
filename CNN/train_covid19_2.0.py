# -*- coding: utf-8 -*-
"""
Spyder Editor

"""
print("Deep Learning Project")
# import the necessary modules
from tensorflow.compat.v1.keras.preprocessing.image import ImageDataGenerator
from tensorflow.compat.v1.keras.applications import VGG16
from tensorflow.compat.v1.keras.layers import AveragePooling2D
from tensorflow.compat.v1.keras.layers import BatchNormalization
from tensorflow.compat.v1.keras.layers import Dropout
from tensorflow.compat.v1.keras.layers import Reshape, Flatten
from tensorflow.compat.v1.keras.layers import Dense
from tensorflow.compat.v1.keras.layers import Input
from tensorflow.compat.v1.keras.models import Model
from tensorflow.compat.v1.keras.optimizers import Adam
from tensorflow.compat.v1.keras.utils import to_categorical
from tensorflow.python.client import device_lib 
from tensorflow.compat.v1.keras.layers import Conv2D, Conv2DTranspose
from tensorflow.compat.v1.keras.layers import LeakyReLU
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import cv2
import os
import tensorflow.compat.v1 as tf
import keras

config = tf.ConfigProto(
        device_count = {'GPU': 0}
    )
sess = tf.Session(config=config)

print(device_lib.list_local_devices())

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
	help="path to input dataset")
ap.add_argument("-p", "--plot", type=str, default="plot.png",
	help="path to output loss/accuracy plot")
ap.add_argument("-m", "--model", type=str, default="covid19.model",
	help="path to output loss/accuracy plot")
#args = vars(ap.parse_args())


# initialize the initial learning rate, number of epochs to train for, and batch size
INIT_LR = 1e-4 # initial learning rate
EPOCHS = 50 #25 # epochs
BS = 40 # batch size
MOM = 0.833 # momentum
DEC = 1e-5 # decay
VAL_PERCENT = 0.3 # percent of data to use as test data
DROP = 0.5 # dropout rate
init_stddev= 0.02

# grab the list of images in our dataset directory, then initialize the list of data (i.e., images) and class images
print("[INFO] loading images...")
imagePaths = list(paths.list_images("dataset"))
data = []
labels = []


# loop over the image paths
for imagePath in imagePaths:
	# extract the class label from the filename
	label = imagePath.split(os.path.sep)[-2]

	# load the image, swap color channels, and resize it to be a fixed
	# 224x224 pixels while ignoring aspect ratio
	image = cv2.imread(imagePath)
	image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	image = cv2.resize(image, (224, 224))

	# update the data and labels lists, respectively
	data.append(image)
	labels.append(label)


# convert the data and labels to NumPy arrays while scaling the pixel
# intensities to the range [0, 255]
data = np.array(data) / 255.0
labels = np.array(labels)

# perform one-hot encoding on the labels
lb = LabelBinarizer()
labels = lb.fit_transform(labels)
labels = to_categorical(labels)

# partition the data into training and testing splits using 90% of
# the data for training and the remaining 10% for testing
(trainX, testX, trainY, testY) = train_test_split(data, labels,
	test_size=VAL_PERCENT, stratify=labels, random_state=42)

print(len(trainX))
print(len(trainY))
print(len(testX))
print(len(testY))

# initialize the training data augmentation object
trainAug = ImageDataGenerator(
	rotation_range=15,
	fill_mode="nearest")

# load the VGG16 network, ensuring the head FC layer sets are left off
baseModel = VGG16(weights="imagenet", include_top=False,
	input_tensor=Input(shape=(224, 224, 3)))

print("baseModel.output:", baseModel.output)
print("baseModel.output.shape:", baseModel.output.shape)

# construct the head of the model that will be placed on top of the the base model
headModel = baseModel.output
headModel = Conv2D(224, kernel_size=[5, 5],
                      strides=[2, 2],
                      padding="same",
                      kernel_initializer=keras.initializers.TruncatedNormal(stddev=init_stddev),
                      )(headModel)
headModel = BatchNormalization(momentum=MOM)(headModel)
headModel = LeakyReLU(alpha=0.2)(headModel)
headModel = Conv2D(448, kernel_size=[5, 5],
                      strides=[2, 2],
                      padding="same",
                      kernel_initializer=keras.initializers.TruncatedNormal(stddev=init_stddev),
                      )(headModel)
headModel = BatchNormalization(momentum=MOM)(headModel)
headModel = LeakyReLU(alpha=0.2)(headModel)
headModel = AveragePooling2D(pool_size=(2, 2))(headModel)
headModel = Flatten(name="flatten")(headModel)
headModel = Dense(64, activation="relu")(headModel)
headModel = BatchNormalization(momentum=MOM)(headModel)
headModel = LeakyReLU(alpha=0.2)(headModel)
headModel = Dense(32, activation="relu")(headModel)
headModel = BatchNormalization(momentum=MOM)(headModel)
headModel = LeakyReLU(alpha=0.2)(headModel)
headModel = Dense(32, activation="relu")(headModel)
headModel = BatchNormalization(momentum=MOM)(headModel)
headModel = LeakyReLU(alpha=0.2)(headModel)
headModel = Dropout(DROP)(headModel)
headModel = Dense(2, activation="softmax")(headModel)

# place the head FC model on top of the base model (this will become the actual model we will train)
model = Model(inputs=baseModel.input, outputs=headModel)
model.summary()

# loop over all layers in the base model and freeze them so they will *not* be updated during the first training process
for layer in baseModel.layers:
	layer.trainable = False

# compile our model
print("[INFO] compiling model...")
opt = Adam(lr=INIT_LR, decay=DEC)
model.compile(loss="binary_crossentropy", optimizer=opt,
	metrics=["accuracy"])

# train the head of the network
print("[INFO] training head...")
H = model.fit_generator(
	trainAug.flow(trainX, trainY, batch_size=BS),
	steps_per_epoch=len(trainX) // BS,
	validation_data=(testX, testY),
	validation_steps=len(testX) // BS,
	epochs=EPOCHS)

# make predictions on the testing set
print("[INFO] evaluating network...")
predIdxs = model.predict(testX, batch_size=BS)

# for each image in the testing set we need to find the index of the label with corresponding largest predicted probability
predIdxs = np.argmax(predIdxs, axis=1)

# show a nicely formatted classification report
print(classification_report(testY.argmax(axis=1), predIdxs,
	target_names=lb.classes_))

# compute the confusion matrix and and use it to derive the raw accuracy, sensitivity, and specificity
cm = confusion_matrix(testY.argmax(axis=1), predIdxs)
total = sum(sum(cm))
acc = (cm[0, 0] + cm[1, 1]) / total
sensitivity = cm[0, 0] / (cm[0, 0] + cm[0, 1])
specificity = cm[1, 1] / (cm[1, 0] + cm[1, 1])

# show the confusion matrix, accuracy, sensitivity, and specificity
print(cm)
print("acc: {:.4f}".format(acc))
print("sensitivity: {:.4f}".format(sensitivity))
print("specificity: {:.4f}".format(specificity))

# plot the training loss and accuracy
N = EPOCHS
plt.style.use("ggplot")
plt.figure()
#plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
#plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc")
plt.title("Training Accuracy on COVID-19 Dataset")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig("plot_acc_BS_" + str(BS) + "_LR_" + str(INIT_LR) + "_MOM_" + str(MOM) + "_DEC_" + str(DEC) + "_EP_" + str(EPOCHS) + ".png")

plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
#plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")
#plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss on COVID-19 Dataset")
plt.xlabel("Epoch #")
plt.ylabel("Loss")
plt.legend(loc="lower left")
plt.savefig("plot_loss_BS_" + str(BS) + "_LR_" + str(INIT_LR) + "_MOM_" + str(MOM) + "_DEC_" + str(DEC) + "_EP_" + str(EPOCHS) + ".png")

# serialize the model to disk
print("[INFO] saving COVID-19 detector model...")
model.save("model", save_format="h5")