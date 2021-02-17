import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import sklearn.model_selection
import tensorflow as tf
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from sklearn.metrics import classification_report
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import (
    Input,
    Conv2D,
    Dense,
    Dropout,
    MaxPooling2D,
    GlobalAveragePooling2D,
)
from tensorflow.keras.layers.experimental.preprocessing import Rescaling
from tensorflow.keras.models import load_model
from tqdm import tqdm

MODEL_SAVE_FOLDER_PATH = "./model/"
if not os.path.exists(MODEL_SAVE_FOLDER_PATH):
    os.mkdir(MODEL_SAVE_FOLDER_PATH)

model_path = MODEL_SAVE_FOLDER_PATH + "{epoch:02d}-{val_loss:.4f}.hdf5"

cb_checkpoint = ModelCheckpoint(
    filepath=model_path, monitor="val_loss", verbose=1, save_best_only=True
)
es_checkpoint = EarlyStopping(monitor="val_loss", mode="min", verbose=1, patience=7)


def image_to_numpy():
    images_list = os.listdir(image_directory)
    print(len(images_list))

    exterior = np.empty((10001, 300, 300, 3), dtype=np.uint8)
    interior = np.empty((14982, 300, 300, 3), dtype=np.uint8)
    food = np.empty((20017, 300, 300, 3), dtype=np.uint8)

    ei = 0
    ii = 0
    fi = 0
    for img in tqdm(images_list):
        if "exterior" in img:
            im = cv2.imread(image_directory + img)
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            exterior[ei, ...] = im
            ei += 1
        elif "interior" in img:
            im = cv2.imread(image_directory + img)
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            interior[ii, ...] = im
            ii += 1
        elif "food" in img:
            im = cv2.imread(image_directory + img)
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            food[fi, ...] = im
            fi += 1

    label_food = np.zeros((20017, 3))
    for a in range(20017):
        label_food[a][0] = 1
    label_interior = np.zeros((14982, 3))
    for a in range(14982):
        label_interior[a][1] = 1
    label_exterior = np.zeros((10001, 3))
    for a in range(10001):
        label_exterior[a][2] = 1

    images = np.concatenate((food, interior, exterior), axis=0)
    labels = np.concatenate((label_food, label_interior, label_exterior), axis=0)

    np.save(numpy_dir + "images.npy", images)
    np.save(numpy_dir + "labels.npy", labels)

    print(images.shape)
    print(labels.shape)


def l_image(config):
    images = np.load(numpy_dir + "images.npy")
    labels = np.load(numpy_dir + "labels.npy")
    # 음식 0 실내 1 실외 2
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(
        images,
        labels,
        test_size=config["validation_split"],
        shuffle=True,
        random_state=123,
    )

    np.save(numpy_dir + "X_train.npy", X_train)
    np.save(numpy_dir + "X_test.npy", X_test)
    np.save(numpy_dir + "y_train.npy", y_train)
    np.save(numpy_dir + "y_test.npy", y_test)
    return X_train, X_test, y_train, y_test


def train_model(X_train, X_test, y_train, y_test, config):
    model = tf.keras.Sequential(
        [
            Input(shape=(300, 300, 3), name="Input_layer"),
            Rescaling(1.0 / 255),
            Conv2D(64, 3, activation="relu", name="Conv_1"),
            Dropout(0.25),
            Conv2D(64, 3, activation="relu", name="Conv_2"),
            Dropout(0.25),
            MaxPooling2D(),
            Conv2D(128, 3, activation="relu", name="Conv_3"),
            Dropout(0.25),
            MaxPooling2D(),
            Conv2D(256, 3, activation="relu", name="Conv_4"),
            Dropout(0.25),
            MaxPooling2D(),
            GlobalAveragePooling2D(),
            Dense(512, activation="relu"),
            Dropout(0.3),
            Dense(512, activation="relu"),
            Dropout(0.3),
            Dense(config["num_class"], activation="softmax"),
        ]
    )

    model.compile(
        optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
    )

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_test, y_test),
        batch_size=config["batch_size"],
        epochs=config["epoch"],
        callbacks=[cb_checkpoint, es_checkpoint],
    )

    model.summary()

    return model, history


def plot_loss_curve(history):
    plt.figure(figsize=(15, 10))
    plt.plot(history["loss"])
    plt.plot(history["val_loss"])
    plt.title("model loss")
    plt.ylabel("loss")
    plt.xlabel("epoch")
    plt.legend(["train", "test"], loc="upper right")
    plt.show()

    print(history)
    print("train loss=", history["loss"][-1])
    print("validation loss=", history["val_loss"][-1])


def plot_acc_curve(history):
    plt.figure(figsize=(15, 10))
    plt.plot(history["accuracy"])
    plt.plot(history["val_accuracy"])
    plt.title("model accuracy")
    plt.ylabel("acc")
    plt.xlabel("epoch")
    plt.legend(["train", "test"], loc="upper right")
    plt.show()

    print(history)
    print("train acc=", history["accuracy"][-1])
    print("validation acc=", history["val_accuracy"][-1])


def predict_image_sample(X_test, y_test, y_pred_onehot):
    wrong = []
    wrong_label = []
    good = []
    good_label = []
    y_test = np.array(y_test)
    for i, onehot_element in enumerate(y_pred_onehot):
        y_test_element = y_test[i]
        print(y_test_element, onehot_element)
        if not (
            (y_test_element[0] == onehot_element[0])
            and (y_test_element[1] == onehot_element[1])
            and (y_test_element[2] == onehot_element[2])
        ):
            if len(wrong_label) < 2:
                wrong.append(i)
                wrong_label.append((y_test_element, onehot_element))
        else:
            if len(good_label) < 2:
                good.append(i)
                good_label.append((y_test_element, onehot_element))

        if len(wrong_label) == 2 and len(good_label) == 2:
            break

    print(len(wrong_label))
    print(len(good_label))
    f = plt.figure()
    ax1 = f.add_subplot(2, 2, 1)
    ax1.imshow(X_test[good[0]])
    ax1.set_title("correct")
    ax1.set_xlabel(good_label[0])
    print(good_label[0])

    ax2 = f.add_subplot(2, 2, 2)
    ax2.imshow(X_test[good[1]])
    ax2.set_title("correct")
    ax2.set_xlabel(good_label[1])
    print(good_label[1])

    ax3 = f.add_subplot(2, 2, 3)
    ax3.imshow(X_test[wrong[0]])
    ax3.set_title("wrong")
    ax3.set_xlabel(wrong_label[0])
    print(wrong_label[0])

    ax4 = f.add_subplot(2, 2, 4)
    ax4.imshow(X_test[wrong[1]])
    ax4.set_title("wrong")
    ax4.set_xlabel(wrong_label[1])
    print(wrong_label[1])

    plt.show()


if __name__ == "__main__":
    numpy_dir = "C:/Users/GY/Desktop/건국대학교/강의자료/3-2/Introduction to Data Science/images_numpy/"
    image_directory = "C:/Users/GY/Desktop/DEV/images/"
    model_directory = "C:/Users/GY/Desktop/건국대학교/강의자료/3-2/Introduction to Data Science/model/"
    image_class_name = ["food", "interior", "exterior"]
    # keras.Sequential(
    #     [
    #         layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical"),
    #         layers.experimental.preprocessing.RandomRotation(0.2),
    #     ]
    # )
    config = {
        "mode": "test",
        "batch_size": 20,
        "epoch": 30,
        "num_class": 3,
        "validation_split": 0.1,
    }

    ##최초 1회만 실행
    # image_to_numpy()

    # X_train, X_test, y_train, y_test = l_image(config)
    X_train = np.load(numpy_dir + "X_train.npy")
    X_test = np.load(numpy_dir + "X_test.npy")
    y_train = np.load(numpy_dir + 'y_train.npy')
    y_test = np.load(numpy_dir + "y_test.npy")
    # X_train = tf.convert_to_tensor(X_train)
    # X_test = tf.convert_to_tensor(X_test)
    # y_train = tf.convert_to_tensor(y_train)
    # y_test = tf.convert_to_tensor(y_test)
    # print(X_train.shape)
    print(X_test.shape)
    # print(y_train.shape)
    print(y_test.shape)

    if config["mode"] == "test":
        model = load_model("model-201611294")
        model.summary()
        model.evaluate(X_test, y_test, batch_size=config["batch_size"])
        y_pred = model.predict(X_test, batch_size=config["batch_size"])
        print(y_pred.shape)
        y_pred_onehot = np.zeros(y_pred.shape)
        print(y_pred_onehot.shape)
        for i, y in enumerate(y_pred):
            y_pred_onehot[i][np.argmax(y, axis=0)] = 1
        print(y_pred_onehot)
        print(
            classification_report(y_test, y_pred_onehot, target_names=image_class_name)
        )
        predict_image_sample(X_test, y_test, y_pred_onehot)

    elif config["mode"] == "train":
        model, history = train_model(X_train, X_test, y_train, y_test, config)
        plot_loss_curve(history.history)
        plot_acc_curve(history.history)
        model.save("model-201611294_final-1421.model")

    elif config["mode"] == "load":
        model = load_model("25-0.2736.hdf5")
        model.save("model-201611294_final_2.model")
        model.evaluate(X_test, y_test, batch_size=config["batch_size"])
