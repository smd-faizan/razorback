#!/usr/bin/env python

# from graphics import *
import csv
import json

from tensorflow import keras
from tensorflow.python.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow.python.keras.metrics import top_k_categorical_accuracy
from tensorflow.keras.models import load_model

from Geometry import *
import numpy as np
from PIL import Image, ImageDraw
import ast

from model import MyModelGenerator

from test import tester
from datetime import datetime
import sys,os

TEST_FILENAME = "../data/test_simplified.csv"
TRAIN_FOLDER = "../data/train_simplified/"
MODEL_SAVE_PATH = "../savedModels/model-"
EXTENSION = ".csv"
IMAGES_PER_CLASS = sys.maxint # Integer.max for all images
imheight = 256
imwidth = 256
num_classes = 340
# Training parameters
BATCH_SIZE = 32
EPOCHS = 22

def draw_it(strokes):
    image = Image.new("P", (256,256), color=255)  # BACKGROUND COLOR is BLACK?
    image_draw = ImageDraw.Draw(image)

    for stroke in strokes:
        for i in xrange(0, len(stroke.points) - 1):
            A = stroke.points[i]
            B = stroke.points[i + 1]
            # line = Line(Point(A.x, A.y), Point(B.x, B.y))
            image_draw.line([A.x, A.y, B.x, B.y], fill=0, width=5)   # FILL THE LINE WITH WHITE COLOR?, width=5
    image = image.resize((imheight, imwidth))
    return np.array(image)/255.


def load_file(filename):
    d = {}
    with open(filename) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            if line_count == 0:
                line_count += 1
            else:
                data = json.loads(row[1])
                drawing = Drawing()
                drawing.country = row[0]
                drawing.keyID = row[2]
                drawing.recognized = True if row[3].lower() == 'true' else False
                drawing.timestamp = row[4]
                drawing.word = row[5]
                for stroke in data:
                    xCordiantes = stroke[0]
                    yCordinates = stroke[1]
                    st = Stroke()
                    for i in xrange(len(xCordiantes)):
                        st.addPoint(xCordiantes[i], yCordinates[i])
                    drawing.addStroke(st)
                d[drawing.keyID] = drawing
                line_count += 1
            if line_count>IMAGES_PER_CLASS:
                break
        print("Processed " + str(line_count) + "lines.")

    result = [] #np.array() # should this be an np array?
    for key, value in d.iteritems():
        # visualize(key, value);
        np_image = draw_it(value.strokes)
        # visualize_np_array(np_image, key)
        result.append(np_image)

    del d

    return np.array(result)


def load_all_files():
    labels = {}
    i=0;
    grand = []
    for filename in os.listdir(TRAIN_FOLDER):
        if filename.endswith(EXTENSION):
            images = load_file(TRAIN_FOLDER+filename)
            labelarray = np.full((images.shape[0], 1), i)
            images = images.reshape(images.shape[0], -1)
            images = np.concatenate((labelarray, images), axis=1)
            for image in images:
                grand.append(image)
            labels[i] = filename
            i += 1
    return grand

def shuffle(grand):
    np.random.shuffle(grand)

def divide(grand):
    valfrac = 0.1
    cutpt = int(valfrac * grand.shape[0])
    Y_train, X_train = grand[cutpt:, 0], grand[cutpt:, 1:]
    Y_val, X_val = grand[0:cutpt, 0], grand[0:cutpt, 1:]

    del grand
    return X_train, Y_train, X_val, Y_val


def top_3_accuracy(x, y):
    t3 = top_k_categorical_accuracy(x, y, 3)
    return t3

def main():
    grand = load_all_files()
    grand = np.array(grand)
    shuffle(grand)
    X_train, Y_train, X_val, Y_val = divide(grand)

    y_train = keras.utils.to_categorical(Y_train, num_classes)
    X_train = X_train.reshape(X_train.shape[0], imheight, imwidth, 1)
    y_val = keras.utils.to_categorical(Y_val, num_classes)
    X_val = X_val.reshape(X_val.shape[0], imheight, imwidth, 1)

    print(y_train.shape, "\n",
          X_train.shape, "\n",
          y_val.shape, "\n",
          X_val.shape)

    reduceLROnPlat = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3,
                                       verbose=1, mode='auto', min_delta=0.005, cooldown=5, min_lr=0.0001)
    earlystop = EarlyStopping(monitor='val_top_3_accuracy', mode='max', patience=5)
    callbacks = [reduceLROnPlat, earlystop]

    modelGenerator = MyModelGenerator(imheight, imwidth, 340);
    model = modelGenerator.getModel();

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy', top_3_accuracy])

    model.fit(x=X_train, y=y_train,
              batch_size=BATCH_SIZE,
              epochs=EPOCHS,
              validation_data=(X_val, y_val),
              callbacks=callbacks,
              verbose=1)

    # save model
    modelSavePath = MODEL_SAVE_PATH + datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    model.save(modelSavePath)
    del model
    model = load_model(modelSavePath, custom_objects={'top_3_accuracy': top_3_accuracy})


    tester(model, TEST_FILENAME, TRAIN_FOLDER, imheight, imwidth)


if __name__ == '__main__':
    main()

