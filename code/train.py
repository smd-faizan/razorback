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
import sys,os, subprocess

TEST_FILENAME = "../data/test_simplified.csv"
TRAIN_FOLDER = "../data/train_simplified/"
MODEL_SAVE_PATH = "../savedModels/model-"
EXTENSION = ".csv"
# IMAGES_PER_CLASS = sys.maxint # Integer.max for all images
IMAGES_PERCENTAGE_PER_BATCH = 5
STEPS_PER_EPOCH=18
VALIDATION_PERCENTAGE = 10
imheight = 256
imwidth = 256
num_classes = 340
# Training parameters
BATCH_SIZE = 32
EPOCHS = 22




def read_line(line, fileNumber):
    line = line[line.index("\"")+1:line.rfind("\"")]
    data = json.loads(line)
    drawing = Drawing()
    for stroke in data:
        xCordiantes = stroke[0]
        yCordinates = stroke[1]
        st = Stroke()
        for i in xrange(len(xCordiantes)):
            st.addPoint(xCordiantes[i], yCordinates[i])
        drawing.addStroke(st)
    x = draw_it(drawing.strokes)
    # y = keras.utils.to_categorical(fileNumber, num_classes)
    return x


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

numberOfLines = []

def load_file_validation(filename, i, validation_percentage):
    wcFilename = filename
    if wcFilename.find(" ")>0:
        while wcFilename.find(" ")>0:
            idx = wcFilename.find(" ")
            wcFilename = wcFilename[0:idx] + "\\" + wcFilename[idx+1:]
        wcFilename = wcFilename.replace("\\", "\\" + " ")
    Number_lines = int((subprocess.Popen('wc -l {0}'.format(wcFilename), shell=True, stdout=subprocess.PIPE).stdout).readlines()[0].split()[0])
    numberOfLines.append(Number_lines)
    cutpt = int(validation_percentage * (Number_lines/100))

    result = []
    fp = open(filename, "r")
    for n in xrange(cutpt):
        content = fp.readline()
        if(content.find("\"")<0):
            continue
        x = read_line(content,i)
        result.append(x)
    print "loaded "+ str(cutpt) + " lines"
    endOffset = fp.tell()
    fp.close()
    return np.array(result), endOffset



def load_files_for_validation():
    file_pointers = []
    i=0
    grand = []
    for filename in os.listdir(TRAIN_FOLDER):
        if filename.endswith(EXTENSION):
            images, endOffset = load_file_validation(TRAIN_FOLDER+filename, i, VALIDATION_PERCENTAGE)
            file_pointers.append(endOffset)
            labelarray = np.full((images.shape[0], 1), i)
            images = images.reshape(images.shape[0], -1)
            images = np.concatenate((labelarray, images), axis=1)
            for image in images:
                grand.append(image)
            i += 1

    return grand, file_pointers

def generate_arrays_from_file(file_pointers, numberOfLines):

    current_file_pointers = []
    for pointer in file_pointers:
        current_file_pointers.append(pointer)

    while 1:
        i = 0
        grand = []
        for filename in os.listdir(TRAIN_FOLDER):
            result = []
            if filename.endswith(EXTENSION):
                fp = open(TRAIN_FOLDER+filename)
                fp.seek(current_file_pointers[i])
                for n in xrange(int((numberOfLines[i]/100)*IMAGES_PERCENTAGE_PER_BATCH)):
                    content = fp.readline()
                    if (content == ''):
                        fp.seek(file_pointers[i])
                        continue
                    x = read_line(content, i)
                    y = keras.utils.to_categorical(i, num_classes)
                    result.append(x)

                    # create numpy arrays of input data
                    # and labels, from each line in the file
                    # x, y = process_line(line)
                    # img = load_images(x)
                    # yield (img, y)
                current_file_pointers[i] = fp.tell()
                fp.close()
                images = np.array(result)
                labelarray = np.full((images.shape[0], 1), i)
                images = images.reshape(images.shape[0], -1)
                images = np.concatenate((labelarray, images), axis=1)
                for image in images:
                    grand.append(image)
                i += 1
        grand = np.array(grand)
        shuffle(grand)
        X_train, Y_train = divide(grand)

        y_train = keras.utils.to_categorical(Y_train, num_classes)
        x_train = X_train.reshape(X_train.shape[0], imheight, imwidth, 1)
        yield (x_train, y_train)


def shuffle(grand):
    np.random.shuffle(grand)

def divide(grand):
    Y_val, X_val = grand[:, 0], grand[:, 1:]

    del grand
    return X_val, Y_val


def top_3_accuracy(x, y):
    t3 = top_k_categorical_accuracy(x, y, 3)
    return t3

def main():
    grand, filePointers = load_files_for_validation()
    grand = np.array(grand)
    shuffle(grand)
    X_val, Y_val = divide(grand)

    # y_train = keras.utils.to_categorical(Y_train, num_classes)
    # X_train = X_train.reshape(X_train.shape[0], imheight, imwidth, 1)
    y_val = keras.utils.to_categorical(Y_val, num_classes)
    X_val = X_val.reshape(X_val.shape[0], imheight, imwidth, 1)

    print(y_val.shape, "\n",
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

    model.fit_generator(generate_arrays_from_file(filePointers, numberOfLines),
              steps_per_epoch=STEPS_PER_EPOCH,
              epochs=EPOCHS,
              validation_data=(X_val, y_val),
              callbacks=callbacks,
              verbose=1)

    # save model
    modelSavePath = MODEL_SAVE_PATH + datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    model.save(modelSavePath)


    tester(model, TEST_FILENAME, TRAIN_FOLDER, imheight, imwidth)


if __name__ == '__main__':
    main()

