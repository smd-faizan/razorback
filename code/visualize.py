from graphics import *
import csv
import json

from tensorflow import keras
from tensorflow.python.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow.python.keras.metrics import top_k_categorical_accuracy

from Geometry import *
import numpy as np
from PIL import Image, ImageDraw
import ast

from model import MyModelGenerator

from test import tester

TEST_FILENAME = "/Users/s0f00xx/PycharmProjects/razorback/data/test_simplified.csv"
TRAIN_FOLDER = "/Users/s0f00xx/PycharmProjects/razorback/data/train_simplified/"
EXTENSION = ".csv"
IMAGES_PER_CLASS = 6 # Integer.max for all images
imheight = 32
imwidth = 32
num_classes = 340
# Training parameters
BATCH_SIZE = 32
EPOCHS = 5

def draw_it(strokes):
    image = Image.new("P", (256,256), color=255)  # BACKGROUND COLOR is BLACK?
    image_draw = ImageDraw.Draw(image)

    for stroke in strokes:
        for i in xrange(0, len(stroke.points) - 1):
            A = stroke.points[i]
            B = stroke.points[i + 1]
            line = Line(Point(A.x, A.y), Point(B.x, B.y))
            image_draw.line([A.x, A.y, B.x, B.y], fill=0, width=5)   # FILL THE LINE WITH WHITE COLOR?, width=5
    image = image.resize((imheight, imwidth))
    return np.array(image)/255.

def visualize(key, value):
    win = GraphWin('Drawing-' + key, 256, 256)  # give title and dimensions
    print "NUmber of strokes in this drawing " + str(len(value.strokes))
    j = 1
    label = Text(Point(100, 120), 'True' if value.recognized else 'False')
    label.draw(win)
    for stroke in value.strokes:
        for i in xrange(0, len(stroke.points) - 1):
            A = stroke.points[i]
            B = stroke.points[i + 1]
            line = Line(Point(A.x, A.y), Point(B.x, B.y))
            line.draw(win)
        print "drew stroke" + str(j)
        j += 1

    win.getMouse()
    win.close()

def visualize_np_array(np_array, title):
    win = GraphWin('Drawing-' + title, 256, 256)  # give title and dimensions
    for x in xrange(np_array.shape[0]):
        for y in xrange(np_array.shape[1]):
            if(np_array[x][y]==1.):
                pt = Point(x, y)
                # pt.draw(win) # TAKES A LOT OF TIME
    win.getMouse()
    win.close()

def load_file(filename):
    d = {}
    with open(filename) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            if line_count == 0:
                line_count += 1
            else:
                data = json.loads(row[2])
                drawing = Drawing()
                drawing.country = row[1]
                drawing.keyID = row[0]
                drawing.recognized = False
                drawing.timestamp = None
                drawing.word = None
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
        visualize(key, value);
        np_image = draw_it(value.strokes)
        #visualize_np_array(np_image, key)
        result.append(np_image)

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
    images = load_file(TEST_FILENAME)


if __name__ == '__main__':
    main()

