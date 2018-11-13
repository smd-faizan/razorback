from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D, Dropout, Dense, Flatten

imheight = 256
imwidth = 256
num_classes = 340

class MyModelGenerator:
    imheight = 0
    imwidth = 0
    num_classes = 0
    model = None

    def __init__(self, imheight, imwidth, num_classes):
        self.imheight = imheight
        self.imwidth = imwidth
        self.num_classes = num_classes
        model = Sequential()
        model.add(Conv2D(32, kernel_size=(3, 3), padding='same', activation='relu', input_shape=(imheight, imwidth, 1)))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(64, kernel_size=(3, 3), padding='same', activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.2))
        model.add(Flatten())
        model.add(Dense(680, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(num_classes, activation='softmax'))
        model.summary()
        self.model=model

    def getModel(self):
        return self.model

def main():
    m = MyModelGenerator(32, 32, 340);
    m.getModel();

if __name__ == '__main__':
    main()