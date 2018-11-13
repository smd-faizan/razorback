class Drawing:
    strokes = []
    country = ""
    keyID = ""
    recognized = ""
    word = ""
    timestamp = ""

    def __init__(self):
        self.strokes = []
        self.country = ""
        self.keyID = ""
        self.recognized = False
        self.word = ""
        self.timestamp = ""

    def addStroke(self, stroke):
        self.strokes.append(stroke)


class Stroke:
    points = []

    def __init__(self):
        self.points = []

    def addPoint(self, x, y):
        self.points.append(MyPoint(x, y))


class MyPoint:
    x = 0
    y = 0
    t = 0

    def __init__(self, x, y, t=None):
        self.x = x
        self.y = y
        self.t = t
