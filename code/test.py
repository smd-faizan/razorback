import ast
import os

from dask import bag
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image, ImageDraw

# TODO: CHANGE THE SIZE ON LINE 22
# faster conversion function
def draw_it(strokes):
    image = Image.new("P", (256,256), color=255)
    image_draw = ImageDraw.Draw(image)
    for stroke in ast.literal_eval(strokes):
        for i in range(len(stroke[0])-1):
            image_draw.line([stroke[0][i],
                             stroke[1][i],
                             stroke[0][i+1],
                             stroke[1][i+1]],
                            fill=0, width=5)
    image = image.resize((32, 32))
    return np.array(image)/255.


def tester(model, TEST_FILENAME, TRAIN_FOLDER, imheight, imwidth):
    ttvlist = []
    reader = pd.read_csv(TEST_FILENAME, index_col=['key_id'],
                         chunksize=2048)
    for chunk in tqdm(reader, total=55):
        imagebag = bag.from_sequence(chunk.drawing.values).map(draw_it)
        testarray = np.array(imagebag.compute())
        testarray = np.reshape(testarray, (testarray.shape[0], imheight, imwidth, 1))
        testpreds = model.predict(testarray, verbose=0)
        ttvs = np.argsort(-testpreds)[:, 0:3]  # top 3
        ttvlist.append(ttvs)

    ttvarray = np.concatenate(ttvlist)

    preds_df = pd.DataFrame({'first': ttvarray[:, 0], 'second': ttvarray[:, 1], 'third': ttvarray[:, 2]})

    # %% set label dictionary and params
    classfiles = os.listdir(TRAIN_FOLDER)
    numstonames = {i: v[:-4].replace(" ", "_") for i, v in enumerate(classfiles)}  # adds underscores


    preds_df = preds_df.replace(numstonames)
    preds_df['words'] = preds_df['first'] + " " + preds_df['second'] + " " + preds_df['third']

    sub = pd.read_csv(TRAIN_FOLDER+"/../sample_submission.csv", index_col=['key_id'])
    sub['word'] = preds_df.words.values
    sub.to_csv('subcnn_small.csv')
    sub.head()

def main():
    # write tests
    print "test"

if __name__ == '__main__':
    main()
