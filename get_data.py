import sqlite3
import requests
import io
import numpy as np
import cv2
import random
from connect import reddit


'''MAKE INTO A CLASS '''
# MAKE A CLASS WITH ALL THE DATA NEEDED
# CURRENT DATA NEEDED UNKOWN
# CHECK FOR MORE IMG_TYPES


def db_update(data):
    db_path = 'db/red_db.db'
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute("""INSERT INTO memes
            (unid, sub, url, class)
            values (?,?,?,?)""",
            (data['id'], data['sub'], data['url'], data['class']))
    conn.commit()
    conn.close()


def get_data(subred):
    img_type = ['jpg', 'png', 'jpeg']
    url_list = []
    for sub in reddit.subreddit(subred).hot(limit=20):
        if sub.is_self == False:
            url = sub.url
            url = [url for x in img_type if x in url]
            if len(url) is not 0:
                data = {
                    'sub': subred,
                    'id': sub.id,
                    'url': url[0],
                    'class': random.randint(0,1)
                }
                url_list.append(data)
    return url_list


def analyize(img):
    img = img.reshape(-1, 299, 299, 3).astype('float32')
    img /= 255
    preds = model.predict(img)
    print(preds)
    if preds[0][0] == 1:
        return 1
    else:
        return 0

def load_img(subred, pred):
    imgs = get_data(subred)
    for img in imgs:
        img_name = img['sub'] + '-' + str(img['id'])
        img_data = io.BytesIO(requests.get(img['url']).content)
        image = cv2.imdecode(np.fromstring(img_data.read(), np.uint8), 1)
        image = cv2.resize(image, (299, 299))
        '''self predict '''
        if pred == 'self':
            cv2.imshow("output", image)
            if cv2.waitKey(0) & 0xFF == ord('m'):
                img['class'] = 1
            else:
                img['class'] = 0
        elif pred == 'model':
            label = analyize(image)
            img['class'] = label
        db_update(img)


if __name__ == '__main__':
    pred = 'self'
    if pred == 'self':
        model = ''
    else:
        from tensorflow import keras
        model = keras.models.load_model('model/keras_model/saves/cnn_1.h5')
    i = 0
    while i < 100:
        subr = reddit.random_subreddit()
        print(subr)
        if subr.over18 == True:
            pass
        else:
            load_img(str(rand), pred=pred)
            i += 1

