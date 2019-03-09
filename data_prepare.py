import sqlite3
import numpy as np
import cv2
import io
import requests
import random

def get_data():
    conn = sqlite3.connect('db/red_db.db')
    c = conn.cursor()
    c.execute("SELECT * FROM memes")
    data = []
    for row in c.fetchall():
        data.append([row[2], row[3]])
    return data

def encode_label(x):
    if x == 1:
        #return 1
        return [1, 0]
    else:
        #return 0
        return [0, 1]

def load_img():
    imgs = get_data()
    train_img, train_label = [], []
    test_img, test_label = [], []
    for img in imgs:
        img_data = io.BytesIO(requests.get(img[0]).content)
        image = cv2.imdecode(np.fromstring(img_data.read(), np.uint8), 1)
        image = cv2.resize(image, (299, 299))
        rand = random.randint(1,10)
        if rand < 8:
            train_img.append(image)
            train_label.append(encode_label(img[1]))
        else:
            test_img.append(image)
            test_label.append(encode_label(img[1]))
    np.save('data/train_img', train_img)
    np.save('data/train_label', train_label)
    np.save('data/test_img', test_img)
    np.save('data/test_label', test_label)


def return_data(new_data):
    if new_data:
        load_img()
    train_img = np.load('data/train_img.npy')
    train_label = np.load('data/train_label.npy')
    test_img = np.load('data/test_img.npy')
    test_label = np.load('data/test_label.npy')
    return (train_img, train_label), (test_img, test_label)

if __name__ == '__main__':
    print(return_data(False))
