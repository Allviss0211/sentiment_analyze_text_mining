import numpy as np
from numpy import round
from PyQt5 import QtGui, QtWidgets, QtDesigner, QtCore
from PyQt5 import uic
from PyQt5.QtCore import pyqtSlot
from PyQt5.QtWidgets import QMainWindow, QApplication, QLabel, QVBoxLayout, QFileDialog, QPushButton, QAction
from PyQt5.QtCore import pyqtSlot
import sys
from bs4 import BeautifulSoup as bs
import requests
import pandas as pd
from datetime import datetime as dt
import tensorflow as tf
from PyQt5.QtGui import QPixmap, QIcon, QImage
from matplotlib import pyplot as plt
# import function as fn
import sentiment_analyze as sa

page = requests.get(f"https://www.imdb.com/title/tt6723592/reviews?ref_=tt_ql_3",headers={'Accept-Language': 'en-US,en;q=0.8'})
soup = bs(page.content, "html.parser")
model = tf.keras.models.load_model("model/model.hdf5")
tables = soup.find(id="main")
body = tables.findAll("div", {"class" : 'lister-item mode-detail imdb-user-review collapsable'})
listRate = []
listReview = []
for review in body:
    rate = review.find("span", {"class" : "rating-other-user-rating"})
    if rate != None:
        tmp = rate.text.replace('\n', '')
        listRate.append(tmp)
        print(tmp)
    else :
        print("No rating")
    content = review.find("div" , {"class" : "text show-more__control"})
    if content != None:
        tmp = content.text.replace("\n", '')
        listReview.append(tmp)
        predicion = sa.sample_predict(tmp, pad = True,model_ = model)
        if(predicion >= 0.5):
            print('positive')
        else:
            print('negative')
        print(predicion[0])
    
# print(listReview)


# text = "I tried, and it just didn't go well, similar to the storyline and acting shown here so terrible."

