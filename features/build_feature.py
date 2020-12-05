import numpy as np
from bs4 import BeautifulSoup as bs
import requests
import pandas as pd
from datetime import datetime as dt
import tensorflow as tf
from matplotlib import pyplot as plt
from models import train_model as train

def make_predict(link):

    model = train.load_model("model.hdf5")
    page = requests.get(f"{link}",headers={'Accept-Language': 'en-US,en;q=0.8'})
    soup = bs(page.content, "html.parser")
    tables = soup.find(id="main")
    body = tables.findAll("div", {"class" : 'lister-item mode-detail imdb-user-review collapsable'})
    listRate = []
    listReview = []
    i = 0
    for review in body:
        print(f"Review {i}:",end=" ")
        rate = review.find("span", {"class" : "rating-other-user-rating"})
        if rate != None:
            tmp = rate.text.replace('\n', '')
            listRate.append(tmp)
            print(f"User rating {tmp}")
        else :
            print("No user rating")
        content = review.find("div" , {"class" : "text show-more__control"})
        if content != None:
            tmp = content.text.replace("\n", '')
            listReview.append(tmp)
            print(tmp[:100] + "...")
            predicion = train.sample_predict(tmp, pad = True,model_ = model)
            if(predicion >= 0.5):
                print('This review is positive')
            else:
                print('This review is negative')
            print('\n')
        i += 1