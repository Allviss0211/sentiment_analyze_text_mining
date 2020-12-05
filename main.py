import numpy as np
from numpy import round
from bs4 import BeautifulSoup as bs
import requests
import pandas as pd
from datetime import datetime as dt
import tensorflow as tf
from matplotlib import pyplot as plt
import sentiment_analyze as sa
from features import build_feature as fea


print("Type your link review (ex: https://www.imdb.com/title/tt6723592/reviews?ref_=tt_ql_3) :")
x = input()
fea.make_predict(x)

# text = "I tried, and it just didn't go well, similar to the storyline and acting shown here so terrible."

