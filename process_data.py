import pandas as pd
from bs4 import BeautifulSoup

with open("data/train.txt", "r") as f:
    unlabeledTrain = [line.strip().split("\t") for line in f.readlines() if len(line.strip().split("\t")) == 2]

unlabel = pd.DataFrame(unlabeledTrain[0: ], columns=['sentiment','review'])

unlabel.to_csv("data/output/wordEmbedding.txt", columns=['review'],header= False,index=False)