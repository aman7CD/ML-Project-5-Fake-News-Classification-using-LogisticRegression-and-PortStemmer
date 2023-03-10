## Importing the Dependecies 
import numpy as np 
import pandas as pd
import re

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

import nltk
nltk.download("stopwords")




## Data Colection and Preprocessing
data = pd.read_csv("/kaggle/input/fake-news/train.csv", index_col= 'id')
data1 = pd.read_csv("/kaggle/input/fake-news/test.csv", index_col= 'id')

data.isnull().sum()
data1.isnull().sum()

data.fillna(' ', inplace=True)
data1.fillna(' ', inplace=True)

x = data
x1 = data1

x["content"] = x["author"]+' '+x["title"]
x1["content"] = x1["author"]+' '+x1["title"]

x = x.drop(["title","author","label","text"], axis = 1)
x1 = x1.drop(["title","author","text"], axis = 1)

##stemming
port_stem = PorterStemmer()

def stem(article):
    stemmed_article = re.sub("[^a-zA-Z]"," ",str(article))
    stemmed_article = stemmed_article.lower()
    stemmed_article = stemmed_article.split()
    stemmed_article = [port_stem.stem(word)  for word in stemmed_article if word not in stopwords.words("english")]
    stemmed_article = " ".join(stemmed_article)
    return stemmed_article

x["content"] = x["content"].apply(stem)
x1["content"] = x1["content"].apply(stem)


##Converting Data From Text to Numerical
X = x["content"].values
X1 = x1["content"].values
y = data["label"].values

vt = TfidfVectorizer(max_features=9140)

X = vt.fit_transform(X)
X1 = vt.fit_transform(X1)


## Splitting the Data
x_train, x_test, y_train, y_test = train_test_split(X,y, test_size = 0.05, stratify=y, random_state=1)


## Training The Model
model = LogisticRegression()
model.fit(X,y)



## Model Evaluation Through r2 Score and MSE

y1_pred =  model.predict(X1)
y_pred = model.predict(X)

accuracy_score = accuracy_score(y,y_pred)


x_new = x_test[3]

prediction = model.predict(x_new)

if (prediction[0]==0):
  print('This news article is Real')
else:
  print('This news article is Fake')

print(f"The accuracy of the model in identifying fake news is {accuracy_score}")
