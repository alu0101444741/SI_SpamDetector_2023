# ********************************************************
# ****************   INICIALIZACION   ********************
# ********************************************************
# ** Crear ejecutable: pyinstaller --onefile gui_v3.py ***
# ********************************************************
# ***************** Paquetes utilizados ******************
import sys
import subprocess
def check_packages():
    package_names = [ 'scikit-learn', 'matplotlib', 'numpy', 'pandas',
                  'seaborn', 'tk', 'Pillow', 'nltk', 're']
    for package in package_names:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])

#check_packages()
# ********************************************************
import nltk 
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer

import numpy as np
import pandas as pd
import re

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.gaussian_process.kernels import RBF
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

data = pd.read_csv("./input/spam_ham_dataset.csv")

names = [
  "MultibinomialNB",
  "Nearest Neighbors",
  "Linear SVM",
  "RBF SVM",
  "Decision Tree",
  "Random Forest",
  "LogisticRegression"
]

classifiers = [
  MultinomialNB(),
  KNeighborsClassifier(3),
  SVC(kernel="linear", C=0.025),
  SVC(gamma=2, C=1),
  DecisionTreeClassifier(max_depth=5),
  RandomForestClassifier(n_estimators=100, criterion="gini"),
  LogisticRegression(solver = 'liblinear', C=10, penalty = 'l2')
]

# Preprocesamiento de texto
stop_words = stopwords.words('english')
stemmer = SnowballStemmer('english')

text_cleaning_re = "@\S+|https?:\S+|http?:\S+|[^A-Za-z0-9]:\S+|subject:\S+|nbsp"

def preprocess(text, stem=False):
  text = re.sub(text_cleaning_re, ' ', str(text).lower()).strip()
  tokens = []
  for token in text.split():
    if token not in stop_words:
      if stem:
        tokens.append(stemmer.stem(token))
      else:
        tokens.append(token)
  return " ".join(tokens)

data.text = data.text.apply(lambda x: preprocess(x))

# Variables para el entrenamiento
x, y = data.text, data.label
x_train , x_test, y_train , y_test = train_test_split(x, y, test_size=0.3)
Vectorizer = CountVectorizer()
count= Vectorizer.fit_transform(x_train.values)

import classifiersinfo

text = '''McAfee(TM)
Recommended by:   Lenovo
BUY NOW
Your trial expired
13 Mar 2022
Your McAfee protection expired 3 days ago

Save an extra 10% with
this email exclusive!
That's a total savings of 70%
on protection!
Get protection and save
Your all-in-one protection
includes these great features


Online privacy with Secure VPN

Award-winning antivirus

Mobile protection app

Safer web browsing

Multi-device compatibility
Award-winning internet security
Protecting more than 600 million consumer‑connected devices.
PC EDITORS CHOICE
AV TEST | TOP PRODUCT'''