import pandas as pd
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
import joblib
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score


df = pd.read_csv('final.csv')
x=df['text']
y=df['sentiment']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=42)
pipe_lr = Pipeline(steps=[('cv', CountVectorizer()),('lr', LogisticRegression())])
pipe_lr.fit(x_train,y_train)
print(pipe_lr.score(x_test,y_test))

pipeline_file=open('text_emotion.pkl','wb')
joblib.dump(pipe_lr,pipeline_file)
pipeline_file.close()
