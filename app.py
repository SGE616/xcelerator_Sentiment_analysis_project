# -*- coding: utf-8 -*-
import dash
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
from dash.dependencies import Input, Output

df = pd.read_excel('https://github.com/SGE616/xcelerator_Sentiment_analysis_project/raw/master/Datafiniti_Hotel_Reviews.xlsx')
rev = df[['reviews.title','reviews.text','reviews.rating']]
def round_of_rating(number):
    return round(number)
rev.loc[:,'reviews.rating'] = rev.loc[:,'reviews.rating'].copy().apply(round_of_rating)
cnts = rev.groupby('reviews.rating').count()
cnts['reviews.rating'] = cnts.index

import plotly.graph_objs as go
counts = [go.Bar(x=cnts['reviews.rating'],y=cnts['reviews.title'])]

import re
def cleanTxt(text):
    text = str(text)
    text = text.lower()
    text = re.sub(r'[;:!\\"\'()\[\]]',"", text)
    text = re.sub(r'(<br\s*/><br\s*/>)|(\-)|(\/)'," ", text)
    return text
rev.loc[:,'reviews.title']=rev.loc[:,'reviews.title'].apply(cleanTxt)
rev.loc[:,'reviews.text']=rev.loc[:,'reviews.text'].apply(cleanTxt)

from nltk.stem import PorterStemmer 
from nltk.tokenize import word_tokenize
ps = PorterStemmer() 
def stemmer(para):
    w = word_tokenize(para)
    l = [ps.stem(j) for j in w]
    link = ' '
    s = link.join(l)
    return s
#import nltk
#nltk.download('punkt')
rev.loc[:,'reviews.title'] = rev.loc[:,'reviews.title'].apply(stemmer)
rev.loc[:,'reviews.text'] = rev.loc[:,'reviews.text'].apply(stemmer)

from textblob import TextBlob
def n_grams(text):
  return TextBlob(text).ngrams(n=3)
rev['ngrams'] = rev['reviews.text'].apply(n_grams)

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
sid = SentimentIntensityAnalyzer()
def pol_score(text):
  return sid.polarity_scores(text)['compound']
rev['polarity_score'] = rev['reviews.text'].apply(pol_score)

import numpy as np
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score
from sklearn.model_selection import train_test_split 
from sklearn.feature_extraction.text import TfidfVectorizer

text_based = rev[['reviews.title','reviews.text']] #storing the text based attributes
text_based['all text'] = text_based['reviews.title'] + ' ' + text_based['reviews.text']

X = text_based['all text']
y = rev['reviews.rating']
y = y.astype('int')

X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.3,random_state=42)

from sklearn.pipeline import Pipeline
#using Naive Bayes first
from sklearn.naive_bayes import MultinomialNB
NB = Pipeline([('tfidf',TfidfVectorizer()),('model',MultinomialNB())])
NB.fit(X_train,y_train)
NB_pred = NB.predict(X_test)
#print(confusion_matrix(y_test,NB_pred))
#print(classification_report(y_test,NB_pred))
#print(accuracy_score(y_test,NB_pred))
#using Decision trees
from sklearn.tree import DecisionTreeClassifier
tree = Pipeline([('tfidf',TfidfVectorizer()), ('model',DecisionTreeClassifier())])
tree.fit(X_train,y_train)
tree_pred = tree.predict(X_test)
#print(confusion_matrix(y_test,tree_pred))
#print(classification_report(y_test,tree_pred))
#print(accuracy_score(y_test,tree_pred))
#using random forest
from sklearn.ensemble import RandomForestClassifier
rfc = Pipeline([('tfidf',TfidfVectorizer()), ('model',RandomForestClassifier())])
rfc.fit(X_train,y_train)
rfc_pred = rfc.predict(X_test)
#print(confusion_matrix(y_test,rfc_pred))
#print(classification_report(y_test,rfc_pred))
#print(accuracy_score(y_test,rfc_pred))

NB_acc = accuracy_score(y_test,NB_pred)
tr_acc = accuracy_score(y_test,tree_pred)
rfc_acc = accuracy_score(y_test,rfc_pred)
accuracy = [go.Bar(x=['NaiveBayes','Decision tree','Random forest'],y=[NB_acc,tr_acc,rfc_acc])]

# -*- flask app -*--

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
colors = {
    'background': '#FFFDBE',
    'text': '#4F4F4F'
}

app.layout = html.Div(children=[

    html.H1(children='Dashboard'),

    html.Div(children='''
        Dash: A web application framework for Python.
    '''),

    dcc.Graph(
        id='Reviews Breakup',
        figure={
            'data': counts,
            'layout': {
                'title': 'Reviews Break-up'
            }
        }
    ),
    dcc.Graph(
        id='Accuracy',
        figure={
            'data': accuracy,
            'layout': {
                'title': 'Accuracy graph'
            }
        }
    ),
    dcc.Input(id='Input review', value='initial value', type='text'),
    dcc.Graph(id='estimate'),
])
@app.callback(
    Output(component_id='estimate', component_property='figure'),
    [Input(component_id='Input review', component_property='text')])
def update_figure(review):
    review = cleanTxt(review)
    review = stemmer(review)
    n_prd = NB.predict(review)
    t_prd = tree.predict(review)
    r_prd = rfc.predict(review)
    estimation = [go.Bar(x=cnts['NaiveBayes','Decision tree','Random forest'],y=[n_prd,t_prd,r_prd])]

    return {
        'data': estimation,
        'layout': dict(
            xaxis={'title': 'Model used'},
            yaxis={'title': 'Prediction'},
            margin={'l': 40, 'b': 40, 't': 10, 'r': 10},
        )
    }

if __name__ == '__main__':
    app.run_server(debug=True)
