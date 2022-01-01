from flask import Flask,render_template,url_for,request
import pandas as pd 
import pickle
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
# from sklearn.externals import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB


app = Flask(__name__)

@app.route('/')
def home():
	return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():
	df= pd.read_csv("review2.csv", encoding="latin-1")

	# Features and Labels
	df['label'] = df['type'].map({'positif': 0, 'negatif': 1})
	X = df['text']
	y = df['label']

	# Extract Feature With CountVectorizer
	# Extract Feature With CountVectorizer :cleaning involved converting all of our data to lower case and removing all punctuation marks. 
	cv = CountVectorizer()
	X = cv.fit_transform(X) 
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
	clf = MultinomialNB() #NAIVE BAYES
	clf.fit(X_train,y_train)
	clf.score(X_test,y_test)
	
	if request.method == 'POST':
		message = request.form['message']
		data = [message]
		vect = cv.transform(data).toarray()
		my_prediction = clf.predict(vect)
	return render_template('result.html',prediction = my_prediction)

if __name__ == '__main__':
	app.run(debug=True)