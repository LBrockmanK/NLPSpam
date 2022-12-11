# TODO:
# Need a mode to generate models from training data, and to apply a selected model to input data, this might be two files
import pandas as pd
import numpy as np
import os
import re
import string
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
import sklearn
nltk.download('wordnet')
nltk.download('omw-1.4')
from sklearn.feature_extraction.text import TfidfVectorizer
nltk.download('words')
from nltk.corpus import words
import svc
import cnn
import nn
from ast import literal_eval
import time

def main(newData):
	starttime = time.time()
	# Read in training data
	if(newData == True): # Re-process training data
		# Collect bag of words for each training sample
		df = dataRead('TrainingData/Ham',False)
		df = pd.concat([df, dataRead('TrainingData/Spam',True)], axis=0)

		# Vectorize training data
		corpus = df.Content.values.tolist()
		
		tfidfvectorizer = TfidfVectorizer(lowercase = 'false', max_features = None)

		tfidf_wm = tfidfvectorizer.fit_transform(corpus)
		tfidf_tokens = tfidfvectorizer.get_feature_names_out()
		df_tfidfvect = tfidf_wm.toarray()
		
		# Replace content with tf-idf
		for index, row in df.iterrows():
			tfidflist = df_tfidfvect.tolist()
			df['Content'] = tfidflist

		# Save data for future use
		df.reset_index(drop=True, inplace=True)
		df.to_csv('TrainingData/tfidf.csv')
		
	else: # Read preprocessed training data from file
		#load dataframe from csv
		df = pd.read_csv('TrainingData/tfidf.csv')
		df.drop(df.columns[df.columns.str.contains('unnamed',case = False)],axis = 1, inplace = True)

		#This converts our lists of tfidf into a string from a list, need to convert it back to a list of numbers
		df['Content'] = df['Content'].apply(literal_eval)
	print("Time to prepare data: ")
	print(time.time() - starttime)
	starttime = time.time()

	svc.svcmodel(df)
	print("Time to prepare SVC: ")
	print(time.time() - starttime)
	starttime = time.time()

	cnn.cnnmodel(df)
	print("Time to prepare CNN: ")
	print(time.time() - starttime)
	starttime = time.time()

	nn.nnmodel(df)
	print("Time to prepare NN: ")
	print(time.time() - starttime)
	starttime = time.time()

def dataRead(directory,spamham):
	# Read Data
	i = 0 # Temporary limit TODO: Remove
	df = pd.DataFrame(columns=['Class','Content'])
	for file in os.listdir(directory):
		f = open(directory+'/'+file,'r', errors="ignore")
		body = f.read()

		body = body.encode("ascii", "ignore")
		body = body.decode()

		# delete all text before the first blank line (removes header info)
		header = body.find('\n\n')+2
		body = body[header:]

		# delete text between <''> (removes formatting info)
		body = re.sub(r'\<.*?\>',' ',body)

		# remove all whitespace related characters
		body = " ".join(body.split())

		# Remove punctuation
		body = body.translate(str.maketrans('', '', string.punctuation))

		# Replace digits with 0, all numbers will be considered the same word for our analysis
		body = re.sub(r'\d+', '0', body)
		numbers = body.count('0')

		# Convert to a list
		wordlist = list(body.split(" "))

		# Remove stop words
		for token in wordlist:
		    if token.lower() in stopwords.words('english'):
		        wordlist.remove(token)

		# Stem and lemmatize words (convert into base form)
		ps = nltk.PorterStemmer()
		wordlist = [ps.stem(word) for word in wordlist]
		wn = nltk.WordNetLemmatizer()
		wordlist = [wn.lemmatize(word, pos = 'v') for word in wordlist]
		wordlist = [wn.lemmatize(word, pos = 'n') for word in wordlist]

		# Remove invalid words
		wordlist = list(filter(lambda x: x != '0', wordlist))
		mispelled = len(wordlist)
		wordlist = list(filter(lambda x: x in words.words(), wordlist))
		mispelled = mispelled-len(wordlist)

		# Add placeholders for numbers and mispelled words (000 and 111 respectively)
		while numbers > 0:
			wordlist.append('000')
			numbers = numbers-1

		while mispelled > 0:
			wordlist.append('111')
			mispelled = mispelled-1

		# Convert list back to string for input to vectorizer later
		body = ' '.join([str(item) for item in wordlist])

		# Add data to frame
		df.loc[len(df.index)] = [spamham, body] 
		f.close()

		i = i+1
		if(i == 1000):
			break

	return df

main(False)