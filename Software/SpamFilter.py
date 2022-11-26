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
from sklearn.feature_extraction.text import CountVectorizer

def main(newData):
	# Read in training data
	# Delete the first blank line and everything preceding it
	#df = pd.DataFrame(columns=['Class','Content'])

	if(newData == True): # Re-process training data
		df = dataRead('TrainingData/Ham',False)
		df = pd.concat([df, dataRead('TrainingData/Spam',True)], axis=0)

		wordslist = df.Content.values.tolist()
	else: # Read preprocessed training data from file
		print("TODO: Implement preprocessed data read")

	print(wordslist)

def dataRead(directory,spamham):
	# Read Data
	df = pd.DataFrame(columns=['Class','Content'])
	for file in os.listdir(directory):
		f = open(directory+'/'+file,'r')
		body = f.read()

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

		# Convert to a list
		wordlist = list(body.split(" "))

		# Remove stop words
		for token in wordlist:
		    if token.lower() in stopwords.words('english'):
		        wordlist.remove(token)

		# Add data to frame
		df.loc[len(df.index)] = [spamham, wordlist] 
		f.close()
		break

	return df

main(True)