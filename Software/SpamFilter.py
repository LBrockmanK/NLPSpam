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

def main(newData):
	# Read in training data
	# Delete the first blank line and everything preceding it

	if(newData == True): # Re-process training data
		df = dataRead('TrainingData/Ham')
		df = pd.DataFrame(columns=['Class', 'Content', 'Length'])

		directory = 'TrainingData/Ham'

		# Read Ham Data
		directory = 'TrainingData/Ham'
		for file in os.listdir(directory):
			f = open(directory+'/'+file,'r')
			body = f.read()

			# delete all text before the first blank line (removes header info)
			header = body.find('\n\n')+2
			body = body[header:]

			# delete text between <''> (removes formatting info)
			body = re.sub(r'\<.*?\>','',body)

			# Get length of body
			length = len(body)

			# Remove punctuation
			body = body.translate(str.maketrans('', '', string.punctuation))

			# Convert to a list
			wordlist = list(body.split(" "))
			#TODO: Still have some bad characters / strings in here

			# Remove stop words
			for token in wordlist:
			    if token.lower() in stopwords.words('english'):
			        wordlist.remove(token)

			print(wordlist)
			f.close()
			break
	else: # Read preprocessed training data from file

def dataRead(directory):
	# Read Data
	for file in os.listdir(directory):
		f = open(directory+'/'+file,'r')
		body = f.read()

		# delete all text before the first blank line (removes header info)
		header = body.find('\n\n')+2
		body = body[header:]

		# delete text between <''> (removes formatting info)
		body = re.sub(r'\<.*?\>','',body)

		# Get length of body
		length = len(body)

		# Remove punctuation
		body = body.translate(str.maketrans('', '', string.punctuation))

		# Convert to a list
		wordlist = list(body.split(" "))
		#TODO: Still have some bad characters / strings in here

		# Remove stop words
		for token in wordlist:
		    if token.lower() in stopwords.words('english'):
		        wordlist.remove(token)

		print(wordlist)
		f.close()
		break

main(True)