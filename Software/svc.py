# TODO:
# Create a function to take parameter inputs and return a trained svc model, possibly check if the model already exists, possibly return some diagnostic data
from sklearn.svm import SVC
from collections import Counter
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix
import numpy as np

def svcmodel(df):
	#Format data
	col_list = df.Content.values.tolist()
	vectorarray = np.array(col_list)

	X_train, X_test, y_train, y_test = train_test_split(vectorarray, df['Class'], test_size=0.20, random_state = 50)    
	clf = SVC(kernel='linear').fit(X_train, y_train)

	predictions = clf.predict(X_test)
	print("SVC Report: ")
	print (classification_report(y_test, predictions))
	print("SVC Confusion Matrix: ")
	print(confusion_matrix(y_test,predictions))