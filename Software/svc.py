# TODO:
# Create a function to take parameter inputs and return a trained svc model, possibly check if the model already exists, possibly return some diagnostic data
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from collections import Counter
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.model_selection import GridSearchCV
import numpy as np

def svcmodel(df):
	# print(df.shape[0])
	# print(len(df["Content"].values[0]))
	# vectors = np.zeros(shape=(df.shape[0],len(df["Content"].values[0])))
	col_list = df.Content.values.tolist()
	vectorarray = np.array(col_list)

	X_train, X_test, y_train, y_test = train_test_split(vectorarray, df['Class'], test_size=0.20, random_state = 50)    
	clf = SVC(kernel='linear').fit(X_train, y_train)

	predictions = clf.predict(X_test)
	print('predicted', predictions)