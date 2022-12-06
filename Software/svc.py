# TODO:
# Create a function to take parameter inputs and return a trained svc model, possibly check if the model already exists, possibly return some diagnostic data
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from collections import Counter
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.model_selection import GridSearchCV

def svcmodel(data):
	X_train, X_test, y_train, y_test = train_test_split(data['Content'], data['Class'], test_size=0.20, random_state = 50)    
	clf = SVC(kernel='linear').fit(X_train, y_train)

	predictions = clf.predict(X_test)
	print('predicted', predictions)