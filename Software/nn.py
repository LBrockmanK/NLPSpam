# TODO:
# Create a function to take parameter inputs and return a trained nn model, possibly check if the model already exists, possibly return some diagnostic data
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Rescaling, Dense, Flatten, MaxPooling1D, Dropout
# Create based on other project, will need to change input format see GloVe
def nnmodel(df):
	#Format data
	col_list = df.Content.values.tolist()
	vectorarray = np.array(col_list)

	class_list = df.Class.values.tolist()
	classarray = np.zeros((len(class_list),2))

	# Convert classification into 2 dimensions to match the 2 output nodes for the neural network
	for idx, i in enumerate(classarray):
		classarray[idx] = [class_list[idx],1-class_list[idx]]

	X_train, X_test, y_train, y_test = train_test_split(vectorarray, classarray, test_size=0.20, random_state = 50)
	
	# build the model
	m = Sequential()
	m.add(Dense(32, activation='relu',input_shape=(vectorarray.shape[1],1)))
	m.add(MaxPooling1D())
	m.add(Flatten())
	m.add(Dense(128, activation='relu'))
	m.add(Dense(2, activation='softmax'))

	print(m.summary())

	# setting and training
	m.compile(loss="categorical_crossentropy", metrics=['accuracy'])
	history  = m.fit(X_train, y_train,verbose=0)
	print(history.history)
	print(history.history["accuracy"])

	#convert neural network output into true false list
	y_pred=m.predict(X_test) 
	y_pred=np.argmax(y_pred, axis=1)
	predictions=np.argmax(y_test, axis=1)

	#list ends up inverts true false from our original calssifications
	for idx, i in enumerate(predictions):
		predictions[idx] = 1-predictions[idx]

	print(predictions)
	print("NN Report: ")
	X_train, X_test, y_train, y_test = train_test_split(vectorarray, df['Class'], test_size=0.20, random_state = 50)
	print (classification_report(y_test, predictions))
	print("NN Confusion Matrix: ")
	print(confusion_matrix(y_test,predictions))