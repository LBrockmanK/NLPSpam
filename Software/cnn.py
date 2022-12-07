# TODO:
# Create a function to take parameter inputs and return a trained cnn model, possibly check if the model already exists, possibly return some diagnostic data
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Rescaling, Conv1D, Dense, Flatten, MaxPooling1D, Dropout
# Create based on other project, will need to change input format see GloVe
def cnnmodel(df):
	# print(df.shape[0])
	# print(len(df["Content"].values[0]))
	# vectors = np.zeros(shape=(df.shape[0],len(df["Content"].values[0])))
	col_list = df.Content.values.tolist()
	vectorarray = np.array(col_list)

	X_train, X_test, y_train, y_test = train_test_split(vectorarray, df['Class'], test_size=0.20, random_state = 50)
	
	# build the model
	m = Sequential()
	m.add(Conv1D(32, 2, activation='relu',input_shape=(vectorarray.shape[1],1)))
	m.add(Dense(32, activation='relu'))
	m.add(MaxPooling1D())
	# m.add(Conv1D(64, kernel_size=(3, 3), activation='relu'))
	# m.add(MaxPooling2D(pool_size=(2, 2)))
	# m.add(Conv1D(128, kernel_size=(3, 3), activation='relu'))
	# m.add(MaxPooling2D(pool_size=(2, 2)))
	# m.add(Conv1D(256, kernel_size=(3, 3), activation='relu'))
	# m.add(MaxPooling2D(pool_size=(2, 2)))
	m.add(Flatten())
	m.add(Dense(128, activation='relu'))
	m.add(Dense(2, activation='softmax'))

	print(m.summary())

	# setting and training
	m.compile(loss="categorical_crossentropy", metrics=['accuracy'])
	history  = m.fit(X_train, y_train,verbose=0)
	print(history.history["accuracy"])

	# testing
	predictions = m.predict(X_test)
	print("CNN Report: ")
	print (classification_report(y_test, predictions))
	print("CNN Confusion Matrix: ")
	print(confusion_matrix(y_test,predictions))