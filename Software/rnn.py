# TODO:
# Create a function to take parameter inputs and return a trained rnn model, possibly check if the model already exists, possibly return some diagnostic data
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix
import numpy as np
# Similar to CNN but replace convolutional layers