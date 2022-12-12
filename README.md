# NLPSpam

Instructions:
To run program, execute Train.bat in root folder
	At the end of SpamFilter.py is a main call with a boolean argument, if set to false the program will use saved data and take ~30 seconds to complete, if set to true it will recreate vectorized data at take roughly 16 hours to complete

Folders

Models: Stores previousely trained models
	File names will be a shorthand for the model details that is TBD

Software
	Python files

TestData
	My own emails used for final evaluation

TestData
	Large data set sourced from online

Root Files
	models.csv : Record of previous training results storing model details for easy review, info will probably be redundant with filename shorthand
	train.bat to execute training

TODO:
	Find and download training data
	Develop based model generation and testing

Resources:
	Statistics
		https://medium.com/analytics-vidhya/confusion-matrix-accuracy-precision-recall-f1-score-ade299cf63cd
	Machine Learning
		https://realpython.com/python-keras-text-classification/#convolutional-neural-networks-cnn
		https://towardsdatascience.com/deep-learning-techniques-for-text-classification-78d9dc40bf7c
		https://www.linkedin.com/pulse/build-your-spam-filter-nlp-machine-learning-david-lim/
		https://towardsdatascience.com/how-to-identify-spam-using-natural-language-processing-nlp-af91f4170113
	Other
		https://encyclopedia.kaspersky.com/knowledge/damage-caused-by-spam/
	References for final report
		https://dl.acm.org/doi/10.1145/3418994.3419002
		https://dl.acm.org/doi/10.5555/3176748.3176757
		https://dl.acm.org/doi/10.1145/3407023.3407079
		https://dl.acm.org/doi/10.1145/3485832.3488024
		https://dl.acm.org/doi/10.1145/3433174.3433605