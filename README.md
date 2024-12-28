# Social-Engineering-Detection
This project takes  sentiment analysis data from text to detect social engineering and phishing attempts.
The dataset was taken from Kaggle. Only a small portion of the original dataset was used. https://www.kaggle.com/datasets/subhajournal/phishingemails
First, the dataset was appended with a sentiment analysis score in making_dataset.py. The new dataset was then used to train a random forest classifier in training_rf.py. 
Sentiment analysis.py contains main to use and test.
