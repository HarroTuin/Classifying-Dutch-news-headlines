## Classifying Dutch news headlines of "de Speld"  and "Nu.nl" 

#### Context on data
This dataset contains news headlines of two Dutch newswebsites. Data was collected by myself using a webscraper.
All sarcastic headlines were collected from the Speld.nl (the Dutch equivalent of The Onion) whereas all "normal" headlines were collected from newswebsite Nu.nl.

#### Data preprocessing
The preprocessing file tokenizes the headlines and cleans the headlines of stop words and unnecessary symbols. The cleaning of the headlines could be further improved, for example it now only keeps symbols present in the alphabet.
Subsequently the headlines are replaced by TF-IDF scores so that the data can be used to classify.

#### Classifiers
In the classifier file two types of classifiers are implemented; SVM and Naive Bayes. For SVM Gridsearch was used to optimize the hyperparameters though this did not influence the accuracy a lot (+- few percent).
SVM classifies over 72% correct whereas the Naive Bayes classifier gets stuck at 61.7%. 

#### Acknowledgements
The idea for creating this dataset was sparked by the dataset from Rishabh Misra on Kaggle which contains English news headlines.

