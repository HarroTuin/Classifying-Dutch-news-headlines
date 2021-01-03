import collections
import math
import pandas as pd
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
from collections import Counter
from sklearn import svm, naive_bayes
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import StratifiedKFold
import timeit

start = timeit.default_timer()

# Import dataset
df = pd.read_csv('headlines_dataset.csv')

# Drop columns that are not needed
df = df.drop(['link', 'source', 'is_binnenland', 'is_buitenland', 'is_politiek'], axis=1)
df = df.sample(frac=1).reset_index(drop=True)

# Remove duplicates (news can be in multiple categories)
df = df.drop_duplicates(subset=['headline'])#.reset_index(drop=True)
#df = df[0:500] # to test code for smaller dataset

# Tokenize headlines
tokenized_headline = []
for headline in df['headline']:
    word = word_tokenize(headline)
    tokenized_headline.append(word)

# Set stop words to Dutch and set unnecessary symbols
stop_words = set(nltk.corpus.stopwords.words('dutch')) 

# Make words lowercase and filter stop words + unnecessary symbols 
headlines_filtered = []
for headline in tokenized_headline:
    new_headline = []
    for word in headline:
        word = word.lower()
        word = "".join(char for char in word if char.isalpha())
        if word not in stop_words and len(word) > 1:
            new_headline.append(word)
    headlines_filtered.append(new_headline)

# Replaces headlines with tokenized and filtered headlines and shuffle rows
df['headline'] = headlines_filtered       
    
# Count how often words occur   
word_counts = collections.Counter(word for words in df['headline'] for word in words)

# TF-IDF
tfidf_list = []
for headline in headlines_filtered:
    # count words in every headline
    c = Counter(headline)
    tfidf_message = []
    for word in headline:
        # calculate TF
        freq_term = int(('%d' % (c[word])))
        num_word = len(headline)
        tf = (freq_term / num_word)   
        # calculate IDF
        documents_w_t = 0 # = denominator of IDF formula
        for i in headlines_filtered:
            for j in i:
                if j == word:
                    documents_w_t += 1
        idf = math.log(len(headlines_filtered)/documents_w_t)
        
        # calculate TF-IDF (TF * IDF) 
        tfidf = tf*idf
        tfidf_message.append(tfidf)
    tfidf_list.append(tfidf_message)

# Pad data
max_words = 0 # max number of words per message
for i in tfidf_list:
    if len(i) > max_words:
        max_words = len(i)
        
for i in tfidf_list: 
    while len(i) < max_words: # pad so every message matches length of longest message
        i.append(0)
        
# Convert list of lists into NumPy array
length = max(map(len, tfidf_list))
array = np.array([xi+[None]*(length-len(xi)) for xi in tfidf_list])

# Scale data between 0 and 1
scaler = MinMaxScaler()

# Make dataframes and save data
array = scaler.fit_transform(array)
X = pd.DataFrame(array)
y = df['is_sarcastic']

X.to_csv('input.csv', index=False, header=False)
y.to_csv('labels.csv', index=False, header=False)

stop = timeit.default_timer()

print('Time: ', stop - start)  