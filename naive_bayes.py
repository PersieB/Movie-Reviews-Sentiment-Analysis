# -*- coding: utf-8 -*-
"""
Author: Percy Brown
Date: 25/03/2021
Topic: Naive Bayes Classifier
"""

# Importing libraries

import pandas as pd
import math
import numpy as np
import re

"""Loading the Training set"""

training_set = pd.read_csv('train.tsv', sep='\t')

"""View Training Data"""

training_set.info()

training_set.head()

training_set['sentence'].values

"""Array of available classes"""

np.unique(training_set['label'].values)

"""Counts per class"""

training_set['label'].value_counts()

"""function ***tokenization*** to split sentences into words using regular expression.
    It takes a sentence as a parameter and returns the splitted words in a list.
    Other means can be used to split sentences into words
"""

def tokenization(sentence):
  result = re.sub(r'[^\w\s]', '', sentence)  #remove punctuations in sentence
  tokens = result.split()  #split them into words
  lower = [i.lower() for i in tokens]
  return lower

""" function ***train*** which implements the naive Bayes train algorithm.
 It takes in parameters **D** (the training document) and **C** (a list containing the classes) and
  returns the **loglikelihood**, **logprior** and **V** (a set of vocabulary)"""

def train(D, C):
  N_doc = len(D)  #number of documents (reviews) in D
  
  #intitialising dictionary to hold logpriors
  log_priors = {x:0 for x in C}

  #intitialising dictionary to hold loglikelihoods
  log_likelihood = {x:{} for x in C} 

  Vocabulary = [] #Vocabulary initialised empty
  big_doc_c = {k:{} for k in C} 
  D['sentence'] = D['sentence'].apply(lambda review: tokenization(review))

  for c in C:
    class_reviews = training_set[training_set['label']==c]  # all reviews belonging to a class
    N_c = len(class_reviews)  # number of reviews belonging to a particular class
    log_priors[c] = math.log(N_c/N_doc) # log prior of c

    for i, j in class_reviews.iterrows():
      temp = j['sentence']
      for word in temp:
        if word not in big_doc_c[c].keys():
          big_doc_c[c][word] = 0  # append(d) for d in D with class c
        big_doc_c[c][word] += 1   # increase word count
        Vocabulary.append(word)
  
  count = 0
  for c in C:
    count = sum(big_doc_c[c].values())
    Vocabulary = set(Vocabulary)  # only unique words in vocabulary
    for token in Vocabulary:
      count_w_c_in_V = 0
      if token in big_doc_c[c]:
        count_w_c_in_V = big_doc_c[c][token]
      log_likelihood[c][token] = math.log((count_w_c_in_V + 1)/(count + len(Vocabulary))) #loglikelihoods for each word in a partlicular class
  return log_priors, log_likelihood, Vocabulary

"""function ***predict*** which implements the prediction or test function for the naive Bayes algorithm."""

def predict(testdoc, logprior, loglikelihood, C, V):
  sum_c = {c:logprior[c] for c in C}
  for c in C:
    for word in tokenization(testdoc):
      if word in V:
        sum_c[c] += loglikelihood[c][word]

 #arg_max of sum_c  #key (class) with the largest value
  v = list(sum_c.values())
  k = list(sum_c.keys())
  chosen_class = k[v.index(max(v))] #return the key with the largest values
  if(chosen_class == 0):
    return "NEGATIVE"
  else:
    return "POSITIVE"

"""Call ***train*** function and input arguments
**train** function returns variables which are then passed in the ***predict*** function.
 The function prints POSITIVE/NEGATIVE as the class of a review
"""
C = np.unique(training_set['label'].values)
logprior, loglikelihood, V = train(training_set, C)

'''
Main function tests our Naive Bayes Classifier Model.
If no test file is passed via the command line (python hw3.py), then it predicts using the test set from the SST Dataset
If any test file is passed via the command line (python hw3.py test_file.txt), it predicts the sentiments based on the new test set passed
The method prints POSITIVE/NEGATIVE on each line corresponding to each review in the test set
'''
def main():
  import sys
  if len(sys.argv) > 1:
    if sys.argv[1]=='test_file.txt':  
      test_doc = pd.read_csv(sys.argv[1], sep='\t') # Load Test set
      # Predict using the file passed via the command line (python hw3.py test_file.txt)
      print()
      print("Sentiments match corresponding reviews in the file")
      for review in test_doc['sentence'].values:
        print(predict(review, logprior, loglikelihood, C, V))
  else:
    # Predict using the test.tsv data
    test_doc = pd.read_csv('test.tsv', sep='\t') # Load Test set
    print()
    print("Sentiments match corresponding reviews in the file")
    for review in test_doc['sentence'].values:
      print(predict(review, logprior, loglikelihood, C, V))

main()