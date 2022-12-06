# Naive-Bayes-Algorithm-Implementation

The repository contains the implementation of the Naive Bayes Algorithm in Python with the help of Pandas and Numpy Libraries

The training and testing data consist of movie reviews and their labels as either positive(1) or negative(9). Hence this project uses the supervised learning approach in machine learning.

The repository consists of four files, namely: naive_bayes.py, train.tsv, test.csv and requirements.txt

The following functions are present in naive_bayes.py:

1. def tokenization(sentence): takes a sentences and splits into words

2. def train(D, C): implements the train algorithm. The function takes in parameters **D** (the training document) and **C** (a list containing the classes) and
  returns the **loglikelihood**, **logprior** and **V** (a set of vocabulary)
  
3. def predict(testdoc, logprior, loglikelihood, C, V): implements the prediction or test function for the naive Bayes algorithm

4. def main(): Main function tests our Naive Bayes Classifier Model.
  If no test file is passed via the command line (python hw3.py), then it predicts using the test set from the SST Dataset
  If any test file is passed via the command line (python hw3.py test_file.txt), it predicts the sentiments based on the new test set passed
  The method prints POSITIVE/NEGATIVE on each line corresponding to each review in the test set
