#Unzip the data because it's in Zip folder 
#We have there Train_pos,Train_neg,Test_pos,Test_neg
!unzip -q arabic-sentiment-twitter-corpus.zip -d .

#Import The Packages
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import time ###################################
#sklearn is machine learning library for python 
import sklearn 
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
#import random value 
import random
import os
#save model file and import it 
import pickle
#library used to plot results 
import matplotlib.pyplot as plt

#Function That can read the data files 
#It takes each data file and return 
#the tweet and the label(pos or neg)
def read_tsv(data):
    tweet = list()
    labels = list()
    infile = open(data, encoding='utf-8')
    for line in infile:
      #loop for each line 
      #look for long space in between and seperate 
      #depending on it  
        if not line.strip():
            continue
        label, text = line.split('\t')
        tweet.append(text)
        labels.append(label)
    return tweet, labels
 
 #Function that can take the 4 files 
#and gives us 4 arguments
#x_train ---> all train tweets (pos & neg)
#y_train ---> all labels of train tweets (pos & neg)
#x_test  ---> all test tweets (pos & neg)
#y_test  ---> all labels of test tweets (pos & neg)
def load(pos_train_file, neg_train_file, pos_test_file, neg_test_file):
    pos_train_data, pos_train_labels = read_tsv(pos_train_file)
    neg_train_data, neg_train_labels = read_tsv(neg_train_file)

    pos_test_data, pos_test_labels = read_tsv(pos_test_file)
    neg_test_data, neg_test_labels = read_tsv(neg_test_file)
    print('------------------------------------')

    sample_size = 5
    print('{} random train tweets (positive) .... '.format(sample_size))
    print(np.array(random.sample(pos_train_data, sample_size)))
    print('------------------------------------')
    print('{} random train tweets (negative) .... '.format(sample_size))
    print(np.array(random.sample(neg_train_data, sample_size)))
    print('------------------------------------')

    x_train = pos_train_data + neg_train_data
    y_train = pos_train_labels + neg_train_labels

    x_test = pos_test_data + neg_test_data
    y_test = pos_test_labels + neg_test_labels

    print('train data size:{}\ttest data size:{}'.format(len(y_train), len(y_test)))
    print('train data: # of pos:{}\t# of neg:{}\t'.format(y_train.count('pos'), y_train.count('neg')))
    print('test data: # of pos:{}\t# of neg:{}\t'.format(y_test.count('pos'), y_test.count('neg')))
    print('------------------------------------')
    return x_train, y_train, x_test, y_test
    
#function that saves the model after training
#in files that can be called again to test 
#any other data
def save_model(model, model_filepath,n_gram):
    pkl_filename =  "/content/drive/My Drive/{}_{}.pkl".format(model_filepath,n_gram)
    with open(pkl_filename, 'wb') as file:
        pickle.dump(model, file)
#Function of the model 
def do_sa(n, my_classifier, name, my_data):


    x_train, y_train, x_test, y_test = my_data
    print('parameters')
    print('n grams:', n)
    print('classifier:', my_classifier.__class__.__name__)
    print('------------------------------------')
    #model pipeline
    start_time = time.clock()###############################
    pipeline = Pipeline([
        ('vect', TfidfVectorizer(min_df=0.0001, max_df=0.95,
                                 analyzer='word', lowercase=False,
                                 ngram_range=(1, n))),
        ('clf', my_classifier),
    ])
    #fit training data to the pipeline 
    pipeline.fit(x_train, y_train)
    print (f"The runtime the algorithm took is {(time.clock() - start_time):.2f} seconds")########################
    #sample of features depending on ngram 
    feature_names = pipeline.named_steps['vect'].get_feature_names()
    #the model tries to predict the test tweets' labels 
    y_predicted = pipeline.predict(x_test)
    #Saving the models 
    save_model(pipeline,  my_classifier.__class__.__name__,n)
    # Print the classification report
    print(metrics.classification_report(y_test, y_predicted,
                                        target_names=['pos', 'neg']))

    # Print the confusion matrix
    cm = metrics.confusion_matrix(y_test, y_predicted)
    print(cm)
    print('# of features:', len(feature_names))
    print('sample of features:', random.sample(feature_names, 40))
    accuracy = accuracy_score(y_test, y_predicted)
    precision = precision_score(y_test, y_predicted, average='weighted')
    recall =  recall_score(y_test, y_predicted, average='weighted')
    return name, n, accuracy, precision, recall
    
#give all the function above the arguments 
#they need and run the model
ngrams = (1, 2, 3)  #n_gram values we used
results = []
pos_training = '/content/train_Arabic_tweets_positive_20190413.tsv'
neg_training = '/content/train_Arabic_tweets_negative_20190413.tsv'

pos_testing = '/content/test_Arabic_tweets_positive_20190413.tsv'
neg_testing = '/content/test_Arabic_tweets_negative_20190413.tsv'

#Algorithms we used
classifiers = [LinearSVC(), SVC(), MultinomialNB(),
               BernoulliNB(), SGDClassifier(), DecisionTreeClassifier(max_depth=5),
               RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
               KNeighborsClassifier(3)
               ]
#Do "for loop" for 3 values of n_gram 
#for each n_gram we get results 
for g in ngrams:
    dataset = load(pos_training, neg_training, pos_testing, neg_testing)
    for alg in classifiers:
        alg_name = alg.__class__.__name__
        r = do_sa(g, alg, alg_name, dataset)
        results.append(r)
  
#Result Summary 
print('{0:25}{1:10}{2:10}{3:10}{4:10}'.format('algorithm', 'ngram', 'accuracy', 'precision', 'recall'))
print('---------------------------------------------------------------------')
for r in results:
    print('{0:25}{1:10}{2:10.3f}{3:10.3f}{4:10.3f}'.format(r[0], r[1], r[2], r[3], r[4]))
    
#Plot the accuracy for each algorithm at each n_gram value
classifiers_names = []
accuracy_classifier = []
for r in results:
  classifiers_names.append(r[0])
  accuracy_classifier.append(r[2]*100)
#Accuracy at n_gram =1
plt.scatter(accuracy_classifier[0:7],classifiers_names[0:7])
plt.title("Accuracy when n_gram = 1")
plt.show()
#Accuracy at n_gram =2
plt.scatter(accuracy_classifier[7:14],classifiers_names[7:14])
plt.title("Accuracy when n_gram = 2")
plt.show()
#Accuracy at n_gram =3
plt.scatter(accuracy_classifier[14:21],classifiers_names[14:21])
plt.title("Accuracy when n_gram = 3")
plt.show()
