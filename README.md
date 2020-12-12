## Introduction:
we are trying to use sklearn to classify Arabic tweets (Negative or Positive) . used 8 classifiers at 3 values of n_gram (1,2,3) .
After the training we saved the models on google drive to easy import after that to test.


## Test: 
Testing is done using two ways : 
- Test using one tweet
- Test using a file of tweets and print out how much of it is positive and how much is negative

## Dataset:
This dataset is collected in April 2019. It contains 58K Arabic tweets (47K training, 11K test) tweets annotated in positive and negative labels. The dataset is balanced and collected using positive and negative emojis lexicon.

Data format: Tab-separated values TSV
label

## Result:


algorithm                ngram     accuracy  precision recall    
---------------------------------------------------------------------
LinearSVC                         1     0.770     0.770     0.770
SVC                               1     0.800     0.803     0.800
MultinomialNB                     1     0.761     0.762     0.761
BernoulliNB                       1     0.758     0.763     0.758
SGDClassifier                     1     0.764     0.768     0.764
DecisionTreeClassifier            1     0.572     0.693     0.572
RandomForestClassifier            1     0.512     0.655     0.512
KNeighborsClassifier              1     0.698     0.748     0.698
LinearSVC                         2     0.774     0.774     0.774
SVC                               2     0.800     0.802     0.800
MultinomialNB                     2     0.769     0.770     0.769
BernoulliNB                       2     0.762     0.772     0.762
SGDClassifier                     2     0.772     0.776     0.772
DecisionTreeClassifier            2     0.572     0.694     0.572
RandomForestClassifier            2     0.513     0.635     0.513
KNeighborsClassifier              2     0.698     0.750     0.698
LinearSVC                         3     0.772     0.772     0.772
SVC                               3     0.799     0.802     0.799
MultinomialNB                     3     0.771     0.772     0.771
BernoulliNB                       3     0.761     0.775     0.761
SGDClassifier                     3     0.773     0.777     0.773
DecisionTreeClassifier            3     0.572     0.694     0.572
RandomForestClassifier            3     0.511     0.713     0.511
KNeighborsClassifier              3     0.699     0.750     0.699
