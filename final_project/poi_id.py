#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
#features_list = ['poi','salary'] # You will need to use more features
features_list = ['poi']
financial_features = ['salary', 'deferral_payments', 'total_payments', 'loan_advances', 'bonus', 'restricted_stock_deferred', 'deferred_income', 'total_stock_value', 'expenses', 'exercised_stock_options', 'other', 'long_term_incentive', 'restricted_stock', 'director_fees']
email_features = ['to_messages', 'from_poi_to_this_person', 'from_messages', 'from_this_person_to_poi', 'shared_receipt_with_poi'] # remove email_address

features_list += financial_features + email_features

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

### Task 2: Remove outliers
del data_dict['TOTAL']

### Task 3: Create new feature(s)

### Store to my_dataset for easy export below.
my_dataset = data_dict
for person in my_dataset:
    my_dataset[person]['to_poi_message_ratio'] = 0
    my_dataset[person]['from_poi_message_ratio'] = 0
    if float(my_dataset[person]['from_messages']) > 0:
        my_dataset[person]['to_poi_message_ratio'] = float(my_dataset[person]['from_this_person_to_poi'])/float(my_dataset[person]['from_messages'])
    if float(my_dataset[person]['to_messages']) > 0:
        my_dataset[person]['from_poi_message_ratio'] = float(my_dataset[person]['from_poi_to_this_person'])/float(my_dataset[person]['to_messages'])
    
features_list.extend(['to_poi_message_ratio', 'from_poi_message_ratio'])


### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html


from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.cluster import KMeans
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier

combined_features = FeatureUnion([("pca", PCA(n_components=19)), ("univ_select", SelectKBest(k=1))])

clfs = {}
clfs['logisticRegression'] = LogisticRegression(C=100000)

clfs['gaussianNB'] = GaussianNB()

clfs['kMeans'] = KMeans(n_clusters=2, tol=0.001)

clfs['adaboost'] = AdaBoostClassifier()

clfs['svc'] = SVC(kernel='rbf', C=1000)

clfs['randomForest'] = RandomForestClassifier()

clfs['sdg'] = SGDClassifier(loss='log')


### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!
#from sklearn.cross_validation import train_test_split
#features_train, features_test, labels_train, labels_test = \
#    train_test_split(features, labels, test_size=0.3, random_state=42)

from tester import dump_classifier_and_data, test_classifier


for clfName in clfs:
    #print clfName
    #print clfs[clfName]
    clf = Pipeline([("minmax", MinMaxScaler()), ("features", combined_features), ('clf', clfs[clfName])])
    print clf
    print clf.steps
    test_classifier(clf, my_dataset, features_list)

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)
