#!/usr/bin/python

""" 
    Starter code for exploring the Enron dataset (emails + finances);
    loads up the dataset (pickled dict of dicts).

    The dataset has the form:
    enron_data["LASTNAME FIRSTNAME MIDDLEINITIAL"] = { features_dict }

    {features_dict} is a dictionary of features associated with that person.
    You should explore features_dict as part of the mini-project,
    but here's an example to get you started:

    enron_data["SKILLING JEFFREY K"]["bonus"] = 5600000
    
"""

import pickle

enron_data = pickle.load(open("../final_project/final_project_dataset.pkl", "r"))

count_salary = 0
count_email = 0
count_payment = 0
count = 0
for person in enron_data:
    if not enron_data[person]['poi']:
        continue
    count += 1
    #if 'SKILLING' in person or 'LAY' in person or 'FASTOW' in person:
    #    print person
    #    print enron_data[person]
    if enron_data[person]['salary'] != 'NaN':
        count_salary += 1
    if enron_data[person]['email_address'] != 'NaN':
        count_email += 1
    if enron_data[person]['total_payments'] == 'NaN':
        count_payment += 1
print count_salary
print count_email
print count_payment
print count
print "%.4f" % (count_payment * 1.0 /count)
