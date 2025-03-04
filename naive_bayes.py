#-------------------------------------------------------------------------
# AUTHOR: Brandon Trieu
# FILENAME: naive_bayes.py
# SPECIFICATION: This program reads in a training dataset and a test dataset to train a Naive Bayes model and make predictions on the test dataset.
# FOR: CS 4210- Assignment #2
# TIME SPENT: 8 hours
#-----------------------------------------------------------*/

#IMPORTANT NOTE: DO NOT USE ANY ADVANCED PYTHON LIBRARY TO COMPLETE THIS CODE SUCH AS numpy OR pandas. You have to work here only with standard
# dictionaries, lists, and arrays

#Importing some Python libraries
from sklearn.naive_bayes import GaussianNB
import csv

#Reading the training data in a csv file
#--> add your Python code here
training_data = []
with open('assignment2/weather_training.csv', 'r') as file:
    reader = csv.reader(file)
    next(reader)
    for row in reader:
        training_data.append(row)

#Transform the original training features to numbers and add them to the 4D array X.
#For instance Sunny = 1, Overcast = 2, Rain = 3, X = [[3, 1, 1, 2], [1, 3, 2, 2], ...]]
#--> add your Python code here
feature_map = {'Sunny': 1, 'Overcast': 2, 'Rain': 3, 'Hot': 1, 'Mild': 2, 'Cool': 3, 
                   'High': 1, 'Normal': 2, 'Weak': 1, 'Strong': 2}
X = []
for row in training_data:
    X.append([feature_map[row[1]], feature_map[row[2]], feature_map[row[3]], feature_map[row[4]]])

#Transform the original training classes to numbers and add them to the vector Y.
#For instance Yes = 1, No = 2, so Y = [1, 1, 2, 2, ...]
#--> add your Python code here
class_map = {'Yes': 1, 'No': 2}
Y = [class_map[row[5]] for row in training_data]

#Fitting the naive bayes to the data
clf = GaussianNB(var_smoothing=1e-9)
clf.fit(X, Y)

#Reading the test data in a csv file
#--> add your Python code here
test_data = []
with open('assignment2/weather_test.csv', 'r') as file:
    reader = csv.reader(file)
    next(reader) 
    for row in reader:
        test_data.append(row)

#Printing the header os the solution
#--> add your Python code here
print("Day, Outlook, Temperature, Humidity, Wind, PlayTennis, Confidence")

#Use your test samples to make probabilistic predictions. For instance: clf.predict_proba([[3, 1, 2, 1]])[0]
#--> add your Python code here
for row in test_data:
    features = [feature_map[row[1]], feature_map[row[2]], feature_map[row[3]], feature_map[row[4]]]
    probabilities = clf.predict_proba([features])[0]
    prediction = clf.predict([features])[0]
    confidence = max(probabilities)
    if confidence >= 0.75:
        predicted_class = 'Yes' if prediction == 1 else 'No'
        print(f"{row[0]}, {row[1]}, {row[2]}, {row[3]}, {row[4]}, {predicted_class}, {confidence:.4f}")



