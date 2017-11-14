#STEP 1 - Getting data - import a dataset

from sklearn import datasets
iris = datasets.load_iris()

#it is usual call features X and labels Y ...remember famous functionf(x)=y
x = iris.data
y = iris.target

from sklearn.cross_validation import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = .5) #test_size above define that 50% of the data will be used as a test and other half as a trainning

#STEP 2 - Trainning

#using DecisionTree Classifier
from sklearn import tree
my_classifier = tree.DecisionTreeClassifier()
my_classifier.fit(x_train, y_train)
#step 3 - testing with decisiontree classifier
predictions = my_classifier.predict(x_test)
print ("Predictions with DecisionTree classifier: ", predictions)

#using KNeighbhors Classifier
from sklearn.neighbors import KNeighborsClassifier
my_classifier_kn = KNeighborsClassifier()
my_classifier_kn.fit(x_train, y_train)
#step 3 - testing with kn classifier
predictions_kn = my_classifier_kn.predict(x_test)
print ("Predictions with KN classifier: ", predictions_kn)


#checking the accuracy of the model
from sklearn.metrics import accuracy_score
print ("DecisionTree accuracy: ", accuracy_score(y_test, predictions))
print ("KN accuracy: ", accuracy_score(y_test, predictions_kn))

