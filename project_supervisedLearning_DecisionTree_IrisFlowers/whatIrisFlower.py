import numpy as np
import graphviz as gp
from sklearn.datasets import load_iris
from sklearn import tree


#http://scikit-learn.org/stable/modules/tree.html

#The iris dataset is already included in the sklearn as a example dataset

#this is project is based in the famous Iris Flower identification problem
#https://en.wikipedia.org/wiki/Iris_flower_data_set


#STEP 1 - Loading data



iris = load_iris()
print ("These are the features: ", iris.feature_names)
print ("These are the text labels: ", iris.target_names)

print ("These are the features from first example: ", iris.data[0])
print ("This is the label from first example: ", iris.target[0])


# these will be the training data
train_target = iris.target
train_data = iris.data


#STEP2 - Training data

print("Trainning the model...")
clf = tree.DecisionTreeClassifier()
clf.fit(train_data, train_target)

#STEP 3 - Make predictions

#print ("Doing the predictions... Ready!, Here the results:.. ")
#print (clf.predict(test_target))

#STEP 4 - See graphical result 

#exporting in a black and white format
dot_data = tree.export_graphviz(clf, out_file="irisExampleGraph") 
graph = gp.Source(dot_data) 

#i am not using the function below because Graphviz and Anaconda do not work well
#graph.render("iris") 

#exporting node format more colorful
dot_datab = tree.export_graphviz(clf, out_file="irisNodeFormat", 
                         feature_names=iris.feature_names,  
                         class_names=iris.target_names,  
                         filled=True, rounded=True,  
                         special_characters=True)  

graph = gp.Source(dot_datab)
#i am not using the function below because Graphviz and Anaconda do not work well
#graph.render("iris")



