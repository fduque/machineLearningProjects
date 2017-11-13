from  sklearn import tree

#Project - HelloWorld Example Supervised Learning 

#STEP 1 - Getting the data
#features = [[140,"smooth"], [130,"smooth"],[150,"bumpy"],[170,"bumpy"]
#changing text labels for numerical labels

features = [[140,1], [130,1],[150,0],[170,0]]

#labels = ["apple", "apple", "orange", "orange"]
#changing text labels for numerical labels

labels = [0, 0, 1, 1]

#STEP 2 - Trainning the model
#assigning the method classifir by decision tree to the variable clf
clf = tree.DecisionTreeClassifier()

#the function fit below... is like find patterns in the data
#passing features and labels as arguments to the fit function
clf = clf.fit(features, labels)

#STEP 3 - Making predictions
#The prediction below informs based on the weight of 150g and been a zero (0) = "bumpy", what would be the possible fruit (0=apple and 1=orange)?

print ("This is a fruit type: ")
print (clf.predict([[150,0]]))