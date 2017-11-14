#myOwnDeply of KNN classifier
from scipy.spatial import distance

def euc(a,b):
    return distance.euclidean(a,b)

class ScrappyKNN():
    def fit(self, x_train, y_train):
        self.x_train = x_train
        self.y_train = y_train
        
    def predict(self, x_test):
        predictions = []
        for row in x_test:
            label = self.closest(row)
            predictions.append(label)
        return predictions
    
    def closest(self, row):
        best_dist = euc(row, self.x_train[0])
        best_index = 0
        for i in range(1, len(self.x_train)):
            dist = euc(row, self.x_train[i])
            if dist < best_dist:
                best_dist = dist
                best_index = i
        return self.y_train[best_index]
            
                       


#STEP 1 - Getting data - import a dataset

from sklearn import datasets
iris = datasets.load_iris()

#it is usual call features X and labels Y ...remember famous functionf(x)=y
x = iris.data
y = iris.target

from sklearn.cross_validation import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = .5) #test_size above define that 50% of the data will be used as a test and other half as a trainning

#STEP 2 - Trainning
#using KNeighbhors Classifier
#from sklearn.neighbors import KNeighborsClassifier
#my_classifier_kn = KNeighborsClassifier()
my_classifier_kn = ScrappyKNN()
my_classifier_kn.fit(x_train, y_train)
#step 3 - testing with kn classifier
predictions_kn = my_classifier_kn.predict(x_test)
print ("Predictions with KN classifier: ", predictions_kn)


#checking the accuracy of the model
from sklearn.metrics import accuracy_score
print ("KN accuracy: ", accuracy_score(y_test, predictions_kn))