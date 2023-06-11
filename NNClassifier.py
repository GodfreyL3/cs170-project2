import numpy as np

class NNClassifier:
    def __init__(self,Y,X,K) -> None:
        self.y = Y
        self.X = X
        #n and d are the number of rows and columns in X
        self.n = self.X.shape[0]
        self.d = self.X.shape[1]
        #K is the number of nearest neighbors to use
        self.K = K
    
    # Predict the class of a single point x
    def predict(self,x):
        #calculate the euclidian distance between x and all the points in X
        dist = np.linalg.norm(self.X-x,axis=1)
        #sort the distance where y is the same from closest to furthest
        sorted_index = np.argsort(dist)
        #print(sorted_index)
        #print(self.y[sorted_index])
    
        #get the first K elements and their y values
        K_nearest = self.y[sorted_index[:self.K]]
        #given the K nearest neighbors, predict the class of x
        classes,counts = np.unique(K_nearest,return_counts=True)
        #print("Classes:",classes)
        #print("Counts:",counts)
        #return the class with the most counts
        return classes[np.argmax(counts)]
    
    # Predict the class of all points in X
    def predict_all(self,X):
        y_pred = np.zeros(X.shape[0])
        for i in range(X.shape[0]):
            y_pred[i] = self.predict(X[i,:])
        return y_pred


class Validator:
    def __init__(self,X,Y, classifier, feature_set) -> None:
        #X is the data
        self.X = X
        #Y is the labels
        self.Y = Y
        #classifier is the classifier object
        self.classifier = classifier
        #feature_set is the list of features to use
        self.feature_set = feature_set

    def leave_one_out(self, true_Y):
        #get the number of rows in the data
        n = self.X.shape[0]
        print("n:",n)
        #get the number of features in the data
        d = self.X.shape[1]
        print("d:",d)
        #build the training set using the feature set
        #get the columns of the data that are in the feature set 
        for i in range(len(self.feature_set)):
            #NOTE that we are using the feature set as a list of columns and the lists starts at 1 not 0
            if i == 0:
                X_train = self.X[:,self.feature_set[i]-1]
            else:
                X_train = np.vstack((X_train,self.X[:,self.feature_set[i]-1]))
        #transpose the training set so that the rows are the features and the columns are the data points
        X_train = X_train.T
        #get the shape of the training set
        #print("X_train shape:",X_train.shape)
        #get the shape of the labels
        #print("Y shape:",self.Y.shape)
        #create a list to hold the predictions
        y_pred = np.zeros(n)
        #for each row in the data
        for i in range(n):
            #get the ith row of the data
            x = X_train[i,:]
            #get the ith row of the labels
            y = self.Y[i]
            #remove the ith row from the training set
            X_train_new = np.delete(X_train,i,axis=0)
            #remove the ith row from the labels
            Y_train_new = np.delete(self.Y,i)
            #create a new classifier object
            classifier = NNClassifier(Y_train_new,X_train_new,1)
            #predict the class of the ith row
            y_pred[i] = classifier.predict(x)

        accuracy = np.mean(y_pred == true_Y)
        
        return accuracy


    
def main():
    filename = "large-test-dataset.txt"
    # load the data for the large test dataset set to 41 for small set to 11
    data = np.loadtxt(filename, usecols=range(41))
    Y = data[:, 0]
    # get the rest of the columns and set to X
    X = data[:, 1:]

    # create a classifier object
    classifier = NNClassifier(Y, X, 1)
    # create a feature set columns are in 1,2,3,4,5,6,7,8,9,10
    feature_set = [1,15,27]
    # create a validator object
    validator = Validator( X,Y, classifier, feature_set)

    accuracy = validator.leave_one_out(Y)
    #print("Predictions:", y_pred)
    
    # Compare y_pred with actual values
    print("Accuracy:", accuracy)

main()
