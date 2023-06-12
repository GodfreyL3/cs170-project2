import numpy as np
import matplotlib.pyplot as plt

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

    def leave_one_out(self):
        #get the number of rows in the data
        n = self.X.shape[0]
        #print("n:",n)
        #get the number of features in the data
        d = self.X.shape[1]
        #print("d:",d)
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
            classifier = NNClassifier(Y_train_new,X_train_new,self.classifier.K)
            #predict the class of the ith row
            y_pred[i] = classifier.predict(x)
        return y_pred



class Visualizer:
    def __init__(self,X,Y) -> None:
        self.X = X
        self.Y = Y

    def plot(self,feature_set):
        #check length of feature set (if 2 then 2d plot if 3 then 3d plot)
        if len(feature_set) == 2:
            is_2d = True
            is_3d = False
        elif len(feature_set) == 3:
            is_2d = False
            is_3d = True
        else:
            is_2d = False
            is_3d = False
        
        for i in range(len(feature_set)):
            #NOTE that we are using the feature set as a list of columns and the lists starts at 1 not 0
            if i == 0:
                X_features = self.X[:,feature_set[i]-1]
            else:
                X_features = np.vstack((X_features,self.X[:,feature_set[i]-1]))
        #transpose the training set so that the rows are the features and the columns are the data points
        X_features = X_features.T
        # dot plot of the data set the labels as feature values
        if is_2d:
            plt.scatter(X_features[:, 0], X_features[:, 1], c=self.Y)
            plt.xlabel("Feature {}".format(feature_set[0]))
            plt.ylabel("Feature {}".format(feature_set[1]))
            plt.show()
        elif is_3d:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(X_features[:, 0], X_features[:, 1], X_features[:, 2], c=self.Y)
            ax.set_xlabel("Feature {}".format(feature_set[0]))
            ax.set_ylabel("Feature {}".format(feature_set[1]))
            ax.set_zlabel("Feature {}".format(feature_set[2]))
            plt.show()
        return None


def main():
    filename = "CS170_Spring_2023_Large_data__46.txt"
    # load the data for the large test dataset set to 41 for small set to 11
    data = np.loadtxt(filename, usecols=range(41))
    Y = data[:, 0]
    # get the rest of the columns and set to X
    X = data[:, 1:]

    # create a classifier object (BE ODD)s
    classifier = NNClassifier(Y, X, 3)
    # create a feature set columns are in 1,2,3,4,5,6,7,8,9,10
    feature_set = [33,29]
    # create a validator object
    validator = Validator( X,Y, classifier, feature_set)

    y_pred = validator.leave_one_out()
    #print("Predictions:", y_pred)
    
    # Compare y_pred with actual values
    accuracy = np.mean(y_pred == Y)
    print("Accuracy:", accuracy)

    # Visualize the data
    visualizer = Visualizer(X,Y)
    visualizer.plot(feature_set)

main()
