import random
import numpy as np
import math
# Change this to change the dataset used
filename = 'small-test-dataset.txt'
filename2 = 'large-test-dataset.txt'

featureSmall = [3, 5, 7]
featureLarge = [1, 15, 27]

# This will be used to keep track of the best accuracy found when exploring feature sets.
best_acc = 0


class Classifier:

    def __init__(self, data):

        # init feature list
        self.best_features = []

        # numpy data array
        self.data = data

        # numpy data consisting of the best features found through training
        self.best_feature_data = []

    def leave_one_out_cross_val(self, current_set, feature_to_add):
        num_correctly_classified = 0

        features = current_set.copy()
        features.append(feature_to_add)

        selected_data = self.data[:, features]  # Select only the required features

        for i, object_to_classify in enumerate(selected_data):
            distances = np.sqrt(np.sum((selected_data - object_to_classify)**2, axis=1))
            distances[i] = np.inf  # Exclude the distance to itself
            nn_location = np.argmin(distances)
            nn_label = self.data[nn_location, 0]  # Get the label of the nearest neighbor

            if self.data[i, 0] == nn_label:
                num_correctly_classified += 1

        accuracy = num_correctly_classified / len(self.data)
        return accuracy

    def train(self):

        ## Init empty set for current set
        current_set_of_features = []
        best_overall = 0

        
        for i in range(1, len(self.data[0])):
            print("On the " + str(i) + "th level of the search tree...")
            feature_to_add = 0
            best_accuracy = 0

            # Go over every feature
            for k in range(1, len(self.data[0])):
                
                # If feature is already in current set, then skip
                if k in current_set_of_features:
                    continue
                print("--Considering adding the " + str(k) + "th feature...")
                
                # Run cross-validation
                accuracy = self.leave_one_out_cross_val(current_set_of_features, k)

                # Here we are trying to find the best feature to add given our current set of features that would give us the best score
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    feature_to_add = k

            current_set_of_features.append(feature_to_add) 

            # If the accuracy found from the cross-validation test is better than anything previously
            # replace and record the set that got this best combination and accuracy
            if best_accuracy > best_overall:
                best_overall = best_accuracy
                self.best_features = current_set_of_features.copy()

            # once we find our best feature add it to the current set of features       
            print("On level " + str(i) + " the feature " + str(feature_to_add) + " was added to the current set\n")


        # print result :)
        print('Best Overall was ' + str(best_overall))
        print('Feature Trace is ' + str(self.best_features))

    # Now that we have our best features, lets cxreate the data with only our best features being present

        # Each full row 
        for feature in self.data:

            # init at 0
            i = 0

            # declare an empty instance
            instance = []

            # append the classifier by default
            instance.append(float(feature[0]))

            # check if each data category is inside our best feature trace, if no do not add
            for data_point in feature:
                i += 1
                if i in self.best_features:
                    instance.append(float(data_point))

            # append new data
            self.best_feature_data.append(instance)

        # debugging :P
        # print(self.best_feature_data)

    def test(self, featureSet):

        features = []
        for feature in featureSet:
            accuracy = self.leave_one_out_cross_val(features, feature)
            print("Test accuracy with feature ", feature, ": ", str(accuracy))
            features.append(feature)


def main():
    # data = []
    # data.append(1)
    # data.append(2)
    # data.append(3)
    # data.append(4)

    # feature_search_demo(data)

    data = np.loadtxt(filename, usecols=range(11))
    data2 = np.loadtxt(filename2, usecols = range(41))
    
    classifier = Classifier(data)
    classifier2 = Classifier(data2)
    classifier.train()
    # classifier2.train()
    classifier.test(featureSmall)
    # classifier2.test(featureLarge)
    


if __name__ == "__main__":
    main()
