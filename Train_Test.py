import random
import numpy as np
import math

# Change this to change the dataset used
filename = 'small-test-dataset.txt'

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


    def leave_one_out_cross_val(self, input_data, current_set_of_features, feature_to_add):

        # data is simply the data being inputted
        # current_set is an array of nums, showing the col added so far
        # feature is the feature being tested, id'd by a number

        # start from 1, since col 0 is just the class label
        data = np.copy(input_data)
        
        # pyton tend to reference the same spot in memory when you just do "="
        # Gotta create a copy of the current_set_of_features copy
        current_set = current_set_of_features.copy()
        

        # add the feature-to-add to current set, but first lets copy
        test_set = current_set
        # Feature we are testing to add
        test_set.append(feature_to_add)
        
        test_set.append(0)  # Always keep the class label, so adding 0 is a must

        
        # keeps track of the data row
        row_num = 0
        for row in data:
            # Keeps track of the feature point we are looking at
            data_point = 0
            for point in col:
                # If the data point (represented by a number) is not in the test set, then exlude it by setting itto 0
                if data_point not in test_set:
                    point = '0' # Do we need this?
                    
                    #   Set to 0
                    data[col_num][data_point] = '0'
                
                # Increments
                data_point += 1
            col_num += 1

        # Once we are done exluding our feature set being tested, now we must test the model
        
        # Set to 0, use for later
        num_correctly_classified = 0

        # iterator
        i = 1

        # For each row in data, their class label will always be the first point (1 or 2)
        for row in data:
            # Object to classify 
            label = float(row[0])

            # object 
            object = point[1:]

            #   Set both nearest-neighbor distance and locations to infinity by default
            nn_distance = float('inf')
            nn_location = float('inf')

            #   iterator
            k = 1

            # For each data point (neighbor) in data
            for neighbor in data:
                
                # assuming we are not comparing the same datapoint
                if(k != i):

                    # get euclidian distance 
                    distance = 0
                    for x in range(1, len(point)):
                        distance += math.pow((float(point[x]) - float(neighbor[x])), 2)
                        distance = math.sqrt(distance)
                        
                    # If the distance of the neighbor data point (k) is closer to the point being tested (i) than we is currently
                    # the clossest nearest0neighbor, set a new nearest neighbor (k)
                    if distance < nn_distance:
                        
                        #   Set distance
                        nn_distance = distance
                        #   Set the locaton (id) of neighbor (-1 is added because i starts at 1)
                        nn_location = k - 1
                        #   Get the class label of the point
                        nn_label = float(data[nn_location][0])
                        #print("Nearest Neighbor Label: " + str(nn_label))
                        
                # Increment inner loop
                k += 1

            # Once we are done, print result
            print('Object ' + str(i) + ' is class ' + str(label))
            print('Its nearest neighbor is object ' + str(nn_location + 1) + ' which is of class ' + str(nn_label))  

            # If the label assigned is equal to what the actual nearest-number label is, the model predicted correctly
            if(label == nn_label):
                num_correctly_classified += 1

            # increment outer loop
            i += 1

        # calculate and return accuracy
        accuracy = num_correctly_classified / len(data)

        return accuracy

    def train(self):

        ## Init empty set for current set
        current_set_of_features = []
        best_overall = 0

        
        for i in range(1, len(self.data[0]) + 1):
            print("On the " + str(i) + "th level of the search tree...")
            feature_to_add = 0
            best_accuracy = 0

            # Go over every feature
            for k in range(1, len(self.data[0]) + 1):
                
                # If feature is already in current set, then skip
                if k in current_set_of_features:
                    continue
                print("--Considering adding the " + str(k) + "th feature...")
                
                # Run cross-validation
                accuracy = self.leave_one_out_cross_val(self.data, current_set_of_features, k)

                # Here we are trying to find the best feature to add given our current set of features that would give us the best score
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    feature_to_add = k

            # If the accuracy found from the cross-validation test is better than anything previously
            # replace and record the set that got this best combination and accuracy
            if best_accuracy > best_overall:
                best_overall = best_accuracy
                self.best_features = current_set_of_features.copy()

            # once we find our best feature add it to the current set of features
            current_set_of_features.append(feature_to_add)        
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
        print(self.best_feature_data)
        


    


    





def main():
    # data = []
    # data.append(1)
    # data.append(2)
    # data.append(3)
    # data.append(4)

    # feature_search_demo(data)

    data = np.loadtxt(filename, dtype=str, usecols=range(11))

    classifier = Classifier(data)
    classifier.train()




if __name__ == "__main__":
    main()
