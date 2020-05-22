"""
@author: Oğuzhan Yıldırım 240201056
"""
import matplotlib.pyplot as plt
import pandas as pd
import math


def min_max_normalization(df,column,min,max):
    

    df.loc[:,column] = df.loc[:,column].apply(lambda x : (x - min) / (max-min))
        
    return ""

def euclidean(train,test):
    
    sum_of_squared_differences = 0.0
    for i in range(len(train)):
        sum_of_squared_differences += (train[i]-test[i])**2
    distance = math.sqrt(sum_of_squared_differences)
    
    return distance

def manhattan(train,test):
    distance = 0.0

    for i in range(len(train)):
        
        distance += (abs(train[i]-test[i]))
    return distance


def knn(train_data, test_data, distance_type, k):
    
    # Calculating Distances Part
        
    distance_list = [] # [ [index, calculated_distance] ]
    
    for row_idx in range(len(train_data)):
        
        distance = 0.0
        
        if (distance_type == 'Manhattan'):
            distance = manhattan(train_data[row_idx][:-1],test_data[:-1])
        
        elif (distance_type == 'Euclidean'):
            distance = euclidean(train_data[row_idx][:-1],test_data[:-1])
        
        d = [row_idx,distance]
        distance_list.append(d)
    
    distance_list = sorted(distance_list, key=lambda x: x[1]) # sorting according to nearest
    
    # Getting first k nearest neighbors
    k_neighbors = distance_list[0:k]
    # -------------------------------------------------------
    # Choosing class label according to weighted majority vote
    
    weighted_majority_1 = 0
    weighted_majority_0 = 0
  
    for neighbor in k_neighbors: 
        
        train_index = neighbor[0] # index of neighbor in train data
        dist = neighbor[1] # calculated_distance
        
        if int(train_data[train_index][2]) == 1: # if neighbors class is 1 increase weight vote for class label 1
            if dist != 0.0:
                weighted_majority_1 += 1 / (dist)
            else:
                weighted_majority_1 += 9999
                  
        elif int(train_data[train_index][2]) == 0:  
            if dist != 0.0:
                weighted_majority_0 += 1 / (dist)
            else:
                weighted_majority_0 += 9999
    
    return 1 if weighted_majority_1>weighted_majority_0 else 0
        
    
    


def calculate_accuracy(list):
    
    return list.count(1)/len(list)

def accuracy_graph(accuracies,k_list,title,test):
    
    plt.plot(k_list,accuracies)
    plt.xlabel('k value')
    plt.ylabel('accuracy')
    plt.title(test + " - " +title)
    plt.show()
    
    return ""


def test_different_k(k_max,distance_type,train,test):
    
    accuracy_list = []
    k_list = []
    for k in range(1,k_max+1):
        
        k_list.append(k)
        prediction_set = []
        
        for patient in test:  
            predicted_value = knn(train,patient,distance_type,k)
            if predicted_value == patient[2]:
                prediction_set.append(1)
            else:
                prediction_set.append(0)
            
        accuracy_for_current_k = calculate_accuracy(prediction_set)
        accuracy_list.append(accuracy_for_current_k)
    
    return accuracy_list,k_list
    


def main():
    
    # I put this fixed min max values as mentioned in pdf 
    # cause while implementing I realized that the max or min value can change 
    # for given test sets or train set and this will lead to inconsisten normalization of data.
    # for this sets this is not a problem but it might be a problem with different test set.
    cough_max = 5
    cough_min = 0
    fever_max = 39.9
    fever_min = 35.0
    
    # reading data
    global CovidDF
    CovidDF = pd.read_csv('covid.csv', delimiter=',')
    print(CovidDF.head(),"\n")
    print("Data readed...","\n------------------------------")
    
  
    # Answer of a
    min_max_normalization(CovidDF,'cough_level',cough_min,cough_max)
    min_max_normalization(CovidDF,'fever',fever_min,fever_max)

    print(CovidDF.head(),"\n")
    print("Data normalized...","\n------------------------------")
    
    # Answer of b
    print("\nknn() Function created...","\n------------------------------")
    
    # Answer of c
    print("\nEuclidean Distance Function created...","\n------------------------------")
    
    # Answer of d
    print("\nManhattan Distance Function created...","\n------------------------------")


    # Answer of e
    
    # train & test
    train_set = CovidDF.values.tolist()
    
    # normalizing test
    test_set1 = [[5,39.0,1], [4,35.0,0], [3,38.0,0], [2,39.0,1], [1,35.0,0],
                [0,36.2,0], [5,39.0,1], [2,35.0,0], [3,38.9,1], [0,35.6,0]]
    
    test_set2 = [[5, 39.0, 1], [4, 35.0, 0], [3, 38.0, 0],
              [2, 39.0, 1], [1, 35.0, 0], [0, 36.2, 0],
              [5, 39.0, 1], [2, 35.0, 0], [3, 38.9, 1],
              [0, 35.6, 0], [4, 37.0, 0], [4, 36.0, 1],
              [3, 36.6, 0], [3, 36.6, 1], [4, 36.6, 1]]
    
    
    # normalization of test1   -- first i convert to df to fit to min max function this way i can use the function both with train and test data
    test_df1 = pd.DataFrame.from_records(test_set1, columns = ['cough_level','fever','target'])
        
    min_max_normalization(test_df1,'cough_level',cough_min,cough_max)
    min_max_normalization(test_df1,'fever',fever_min,fever_max)
    
    test_set1 = test_df1.values.tolist()
    
    # normalization of test2
    test_df2 = pd.DataFrame.from_records(test_set2, columns = ['cough_level','fever','target'])
    
    min_max_normalization(test_df2,'cough_level',cough_min,cough_max)
    min_max_normalization(test_df2,'fever',fever_min,fever_max)
    
    test_set2 = test_df2.values.tolist()
    
    
    
    # Answer of f
    k_max = len(train_set)
 
    accuracies_euc1,k_list = test_different_k(k_max, 'Euclidean', train_set, test_set1)
    accuracies_man1,k_list = test_different_k(k_max, 'Manhattan', train_set, test_set1)
    accuracies_euc2,k_list = test_different_k(k_max, 'Euclidean', train_set, test_set2)
    accuracies_man2,k_list = test_different_k(k_max, 'Manhattan', train_set, test_set2)

    accuracy_graph(accuracies_man1,k_list,'Manhattan','Test1')
    accuracy_graph(accuracies_euc1,k_list,'Euclidean','Test1')
    accuracy_graph(accuracies_man2,k_list,'Manhattan','Test2')
    accuracy_graph(accuracies_euc2,k_list,'Euclidean','Test2')
    
    # Answer of g
    
    # In first test set we can see that we have %90 success of predictions in both distance types.
    
    # This tells us that the class label distinction of given test elements' values are precise and 
    # each time only one patient is predicted false. This patient might have the sensitive values that is very close to 
    # both class. We can think it as an outlier of the prediction.
    
    
    # For the second case we should divide our perspective to 2 parts.
    
    # In manhattan distance we see that after k value 10 the accuracy increases to %80 percent
    # however if we continue increasing the k , accuracy drops to %73 percents until k = 30. 
    # from 30 to 35 we see 2 more k values that reach to %80. 
    # we can say that the optimum k value is seen in 10 to 20 in Manhattan.
    
    # In Euclidean until k > 3  the accuracy is low and after 3 until 13 it goes constant as %73 percent and starts 
    # increasing from 13 k value until 20's . And also from 30 to 34 we see again a increase. For this distance type 
    # again between 10's to 20's we get the highest success. 
    
    
    
    # As a result of second test_set I can say that If k is too small, knn() would be more sensitive to outliers.
    # If k is too large, then the neighborhood of the test element will have some neighboors from other class label
    # finding an optimum range of k is the heart of this algorithm.
    
    
if __name__ == "__main__":

    main()




