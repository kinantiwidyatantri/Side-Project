# Evaluate the classifiers using the k-fold cross-validation technique. Output their average precisions, recalls, F-measures and accuracies. Implement the
# validation. Remember that folds need to be of roughly equal size. The template contains a range of
# functions needed to implement for this task.

import os
import Task_1_5
from Task_1_5 import computeMeasure1, computeMeasure2, computeMeasure3
import Task_2
import numpy as np
import Dummy


#Disclaimer: This task does work, it's just take a really long time to process everything


# This function takes the data for cross evaluation and returns training_data a list of lists s.t. the first element
# is the round number, second is the training data for that round, and third is the testing data for that round
#
# INPUT: training_data      : a list of lists that was read from the training data csv (see parse_arguments function)
#        f                  : the number of folds to split the data into (which is also same as # of rounds)
# OUTPUT: folds             : a list of lists s.t. the first element is the round number, second is the training data
#                             for that round, and third is the testing data for that round

def splitDataForCrossValidation(training_data, f):
    fold_size = len(training_data) // f
    folds_data = []

    for i in range(f):
        start = i * fold_size
        end = (i + 1) * fold_size

        testing= training_data[start:end]
        training = training_data[:start] + training_data[end:]

        folds_data.append([i+1, training, testing])
    
    return folds_data


# This function implement validation of the data that is produced by the cross evaluation function PRIOR to
# the addition of rows with the average meaasures.
#
# INPUT:  data              : a list of lists that was produced by the crossEvaluateKNN function
#         f                 : number of folds to validate against
#
# OUTPUT: boolean value     : True if the data contains the header ["Path", "ActualClass", "PredictedClass","FoldNumber"]
#                             (there can be more column names, but at least these four at the start must be present)
#                             AND the values in the "Path" column (if there are any) are file paths
#                             AND the values in the "ActualClass" and "PredictedClass" columns
#                             (if there are any) are classes from the scheme
#                             AND the values in the "FoldNumber" column are integers in [0,f) range
#                             AND there are as many Path entries as ActualClass and PredictedClass and FoldNumber entries
#                             AND the number of entries per each integer in [0,f) range for FoldNumber are approximately
#                             the same (they can differ by at most 1)
#
#                             False otherwise

def validateDataFormat(data, f):
    formatCorrect = False
    
    if data[0] == ["Path", "ActualClass", "PredictedClass","FoldNumber"]:
        
        path = []
        actual = []
        predicted = []
        fold = []
        
        for c, v in enumerate(data):
            if c > 0:
                if os.path.isfile(data[c][0]):
                    path.append(data[c][0])
                
                if data[c][1] in Task_1_5.classification_scheme and data[c][2] in Task_1_5.classification_scheme:
                    actual.append(data[c][1])
                    predicted.append(data[c][2])
                  
                if isinstance(data[c][3], int) and data[c][3] in range (0,f):  
                    fold.append(data[c][3])
        
        if len(path) == len(actual) == len(predicted) == len(fold):
            counter = []
            
            for a in range(0,f):
                counter.append(fold.count(a))
            
            if all(count == counter[0] for count in counter) and max(counter) - min(counter) <= 1:
                formatCorrect = True
                

    return formatCorrect


# This function takes the classified data from each cross validation round and calculates the average precision, recall,
# accuracy and f-measure for them.
# Invoke either the Task 2 evaluation function 
#
# INPUT: classified_data_list
#                           : a tuple consisting of the classified data computed for each cross validation round
#        evaluation_func    : the function to be invoked for the evaluation (by default, it is the one from
#                             Task_2, but you can use dummy)
# OUTPUT: avg_precision, avg_recall, avg_f_measure, avg_accuracy
#                           : average evaluation measures. You are expected to evaluate every classified data in the
#                             tuple and average out these values in the usual way.

def evaluateCrossValidation(*classified_data_list, evaluation_func=Task_2.evaluateKNN):
    avg_precision = float(0)
    avg_recall = float(0)
    avg_f_measure = float(0)
    avg_accuracy = float(0)
    length = len(classified_data_list)
    
    for data in classified_data_list:
        precision, recall, f_measure, accuracy = evaluation_func(data)
        
        avg_precision += precision
        avg_recall += recall
        avg_f_measure += f_measure
        avg_accuracy += accuracy
        
    avg_precision = avg_precision/length
    avg_recall = avg_recall/length
    avg_f_measure = avg_f_measure/length
    avg_accuracy = avg_accuracy/length

    # There are multiple ways to count average measures during cross-validation. For the purpose of this portfolio,
    # it's fine to just compute the values for each round and average them out in the usual way.

    return avg_precision, avg_recall, avg_f_measure, avg_accuracy


# Perform cross-validation where f defines the number of folds to consider.
# "processed" holds the information from training data along with the following information: for each image,
# stated the id of the fold it landed in, and the predicted class it was assigned once it was chosen for testing data.
# After everything is done, add the average measures at the end. The writing to csv is done in a different function.
# Invoke the Task 1 kNN classifier 
#
# INPUT: training_data      : a list of lists that was read from the training data csv (see parse_arguments function)
#        k                  : the value of k neighbours, to be passed to the kNN classifier
#        measure_func       : the function to be invoked to calculate similarity/distance
#        similarity_flag    : a boolean value stating that the measure above used to produce the values is a distance
#                             (False) or a similarity (True)
#        knn_func           : the function to be invoked for the classification (by default, it is the one from
#                             Task_1_5, but you can use dummy)
#        split_func         : the function used to split data for cross validation (by default, it is the one above)
#        f                  : number of folds to use in cross validation
# OUTPUT: processed+r       : a list of lists which expands the training_data with columns stating the fold number to
#                             which a given image was assigned and the predicted class for that image; and with rows
#                             that contain the average evaluation measures



def crossEvaluateKNN(training_data, k, measure_func, similarity_flag, f, knn_func=Task_1_5.kNN,
                     split_func=splitDataForCrossValidation):
    # This adds the header
    processed = np.array([['Path', 'ActualClass', 'PredictedClass', 'FoldNumber']])
    avg_precision = -1.0
    avg_recall = -1.0
    avg_fMeasure = -1.0
    avg_accuracy = -1.0
    
    folds = split_func(training_data, f)
    classified_data_list = []
    #measure_function = None

    '''if measure_func == 'computeMeasure1':
        measure_function = Task_1_5.computeMeasure1
    elif measure_func == 'computeMeasure2':
        measure_function = Task_1_5.computeMeasure2
    elif measure_func == 'computeMeasure3':
        measure_function = Task_1_5.computeMeasure3
    elif measure_func == 'selfComputeMeasure1':
        measure_function = Task_1_5.selfComputeMeasure1
    elif measure_func == 'selfComputeMeasure2':
        measure_function = Task_1_5.selfComputeMeasure2'''
    
    for c, v in enumerate(folds):
        fold_id = folds[c][0]
        training = folds[c][1]
        testing = folds[c][2]
        
        classified_data = knn_func(training, k, measure_func, similarity_flag, testing)
        
        classified_data = knn_func(training, k, measure_func, similarity_flag, testing)
        classified_data_np = np.insert(classified_data, 3, fold_id, axis=1)
        classified_data_list.append(classified_data)
        processed = np.vstack((processed, classified_data_np))
           
    avg_precision, avg_recall, avg_fMeasure, avg_accuracy = evaluateCrossValidation(*classified_data_list)

    # The measures are now added to the end:
    h = ['avg_precision', 'avg_recall', 'avg_f_measure', 'avg_accuracy']
    v = [avg_precision, avg_recall, avg_fMeasure, avg_accuracy]
    r = np.array([[h[i], v[i]] for i in range(len(h))])

    return processed.tolist() + r.tolist()


# This function reads the necessary arguments (see parse_arguments function in Task_1_5),
# and based on them evaluates the kNN classifier using the cross-validation technique. The results
# are written into an appropriate csv file.
def main():
    opts = Task_1_5.parseArguments()
    if not opts:
        exit(1)
    print(f'Reading data from {opts["training_data"]}')
    training_data = Task_1_5.readCSVFile(opts['training_data'])
    print('Evaluating kNN')
    result = crossEvaluateKNN(training_data, opts['k'], eval(opts['measure']), opts['simflag'], opts['f'],
                              eval(opts['al']), eval(opts['sf']))
    path = os.path.dirname(os.path.realpath(opts['training_data']))
    out = f'{path}/{Task_1_5.student_id}_cross_validation.csv'
    print(f'Writing data to {out}')
    Task_1_5.writeCSVFile(out, result)


if __name__ == '__main__':
    main()
