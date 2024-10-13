
# Evaluate the classifiers. Implement a method that will create a confusion matrix based on the
# provided classified data. Then implement methods that will output precision, recall, F-measure, and accuracy of
# the classifier based on your confusion matrix. Use macro-averaging approach and be mindful of edge cases. The
# template contains a range of functions you need to implement for this task.


import Task_1_5
import numpy as np
import Dummy


# This function computes the confusion matrix based on the provided data.
#
# INPUT: classified_data   : a list of lists containing paths to images, actual classes and predicted classes.
#                            Please refer to Task 1 for precise format description.
# OUTPUT: confusion_matrix : the confusion matrix computed based on the classified_data. The order of elements needs
#                            to be the same as  in the classification scheme. The columns correspond to actual classes
#                            and rows to predicted classes. In other words, confusion_matrix[0] should be understood
#                            as the row of values predicted as Female, and [row[0] for row in confusion_matrix] as the
#                            column of values that were actually Female

def confusionMatrix(classified_data):
    classes_name = set()
    for c, v in enumerate(classified_data):
        if c > 0:
            classes_name.add(classified_data[c][1])
            classes_name.add(classified_data[c][2])  
    
    classes_name = sorted(classes_name, key=Task_1_5.classification_scheme.index)
    table_length = len(classes_name)
    
    confusion_matrix = np.zeros((table_length, table_length), dtype=np.int32)
    
    for c2, v2 in enumerate(classified_data):
        if c2 > 0:
            actual = classified_data[c2][1]
            predicted = classified_data[c2][2]
            index_actual = classes_name.index(actual)
            index_predicted = classes_name.index(predicted)
            
            confusion_matrix[index_predicted][index_actual] += 1
    
    return confusion_matrix.tolist()


# These functions compute per-class true positives and false positives/negatives based on the provided confusion matrix.
#
# INPUT: confusion_matrix : the confusion matrix computed based on the classified_data. The order of elements is
#                           the same as  in the classification scheme. The columns correspond to actual classes
#                           and rows to predicted classes.
# OUTPUT: a list of appropriate true positive, false positive or false
#         negative values per a given class, in the same order as in the classification scheme. For example, tps[1]
#         corresponds for TPs for Male class.


def computeTPs(confusion_matrix):
    tps = []
    num_classes = len(confusion_matrix)

    for i in range(num_classes):
        tp = confusion_matrix[i][i]  
        tps.append(tp)

    return tps


def computeFPs(confusion_matrix):
    fps = []
    classification = len(confusion_matrix)

    for i in range(classification):
        fp = sum(confusion_matrix[j][i] for j in range(classification)) - confusion_matrix[i][i]
        fps.append(fp)
        
    return fps


def computeFNs(confusion_matrix):
    fns = []
    classification = len(confusion_matrix)

    for i in range(classification):
        fn = sum(confusion_matrix[i][j] for j in range(classification)) - confusion_matrix[i][i]
        fns.append(fn)

    return fns

# These functions compute the evaluation measures based on the provided values. Not all measures use of all the values.
#
# INPUT: tps, fps, fns, data_size
#                       : the per-class true positives, false positive and negatives, and size of the classified data.
# OUTPUT: appropriate evaluation measures created using the macro-average approach.

def computeMacroPrecision(tps, fps):
    class_length = len(tps)
    precision = float(0)

    for i in range(class_length):
        if tps[i] + fps[i] != 0:
            precision += tps[i] / (tps[i] + fps[i])

    precision = precision / class_length
    
    return precision


def computeMacroRecall(tps, fns):
    class_length = len(tps)
    recall = float(0)

    for i in range(class_length):
        
        if tps[i] + fns[i] != 0:
            recall += tps[i] / (tps[i] + fns[i])

    recall = recall / class_length
    
    return recall


def computeMacroFMeasure(tps, fps, fns):
    class_length = len(tps)
    f_measure = float(0)

    for i in range(class_length):
        precision = 0 
        if tps[i] + fps[i] != 0:
             precision = tps[i] / (tps[i] + fps[i])
             
        recall = 0 
        if tps[i] + fns[i] != 0:
            recall = tps[i] / (tps[i] + fns[i])
                
        if precision + recall != 0:
             f_measure += 2 * (precision * recall) / (precision + recall)

    f_measure =  f_measure / class_length
    
    return f_measure


def computeAccuracy(tps, data_size):
    accuracy = float(0)
    total_tp = sum(tps)
   
    if data_size != 0:
        accuracy = total_tp / data_size

    return accuracy


# This function expected to compute precision, recall, f-measure and accuracy of the classifier using
# the macro average approach.

# INPUT: classified_data   : a list of lists containing paths to images, actual classes and predicted classes.
#                            Please refer to Task 1 for precise format description.
#       confusion_func     : function to be invoked to compute the confusion matrix
#
# OUTPUT: computed measures
def evaluateKNN(classified_data, confusion_func=confusionMatrix):
    confusion_matrix = confusion_func(classified_data)
    
    true_positive = computeTPs(confusion_matrix)
    false_positive = computeFPs(confusion_matrix)
    false_negative = computeFNs(confusion_matrix)
    data_size = len(classified_data)
    
    precision = computeMacroPrecision(true_positive, false_positive)
    recall = computeMacroRecall(true_positive, false_negative)
    f_measure = computeMacroFMeasure(true_positive, false_positive, false_negative)
    accuracy = computeAccuracy(true_positive, data_size)
  

    # once ready, we return the values
    return precision, recall, f_measure, accuracy


# This function reads the necessary arguments (see parse_arguments function in Task_1_5),
# and based on them evaluates the kNN classifier.
def main():
    opts = Task_1_5.parseArguments()
    if not opts:
        exit(1)
    print(f'Reading data from {opts["classified_data"]}')
    classified_data = Task_1_5.readCSVFile(opts['classified_data'])
    print('Evaluating kNN')
    result = evaluateKNN(classified_data, eval(opts['cf']))
    print('Result: precision {}; recall {}; f-measure {}; accuracy {}'.format(*result))


if __name__ == '__main__':
    main()
