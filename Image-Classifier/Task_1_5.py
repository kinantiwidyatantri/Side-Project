import argparse
import csv
import os
import cv2
import numpy as np
from PIL import Image
from scipy import spatial
import distutils 
import Dummy


# Using the kNN approach and three distance or similarity measures, build image classifiers.

# This is the classification scheme you should use for kNN
classification_scheme = ['Female', 'Male', 'Primate', 'Rodent', 'Food']


# In this function, implement validation of the data that is supplied to or produced by the kNN classifier.
#
# INPUT:  data              : a list of lists that was read from the training data or data to classify csv
#                             (see parse_arguments function) or produced by the kNN function
#         predicted         : a boolean value stating whether the "PredictedClass" column should be present
#
# OUTPUT: boolean value     : True if the data contains the header ["Path", "ActualClass"] if predicted variable
#                             is False and ["Path", "ActualClass", "PredictedClass"] if it is True
#                             (there can be more column names, but at least these three at the start must be present)
#                             AND the values in the "Path" column (if there are any) are file paths
#                             AND the values in the "ActualClass" column (if there are any) are classes from scheme
#                             AND (if predicted is True) the values in the "PredictedClass" column (if there are any)
#                             are classes from scheme
#                             AND there are as many Path entries as ActualClass (and PredictedClass, if predicted
#                             is True) entries
#
#                             False otherwise

def validateDataFormat(data, predicted):
    formatCorrect = False
    result = []
   
    if predicted == False :
        for count, value in enumerate(data):
            if count > 0:
                imgPath = data[count][0]
                actualClass = data[count][1]
        
                if os.path.isfile(imgPath) and actualClass in classification_scheme and (imgPath != "" and actualClass != ""):
                    result.append(True)
                else:
                    result.append(False)
                    
                
    elif predicted == True :
        for count, value in enumerate(data):
            if count > 0:
                imgPath = data[count][0]
                actualClass = data[count][1]
                predictedClass = data[count][2]
                
                if os.path.isfile(imgPath) and actualClass in classification_scheme and predictedClass in classification_scheme and (imgPath != "" and actualClass != "" and predictedClass != ""):
                    result.append(True)
                else:
                    result.append(False)
                    
    if False in result:
        formatCorrect = False
    else:
        formatCorrect = True
                    
    return formatCorrect
        


# This function does reading and resizing of an image located in a give path on your drive.
#
# INPUT:  imagePath         : path to image. DO NOT MODIFY - take from the file as-is.
#         width, height     : dimensions to which you are asked to resize your image
#
# OUTPUT: image             : read and resized image, or empty list if the image is not found at a given path

def readAndResize(image_path, width=60, height=30):
    
    image = cv2.imread(image_path)
    
    if image is not None:
        image = cv2.resize(image, (width, height),interpolation= cv2.INTER_LINEAR)
        image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
    else:
        image = []
        
    return image



# These functions compute the distance or similarity value between two images according to a particular
# similarity or distance measure. Return nan if images are empty. These three measures must be
# computed by libraries according to portfolio requirements.
#
# INPUT:  image1, image2    : two images to compare
#
# OUTPUT: value             : the distance or similarity value between image1 and image2 according to a chosen approach.
#                             Defaults to nan if images are empty.
#

def computeMeasure1(image1, image2):
    # Euclidean distance
    
    if image1 == None and not image2 == None:
        value = float('nan')
        return value
        
    else:
        image1 = image1.convert('L')
        image2 = image2.convert('L')
        
        image1Array = np.asarray(image1)
        image2Array = np.asarray(image2)
        
        flat_image1Array = image1Array.flatten()
        flat_image2Array = image2Array.flatten()
        
        flat_image1Array = flat_image1Array/255
        flat_image2Array = flat_image2Array/255
        
        value = float(np.linalg.norm(flat_image1Array - flat_image2Array))
        
    return value


def computeMeasure2(image1, image2):
   #Jaccard similarity
    
    if not image1 and not image2:
        value = float('nan')
        return value
        
    else:    
        image1 = image1.convert('L')
        image2 = image2.convert('L')
            
        image1Array = np.asarray(image1)
        image2Array = np.asarray(image2)
            
        flat_image1Array = image1Array.flatten()
        flat_image2Array = image2Array.flatten()
        
        arrayIntersection = np.intersect1d(flat_image1Array, flat_image2Array)
        arrayUnion = np.union1d(flat_image1Array, flat_image2Array)
        value = len(arrayIntersection) / float(len(arrayUnion))
    
    return value



def computeMeasure3(image1, image2):
    # Cosine similarity
   
    if not image1 and not image2:
        value = float('nan')
        return value
       
    else:
        image1 = image1.convert('L')
        image2 = image2.convert('L')
            
        image1Array = np.asarray(image1)
        image2Array = np.asarray(image2)
            
        flat_image1Array = image1Array.flatten()
        flat_image2Array = image2Array.flatten()
        
        flat_image1Array = flat_image1Array/255
        flat_image2Array = flat_image2Array/255
        
        value = -1 * (spatial.distance.cosine(flat_image1Array, flat_image2Array) - 1)
        
        
    return value


# These functions compute the distance or similarity value between two images according to a particular similarity or
# distance measure. Return nan if images are empty. 
#
# INPUT:  image1, image2    : two images to compare
#
# OUTPUT: value             : the distance or similarity value between image1 and image2 according to a chosen approach.
#                             Defaults to nan if images are empty.
#

def selfComputeMeasure1(image1, image2):
    # Euclidean distance - correspond to the approach in computeMeasure1
   
    if not image1 and not image2:
        value = float('nan')
        return value
        
    else:
        image1 = image1.convert('L')
        image2 = image2.convert('L')
        
        image1Array = np.asarray(image1)
        image2Array = np.asarray(image2)
        
        flat_image1Array = image1Array.flatten()
        flat_image2Array = image2Array.flatten()
        
        flat_image1Array = flat_image1Array/255
        flat_image2Array = flat_image2Array/255
        
        value = np.sum(np.square(flat_image1Array - flat_image2Array))
        value = np.sqrt(value)
    
        
    return value


def selfComputeMeasure2(image1, image2):
    #Jaccard similarity - correspond to the approach in computeMeasure2
    
    if not image1 and not image2:
        value = float('nan')
        return value
       
    else:
        image1 = image1.convert('L')
        image2 = image2.convert('L')
            
        image1Array = np.asarray(image1)
        image2Array = np.asarray(image2)
            
        flat_image1Array = image1Array.flatten()
        flat_image2Array = image2Array.flatten()
        
    
        flat_image1Set = set(flat_image1Array)
        flat_image2Set = set(flat_image2Array)
        
    
        intersection = list(flat_image1Set.intersection(set(flat_image2Set)))
        union = flat_image1Set.union(flat_image2Set)
        value = (len(intersection)/len(union))

    return value


# This function is supposed to return a dictionary of classes and their occurrences as taken from k nearest neighbours.
#
# INPUT:  measure_classes   : a list of lists that contain two elements each - a distance/similarity value
#                             and class from scheme
#         k                 : the value of k neighbours
#         similarity_flag   : a boolean value stating that the measure used to produce the values above is a distance
#                             (False) or a similarity (True)
# OUTPUT: nearest_neighbours_classes
#                           : a dictionary that, for each class in the scheme, states how often this class
#                             was in the k nearest neighbours
#
def getClassesOfKNearestNeighbours(measures_classes, k, similarity_flag):
    nearest_neighbours_classes = {}
    
    if similarity_flag:
        measures_classes = sorted(measures_classes, key=lambda x: x[0], reverse=True)
    else:
        measures_classes = sorted(measures_classes, key=lambda x: x[0], reverse=False)
    
    names = np.array([item[1] for item in measures_classes[:k]])
    unique_names, counts = np.unique(names, return_counts=True)
    
    for name, count in zip(unique_names, counts):
        nearest_neighbours_classes[name] = count
    
    return nearest_neighbours_classes

# Given a dictionary of classes and their occurrences, returns the most common class. In case there are multiple
# candidates, it follows the order of classes in the scheme. The function returns empty string if the input dictionary
# is empty, does not contain any classes from the scheme, or if all classes in the scheme have occurrence of 0.
#
# INPUT: nearest_neighbours_classes
#                           : a dictionary that, for each class in the scheme, states how often this class
#                             was in the k nearest neighbours
#
# OUTPUT: winner            : the most common class from the classification scheme. In case there are
#                             multiple candidates, it follows the order of classes in the scheme. Returns empty string
#                             if the input dictionary is empty, does not contain any classes from the scheme,
#                             or if all classes in the scheme have occurrence of 0
#
def getMostCommonClass(nearest_neighbours_classes):
    winner = '' 

    if not nearest_neighbours_classes or all(value == 0 for value in nearest_neighbours_classes.values()) or all(key not in classification_scheme for key in nearest_neighbours_classes.keys()):
        return winner
    
    else:
        max_value = max(nearest_neighbours_classes.values())
        common_class = []
        
        for k, v in nearest_neighbours_classes.items():
            if v == max_value:
                common_class.append(k)
                
        
        for a in common_class:
            if a not in classification_scheme:
                common_class.remove(a)
        
        
        if len(common_class) > 1:
            common_class = sorted(common_class,key=classification_scheme.index)
            
        winner = common_class[0]
    
    #print(winner)
    return winner


# Implement the kNN classifier. 
#
# INPUT:  training_data       : a list of lists that was read from the training data csv (see parse_arguments function)
#         k                   : the value of k neighbours
#         measure_func        : the function to be invoked to calculate similarity/distance (any of the above)
#         similarity_flag     : a boolean value stating that the measure above used to produce the values is a distance
#                             (False) or a similarity (True)
#         data_to_classify    : a list of lists that was read from the data to classify csv;
#                             this data is NOT be used for training the classifier, but for running and testing it
#                             (see parse_arguments function)
#     most_common_class_func  : the function to be invoked to find the most common class among the neighbours
#                             (by default, it is the one from above)
# get_neighbour_classes_func  : the function to be invoked to find the classes of nearest neighbours
#                             (by default, it is the one from above)
#         read_func           : the function to be invoked to find to read and resize images
#                             (by default, it is the one from above)
#  OUTPUT: classified_data    : a list of lists which expands the data_to_classify with the results on how your
#                             classifier has classified a given image. In case no classification can be performed due
#                             to absence of training_data or data_to_classify, it only contains the header list.
def kNN(training_data, k, measure_func, similarity_flag, data_to_classify,
        most_common_class_func=getMostCommonClass, get_neighbour_classes_func=getClassesOfKNearestNeighbours,
        read_func=readAndResize):
    # This sets the header list
    classified_data = [['Path', 'ActualClass', 'PredictedClass']]
    
    if not training_data or not data_to_classify:
        return classified_data
    
    else:
        for c, v in enumerate(data_to_classify):
                if c > 0:
                    path = data_to_classify[c][0]
                    actual_class = data_to_classify[c][1]
                    
                    actual_image = read_func(path) 
        
                    training = []
                    for c2, v2 in enumerate(training_data):
                        if c2 > 0:
                            path2 = training_data[c2][0]
                            class2 = training_data[c2][1]
                            
                            train_image = read_func(path2)
                        
                            result = measure_func(actual_image,train_image)
                            training.append([result , class2])
                    
                    neighbour = get_neighbour_classes_func(training, k, similarity_flag)
                
                    predicted_class = most_common_class_func(neighbour)
        
                    classified_data.append([path, actual_class, predicted_class])
        # Have fun!
    
    return classified_data


# This function reads the necessary arguments (see parse_arguments function), and based on them executes
# the kNN classifier. If the "unseen" mode is on, the results are written to a file.

def main():
    opts = parseArguments()
    if not opts:
        exit(1)
    print(f'Reading data from {opts["training_data"]} and {opts["data_to_classify"]}')
    training_data = readCSVFile(opts['training_data'])
    data_to_classify = readCSVFile(opts['data_to_classify'])
    unseen = opts['mode']
    print('Running kNN')
    result = kNN(training_data, opts['k'], eval(opts['measure']), opts['simflag'], data_to_classify,
                 eval(opts['mcc']), eval(opts['gnc']), eval(opts['rrf']))
    if unseen:
        path = os.path.dirname(os.path.realpath(opts['data_to_classify']))
        out = f'result_classified_data.csv'
        print(f'Writing data to {out}')
        writeCSVFile(out, result)


# Straightforward function to read the data contained in the file "filename"
def readCSVFile(filename):
    lines = []
    with open(filename, newline='') as infile:
        reader = csv.reader(infile)
        for line in reader:
            lines.append(line)
    return lines


# Straightforward function to write the data contained in "lines" to a file "filename"
def writeCSVFile(filename, lines):
    with open(filename, 'w', newline='') as outfile:
        writer = csv.writer(outfile)
        writer.writerows(lines)


# This function simply parses the arguments passed to main. It looks for the following:
#       -k              : the value of k neighbours
#                         (needed in Tasks 1, 2, 3 and 5)
#       -f              : the number of folds to be used for cross-validation
#                         (needed in Task 3)
#       -measure        : function to compute a given similarity/distance measure
#       -simflag        : flag telling us whether the above measure is a distance (False) or similarity (True)
#       -u              : flag for how to understand the data. If -u is used, it means data is "unseen" and
#                         the classification will be written to the file. If -u is not used, it means the data is
#                         for training purposes and no writing to files will happen.
#                         (needed in Tasks 1, 3 and 5)
#       training_data   : csv file to be used for training the classifier, contains two columns: "Path" that denotes
#                         the path to a given image file, and "Class" that gives the true class of the image
#                         according to the classification scheme defined at the start of this file.
#                         (needed in Tasks 1, 2, 3 and 5)
#       data_to_classify: csv file formatted the same way as training_data; it will NOT be used for training
#                         the classifier, but for running and testing it
#                         (needed in Tasks 1, 2, 3 and 5)
#       mcc, gnc, rrf, vf,cf,sf,al
#                       : staff variables, do not use
#
def parseArguments():
    parser = argparse.ArgumentParser(description='Processes files ')
    parser.add_argument('-k', type=int)
    parser.add_argument('-f', type=int)
    parser.add_argument('-m', '--measure')
    parser.add_argument('-s', '--simflag', type=lambda x:bool(distutils.util.strtobool(x)))
    parser.add_argument('-u', '--unseen', action='store_true')
    parser.add_argument('-train', type=str)
    parser.add_argument('-test', type=str)
    parser.add_argument('-classified', type=str)
    parser.add_argument('-mcc', default="getMostCommonClass")
    parser.add_argument('-gnc', default="getClassesOfKNearestNeighbours")
    parser.add_argument('-rrf', default="readAndResize")
    parser.add_argument('-cf', default="confusionMatrix")
    parser.add_argument('-sf', default="splitDataForCrossValidation")
    parser.add_argument('-al', default="Task_1_5.kNN")
    params = parser.parse_args()

    opt = {'k': params.k,
           'f': params.f,
           'measure': params.measure,
           'simflag': params.simflag,
           'training_data': params.train,
           'data_to_classify': params.test,
           'classified_data': params.classified,
           'mode': params.unseen,
           'mcc': params.mcc,
           'gnc': params.gnc,
           'rrf': params.rrf,
           'cf': params.cf,
           'sf': params.sf,
           'al': params.al
           }
    return opt


if __name__ == '__main__':
    main()
