"""
Thomas Merkh, tmerkh@g.ucla.edu, April 2020

This script creates matlab matrices which contain the learned representations of COVID-Net on several sources of X-ray data.
The data used here is exactly the data set outline in COVID-Net, see https://github.com/lindawangg/COVID-Net.
The COVID-Net model(s) in use here are the pretrained models supplied on the aforementioned Github.  
This script saves both a csv file and a .mat file* containing the representations data
*Note that if the .mat file is too large, a least a couple of Gbs, Python refuses to save the .mat file and will only create a csv. 

To run this file, clone the above github, place this file within the master directory, generate the dataset, and then run this program:
python Extract_COVIDnet_Features.py --weightspath <PATH_TO_MODEL_CHKPTS> --metaname model.meta_train --ckptname <FILE_NAME_MODEL> --imagepath assets/ex-covid.jpeg
For example:
python Extract_COVIDnet_Features.py --weightspath COVID-Net-Large --metaname model.meta_train --ckptname model-2069 --imagepath assets/ex-covid.jpeg

Note that several python modules will need to be installed, including but not limited to tensorflow (using 1.15), opencv, scikit-learn.
"""

from sklearn.metrics import confusion_matrix
import numpy as np
import tensorflow as tf
from tensorflow import keras
import os, argparse, sys
import cv2
import scipy.io
import csv

if __name__ == '__main__':
    """
    User adjustable parameters:

    Save_loc (string) = The path to the location at which the feature matix should be saved. 
    Save_file (string) = The name of the file (excluding suffix) to be saved containing the network's representations
    print_tensors (bool) = True if one wishes for the hundreds of tensors within the graph to be printed to the console
    print_operations (bool) = True if one wishes for the thousands of operations to be printed
    training_features (bool) = True if one wishes to extract the representations of the training data set
    test_features (bool) = True if one wishes to extract the representation of the test data set.  This will take precedence over training_features if both true.
    single_image (bool) = True if one wishes to output the representation of a single X-Ray image specified as a runtime argument.
    
    Several lines down (around line 90), there are a couple of bools for setting which representation size (100352,1024,256) the user wants.
    These can also be switched, though it is likely that users will be most interested in the 256 or 1024 size representations. 
    """
    Save_loc = "/media/tmerkh/a18f683c-46a6-4846-aa3c-f47e5cfb8171"  # os.getcwd()
    Save_file = "representations_train_256"
    print_tensors = False
    print_operations = False
    training_features = True
    test_features = False
    single_image = False

    ####################################### User adjustable parameters done #################################################

    parser = argparse.ArgumentParser(description='COVID-Net Feature Extraction')
    parser.add_argument('--weightspath', default='output', type=str, help='Path to output folder')
    parser.add_argument('--metaname', default='model.meta', type=str, help='Name of ckpt meta file')
    parser.add_argument('--ckptname', default='model', type=str, help='Name of model ckpts')
    parser.add_argument('--testfile', default='test_COVIDx.txt', type=str, help='Name of testfile')
    parser.add_argument('--trainfile', default='train_COVIDx.txt', type=str, help='Name of train file')
    parser.add_argument('--testfolder', default='test', type=str, help='Folder where test data is located')
    parser.add_argument('--trainfolder', default='train', type=str, help='Folder where training data is located')
    parser.add_argument('--imagepath', default='assets/ex-covid.jpeg', type=str, help='Full path to image to be inferenced')
    parser.add_argument('--datadir', default='data', type=str, help='Path to data folder')

    args = parser.parse_args()

    sess = tf.Session()
    tf.get_default_graph()
    saver = tf.train.import_meta_graph(os.path.join(args.weightspath, args.metaname))
    saver.restore(sess, os.path.join(args.weightspath, args.ckptname))
    graph = tf.get_default_graph()

    with open(args.trainfile) as f:
        trainfile = f.readlines()
    with open(args.testfile) as f:
        testfile = f.readlines()

    if(print_tensors):
        # Gets all tensors in this graph
        print("The tensors in this graph are as follows...")
        trainable = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        for item in trainable:
            print(item)

    if(print_operations):
        ## Gets all operations in this graph
        print("The operations in this graph are as follows...")
        all_names = [op.name for op in graph.get_operations()]
        for name in all_names:
            print(name)


    #This network produces 2024 7x7 patches at the final layer which gets flattened into 100352 1D vector.
    representation_1 = False  # The 100352 dense layer pre-activation
    representation_2 = False  # The 1024 dense layer pre-activation
    representation_3 = True   # The 256 dense layer per-activation

    image_tensor = graph.get_tensor_by_name("input_1:0")

    if(representation_1):
        representation_tensor = graph.get_tensor_by_name("flatten_1/Reshape:0") # pre-activation for 100352 dense layer
        SZ = 100352
    elif(representation_2):
        representation_tensor = graph.get_tensor_by_name("dense_1/Relu:0") # pre-activation for 1024 dense layer
        SZ = 1024
    elif(representation_3):
        representation_tensor = graph.get_tensor_by_name("dense_2/Relu:0") # pre-activation for 256 dense layer
        SZ = 256
    
    ## testfile or trainfile can be used
    ## trainfile is so large that the process will get killed if not enough memory is available (I think its 7-11Gbs)
    if(test_features):
        file = testfile
        folder = args.testfolder
    elif(training_features):
        file = trainfile
        folder = args.trainfolder
    if(test_features or training_features):
        covidnet_reps = np.zeros((len(file),SZ))
    
    
    # For generating the representations learned on either the test set or training set
    if(test_features or training_features):
        print("Extracting the representations of the data set, and saving them as " + Save_file + " at location " + Save_loc)
        for i in range(len(file)):
            line = file[i].split()
            x = cv2.imread(os.path.join('data', folder, line[1]))
            x = cv2.resize(x, (224, 224))
            x = x.astype('float32') / 255.0
            output = sess.run(representation_tensor, feed_dict={image_tensor: np.expand_dims(x, axis=0)})
            covidnet_reps[i][:] = output[0]
            if(i % 100 == 0):
                print("Completed " + str(i) + " out of " + str(len(file)) + " ...")

    ## For a single input specified with --imagepath
    if(single_image):
        x = cv2.imread(args.imagepath)
        x = cv2.resize(x, (224, 224))
        x = x.astype('float32') / 255.0
        output = sess.run(representation_tensor, feed_dict={image_tensor: np.expand_dims(x, axis=0)})
        print("The image at " + str(args.imagepath) + " has representation:")
        np.set_printoptions(threshold=sys.maxsize)
        print(output[0])
        
    if(test_features or training_features):
        print("Saving Representations to " + Save_loc)
        os.chdir(Save_loc)
    
        ## This does not work if the matrix to be saved is several Gbs (as it is for the training set representations).        
        with open(Save_file + ".csv", 'w', newline='') as filen:
            writer = csv.writer(filen)
            for i in range(len(file)):
                writer.writerow(list(covidnet_reps[i][:]))
        ## This command will simply terminate itself if that is the case.  A csv file will still be saved.
        scipy.io.savemat(Save_file + ".mat", {Save_file : covidnet_reps})
    

    print("Program finished")
    print("For tips on loading in the saved files, please see the comments at the end of the code")
    print("\n")

    """
    This saves the representations as a csv file.
    In python, this can be loaded in by the script:

    filepath = os.getcwd() + "CSV_FILE_NAME_HERE"
    with open(filepath) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter = ',')  # Reads in each row of the csv file as a list
        line_count = 0
        for row in csv_reader:
            line_count += 1
            if(line_count == 1):
                print("The size of each row, or the dimension of the representation, is: " + str(len(row)))
        print("The line count, or number of data points, is: " + str(line_count))

    In Octave (and MATLAB), this file may be loaded in by the command:
    
    csvread(CSV_FILE_NAME_HERE)


    Potential Problems:
        When trying to save/load a .mat or .csv file, scipy.io and Octave attempt to hold the entire matrix in RAM, at least while loading or saving.
        This causes the program to crash if the machine decides that there isn't sufficient RAM for the size of the matrix, a problem that occurs with the largest representations 16546 by 100352.
        Custom read-in and save-to scripts need to be written for this instance. 
    """