Requirements and dependencies:
    1. Numpy
    2. re  (regular expression)
    3. Math
    4.matplotlib.pyplot
    5.os
    6.random

Loading train and test data:

I have copied the data from web generated php files into text files (1b_training_data.txt,1c_test_data.txt, data_2c2d3c3d_program.txt, data_test_2a3a, data_train_2a3a)
I use custom functions (Load_data, Load_data_2, Load_data_logisticreg_train, Load_data_logisticreg_test) to load and convert .txt files to numpy arrays
Please place input .txt files (1b_training_data.txt,1c_test_data.txt, data_2c2d3c3d_program.txt)  in the current working directory


Inorder to use custom data, copy the data to a text file and use Load_data* functions with appropriate paths to .txt files.

All supporting functions are mainly located in utilities.py
