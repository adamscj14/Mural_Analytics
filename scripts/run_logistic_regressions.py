#!/usr/bin/env python

## Created: 10/29/18

## This script will run my logistic regression analyses


import pandas as pd
import numpy as np
import sys
import os
import getopt
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


def help():
    print("""-----------------------------------------------------------------
ARGUMENTS
    -d => <csv> input logistic regression data file REQUIRED
    -o => <out name> Specifies output location and name REQUIRED
""")

###############################################################################

## MAIN 

def main(argv): 
    try: 
        opts, args = getopt.getopt(sys.argv[1:], "d:o:")
    except getopt.GetoptError:
        print("Error: Incorrect usage of getopts flags!")
        help() 
        sys.exit()

    data_file = False
    output_file = False

    for opt, arg in opts:
        print(opt,arg)

        if opt == "-d":
            data_file = arg

        elif opt == "-o":
            output_file = arg

        else:
            print("Error: Incorrect usage of getopts flags!")
            help()
            sys.exit()

    if not (data_file and output_file):
        print("Error: Missing required inputs.")
        help()
        sys.exit()

    print("Acceptable Inputs Given")

    driver(data_file, output_file)


def driver(data_file, output_file):

    input_df = pd.read_table(data_file, sep = ',')

    # perform simple logistic regressions
    perform_simple_log_regs(input_df)




def perform_simple_log_regs(input_df):

    predictor_vars = list(input_df.columns.values)
    predictor_vars.remove('blockgroupID')
    predictor_vars.remove('mural_presence')

    confusion_dict = {'00': 0, '01': 1, '10': 2, '11': 3}

    error_dict = {}
    confusion_mat_dict = {}
    iterations = 100
    for pred in predictor_vars:
        error_dict[pred] = [0,0]
        confusion_mat_dict[pred] = [0,0]
        #print(pred)
        category_label = np.asarray(input_df['mural_presence'])
        pred_data = np.asarray(input_df[pred])

        pred_dict = {'pred': pred_data, 'category': category_label}
        pred_df = pd.DataFrame(data=pred_dict)

        # remove nan values
        pred_df = pred_df.dropna(0)

        ## just for reference linear regression
        lin_reg = LinearRegression()
        lin_reg.fit(np.asarray(pred_df['pred']).reshape(-1, 1), np.asarray(pred_df['category']))
        print(pred)
        print(lin_reg.score(np.asarray(pred_df['pred']).reshape(-1, 1), np.asarray(pred_df['category'])))

        confusion_tr_matrix = [0,0,0,0]
        confusion_test_matrix = [0, 0, 0, 0]

        for i in range(iterations):
            [X_train, X_test, Y_train, Y_test] = train_test_split(np.asarray(pred_df['pred']).reshape(-1, 1), np.asarray(pred_df['category']), test_size = 0.1)
            #print len(X_test), len(Y_test)

            logreg_fit = LogisticRegression(solver='sag').fit(X_train, Y_train)
            
            #train_error
            train_pred = logreg_fit.predict(X_train)
            #print sum(train_pred)
            for pos in range(len(Y_train)):
                dict_key = "{}{}".format(train_pred[pos], Y_train[pos])

                confusion_tr_matrix[confusion_dict[dict_key]] += 1


            total_tr_errs = float(sum(abs(train_pred-Y_train)))
            total_tr_data_points = len(Y_train)
            train_err_rate = total_tr_errs/total_tr_data_points

            error_dict[pred][0] += train_err_rate
            '''
            print("total_test_errs: {}\ntotal_test_datapoints: {}\nerr_rate: {}\n".format(total_tr_errs,
                                                                                          total_tr_data_points,
                                                                                          train_err_rate))
            '''
            #test error

            test_pred = logreg_fit.predict(X_test)

            for pos in range(len(Y_test)):
                dict_key = "{}{}".format(test_pred[pos], Y_test[pos])

                confusion_test_matrix[confusion_dict[dict_key]] += 1


            total_test_errs = float(sum(abs(test_pred-Y_test)))
            total_test_data_points = len(Y_test)
            test_err_rate = total_test_errs/total_test_data_points

            error_dict[pred][1] += test_err_rate

            '''
            print("total_test_errs: {}\ntotal_test_datapoints: {}\nerr_rate: {}\n-------------------------------".format(
                                                                                                total_test_errs,
                                                                                                total_test_data_points,
                                                                                                test_err_rate))
            '''
        avg_tr_confusion_mat = [x/iterations for x in confusion_tr_matrix]
        avg_test_confusion_mat = [x / iterations for x in confusion_test_matrix]
        confusion_mat_dict[pred][0] = avg_tr_confusion_mat
        confusion_mat_dict[pred][0] = avg_test_confusion_mat


    for pred in error_dict:
        pred_tr_confusion_matrix = confusion_mat_dict[pred][0]
        pred_test_confusion_matrix = confusion_mat_dict[pred][0]

        print(pred)
        print("Confusion Matrix Train:\nInferred x Truth\n    0     1")

        print("0 | {}    {}\n1 | {}    {}".format(pred_tr_confusion_matrix[0], pred_tr_confusion_matrix[1],
                                                  pred_tr_confusion_matrix[2], pred_tr_confusion_matrix[3]))
        print("Confusion Matrix Test:\nInferred x Truth\n    0     1")
        print("0 | {}    {}\n1 | {}    {}".format(pred_test_confusion_matrix[0], pred_test_confusion_matrix[1],
                                                  pred_test_confusion_matrix[2], pred_test_confusion_matrix[3]))
        print("Train Error: {}\nTest Error: {}\n--------------------------------------------".format(error_dict[pred][0] / float(iterations),
                                                                       error_dict[pred][1] / float(iterations)))

if __name__ == '__main__':
    main(sys.argv[1:])
