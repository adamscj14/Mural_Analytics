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
from sklearn.feature_selection import chi2


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

    predictor_vars = list(input_df.columns.values)
    predictor_vars.remove('blockgroupID')
    predictor_vars.remove('mural_presence')

    #predictor_vars = ['comresprop', "blackprop", 'vacantprop']

    # perform simple linear regressions
    #perform_simple_linear_regressions(input_df, predictor_vars)

    # perform simple logistic regressions
    #perform_simple_log_regs(input_df, predictor_vars)

    # perform multiple logistic regressions
    perform_multiple_log_regs(input_df, predictor_vars)


def perform_multiple_log_regs(input_df, predictor_vars):

    confusion_dict = {'00': 0, '01': 1, '10': 2, '11': 3}

    confusion_matrix = [0, 0, 0, 0]
    master_df_columns = list(predictor_vars)
    master_df_columns.append("mural_presence")
    master_df = input_df[master_df_columns]

    # remove nan values
    master_df = master_df.dropna(0)

    #logreg = LogisticRegression(fit_intercept=True, solver='lbfgs', tol=0.0000000001, verbose=True)
    logreg = LogisticRegression()
    Y_train = np.asarray(master_df['mural_presence'])
    X_train = np.asarray(master_df[predictor_vars])
    print master_df[predictor_vars]
    print master_df['mural_presence']

    logreg_fit = logreg.fit(X_train, Y_train)

    prediction = logreg_fit.predict(X_train)

    for pos in range(len(Y_train)):
        dict_key = "{}{}".format(int(prediction[pos]), int(Y_train[pos]))

        confusion_matrix[confusion_dict[dict_key]] += 1

    train_err_rate = 1 - logreg_fit.score(X_train, Y_train)

    print(predictor_vars)
    print("Confusion Matrix Train:\nInferred x Truth\n    0     1")

    print("0 | {}    {}\n1 | {}    {}".format(confusion_matrix[0], confusion_matrix[1],
                                              confusion_matrix[2], confusion_matrix[3]))

    print("Train Error: {}\n--------------------------------------------".format(train_err_rate))


def perform_simple_log_regs(input_df, predictor_vars):

    confusion_dict = {'00': 0, '01': 1, '10': 2, '11': 3}

    error_dict = {}
    confusion_mat_dict = {}

    for pred in predictor_vars:
        error_dict[pred] = 0
        confusion_mat_dict[pred] = 0
        confusion_matrix = [0, 0, 0, 0]

        pred_df = pd.concat([input_df[pred],input_df['mural_presence']], axis = 1, keys = ['pred', 'category'])
        # remove nan values
        pred_df = pred_df.dropna(0)

        logreg = LogisticRegression(fit_intercept=True, solver='lbfgs', tol=0.0000000001, verbose=True, C=np.inf)
        Y_train = np.asarray(pred_df['category'])
        X_train = np.asarray(pred_df['pred']).reshape(-1, 1)

        logreg_fit = logreg.fit(X_train, Y_train)

        prediction = logreg_fit.predict(X_train)

        for pos in range(len(Y_train)):
            dict_key = "{}{}".format(int(prediction[pos]), int(Y_train[pos]))

            confusion_matrix[confusion_dict[dict_key]] += 1

        train_err_rate = 1 - logreg_fit.score(X_train, Y_train)
        error_dict[pred] += float(train_err_rate)

        confusion_mat_dict[pred] = confusion_matrix

    for pred in predictor_vars:
         pred_tr_confusion_matrix = confusion_mat_dict[pred]

         print(pred)
         print("Confusion Matrix Train:\nInferred x Truth\n    0     1")

         print("0 | {}    {}\n1 | {}    {}".format(pred_tr_confusion_matrix[0], pred_tr_confusion_matrix[1],
                                                   pred_tr_confusion_matrix[2], pred_tr_confusion_matrix[3]))

         print("Train Error: {}\n--------------------------------------------".format(error_dict[pred]))


def perform_simple_linear_regressions(input_df, predictor_vars):

    for pred in predictor_vars: #predictor_vars:

        pred_df = pd.concat([input_df[pred],input_df['mural_presence']], axis = 1, keys = ['pred', 'category'])

        # remove nan values
        pred_df = pred_df.dropna(0)

        ## just for reference linear regression
        lin_reg = LinearRegression()
        lin_reg.fit(np.asarray(pred_df['pred']).reshape(-1, 1), np.asarray(pred_df['category']))
        #print(pred)
        print(pred)
        print("Linear Regression R^2",lin_reg.score(np.asarray(pred_df['pred']).reshape(-1, 1), np.asarray(pred_df['category'])))

        scores, pvalues = chi2(np.asarray(pred_df['pred']).reshape(-1, 1), np.asarray(pred_df['category']))
        print("Score: ",scores)
        print("Pvalue: ", pvalues)


if __name__ == '__main__':
    main(sys.argv[1:])
