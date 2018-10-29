#!/usr/bin/env python

## Created: 10/29/18

## This script will run my logistic regression analyses


import pandas as pd
import sys
import os
import getopt
import json


def help():
    print """-----------------------------------------------------------------
ARGUMENTS
    -d => <csv> input logistic regression data file REQUIRED
    -o => <out name> Specifies output location and name REQUIRED
"""

###############################################################################

## MAIN 

def main(argv): 
    try: 
        opts, args = getopt.getopt(sys.argv[1:], "d:o:")
    except getopt.GetoptError:
        print "Error: Incorrect usage of getopts flags!"
        help() 
        sys.exit()

    data_file = False
    output_file = False

    for opt, arg in opts:
        print opt,arg

        if opt == "-d":
            rate_file = arg

        elif opt == "-o":
            output_file = arg

        else:
            print "Error: Incorrect usage of getopts flags!"
            help()
            sys.exit()

    if not (data_file and output_file):
        print "Error: Missing required inputs."
        help()
        sys.exit()

    print "Acceptable Inputs Given"

    driver(data_file, output_file)


def driver(data_file, output_file):
    pass

if __name__ == '__main__':
    main(sys.argv[1:])
