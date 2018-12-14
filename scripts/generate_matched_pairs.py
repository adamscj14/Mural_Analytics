#!/usr/bin/Python

# Created by: Christopher J Adams 12/14/2018
# 

###############################################################################
###
### This script will match block groups given their propensity scores
###
###############################################################################

#import cProfile
import sys
import getopt
import pandas as pd
import json
import csv

def help():
    print """-----------------------------------------------------------------
ARGUMENTS
    -f => <csv> Holds the full dataframe REQUIRED
    -p => <csv> Holds the "present" propensity file REQUIRED
    -a => <csv> Holds the "absent" propensity file REQUIRED
    -o => <Output File name/location> (Specifies the absolute desired location

"""
###############################################################################
###########################  COLLECT AND CHECK ARGUMENTS  #####################
###############################################################################

    
## MAIN ##

def main(argv): 
    try: 
        opts, args = getopt.getopt(sys.argv[1:], "f:p:a:o:")
    except getopt.GetoptError:
        print "Error: Incorrect usage of getopts flags!"
        help() 
        sys.exit()

    full_file = False
    present_file = False
    absent_file = False
    output_file = False

    for opt, arg in opts:
        print opt,arg

        if opt == "-f":
            full_file = arg

        elif opt == "-p":
            present_file = arg

        elif opt == "-a":
            absent_file = arg

        elif opt == "-o":
            output_file = arg

        else:
            print "Error: Incorrect usage of getopts flags!"
            help()
            sys.exit()

    check_arguments(full_file, present_file, absent_file, output_file)

    print "Acceptable Inputs Given"

    driver(full_file, present_file, absent_file, output_file)


## Makes sure that all the arguments given are congruent with one another.
## ONE-TIME CALL -- called by main

def check_arguments(full_file, present_file, absent_file, output_file):
   
   # Ensure that both the required inputs exist
    if not (full_file and present_file and absent_file and output_file):
        print ('Error: At least one of the required inputs does not exist')
        help()
        sys.exit()


###############################################################################
#############################  DRIVER  ########################################
###############################################################################


## drive the script ##
## ONE-TIME CALL -- called by main

def driver(full_file, present_file, absent_file, output_file):

    present_csv = open(present_file, 'r')
    present_reader = csv.DictReader(present_csv, fieldnames=["index", "blockgroup_id", "prop_score"])
    
    absent_csv = open(absent_file)
    absent_reader = csv.DictReader(absent_csv, fieldnames=["index", "blockgroup_id", "prop_score"])
    
    for 

########################################################################################################################
#cProfile.run('main(sys.argv[1:])')
         
if __name__ == "__main__":
    
    main(sys.argv[1:])
