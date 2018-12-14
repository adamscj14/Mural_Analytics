#!/usr/bin/Python

# Created by: Christopher J Adams 8/19/2018
# 

###############################################################################
###
### This script will place the murals into their respective blocks
###
###############################################################################

#import cProfile
import sys
import getopt
import pandas as pd
import json
import csv
from shapely.geometry import Polygon
from shapely.geometry import Point

def help():
    print """-----------------------------------------------------------------
ARGUMENTS
    -m => <json> Holds the mural location data REQUIRED
    -b => <csv> Holds the block data REQUIRED
    -o => <Output File name/location> (Specifies the absolute desired location

"""
###############################################################################
###########################  COLLECT AND CHECK ARGUMENTS  #####################
###############################################################################

    
## MAIN ##

def main(argv): 
    try: 
        opts, args = getopt.getopt(sys.argv[1:], "m:b:o:")
    except getopt.GetoptError:
        print "Error: Incorrect usage of getopts flags!"
        help() 
        sys.exit()

    mural_file = False
    output_file = False
    block_file = False

    for opt, arg in opts:
        print opt,arg

        # Flag for count data location. 
        if opt == "-m":
            mural_file = arg

        # Flag for desired location and name of output file.
        elif opt == "-o":
            output_file = arg

        elif opt == "-b":
            block_file = arg

        else:
            print "Error: Incorrect usage of getopts flags!"
            help()
            sys.exit()

    check_arguments(mural_file, block_file, output_file)

    print "Acceptable Inputs Given"

    driver(mural_file, block_file, output_file)


## Makes sure that all the arguments given are congruent with one another.
## ONE-TIME CALL -- called by main

def check_arguments(mural_file, block_file, output_file):
   
   # Ensure that both the required inputs exist
    if not (mural_file and block_file and output_file):
        print ('Error: At least one of the required inputs does not exist')
        help()
        sys.exit()


###############################################################################
#############################  DRIVER  ########################################
###############################################################################


## drive the script ##
## ONE-TIME CALL -- called by main

def driver(mural_file, block_file, output_file):

    block_mural_dict = {}

    block_csv = open(block_file, 'r')
    reader = csv.DictReader(block_csv, fieldnames=["index", "block", "blockgroup","censustract", "area", "geometry"])
    block_polygon_dict = store_block_polygons(reader)

    mural_dict_list = read_and_interpret_mural_json(mural_file)
    for mural_dict in mural_dict_list:
        lon = mural_dict['lng']
        lat = mural_dict['lat']
        for block_index in block_polygon_dict:
            if block_index not in block_mural_dict:
                block_mural_dict[block_index] = 0
            if determine_presence(block_polygon_dict[block_index], lon, lat):
                block_mural_dict[block_index] += 1



    #print block_mural_dict
    df = pd.DataFrame.from_dict(block_mural_dict, orient='index')
    #print df
    df.to_csv(output_file, sep='\t', header = ['mural_count'])


def store_block_polygons(reader):

    polygon_dict = {}

    for block in reader:
        coords = block['geometry']
        coords_lists = []
        if "c" in coords:
            sub_lists = coords.strip().split('), c(')
            for sub in sub_lists:
                sub = sub.replace(")", "")
                coords_lists.append(sub.strip().split(', '))

        else:
            coords_lists = [coords.strip().split(',')]

        for l in coords_lists:
            lon_coords = l[0:(len(l)/2)]
            lat_coords = l[(len(l)/2):]
            tup_list = []

            for ind in range(len(lon_coords)):
                tup = (float(lat_coords[ind]), float(lon_coords[ind]))
                tup_list.append(tup)

            polygon = Polygon(tup_list)
            try:
                polygon_dict[block['geometry']].append(polygon)
            except KeyError:
                polygon_dict[block['geometry']] = [polygon]

    return polygon_dict


def determine_presence(polygon_list, lon, lat):

    tup = (float(lat), float(lon))
    for p in polygon_list:
        if p.contains(Point(tup)):
            return True

def read_and_interpret_mural_json(mural_file):

    mural_json = open(mural_file, 'r')
    mural_dict = json.load(mural_json)
    return mural_dict['markers']


########################################################################################################################
#cProfile.run('main(sys.argv[1:])')
         
if __name__ == "__main__":
    
    main(sys.argv[1:])
