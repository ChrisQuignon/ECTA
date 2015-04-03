#!/usr/bin/env python
import sys
import os
import time
import csv
import pickle
import numpy as np
import logging

def read_file(thefile):
    with open(thefile, 'rb') as file_handle:
        try:
            return np.load(file_handle)
        except IOError:
            logging.info("IO error with file: >" + thefile + "< but continuing")
            return False

def write_file(thefile, thedata):
    with open(thefile, 'wb') as file_handle:
        thedata.dump(file_handle)

def append_file(thefile, thedata):
    olddata = read_file(thefile)
    with open(thefile, 'wb') as file_handle:
        if olddata is False:
            thedata.dump(file_handle)
        else:
            #TODO: check if axis = 0 is correct
            #axis = 1 does not flattens the data
            olddata = np.concatenate((olddata, thedata), axis=1)
            olddata.dump(file_handle)

def empty_file(thefile):
    open(thefile, 'wb').close()
