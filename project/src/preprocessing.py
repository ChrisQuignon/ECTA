#!/usr/bin/env python

from imp import load_source
from random import randrange, random
import numpy as np
from datetime import datetime, timedelta
import pylab
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
import csv

#import dataset
#equivalent to:
#import ../helpers/csvimport as helper
helper = load_source('dsimport', 'helpers/helper.py')

ds = helper.dsimport()

# ds = helper.stretch(ds)

df = pd.DataFrame(ds)
df.set_index(df.Date, inplace = True)
df.Energie.resample('1Min', fill_method="ffill")
df = df.resample('1Min')
df.Energie.resample('D')
# df.interpolate(inplace=True)
df.fillna(inplace=True, method='ffill')#we at first forwardfill
# df.fillna(inplace=True, method='bfill')#then do a backwards fill

df = df - df.min()
df = df / df.max()

def slice(ds, timedelta_input, timedelta_output, to_predict, stepwidth, input_sampling, output_sampling):
    """
    Slices a dataframe into inputs and outputframes and returns them along with their shape.
    The stepwidth of the sampling, the size of the frames and the sampling inside of the frames can be defined.
    """

    inputs = []
    outputs = []

    start_input_frame = ds.index[0]
    while start_input_frame +  timedelta_input + timedelta_output <= ds.index[-1]:

        end_input_frame = start_input_frame + timedelta_input
        end_output_frame = end_input_frame+timedelta_output

        input_frame = ds[start_input_frame:end_input_frame]
        output_frame = ds[end_input_frame:end_output_frame]

        input_frame = input_frame.resample(input_sampling)
        output_frame = output_frame.resample(output_sampling)

        for k in output_frame.keys():
            if k not in to_predict:
                del output_frame[k]

        inputs.append(input_frame.as_matrix().flatten())
        outputs.append(output_frame.as_matrix().flatten())

        #Move forward
        start_input_frame  = start_input_frame + stepwidth


    return (inputs, input_frame.shape), (outputs, output_frame.shape)

validation_delta = timedelta(days = 10)
timedelta_input = timedelta(hours = 24)
timedelta_output =  timedelta(hours = 24)
to_predict = ['Energie']#, 'Leistung']
input_sampling = '20Min'
output_sampling = '24H'
stepwidth = timedelta(minutes=20)

print 'Learning:'
print 'Input frame: ', timedelta_input
print "Input sampling: ", input_sampling
print 'Output frame: ', timedelta_output
print "Output sampling: ", output_sampling
print 'Stepwidth: ', stepwidth
print 'Features to learn: ', helper.translate(to_predict)

#CUTTING OF THE VALIDATION FRAME
last = df.index[-1] - validation_delta

train_frame = df[:last]
validation_frame = df[last:]

#SAMPLING THE FUNCTION
(inputs, input_shape), (outputs, output_shape) = slice(train_frame, timedelta_input, timedelta_output, to_predict, stepwidth, input_sampling, output_sampling)
(val_in, _), (val_out, _) = slice(validation_frame, timedelta_input, timedelta_output, to_predict, stepwidth, input_sampling, output_sampling)

inputs = np.asarray(inputs)
outputs = np.asarray(outputs)
val_in = np.asarray(val_in)
val_out = np.asarray(val_out)

val_out = val_out[:, 1]
outputs = outputs[:, 1]

np.savetxt('data/inputs_2024.txt', inputs)
np.savetxt('data/outputs_2024.txt', outputs)
np.savetxt('data/val_in_2024.txt', val_in)
np.savetxt('data/val_out_2024.txt', val_out)
