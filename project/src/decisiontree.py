#!/usr/bin/env python

import numpy as np
import pylab
import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_csv('data/ds1_weather.csv', decimal=',',sep=';', parse_dates=True, index_col=[0])

df = df.resample('1Min')
df.interpolate(inplace=True)
df.fillna(inplace=True, method='ffill')#we at first forwardfill
df.fillna(inplace=True, method='bfill')#then do a backwards fill

class decisionLeaf():

    def __init__(self, val):
        self.val = val
        self.mse = float('inf')

    def predict(self, predictionset):
        return [self.val for _ in predictionset.index]


    def update_mse(self,  input_df, outputset):
        result = self.predict(input_df)

        result = np.array(result)
        output = np.array(outputset)

        self.mse = ((result - output) ** 2).mean()
        return self.mse




class decisionTree():
    def __init__(self, feature, split, left_child, right_child):
            self.feature = feature
            self.split = split

            self.mse = float('inf')

            self.right_child = []
            self.set_right_child(right_child)

            self.left_child = []
            self.set_left_child(left_child)


    def set_left_child(self, left_child):
        if isinstance(left_child, float):
            self.left_child = decisionLeaf(left_child)
        else:
            self.left_child = left_child


    def set_right_child(self, right_child):
        if isinstance(right_child, float):
            self.right_child = decisionLeaf(right_child)
        else:
            self.right_child = right_child


    def update_mse(self, input_df, outputset):

        left = pd.DataFrame()
        right = pd.DataFrame()
        leftout = []
        rightout = []

        #split the dataset
        for idx  in input_df.index:
            if input_df[self.feature][idx] < self.split:
                leftout.append(outputset[idx])
                left = left.append(input_df.ix[idx])
            else:
                rightout.append(outputset[idx])
                right = right.append(input_df.ix[idx])

        #updatemse is reculrsive
        self.left_child.update_mse(leftout)
        self.right_child.update_mse(rightout)

        output = []
        output.extend(leftout)
        output.extend(rightout)

        result = self.predict(input_df)

        result = np.array(result)
        output = np.array(output)

        self.mse = ((result - output) ** 2).mean()

        return self.mse


    def predict(self, predictionset):
        left = pd.DataFrame()
        right = pd.DataFrame()

        #split the dataset
        for idx  in predictionset.index:
            if predictionset[self.feature][idx] < self.split:
                left = left.append(predictionset.ix[idx])
            else:
                right = right.append(predictionset.ix[idx])


        left_val = self.left_child.predict(left)
        right_val = self.right_child.predict(right)

        result = []
        result.extend(left_val)
        result.extend(right_val)

        return result



#TESTING

dT = decisionTree('Aussentemperatur', 7, 12.5, 56.5)
dN = decisionTree('Vorlauftemperatur', 13, dT, 57.0)
dN.update_mse(df[0:100], df.Energie[0:100])
dN.predict(df[10:20])
