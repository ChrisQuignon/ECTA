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

    def predict(self, prediction_set):
        mse = float('inf')
        return mse, [self.val for _ in prediction_set]




class decisionTree():
    def __init__(self, feature, split, left_child, right_child):
            self.feature = feature
            self.split = split

            self.right_child = []
            self.set_right_child(right_child)
            self.rmse = float('inf')

            self.left_child = []
            self.set_left_child(left_child)
            self.lmse = float('inf')

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

    def train(self, input_df, outputset):
        left = []
        leftout = []
        right = []
        rightout = []

        #split the dataset
        for idx  in input_df.index:
            if input_df[self.feature][idx] < self.split:
                left.append(input_df.ix[idx])
                leftout.append(outputset[idx])
            else:
                right.append(input_df.ix[idx])
                rightout.append(outputset[idx])

        _, left_result = self.left_child.predict(left)
        _, right_result = self.right_child.predict(right)

        left_result = np.asarray(left_result)
        right_result = np.asarray(right_result)
        leftout = np.array(leftout)
        rightout = np.array(rightout)

        lmse = ((left_result - leftout) ** 2).mean()
        rmse = ((right_result - rightout) ** 2).mean()

        self.lmse = min(self.lmse, lmse)
        self.rmse = min(self.rmse, rmse)


    def predict(self, predictionset):
        left = []
        right = []

        #split the dataset
        for idx  in predictionset.index:
            if predictionset[self.feature][idx] < self.split:
                left.append(predictionset.ix[idx])
            else:
                right.append(predictionset.ix[idx])

        lmse, left_val = self.left_child.predict(left)
        rmse, right_val = self.right_child.predict(right)

        if lmse > rmse:
            # TODO check whether plus is a good operator
            return (self.lmse + lmse, left_val)
        else:
            return (self.rmse + rmse, left_val)


#TESTING

dT = decisionTree('Aussentemperatur', 12, 12.5, 11.5)
dN = decisionTree('Vorlauftemperatur', 12, dT, 14.5)
dT.train(df[0:100], df.Energie[0:100])
dT.predict(df[100:200])
