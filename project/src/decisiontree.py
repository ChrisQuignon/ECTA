#!/usr/bin/env python

import numpy as np
import pylab
import pandas as pd
import matplotlib.pyplot as plt
from imp import load_source
import time

# # TESTING
# df = pd.read_csv('data/ds1_weather.csv', decimal=',',sep=';', parse_dates=True, index_col=[0])
#
# df = df.resample('1Min')
# df.interpolate(inplace=True)
# df.fillna(inplace=True, method='ffill')#we at first forwardfill
# df.fillna(inplace=True, method='bfill')#then do a backwards fill


class DecisionLeaf(object):

  def __init__(self, val):
    self.val = val
    self.mse = float('inf')

  def __iter__(self):
    yield self

  def __repr__(self):
      return str(self.val)

  def __getitem__(self, key):
      #TODO fix
      return self.val

  def __setitem__(self, key, item):
      #TODO fix
       self.val = item

  def predict(self, predictionset):
    return [self.val for _ in predictionset.index]

  def update_mse(self,  input_df, outputset):
    result = self.predict(input_df)

    result = np.array(result)
    output = np.array(outputset)

    self.mse = ((result - output) ** 2).mean()
    return self.mse

  def depth(self):
    return 1

  def size(self):
    return 1

class DecisionTree(object):

  def __init__(self, feature, split, left_child, right_child):
    self.feature = feature
    self.split = split

    self.mse = float('inf')

    self.right_child = []
    self.set_right_child(right_child)

    self.left_child = []
    self.set_left_child(left_child)

  def __iter__(self):
    yield self
    for left in self.left_child:
        yield left
    for right in self.right_child:
        yield right


  def __getitem__(self, key):
    return [x for x in self][key]

  def __setitem__(self, key, item):
    left_items = self.left_child.size()

    if key == 0:
        print "CANNOT CHANGE ITSELF"
        return

    if key == 1:
        self.left_child = item
        return

    if key == left_items + 1:
        self.right_child = item
        return

    if key <= left_items:
        self.left_child[key - 1] = item
        return

    if key > left_items + 1:
        self.right_child[key - left_items-1] = item
        return

  def __repr__(self):
      s = 'if x[' + str(self.feature) + '] <= ' + str(self.split) + ':\n'

      for left_line in str(self.left_child).split('\n'):#indent
          s = s + '  ' + left_line + '\n'
      s = s + 'else:\n'

      for right_line in str(self.right_child).split('\n'):#indent
          s = s + '  ' + right_line + '\n'
      return s

  def set_left_child(self, left_child):
    if isinstance(left_child, float):
      self.left_child = DecisionLeaf(left_child)
    else:
      self.left_child = left_child

  def set_right_child(self, right_child):
    if isinstance(right_child, float):
      self.right_child = DecisionLeaf(right_child)
    else:
      self.right_child = right_child

  def update_mse(self, input_df, outputset):


    li = []
    ri = []

    for i in input_df.index:
        if input_df[self.feature][i] < self.split:
            li.append(i)
        else:
            ri.append(i)


    left = input_df.ix[li]
    left_out = outputset.ix[li]


    right = input_df.ix[ri]
    right_out = outputset.ix[ri]

    #update_mse recursively
    #heavy in computation time.
    # self.left_child.update_mse(left, leftout)
    # self.right_child.update_mse(right, rightout)

    result = self.left_child.predict(left)
    result.extend(self.right_child.predict(right))

    result = np.array(result)
    output = np.concatenate((left_out.as_matrix(), right_out.as_matrix()))
    output.flatten()

    result = (result - output) ** 2
    result = result[~np.isnan(result)] #drop nans


    self.mse = result.mean(axis = 0)


  def predict(self, predictionset):

    li = []
    ri = []

    for i in predictionset.index:
        if predictionset[self.feature][i] <= self.split:
            li.append(i)
        else:
            ri.append(i)

    left = predictionset.ix[li]
    right = predictionset.ix[ri]

    left_val = self.left_child.predict(left)
    right_val = self.right_child.predict(right)

    result = []
    result.extend(left_val)
    result.extend(right_val)

    return result

  def depth(self):
    left_depth = self.left_child.depth()
    right_depth = self.right_child.depth()
    return 1 + (max(left_depth, right_depth))

  def size(self):
    left_size = self.left_child.size()
    right_size = self.right_child.size()
    return 1 + left_size + right_size


# #
# # #TESTING
# #
# dT = DecisionTree('Aussentemperatur', 7, 7.7, 64.0)
# dN = DecisionTree('Vorlauftemperatur', 8.2, 0.2, 64.6)
# dN[1] = dT
# dN.update_mse(df[0:100], df.Energie[0:100])
# print dN.predict(df[10:20])
#
# print dN
