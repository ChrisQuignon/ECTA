import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn import tree

# dummy data:
df = pd.DataFrame({'col1':[0,1,2,3],'col2':[3,4,5,6],'dv':[0,1,0,1]})

# create decision tree
dt = tree.DecisionTreeClassifier(max_depth=5, min_samples_leaf=1)


dt.fit(df.ix[:,:2], df.dv)

dt.tree_.threshold
#Out[6]: array([ 3.5, -2. ,  4.5, -2. ,  2.5, -2. , -2. ])

dt.tree_.threshold =  dt.tree_.threshold + 0.1
#AttributeError: attribute 'threshold' of 'sklearn.tree._tree.Tree' objects is not writable




dt.fit(df.ix[:,:2], df.dv)

def get_code(tree, feature_names):
        left      = tree.tree_.children_left
        right     = tree.tree_.children_right
        threshold = tree.tree_.threshold
        features  = [feature_names[i] for i in tree.tree_.feature]
        value = tree.tree_.value

        def recurse(left, right, threshold, features, node):
                if (threshold[node] != -2):
                        print "if ( " + features[node] + " <= " + str(threshold[node]) + " ) {"
                        if left[node] != -1:
                                recurse (left, right, threshold, features,left[node])
                        print "} else {"
                        if right[node] != -1:
                                recurse (left, right, threshold, features,right[node])
                        print "}"
                else:
                        print "return " + str(value[node])

        recurse(left, right, threshold, features, 0)


get_code(dt, df.columns)
tree.export_graphviz(dt,
    out_file='tree.dot')


#TRY TO SET THRESHOLDS: DOES NOT WORK
#     from sklearn.datasets import load_linnerud
#     from sklearn import tree
#
#     linnerud = load_linnerud()
#     dt = tree.DecisionTreeRegressor(max_leaf_nodes = 4)
#     dt = dt.fit(linnerud.data, linnerud.target)
#
#     dt.tree_.threshold
#     # array([  55. ,   -2. ,  212.5,  103. ,   -2. ,   -2. ,   -2. ])
#
#
#     #Increase the threshold values
#     dt.tree_.threshold = [value * 1.1 if value != -2.0 else -2.0 for value in dt.tree_.threshold ]
#     #AttributeError: attribute 'threshold' of 'sklearn.tree._tree.Tree' objects is not writable
#
