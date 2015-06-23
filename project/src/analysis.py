import pandas as pd
from matplotlib import pylab
import numpy as np


mutate_run = pd.DataFrame.from_csv('img/overnight/mutation_run.csv', sep = ';', index_col=None)

mutate_run=mutate_run.rename(columns = {'min':'Minimal MSE',
                                  'leaf_mutation' : 'Leaf mutation rate',
                                  'node_mutation' : 'Node mutation rate',
                                  'predict_mse' : 'Prediction MSE',
                                 })


relevant_values = ['Minimal MSE', 'Leaf mutation rate', 'Node mutation rate','Prediction MSE',]

#MINIMAL FITNESS
m_min = mutate_run.sort(columns = ['Minimal MSE'], ascending=False)
for idx in relevant_values:
    pylab.plot(m_min[idx].values  / m_min[idx].max(), linestyle='.', marker = '.',  label= idx)
    pylab.plot(m_min['Minimal MSE'].values / m_min['Minimal MSE'].max(), label='Minimal MSE')
    pylab.xlabel('Run index, sorted by minimal MSE (fitness)')
    pylab.ylabel('Normalized values')
    pylab.ylim(0, 1.1)
    pylab.legend()
    pylab.show()


#PREDICTED FITNESS
m_pred = mutate_run.sort(columns = ['Prediction MSE'], ascending=False)
for idx in relevant_values:
    pylab.plot(m_pred[idx].values  / m_pred[idx].max(), linestyle='.', marker = '.',  label= idx)
    pylab.plot(m_pred['Prediction MSE'].values / m_pred['Prediction MSE'].max(), label='Predicted MSE (fitness)')
    pylab.xlabel('Run index, sorted by predicted MSE (fitness)')
    pylab.ylabel('Normalized values')
    pylab.ylim(0, 1.1)
    pylab.legend()
    pylab.show()


#SORTED EVERYTHING
for idx in relevant_values:
    s = m_pred.sort(columns = [idx], ascending=False)
    rest = [x for x in relevant_values]# if x != idx]
    for jdx in rest:
        pylab.plot((s[jdx] / s[jdx].max()).values, linestyle='.', marker = '.',  label=jdx)
    pylab.title(idx)
    pylab.xlabel('Run index, sorted by ' + idx)
    pylab.ylabel('Normalized values')
    pylab.ylim(0, 1.1)
    pylab.legend()
    pylab.show()


#CORRELATION
cm = mutate_run[relevant_values].corr()
cm = cm.as_matrix()
pylab.pcolor(cm, cmap='RdBu')
cbar = pylab.colorbar()
pylab.clim(-1,1)

lable = map (lambda x: x.replace(' ', '\n'), relevant_values)
pylab.yticks(np.arange(0.5,4.5),lable, rotation='horizontal')
pylab.xticks(np.arange(0.5,4.5),lable, rotation='horizontal')
# savefig('../img/correlation.png')

pylab.title("Corellation matrix")
pylab.show()
pylab.clf()
