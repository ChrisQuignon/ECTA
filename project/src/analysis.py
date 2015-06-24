import pandas as pd
from matplotlib import pylab
import numpy as np


# mutate_run = pd.DataFrame.from_csv('img/overnight/mutation_run.csv', sep = ';', index_col=None)
# selection_run = pd.DataFrame.from_csv('img/overnight/select_run.csv', sep = ';', index_col=None)
# mutate_run = pd.DataFrame.from_csv('img/last_runs/mutation_run.csv', sep = ';', index_col=None)
# selection_run = pd.DataFrame.from_csv('img/last_runs/select_run.csv', sep = ';', index_col=None)
#
# all_runs = mutate_run.append(selection_run)
all_runs = pd.DataFrame.from_csv('img/statsrun/statsrun.csv', sep = ',', index_col=None)

all_runs=all_runs.rename(columns = {'min':'Minimal MSE',
                                  'leaf_mutation' : 'Leaf mutation rate',
                                  'node_mutation' : 'Node mutation rate',
                                  'predict_mse' : 'Prediction MSE',
                                  "sigma" : 'Sigma',
                                  "iterations" : 'Iterations',
                                  "selection" : 'Selection Type',
                                  "n_samples" : "MSE Samples"
                                 })


#parse selection
slist = list(set(all_runs['Selection Type'].values))
sindexes = [slist.index(stype) for stype in all_runs['Selection Type'].values]
all_runs['Selection Type'] = sindexes

relevant_values = np.asarray(['Minimal MSE', 'Leaf mutation rate', 'Node mutation rate','Prediction MSE', "Sigma", "Iterations", "Selection Type", "MSE Samples"])


#PREDICTED FITNESS
all_runs = all_runs.sort(columns = ['Prediction MSE'], ascending=False)
for idx in relevant_values:
    pylab.plot(all_runs[idx].values, linestyle='.', marker = '.',  label= idx)
    pylab.plot(all_runs['Prediction MSE'].values * float(all_runs[idx].max()*4), label='Scaled predicted MSE (fitness)')
    pylab.xlabel('Run index, sorted by predicted MSE (fitness)')
    pylab.ylabel(idx)
    axes = pylab.gca()
    ymin, ymax = axes.get_ylim()
    pylab.ylim(ymin*1.1, ymax*1.3)

    if idx in ["Selection Type"]:
    # pylab.ylim(0, 1.1)
        pylab.yticks(np.arange(0.0,4.0), slist)
        pylab.ylim(-.2, ymax*1.3)

    pylab.legend()
    pylab.show()


#PREDICTED MSE
all_runs = all_runs.sort(columns = ['Minimal MSE'], ascending=False)
for idx in relevant_values:
    pylab.plot(all_runs[idx].values, linestyle='.', marker = '.',  label= idx)
    pylab.plot(all_runs['Minimal MSE'].values * float(all_runs[idx].max()*4), label='Scaled minimal MSE (fitness)')
    pylab.xlabel('Run index, sorted by minimal MSE (fitness)')
    pylab.ylabel(idx)
    axes = pylab.gca()
    ymin, ymax = axes.get_ylim()
    pylab.ylim(ymin*1.1, ymax*1.3)

    if idx in ["Selection Type"]:
    # pylab.ylim(0, 1.1)
        pylab.yticks(np.arange(0.0,4.0), slist)
        pylab.ylim(-.2, ymax*1.3)

    pylab.legend()
    pylab.show()


#SORTED EVERYTHING
for idx in relevant_values:
    s = all_runs.sort(columns = [idx], ascending=False)
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
cm = all_runs[relevant_values].corr()
cm = cm.as_matrix()

# #REORDER
sortkey = np.asarray([0, 3, 2, 7, 1, 4, 5, 6])
cm = cm[sortkey]
for i,  row in enumerate(cm):
    cm[i] = row[sortkey]

pylab.pcolor(cm, cmap='RdBu')
cbar = pylab.colorbar()
pylab.clim(-1,1)

lable = map (lambda x: x.replace(' ', '\n'), relevant_values[sortkey])
pylab.yticks(np.arange(0.5,8.5),lable, rotation='horizontal')
pylab.xticks(np.arange(0.5,8.5),lable, rotation='horizontal')
# savefig('../img/correlation.png')

pylab.title("Corellation matrix")
pylab.show()
pylab.clf()
