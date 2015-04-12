#!/usr/bin/env python

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import csv

ds = []
keys = [
    'min',
    'mean',
    'max',
    'learning_rate',
    'inertia',
    'ff',
    'opt',
    'sp']

#IMPORT FILE
input_file = csv.DictReader(open("data.csv"))
for row in input_file:
    ds.append(row)

print len(ds)

se =  [d for d in ds if d['ff'] == 'SquaredError2D']
tm =  [d for d in ds if d['ff'] == 'Trimodal2D']
pt =  [d for d in ds if d['ff'] == 'Plateau3D']

hc =  [d for d in ds if d['opt'] == 'HillClimber']
sd =  [d for d in ds if d['opt'] == 'SteepestDescent']
nm =  [d for d in ds if d['opt'] == 'NewtonMethod']


#SQUARED 3D Overall
# y = [float(d['sp'].strip('[]')) for d in ds if (d['ff'] == 'SquaredError2D' and d['opt'] == 'HillClimber')]
#
#
#
# x = [float(d['max'].strip('[]')) for d in ds if (d['ff'] == 'SquaredError2D' and d['opt'] == 'HillClimber')]
# plt.plot( y, x,c = 'red', linestyle = '-')
#
# # x = [float(d['min'].strip('[]')) for d in ds if (d['ff'] == 'Trimodal2D' and d['opt'] == 'HillClimber')]
# # plt.plot( y, x,c = 'blue')
#
# x = [float(d['mean'].strip('[]')) for d in ds if (d['ff'] == 'SquaredError2D' and d['opt'] == 'HillClimber')]
# plt.plot( y, x,c = 'green', linestyle = '-')
#
#
#
# x = [float(d['max'].strip('[]')) for d in ds if (d['ff'] == 'SquaredError2D' and d['opt'] == 'NewtonMethod')]
# plt.plot( y, x,c = 'red', linestyle = '--')
#
# # x = [float(d['min'].strip('[]')) for d in ds if (d['ff'] == 'Trimodal2D' and d['opt'] == 'HillClimber')]
# # plt.plot( y, x,c = 'blue')
#
# x = [float(d['mean'].strip('[]')) for d in ds if (d['ff'] == 'SquaredError2D' and d['opt'] == 'NewtonMethod')]
# plt.plot( y, x,c = 'green', linestyle = '--')
#
# # plt.show()




# ##PLATEAU 3D Overall
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
#
#
# xs = []
# ys = []
# for d in ds:
#     if (d['ff'] == 'Plateau3D' and d['opt'] == 'SteepestDescent'):
#         s =  d['sp'].strip('[] ')
#         x, y =  s.split()
#         xs.append(float(x))
#         ys.append(float(y))
#         # print (float(x), float(y))
#
# z = [float(d['max'].strip('[]')) for d in ds if (d['ff'] == 'Plateau3D' and d['opt'] == 'SteepestDescent')]
# ax.scatter(xs, ys, z, marker = 'x', s = 40, alpha = 0.20, zorder = 2)
#
# xs = []
# ys = []
# for d in ds:
#     if (d['ff'] == 'Plateau3D' and d['opt'] == 'HillClimber'):
#         s =  d['sp'].strip('[] ')
#         x, y =  s.split()
#         xs.append(float(x))
#         ys.append(float(y))
#         # print (float(x), float(y))
#
# z = [float(d['max'].strip('[]')) for d in ds if (d['ff'] == 'Plateau3D' and d['opt'] == 'HillClimber')]
# ax.scatter(xs, ys, z, marker = 'o',  s = 100, alpha = 1.0, zorder = 10)


fig = plt.figure()
ax = fig.add_subplot(111)
# plt.ylim([0.5, 2.5])

y = [float(d['inertia'].strip('[]')) for d in ds if (d['ff'] == 'Plateau3D' and d['opt'] == 'SteepestDescent')]


x = [float(d['max'].strip('[]')) for d in ds if (d['ff'] == 'Plateau3D' and d['opt'] == 'SteepestDescent')]
plt.scatter( y, x, c = 'green', marker = 'o',  s = 40, alpha = 0.10, zorder = 10)


# x = [float(d['max'].strip('[]')) for d in ds if (d['ff'] == 'Trimodal2D' and d['opt'] == 'SteepestDescent')]
# plt.scatter( y, x, c = 'red', marker = 'o',  s = 40, alpha = 0.10, zorder = 10)

# x = [float(d['min'].strip('[]')) for d in ds if (d['ff'] == 'Trimodal2D' and d['opt'] == 'SteepestDescent')]
# plt.scatter( y, x, c = 'blue')



plt.show()

# print len(se)
# print len(tm)
# print len(pt)
# print len(hc)
# print len(sd)
# print len(nm)
