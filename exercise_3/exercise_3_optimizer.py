
#from random import shuffle, randint
import time
import numpy as np
import matplotlib.pyplot as plt
from math import sqrt
from random import randint, shuffle, random

# import multiprocessing
from multiprocessing import Pool

import matplotlib.pyplot as plt
import csv


#read in file - no utf8
lines = [line.strip() for line in open('100 biggest German cities.txt')]
lines = [line.split("\t") for line in lines]

cities = []

#create dict
for line in lines[1:]:
    cities.append(dict(zip(lines[0], line)))

#type casting
for city in cities:
    city['Lat'] = (float) (city['Lat'])
    city['Long'] = (float) (city['Long'])
    city['Population'] = (int) (city['Population'])

# print len(cities)

#presort
cities = sorted(cities, key=lambda k: k['Long'])


ds = []


# print cities
#
#TSP

# def travel_length(cities):
#     length = 0.0
#     last_city = cities[-1]
#
#     for city in cities:
#         #conect all cities
#         #plt.plot([city['Long'], last_city['Long']], [city['Lat'], last_city['Lat']], c = 'b', alpha = 0.3)
#         lats = [city['Lat'], last_city['Lat']]
#         longs = [city['Long'], last_city['Long']]
#         #l = map(norm, [city['Long'], last_city['Long']], [city['Lat'], last_city['Lat']])
#         z =  zip(lats, longs)
#         dist = sqrt((z[0][0]-z[1][0])**2 + (z[0][1]-z[1][1])**2)
#         length = length + dist
#
#         last_city = city
#     return length
#
def sort_by(list, keys):
    return [val for (idx, val) in sorted(zip(keys, list))]


class Genome():
    def __init__(self, size, mutation_prob, genotype = None):
        self.mutation_prob = mutation_prob
        self.size = size

        if genotype is None:
            self.genotype = range(self.size)
            shuffle(self.genotype)
        else:
            self.genotype = genotype

    def mutate(self):
        if random() < self.mutation_prob:
            a = randint(0, len(self.genotype)-1)
            b = randint(0, len(self.genotype)-1)

            temp = self.genotype[a]
            self.genotype[a] = self.genotype[b]
            self.genotype[b] = temp
        pass

    def breed(self, partner):
        basis = self.genotype
        new_genome = partner.genotype

        c = randint(0, len(new_genome)-1)

        kid = basis[0:c]
        kid = kid + filter(lambda x : x not in kid, new_genome)

        return Genome(self.size, self.mutation_prob, kid)

class TSP():
    def __init__(self, cities, iterations, pop_size, select_perc, mutation_prob):
        self.cities = cities
        self.iterations = iterations
        self.pop_size = pop_size
        self.select_perc = select_perc
        self.mutation_prob = mutation_prob

        self.mins = []
        self.maxs = []
        self.means = []
        self.best_genotype = []

        self.pop = [Genome(len(cities), self.mutation_prob) for _ in range(pop_size)]


    def run(self):
        for i in range(self.iterations):
            self.evaluation()
            self.selection()
            self.crossover()
            self.mutation()
            # print self.fitness(self.pop[0])

            # #PLOT
            # plt.ion()
            # plt.show()
            #
            # if ((i%int(self.runs/100.0)) ==0):
            #     plt.clf()
            #     plt.axes().set_aspect(1.5)
            #     plt.scatter([c['Long'] for c in self.cities], [c['Lat'] for c in self.cities], c='y', s = 40)
            #
            #     plot_cities = sort_by(cities, self.pop[0].genotype)
            #    # plt.plot([c['Long'] for c in plot_cities], [c['Lat'] for c in plot_cities], c='b')
            # plt.fill([c['Long'] for c in plot_cities], [c['Lat'] for c in plot_cities],fill=False, edgecolor='b')
            #     plt.draw()
        self.evaluation()
        # print "DONE"
        # plt.clf()
        # plt.axes().set_aspect(1.5)
        # plt.scatter([c['Long'] for c in self.cities], [c['Lat'] for c in self.cities], c='y', s = 40)
        #
        # plot_cities = sort_by(cities, self.pop[0].genotype)
        # # plt.plot([c['Long'] for c in plot_cities], [c['Lat'] for c in plot_cities], c='b')
        # plt.fill([c['Long'] for c in plot_cities], [c['Lat'] for c in plot_cities],fill=False, edgecolor='b')
        # plt.draw()
        # plt.show()

        return self.mins, self.means, self.maxs, self.best_genotype

    def evaluation(self):
        # print 'eval'
        # print self.pop
        self.pop.sort(key=lambda x: self.fitness(x))

        #update statistics
        self.mins.append(self.fitness(self.pop[0]))
        self.maxs.append(self.fitness(self.pop[-1]))
        self.means.append(sum([self.fitness(x) for x in self.pop])/len(self.pop))

        if not self.best_genotype:
            self.best_genotype = self.pop[0].genotype
        elif self.mins[-1] < min(self.mins):
            self.best_genotype = self.pop[0].genotype


    def selection(self):
        # print 'select'
        n = int(len(self.pop) * self.select_perc)
        # print n
        self.pop = self.pop[:n]

    def crossover(self):
        # print 'crossover'
        new_pop = [self.pop[0]]#elitism
        #TODO: safe best
        # self.pop[0].mutation_prob = 0.0#super-elitism

        for _ in range(self.pop_size):
            p1 = randint(0, len(self.pop) - 1)
            p2 = randint(0, len(self.pop) - 1)

            p1 = self.pop[p1]
            p2 = self.pop[p2]

            new_pop.append(p1.breed(p2), )
        self.pop = new_pop

    def mutation(self):
        # print 'mutate'
        for g in self.pop:
            g = g.mutate()

    def fitness(self, genome):
        order = sort_by(self.cities, genome.genotype)

        length = 0.0
        last_city = order[-1]

        for city in order:
            #conect all cities
            #plt.plot([city['Long'], last_city['Long']], [city['Lat'], last_city['Lat']], c = 'b', alpha = 0.3)
            lats = [city['Lat'], last_city['Lat']]
            longs = [city['Long'], last_city['Long']]
            #l = map(norm, [city['Long'], last_city['Long']], [city['Lat'], last_city['Lat']])
            z =  zip(lats, longs)
            dist = sqrt((z[0][0]-z[1][0])**2 + (z[0][1]-z[1][1])**2)
            length = length + dist

            last_city = city
        return length

def run_tsp(runs, iterations, pop_size, select_perc, mutation_prob):
    final_min = []
    final_max = []
    final_mean = []


    name = str(runs) + '-' + str(iterations) + '-' + str(pop_size) + '-' + str(select_perc) + '-' + str(mutation_prob)

    print name

    # pars = 10
    # p = Pool(pars)
    # for _ in range(4):# 3*10  = 30 runs
    #     p.map(run_tsp(runs, 1000, pop_size, select_perc, mutation_prob), range(pars)) #runs 3 minutes

    for i in range(1, runs+1):
        tsp = TSP(cities = cities,
                  iterations = iterations,
                  pop_size = pop_size,
                  select_perc = select_perc,
                  mutation_prob = mutation_prob)

        _mins, _means, _maxs, _best_genotype = tsp.run()

        final_min.append(_mins[-1])
        final_max.append(_maxs[-1])
        final_mean.append(_means[-1])

        print "run " + str(i) + " of " + str(runs)

        # return final_min, final_max, final_mean

    # print _mins
    # print _maxs
    # print _means
    # print _best_genotype

    name = "{0}".format(round(tsp.fitness(tsp.pop[0]), 2)) + '-' + name

    plt.clf()
    plt.plot(range(len(_mins)), _mins, c = 'green')
    plt.plot(range(len(_mins)), _means, c = 'blue')
    plt.plot(range(len(_mins)), _maxs, c = 'red')
    plt.title(str(tsp.pop_size) + ' individuals, ' + str(tsp.select_perc) + '% selection ' + str(tsp.mutation_prob) + "% mutation")
    plt.xlabel("# iteration")
    plt.ylabel("Fitness in degree Lat/Long (~110km)")
    plt.ylim(plt.ylim()[0], 300)
    # plt.ylim(50, 300)
    # plt.show()
    plt.draw()
    plt.savefig('images/TravellingSalesman/' + name + '-fitness.png')


    plt.clf()
    plt.axes().set_aspect(1.5)
    plt.scatter([c['Long'] for c in tsp.cities], [c['Lat'] for c in tsp.cities], c='y', s = 40)

    plot_cities = sort_by(cities, tsp.pop[0].genotype)
    plt.fill([c['Long'] for c in plot_cities], [c['Lat'] for c in plot_cities],fill=False, edgecolor='b')
    # plt.plot([c['Long'] for c in plot_cities], [c['Lat'] for c in plot_cities], c='b')
    plt.draw()
    # plt.show()
    plt.savefig('images/TravellingSalesman/' + name + '.png')

    d = {'runs':runs,
        'iterations':iterations,
        'pop_size':pop_size,
        'select_perc':select_perc,
        'mutation_prob':mutation_prob,
        'min':np.mean(final_min),
        'max':np.mean(final_max),
        'mean':np.mean(final_mean)}
    return d

def wrapper(args):
    runs, iterations, pop_size, select_perc, mutation_prob = args
    r = run_tsp(runs, iterations, pop_size, select_perc, mutation_prob)
    return r

def analysis():
    runs = 30
    pop_sizes = [10, 100, 200]
    select_percs = [0.1, 0.2, 0.6]
    mutation_probs = [0.01, 0.1, 0.3]
    iterations = 100


    # ds = []

    for pop_size in pop_sizes:
        for select_perc in select_percs:
            # for mutation_prob in mutation_probs:
                # run_tsp(runs, 1000, pop_size, select_perc, mutation_prob)
            args = [(runs, iterations, pop_size, select_perc, x) for x in mutation_probs]

            pars = 4
            p = Pool(pars)
            rvals = p.map(wrapper, args) #runs 3 minutes
            ds.extend(rvals)
    #
    # _min, _max, _mean = run_tsp(runs, iterations, pop_size, select_perc, mutation_prob)

analysis()


keys = ['runs',
        'iterations',
        'pop_size',
        'select_perc',
        'mutation_prob',
        'min',
        'max',
        'mean']

with open('data_tsp.csv', 'wb') as output_file:
        dict_writer = csv.DictWriter(output_file, keys)
        dict_writer.writeheader()
        dict_writer.writerows(ds)







# pars = 4
# p = Pool(pars)
# for _ in range(21):
#     (p.map(run_tsp, range(pars))) #runs 3 minutes
