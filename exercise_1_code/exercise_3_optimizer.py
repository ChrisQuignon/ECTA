
#from random import shuffle, randint
import time
import numpy as np
import matplotlib.pyplot as plt
from math import sqrt
from random import randint, shuffle, random


import matplotlib.pyplot as plt


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


runs = 2
pop_sizes = [10, 50, 100]
select_percs = [0.1, 0.5, 1]
mutation_probs = [0.01, 0.1, 0.3]


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
    def __init__(self, cities, runs, pop_size, select_perc, mutation_prob):
        self.cities = cities
        self.runs = runs
        self.pop_size = pop_size
        self.select_perc = select_perc
        self.mutation_prob = mutation_prob

        self.pop = [Genome(len(cities), self.mutation_prob) for _ in range(pop_size)]


    def run(self):
        for i in range(self.runs):
            self.evaluation()
            self.selection()
            self.crossover()
            self.mutation()
            print self.fitness(self.pop[0])

            #PLOT

            plt.ion()
            plt.show()

            if ((i%int(self.runs/100.0)) ==0):
                plt.clf()
                plt.axes().set_aspect(1.5)
                plt.scatter([c['Long'] for c in self.cities], [c['Lat'] for c in self.cities], c='y', s = 40)

                plot_cities = sort_by(cities, self.pop[0].genotype)
                plt.plot([c['Long'] for c in plot_cities], [c['Lat'] for c in plot_cities], c='b')
                plt.draw()

        print "ok"
        plt.clf()
        plt.axes().set_aspect(1.5)
        plt.scatter([c['Long'] for c in self.cities], [c['Lat'] for c in self.cities], c='y', s = 40)

        plot_cities = sort_by(cities, self.pop[0].genotype)
        plt.plot([c['Long'] for c in plot_cities], [c['Lat'] for c in plot_cities], c='b')
        plt.draw()
        plt.show()

    def evaluation(self):
        # print 'eval'
        # print self.pop
        self.pop.sort(key=lambda x: self.fitness(x))

    def selection(self):
        # print 'select'
        n = int(len(self.pop) * self.select_perc)
        # print n
        self.pop = self.pop[:n]

    def crossover(self):
        # print 'crossover'

        new_pop = [self.pop[0]]#elitism

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


tsp = TSP(cities = cities,
          runs = 2000,
          pop_size = 20,
          select_perc = 0.8,
          mutation_prob = 0.2)
tsp.run()
