#!/usr/bin/env python

from __future__ import unicode_literals
import sys
import os
import time
import csv
import psutil
import logging

from matplotlib.backends import qt_compat

import numpy as np
use_pyside = qt_compat.QT_API == qt_compat.QT_API_PYSIDE
if use_pyside:
    from PySide import QtGui, QtCore
else:
    from PyQt4 import QtGui, QtCore

from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt4agg import NavigationToolbar2QT as NavigationToolbar

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import proj3d
from matplotlib.pyplot import cm

import helpers

progname = os.path.basename(sys.argv[0])
progversion = "0.1"

FRAMERATE = 30

# 0: show all generations
CURRENT_GENERATION = 0

class Canvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=100, nD = 2):
        #plt.xkcd()
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        FigureCanvas.__init__(self, self.fig)
        self.dim = nD
        if self.dim is 2:
            self.axes = self.fig.add_subplot(1, 1, 1)
            pass
        else:
            self.axes = self.fig.add_subplot(1, 1, 1, projection='3d')

            #Rotate
            self.axes.view_init(90, 90)

        self.axes.hold(False)
        self.compute_initial_figure()
        self.setParent(parent)
        FigureCanvas.setSizePolicy(self,
                                   QtGui.QSizePolicy.Expanding,
                                   QtGui.QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)

        #canvas Ticks
        start, end = self.axes.get_xlim()
        self.axes.xaxis.set_ticks(np.arange(start, end, 1.0))
        start, end = self.axes.get_ylim()
        self.axes.yaxis.set_ticks(np.arange(start, end, 1.0))

    def compute_initial_figure(self):
        pass

    def export_pdf(self):
        fname='export.pdf'
        self.fig.savefig(fname)

    def export_jpg(self):
        fname='export.jpg'
        self.fig.savefig(fname)


class FitnessCanvas(Canvas):
    global FRAMERATE

    def __init__(self, data = '', **kwargs):
        self.cfg_data_fitness = data
        Canvas.__init__(self, **kwargs)
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.update_figure)
        self.timer.start(1000/FRAMERATE)

    def compute_initial_figure(self):
        self.last_open_up = time.ctime(os.path.getmtime(self.cfg_data_fitness))
        self.last_open = self.last_open_up
        self.fitness_data = helpers.read_file(self.cfg_data_fitness)
        self.x = np.arange(0., len(self.fitness_data[0]), 1.)
        maxfit = x = np.array(map(float, self.fitness_data[0]))
        meanfit = x = np.array(map(float, self.fitness_data[1]))
        minfit = x = np.array(map(float, self.fitness_data[2]))
        self.l1, = self.axes.plot(self.x, maxfit, 'r')
        self.axes.hold(True)
        self.l2, = self.axes.plot(self.x, meanfit, 'g')
        self.axes.hold(True)
        self.l3, = self.axes.plot(self.x, minfit, 'b')
        self.axes.hold(False)


    def update_figure(self):
        self.timer.setInterval(1000/FRAMERATE)
        changed = False
        self.last_open_up = time.ctime(os.path.getmtime(self.cfg_data_fitness))
        # Make sure both fitness developments are of same length
        while (self.last_open != self.last_open_up) or (len(self.fitness_data[0]) != len(self.fitness_data[1])):
            changed = True
            self.last_open = self.last_open_up
            self.fitness_data = helpers.read_file(self.cfg_data_fitness)
            time.sleep(0.05)
        if changed:
            self.x = np.arange(0., len(self.fitness_data[0]), 1.)
            maxfit = x = np.array(map(float, self.fitness_data[0]))
            meanfit = x = np.array(map(float, self.fitness_data[1]))
            minfit = x = np.array(map(float, self.fitness_data[2]))
            self.l1, = self.axes.plot(self.x, maxfit, 'r')
            self.axes.hold(True)
            self.l2, = self.axes.plot(self.x, meanfit, 'g')
            self.axes.hold(True)
            self.l3, = self.axes.plot(self.x, minfit, 'b')
            self.axes.hold(False)

    def export_pdf(self):
        fname='export_fitness.pdf'
        self.fig.savefig(fname)

    def export_jpg(self):
        fname='export_fitness.jpg'
        self.fig.savefig(fname)

class FitnessLandScapeCanvas(Canvas):
    global FRAMERATE
    global CURRENT_GENERATION

    def __init__(self, ranges, individuals_data = '', fitscape_data='', **kwargs):
        self.cfg_data_individuals = individuals_data
        self.fitscape_data = fitscape_data
        self.ranges = ranges
        Canvas.__init__(self, **kwargs)
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.update_figure)
        self.timer.start(1000/FRAMERATE)

    def compute_initial_figure(self):
        self.curgen = CURRENT_GENERATION
        self.last_open_up_individuals = time.ctime(os.path.getmtime(self.cfg_data_individuals))
        self.last_open_individuals = self.last_open_up_individuals
        self.individuals_data = helpers.read_file(self.cfg_data_individuals)
        self.produce_scatter()

        # Create fitness surface
        self.fitscape = helpers.read_file(self.fitscape_data)
        self.produce_landscape()

    def update_figure(self):
        self.timer.setInterval(1000/FRAMERATE)
        changed = False
        self.last_open_up_individuals = time.ctime(os.path.getmtime(self.cfg_data_individuals))
        while (self.last_open_individuals != self.last_open_up_individuals) or (len(self.individuals_data) != len(self.individuals_data)):
            changed = True
            self.last_open_individuals = self.last_open_up_individuals
            self.individuals_data = helpers.read_file(self.cfg_data_individuals)
            self.annotation_last_step = []
        if changed or CURRENT_GENERATION is not self.curgen:
            self.scatter.remove()
            self.scatter_best.remove()
            self.produce_scatter()

            self.curgen = CURRENT_GENERATION
            self.draw()
        pass

    def produce_scatter(self):
        pass

    def produce_landscape(self):
        pass

class FitnessLandScapeCanvas2D(FitnessLandScapeCanvas):
    def produce_scatter(self):
        self.scatter = self.axes.plot(self.individuals_data[[2*CURRENT_GENERATION],:][0],
                                            self.individuals_data[[2*CURRENT_GENERATION+1][0],:],
                                            color='red',
                                            marker='x',
                                            zorder = 10,
                                            #s=50,
                                            linestyle = '-',
                                            )
        self.axes.hold(True)
        best_individual = np.argmax(self.individuals_data[[2*CURRENT_GENERATION+1],:])
        self.scatter_best = self.axes.scatter(self.individuals_data[[2*CURRENT_GENERATION],best_individual], self.individuals_data[[2*CURRENT_GENERATION+1],best_individual], c='g', marker='o', zorder = 4, s=120)

    def produce_landscape(self):
        self.l1, = self.axes.plot(self.ranges, self.fitscape, 'b')
        self.axes.set_axisbelow(True)
        self.axes.hold(True)

class FitnessLandScapeCanvas3D(FitnessLandScapeCanvas):
    def produce_scatter(self):

        x= self.individuals_data[0]
        y =  self.individuals_data[1]
        z =  self.individuals_data[2]

        # pass
        # x = self.individuals_data[CURRENT_GENERATION, 0::3]
        # y = self.individuals_data[CURRENT_GENERATION, 0::2]
        # z = self.individuals_data[CURRENT_GENERATION, 0::1]
        #
        # print "XXX"
        # print x
        # print y
        # print z

        # print x
        # print y
        # print z


        self.scatter = self.axes.scatter(x, y, z, c='r', marker='o', zorder = 4, s=50)
        self.axes.hold(True)
        best_individual = np.argmax(z)
        self.scatter_best = self.axes.scatter(x[best_individual], y[best_individual], z[best_individual], c='g', marker='o', s=120, zorder = 5)

    def produce_landscape(self):
        self.x, self.y = np.meshgrid(self.ranges, self.ranges)
        surf = self.axes.plot_surface(self.x, self.y, self.fitscape, rstride=1, cstride=1, cmap=cm.jet, alpha=0.4, zorder = 10)
        self.axes.set_axisbelow(True)
        self.axes.hold(True)
        self.axes.mouse_init()

class EAVApplicationWindow(QtGui.QMainWindow):
    global CPU_CHECK_FRAMERATE
    def __init__(self, fitstat_data, individuals_data, fitscape_data, ranges, width=8, height=10, dpi=100, dim=2, max_iterations=100):
        self.dim = dim
        self.max_iterations = max_iterations - 1
        QtGui.QMainWindow.__init__(self)
        self.setAttribute(QtCore.Qt.WA_DeleteOnClose)
        self.setWindowTitle("application main window")

        self.file_menu = QtGui.QMenu('&File', self)
        self.file_menu.addAction('&Quit', self.fileQuit,
                                 QtCore.Qt.CTRL + QtCore.Qt.Key_Q)
        self.menuBar().addMenu(self.file_menu)

        self.help_menu = QtGui.QMenu('&Help', self)
        self.menuBar().addSeparator()
        self.menuBar().addMenu(self.help_menu)

        self.help_menu.addAction('&Info', self.info)
        self.help_menu.addAction('&About', self.about)

        self.main_widget = QtGui.QWidget(self)
        l = QtGui.QVBoxLayout(self.main_widget)

        self.label_sld_framerate = QtGui.QLabel("Auto-refresh rate")
        self.sld_framerate = QtGui.QSlider(QtCore.Qt.Horizontal, self)
        self.sld_framerate.valueChanged[int].connect(self.changeFramerate)
        self.sld_framerate.setTickPosition(2)
        self.sld_framerate.setTickInterval(5)

        self.label_sld_generation = QtGui.QLabel("Maximum generation displayed")
        self.label_current_generation = QtGui.QLabel(str(CURRENT_GENERATION))

        #self.sld_generation = QtGui.QSlider(0, max_iterations, 1, 0, QtCore.Qt.Horizontal, self)
        self.sld_generation = QtGui.QSlider(QtCore.Qt.Horizontal, self)
        self.sld_generation.setMinimum(0)
        self.sld_generation.setMaximum(self.max_iterations)
        self.sld_generation.setTickInterval(1)
        self.sld_generation.setValue(0)
        self.sld_generation.valueChanged[int].connect(self.changeGeneration)
        self.sld_generation.setTickPosition(2)
        self.sld_generation.setTickInterval(10)

        self.btn_export_pdf = QtGui.QPushButton("Export to PDF", self)
        self.btn_export_pdf.clicked.connect(self.export_pdf)
        self.btn_export_jpg = QtGui.QPushButton("Export to JPG", self)
        self.btn_export_jpg.clicked.connect(self.export_jpg)

        self.frame_sliders = QtGui.QFrame()
        self.hBoxLayout = QtGui.QHBoxLayout()

        self.hBoxLayout.addWidget(self.label_sld_framerate)
        self.hBoxLayout.addWidget(self.sld_framerate)
        self.hBoxLayout.addWidget(self.label_sld_generation)
        self.hBoxLayout.addWidget(self.sld_generation)
        self.hBoxLayout.addWidget(self.label_current_generation)
        self.hBoxLayout.addWidget(self.btn_export_pdf)
        self.hBoxLayout.addWidget(self.btn_export_jpg)
        self.frame_sliders.setLayout(self.hBoxLayout)

        # Add fitness canvas
        self.canvas_fit = FitnessCanvas(data=str(fitstat_data), parent=self.main_widget, width=width, height=height, dpi=dpi, nD = 2)


        # Add fitness landscape and individuals
        self.dim = dim
        if dim is 3:
            self.canvas_fitscape = FitnessLandScapeCanvas3D(parent=self.main_widget, width=width, height=height, dpi=dpi, nD = self.dim, ranges=ranges, individuals_data=str(individuals_data), fitscape_data=str(fitscape_data))
        elif dim is 2:
            self.canvas_fitscape = FitnessLandScapeCanvas2D(parent=self.main_widget, width=width, height=height, dpi=dpi, nD = self.dim, ranges=ranges, individuals_data=str(individuals_data), fitscape_data=str(fitscape_data))
        else:
            logging.error("Failed visualization of fitness landscape. Dimensionality = " + str(dim))

        self.frame_nav = QtGui.QFrame()
        self.hBoxLayout_nav = QtGui.QHBoxLayout()
        self.toolbar = NavigationToolbar(self.canvas_fitscape, self)
        self.hBoxLayout_nav.addWidget(self.toolbar)
        self.frame_nav.setLayout(self.hBoxLayout_nav)

        l.addWidget(self.canvas_fit)
        #l.addWidget(self.frame_nav)
        l.addWidget(self.canvas_fitscape)
        #l.addWidget(self.frame_sliders)

        self.main_widget.setFocus()
        self.setCentralWidget(self.main_widget)

        self.statusBar().showMessage("Have fun!", 2000)

    def keyPressEvent(self, e):
        if e.key() == QtCore.Qt.Key_Escape:
            self.close()
        if e.key() == QtCore.Qt.Key_Plus:
            self.incGeneration()
        if e.key() == QtCore.Qt.Key_Minus:
            self.decGeneration()

    def set_label_current_generation(self):
        self.label_current_generation.setText(str(CURRENT_GENERATION))

    def changeFramerate(self, val):
        global FRAMERATE
        val = val + 1
        FRAMERATE = val
        logging.info("Framerate set to: " + str(val))

    def changeGeneration(self, val):
        global CURRENT_GENERATION
        CURRENT_GENERATION = val
        self.set_label_current_generation()

    def incGeneration(self):
        global CURRENT_GENERATION
        CURRENT_GENERATION = CURRENT_GENERATION + 1
        self.set_label_current_generation()

    def decGeneration(self):
        global CURRENT_GENERATION
        CURRENT_GENERATION = CURRENT_GENERATION - 1
        self.set_label_current_generation()

    def fileQuit(self):
        self.close()

    def closeEvent(self, ce):
        self.fileQuit()

    def info(self):
        QtGui.QMessageBox.about(self,   "Info",
                                        """Visualization tool for demos and exercises\n\n Left-click-drag to rotate graph, right-click-drag to zoom graph"""
)

    def about(self):
        QtGui.QMessageBox.about(self,   "About","""Bonn-Rhine-Sieg University of Applied Sciences.\nCourse on Evolutionary Algorithms"""
)

    def export_pdf(self):
        self.canvas_fit.export_pdf()
        self.canvas_fitscape.export_pdf()
        self.statusBar().showMessage("PDF exported", 2000)

    def export_jpg(self):
        self.canvas_fit.export_jpg()
        self.canvas_fitscape.export_jpg()
        self.statusBar().showMessage("JPG exported", 2000)
