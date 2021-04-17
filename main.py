# -*- coding: utf-8 -*-
"""
Created on Fri Jan 29 17:10:06 2021

@author: Leonardo Eiji Tamayose & Guilherme Ferrari Fortino 

Main File

"""

import sys
import os
import platform
from matplotlib_backend_qtquick.qt_compat import QtGui, QtQml, QtCore
from matplotlib_backend_qtquick.backend_qtquickagg import FigureCanvasQtQuickAgg
from numpy.core.records import array
from operator import mod
from src.MatPlotLib import DisplayBridge
from src.Model import Model
from src.Calculators import CalculatorCanvas, interpreter_calculator, Plot
from src.ProjectManager import ProjectManager
from src.MultiPlot import Multiplot
from src.UpdateChecker import UpdateChecker

# Instantiating the display bridge || Global variable
displayBridge = DisplayBridge()

# Instantiating the fit class
model = Model() 

class Bridge(QtCore.QObject):
    # Signal to Properties page
    signalPropPage = QtCore.Signal()

    # Signal to write infos
    writeInfos = QtCore.Signal(str, arguments='expr')
    writeCalculator = QtCore.Signal(str, arguments='expr')
    emitData = QtCore.Signal()

    @QtCore.Slot(QtCore.QJsonValue)
    def getProps(self, props):
        self.emitData.emit()
        props = props.toVariant()

        displayBridge.setSigma(props['sigmax'], props['sigmay'])

        # Setting up initial parameters
        p0_tmp = list()
        p0 = props['p0']
        if p0 != '':
            # Anti-dummies system
            p0 = p0.replace(';', ',')
            p0 = p0.replace('/', ',')
            for i in p0.split(','):
                p0_tmp.append(float(i))
            model.set_p0(p0_tmp)

        # Anti-dummies system 2
        expression = props['expr']
        expression = expression.replace('^', '**')
        expression = expression.replace('arctan', 'atan')
        expression = expression.replace('arcsin', 'asin')
        expression = expression.replace('arccos', 'acos')
        expression = expression.replace('sen', 'sin')
        
        # Setting expression
        if model.exp_model != expression:
            model.set_expression(expression)

        curveStyles = {
            'Sólido':'-',
            'Tracejado':'--',
            'Ponto-Tracejado':'-.'
            }
        symbols = {
            'Círculo':'o',
            'Triângulo':'^',
            'Quadrado':'s',
            'Pentagono':'p',
            'Octagono':'8',
            'Cruz':'P',
            'Estrela':'*',
            'Diamante':'d',
            'Produto':'X'
            }

        # Setting style of the plot 
        model.set_title(props['titulo'])
        model.set_x_axis(props['eixox'])
        model.set_y_axis(props['eixoy'])
        displayBridge.setStyle( props['logx'],
                                props['logy'],
                                props['markerColor'],
                                props['markerSize'],
                                symbols[props['marker']],
                                props['curveColor'],
                                props['curveThickness'],
                                curveStyles[props['curveType']],
                                props['legend'],
                                model.exp_model.replace('**', '^'))

        # Making plot
        displayBridge.Plot(model, props['residuos'], props['grade'],
         props['xmin'], props['xmax'], props['xdiv'],
         props['ymin'], props['ymax'], props['ydiv'],
         props['resMin'], props['resMax'])

    @QtCore.Slot(str)
    def loadData(self, file_path):
        """Gets the path to data's file and fills the data's table"""
        model.load_data(QtCore.QUrl(file_path).toLocalFile())

    @QtCore.Slot(str)
    def savePlot(self, save_path):
        """Gets the path from input and save the actual plot"""
        displayBridge.figure.savefig(QtCore.QUrl(save_path).toLocalFile(), dpi = 400)

    @QtCore.Slot(str, str, str, str, str, str)
    def calculator(self, function, opt1, nc, ngl, mean, std):
        functionDict = {
            'Chi²':0,
            'Chi² Reduzido':1,
            'Gaussiana':2,
            'Student':3
        }
        methodDict = {
            'Simétrico de Dois Lados':0,
            'Apenas Limite Inferior':1,
            'Apenas Limite Superior':2
        }   
        try:
            nc = nc.replace(',', '.')
            nc = float(nc)
        except:
            pass
        try:
            ngl = ngl.replace(',', '.')
            ngl = float(ngl)
        except:
            pass
        try:
            mean = mean.replace(',', '.')
            mean = float(mean)
        except:
            pass
        try:
            std = std.replace(',', '.')
            std = float(std)
        except:
            pass

        s, x, y, x_area, y_area = interpreter_calculator(functionDict[function], methodDict[opt1], nc, ngl, mean, std)
        Plot(displayBridge, x, y, x_area, y_area)
        self.writeCalculator.emit(s)

if __name__ == "__main__":
    # Matplotlib stuff
    QtQml.qmlRegisterType(FigureCanvasQtQuickAgg, "Canvas", 1, 0, "FigureCanvas")
    QtCore.QCoreApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling)

    # Setting up app
    app = QtGui.QGuiApplication(sys.argv)
    app.setOrganizationName("High Elo Devs")
    app.setOrganizationDomain("https://github.com/leoeiji/Analysis-Tool-for-Undergrad-Students---ATUS")
    app.setApplicationName("Analysis Tool for Undergrad Students")
    app.setWindowIcon(QtGui.QIcon(os.path.join(os.path.dirname(__file__), "ATUS Icon 3.png")))
    engine = QtQml.QQmlApplicationEngine()

    # Creating bridge
    bridge    = Bridge()
    multiplot = Multiplot(displayBridge)
    updater = UpdateChecker()

    # Project Manager
    projectMngr = ProjectManager(displayBridge, model)

    # Creating 'link' between front-end and back-end
    context = engine.rootContext()
    context.setContextProperty("displayBridge", displayBridge)
    context.setContextProperty("backend", bridge)
    context.setContextProperty("model", model)
    context.setContextProperty("projectMngr", projectMngr)
    context.setContextProperty("multiplot", multiplot)
    context.setContextProperty("updater", updater)
    
    # Loading QML files
    plat = platform.system()
    # print('Sistema Operacional: ' + plat)

    if(plat == 'Darwin'):
        engine.load(QtCore.QUrl.fromLocalFile(os.path.join(os.path.dirname(__file__), "qml/main_mac.qml")))
    else:
        engine.load(QtCore.QUrl.fromLocalFile(os.path.join(os.path.dirname(__file__), "qml/main_windows.qml")))
    # Updating canvasPlot with the plot
    win = engine.rootObjects()[0]
    displayBridge.updateWithCanvas(win.findChild(QtCore.QObject, "canvasPlot"))
    
    # Stopping program if PySide fails loading the file
    if not engine.rootObjects():
        sys.exit(-1)    

    # Starting program
    sys.exit(app.exec_())