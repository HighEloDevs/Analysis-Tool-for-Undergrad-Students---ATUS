# -*- coding: utf-8 -*-
"""
MIT License

Copyright (c) 2021 Leonardo Eiji Tamayose, Guilherme Ferrari Fortino

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

from matplotlib_backend_qtquick.qt_compat import QtCore
from src.Calculators import interpreter_calculator, Plot
import pandas as pd
import json
import platform

class SinglePlot(QtCore.QObject):
    '''Class that controls the single-plot page'''

    # Signal to write infos
    writeCalculator       = QtCore.Signal(str, arguments='expr')
    fillPlotPageSignal    = QtCore.Signal(QtCore.QJsonValue, arguments='props')
    plot                  = QtCore.Signal()

    def __init__(self, canvas, model, messageHandler):
        super().__init__()
        self.canvas = canvas
        self.model  = model
        self.path   = ''
        self.msg    = messageHandler

        # Default properties for the singlePlot page
        self.props = {
            'id': '',
            'dataProps': {
                'marker_color'    : '#000',
                'marker_size'     : 3,
                'marker'          : 'o',
                'curve_color'     : '#000',
                'curve_thickness' : 3,
                'curve_style'     : '-',
            },
            'canvasProps': {
                'xaxis'     : '',
                'yaxis'     : '',
                'title'     : '',
                'log_x'     : False,
                'log_y'     : False,
                'legend'    : False,
                'grid'      : False,
                'residuals' : False,
                'xmin'      : '',
                'xmax'      : '',
                'xdiv'      : '',
                'ymin'      : '',
                'ymax'      : '',
                'ydiv'      : '',
                'resmin'    : '',
                'resmax'    : '',
            },
            'fitProps': {
                'expr'       : '',
                'p0'         : '',
                'wsx'        : True,
                'wsy'        : True,
                'parameters' : {}
            },
            'data': []
        }

    @QtCore.Slot(QtCore.QJsonValue)
    def getPlotData(self, plotData):
        plotData    = plotData.toVariant()
        # id          = plotData['id']
        canvasProps = plotData['canvasProps']
        dataProps   = plotData['dataProps']
        fitProps    = plotData['fitProps']
        data        = plotData['data']

        # Loading data from the table
        self.model.loadDataTable(data)

        # Getting function to fit
        # Anti-dummies system
        expr = fitProps['expr']
        expr = expr.replace('^', '**')
        expr = expr.replace('arctan', 'atan')
        expr = expr.replace('arcsin', 'asin')
        expr = expr.replace('arccos', 'acos')
        expr = expr.replace('sen', 'sin')

        # Setting expression
        if self.model._exp_model != expr:
            self.model.set_expression(expr)

        # Getting initial parameters
        if fitProps['p0'] != '':
            p0 = fitProps['p0']
            p0 = p0.replace(';', ',')
            p0 = p0.replace('/', ',')
            self.model.set_p0(p0)

        # Setting style of the plot 
        self.canvas.setCanvasProps(canvasProps, expr)
        self.canvas.setDataProps(dataProps, fitProps)
        self.canvas.Plot(self.model)

    def fillPlotPage(self, props=None):
        # If no properties passed, emit the default values
        if props is None:
            self.fillPlotPageSignal.emit(QtCore.QJsonValue.fromVariant(self.props))
        else:
            self.fillPlotPageSignal.emit(QtCore.QJsonValue.fromVariant(props))
        
    @QtCore.Slot()
    def new(self):
        # Reseting canvas and model
        self.model.reset()
        self.canvas.reset()

        # Fill singlePlot page with default values
        self.fillPlotPage()

        # Reseting path
        self.path = ''

    @QtCore.Slot(str)
    def load(self, path):
        # Reseting frontend
        self.new()

        # Getting path
        self.path = QtCore.QUrl(path).toLocalFile()

        # Getting props
        with open(self.path, encoding='utf-8') as file:
            props = json.load(file)

        if "key" in props:
            # Loading data from the table
            self.model.load_data(df_array=props['data'])
            if props["key"][0] != "2":
                self.msg.raiseWarn("O carregamento de arquivos antigos está limitado à uma versão anterior. Adaptação feita automaticamente.")
        else:
            self.msg.raiseWarn("O carregamento de arquivos antigos está limitado à uma versão anterior. Adaptação feita automaticamente.")
            props = self.loadOldJson(props)
            self.model.load_data(df=props['data'])

        self.fillPlotPage(props)
        # self.plot.emit()

    def loadOldJson(self, props):
        props_tmp = self.props.copy()

        # Shaping old json into the new one
        props_tmp['id']                           = props['projectName']
        props_tmp['dataProps']['marker_color']    = props['symbol_color']
        props_tmp['dataProps']['marker_size']     = props['symbol_size']
        props_tmp['dataProps']['marker']          = props['symbol']
        props_tmp['dataProps']['curve_color']     = props['curve_color']
        props_tmp['dataProps']['curve_thickness'] = props['curve_thickness']
        props_tmp['dataProps']['curve_style']     = props['curve_style']
        props_tmp['canvasProps']['xaxis']         = props['xaxis']
        props_tmp['canvasProps']['yaxis']         = props['yaxis']
        props_tmp['canvasProps']['title']         = props['title']
        props_tmp['canvasProps']['log_x']         = props['log_x']
        props_tmp['canvasProps']['log_y']         = props['log_y']
        props_tmp['canvasProps']['legend']        = props['legend']
        props_tmp['canvasProps']['grid']          = props['grid']
        props_tmp['canvasProps']['residuals']     = props['residuals']
        props_tmp['canvasProps']['xmin']          = props['xmin']
        props_tmp['canvasProps']['xmax']          = props['xmax']
        props_tmp['canvasProps']['xdiv']          = props['xdiv']
        props_tmp['canvasProps']['ymin']          = props['ymin']
        props_tmp['canvasProps']['ymax']          = props['ymax']
        props_tmp['canvasProps']['ydiv']          = props['ydiv']
        props_tmp['canvasProps']['resmin']        = props['resmin']
        props_tmp['canvasProps']['resmax']        = props['resmax']
        props_tmp['fitProps']['expr']             = props['expr']
        props_tmp['fitProps']['p0']               = props['p0']
        props_tmp['fitProps']['wsx']              = props['wsx']
        props_tmp['fitProps']['wsy']              = props['wsy']
        props_tmp['fitProps']['parameters']       = props['parameters']
        props_tmp['data']                         = pd.read_json(props['data'], dtype=str)

        return props_tmp

    @QtCore.Slot(QtCore.QJsonValue, result=int)
    def save(self, props):
        # If there's no path for saving, saveAs()
        if self.path == '':
            return 1

        # Getting properties
        props                           = props.toVariant()
        props["fitProps"]["parameters"] = self.model._params.valuesdict()

        if platform.system() == "Linux":
            with open(self.path + ".json", 'w', encoding='utf-8') as file:
                json.dump(props, file, ensure_ascii=False, indent=4)
        else:
            with open(self.path, 'w', encoding='utf-8') as file:
                json.dump(props, file, ensure_ascii=False, indent=4)

        return 0
    
    @QtCore.Slot(str, QtCore.QJsonValue)
    def saveAs(self, path, props):
        # Getting path
        self.path = QtCore.QUrl(path).toLocalFile()

        # Getting properties
        props = props.toVariant()
        props['fitProps']['parameters'] = self.model._params.valuesdict()

        if platform.system() == "Linux":
            with open(self.path + ".json", 'w', encoding='utf-8') as file:
                json.dump(props, file, ensure_ascii=False, indent=4)
        else:
            with open(self.path, 'w', encoding='utf-8') as file:
                json.dump(props, file, ensure_ascii=False, indent=4)

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
        if std <= 0:
            self.msg.raiseError("Desvio padrão deve ser sempre maior que zero. Rever desvio padrão.")
            return None
        if ngl >= 1:
            self.msg.raiseError("Número de graus de liberdade deve ser sempre menor que 1. Rever número de graus de liberdade.")
            return None
        s, x, y, x_area, y_area = interpreter_calculator(functionDict[function], methodDict[opt1], nc, ngl, mean, std)
        Plot(self.canvas, x, y, x_area, y_area)
        self.writeCalculator.emit(s)