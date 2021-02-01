# -*- coding: utf-8 -*-
"""
Created on Fri Jan 29 17:10:06 2021

@author: Leonardo Eiji Tamayose & Guilherme Ferrari Fortino 

Model Class

"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from copy import deepcopy
from lmfit.models import ExpressionModel
from lmfit import Parameters

class Model():
    def __init__(self):
        self.data       = None
        self.eixos      = [["Eixo x"], ["Eixo y"], ["Título"]]
        self.exp_model  = ""
        self.model      = None
        self.report_fit = ""
        self.result     = None
        self.coef       = list()
        self.params     = Parameters()
        self.dict       = dict()
        self.mode       = 0
        
    def __str__(self):
        return self.report_fit
        
    def load_data(self, data_path):
        """ Loads the data from a given path """
        
        # Janela que seleciona dados -> path
        
        # Making data frame and renaming columns
        df = pd.read_csv(data_path, sep='\t', header=None, dtype = str).dropna()
        # Making sure dot is a decimal separator
        for i in df.columns:
            df[i] = [x.replace(',', '.') for x in df[i]]
            df[i] = df[i].astype(float)
            
        self.mode = len(df.columns) - 2
        
        # Naming columns
        
        if self.mode == 0:
            df["sy"] = [1]*len(df[0])
            df["sx"] = [1]*len(df[0])
        elif self.mode == 1:
            df["sx"] = [1]*len(df[0])
        df.columns= ['x', 'y', 'sy', 'sx']
            
        self.data = deepcopy(df)
        
    def set_x_axis(self, name = ""):
        """ Set new x label to the graph """
        self.eixos[0] = [name]
    
    def set_y_axis(self, name = ""):
        """ Set new y label to the graph """
        self.eixos[1] = [name]

    def set_title(self, name = ""):
        """ Set new title to the graph """
        self.eixos[2] = [name]
        
    def set_expression(self, exp = ""):
        """ Set new expression to model"""
# =============================================================================
#         if type(exp) != str:
#             print("Expression is not a string. Setting to default")
#             return None
# =============================================================================
        self.exp_model = exp
        self._set_model()
        
    def _set_model(self):
        """ Creates the new model """
        self.model = ExpressionModel(self.exp_model)
        self.coef = list()
        self.params = Parameters()
        for i in self.model.param_names:
            self.coef.append(i)            
            self.params.add(i, value=1)
        
    def fit(self):
        self.result = self.model.fit(data=self.data["y"].to_numpy(), x = self.data["x"].to_numpy(),
                                     params=self.params, scale_covar = False,
                                     weights = 1/self.data["sy"].to_numpy())
        self._set_param_values()
        self._set_report()
        
    def get_params(self):
        ''' Return a dict with parameters as keys and returns a list with [value, uncertainty] '''
        return self.dict
        
    def _set_param_values(self):
        self.dict.clear()
        for i in range(len(self.coef)):
            self.params.add(self.coef[i], self.result.values[self.coef[i]])
            self.dict.update({self.coef[i]: [self.result.values[self.coef[i]], np.sqrt(self.result.covar[i][i])]})
        
    def _set_report(self):
        self.report_fit = ""
        self.report_fit += "\nAjuste: y = %s\n"%self.exp_model
        self.report_fit += "\nNGL  = %d"%(len(self.data["x"]) - len(self.coef))
        self.report_fit += "\nChi² = %f"%self.result.chisqr
        self.report_fit += "\nMatriz de covariância:\n" + str(self.result.covar) + "\n\n"

    def plot_data(self, figsize = None, dpi = 120, size = 1, lw = 1, mstyle = '.', color = 'blue'):
        """ Scatter the data """
        fig = plt.figure(figsize = figsize, dpi = dpi)
        plt.scatter(x = self.data["x"].to_numpy(), y = self.data["y"].to_numpy(), s = size,
                    c = color, marker = mstyle, linewidths = 1)
        plt.errorbar(x = self.data["x"].to_numpy(), y = self.data["y"].to_numpy(),
                     yerr=self.data["sy"].to_numpy(), xerr = self.data["sx"].to_numpy(),
                     fmt = 'ko', ecolor = 'black', capsize = 2, ms = size, elinewidth = 1)
        fig.show()
        
    def get_coefficients(self):
        ''' Deprecated function '''
        return self.coef 
    
    def get_data(self, *args):
        ''' Return data arrays based on mode attribute '''
        if self.mode == 0:
            return self.data["x"].to_numpy(), self.data["y"].to_numpy()
        elif self.mode == 1:
            return self.data["x"].to_numpy(), self.data["y"].to_numpy(), self.data["sy"].to_numpy()
        return self.data["x"].to_numpy(), self.data["y"].to_numpy(), self.data["sy"].to_numpy(), self.data["sx"].to_numpy()
        
    def get_predict(self):
        x = self.data["x"].to_numpy()
        return x, self.result.eval(x = self.data["x"].to_numpy())
        
        