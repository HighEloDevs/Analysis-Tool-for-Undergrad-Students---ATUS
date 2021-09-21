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

import numpy as np
import pandas as pd
from PyQt5.QtGui import QGuiApplication
from PyQt5.QtCore import QObject, QJsonValue, QUrl, QVariant, pyqtSignal, pyqtSlot
from scipy.odr import ODR, Model as SciPyModel, RealData
from lmfit.models import ExpressionModel
from lmfit import Parameters
from copy import deepcopy
from io import StringIO

class Model(QObject):
    """
    Class used for fit.
    """
    # Signals
    fillDataTable   = pyqtSignal(str, str, str, str, str, str, arguments=['x', 'y', 'sy', 'sx', 'filename'])
    fillParamsTable = pyqtSignal(str, float, float, arguments=['param', 'value', 'uncertainty'])
    writeInfos      = pyqtSignal(str, arguments='expr')
    uploadData      = pyqtSignal(QVariant, str, arguments=['data', 'fileName'])

    def __init__(self, messageHandler):
        super().__init__()
        pd.set_option('display.expand_frame_repr', False)
        self._msgHandler = messageHandler
        self._data       = None
        self._data_json  = None
        self._exp_model  = ""
        self._indVar     = "x"
        self._model      = None
        self._report_fit = ""
        self._result     = None
        self._coef       = list()
        self._params     = Parameters()
        self._dict       = dict()
        self._dict2      = dict()
        self._p0         = None
        self.xmin_adj    = 0.
        self.xmax_adj    = 0.
        self._mode       = 0
        self._has_data   = False
        self._isvalid    = False
        self._has_sx     = True
        self._has_sy     = True
        
    def __str__(self):
        return self._report_fit
        
    @pyqtSlot(QJsonValue)
    def loadDataTable(self, data = None):
        """ Getting data from table. """
        df = pd.DataFrame.from_records(data)
        df.columns = ['x', 'y', 'sy', 'sx', 'bool']

        # Removing not chosen rows
        df = df[df['bool'] == 1]
        del df['bool']
        uniqueSi = df["sy"].unique().astype(float)
        if 0. in uniqueSi:
            if len(uniqueSi) > 1:
                self._msgHandler.raiseWarn("Um valor nulo foi encontrado nas incertezas em y, removendo coluna de sy.")
            self._has_sy = False
        uniqueSi = df["sx"].unique().astype(float)
        if 0. in uniqueSi:
            if len(uniqueSi) > 1:
                self._msgHandler.raiseWarn("Um valor nulo foi encontrado nas incertezas em x, removendo coluna de sx.")
            self._has_sx = False

        self._data_json = deepcopy(df)

        # Turn everything into number (str -> number)
        df = df.astype(float)
        self._data     = deepcopy(df)
        self._has_data = True

    @pyqtSlot()
    def loadDataClipboard(self):
        """Pega a tabela de dados do Clipboard."""
        # Instantiating clipboard
        clipboard = QGuiApplication.clipboard()
        clipboardText = clipboard.mimeData().text()
        # try:
            # Creating a dataframe from the string
        df = pd.read_csv(StringIO(clipboardText), sep = '\t', header = None, dtype = str).replace(np.nan, "0")
        # Replacing all commas for dots
        for i in df.columns:
            df[i] = [x.replace(',', '.') for x in df[i]]
            df[i] = df[i].astype(str)
        self.load_data(df=df)
        # except Exception:
        #     self._msgHandler.raiseError("Falha ao carregar clipboard. Rever dados de entrada.")
        #     return None
                
    @pyqtSlot(str)
    def load_data(self, data_path='', df=None, df_array=None):
        """Loads the data from a given path or from a given dataframe."""

        # Name of the loaded file
        fileName = 'Dados Carregados do Projeto'

        # If no dataframe passed, loading data from the given path
        if len(data_path) > 0:
            # Loading from .csv or (.txt and .tsv)
            data_path = QUrl(data_path).toLocalFile()
            if data_path[-3:] == "csv":
                try:
                    df = pd.read_csv(data_path, sep=',', header=None, dtype = str).replace(np.nan, "0")
                except pd.errors.ParserError:
                    self._msgHandler.raiseError("Separação de colunas de arquivos csv são com vírgula (","). Rever dados de entrada.")
                    return None
            else:
                try:
                    df = pd.read_csv(data_path, sep='\t', header=None, dtype = str).replace(np.nan, "0")
                except pd.errors.ParserError:
                    self._msgHandler.raiseError("Separação de colunas de arquivos txt e tsv são com tab. Rever dados de entrada.")
                    return None
            # Getting file name
            fileName = data_path.split('/')[-1]
        elif df is None:
            df = pd.DataFrame.from_records(df_array)
            df.columns = ['x', 'y', 'sy', 'sx', 'bool']
            bools = df['bool'].astype(str)
            del df['bool']
            uniqueSi = df["sy"].unique().astype(float)
            if 0. in uniqueSi:
                if len(uniqueSi) > 1:
                    self._msgHandler.raiseWarn("Um valor nulo foi encontrado nas incertezas em y, removendo coluna de sy.")
                self._has_sy = False
            uniqueSi = df["sx"].unique().astype(float)
            if 0. in uniqueSi:
                if len(uniqueSi) > 1:
                    self._msgHandler.raiseWarn("Um valor nulo foi encontrado nas incertezas em x, removendo coluna de sx.")
                self._has_sx = False
            
        # Saving the dataframe in the class
        self._data_json = deepcopy(df)

        # Applying some filters over the df
        for i in df.columns:
            # Replacing comma for dots
            df[i]              = [x.replace(',', '.') for x in df[i]]
            self._data_json[i] = [x.replace(',', '.') for x in self._data_json[i]]
            try:
                df[i] = df[i].astype(float)
            except ValueError:
                self._msgHandler.raiseError("A entrada de dados só permite entrada de números. Rever arquivo de entrada.")
                return None

        self._has_sx = True
        self._has_sy = True
        self._mode   = len(df.columns) - 2

        # Naming columns
        if self._mode == -1:
            self._has_sy            = not self._has_sy
            self._has_sx            = not self._has_sx
            df["x"]                 = np.arange(len(df), dtype = float)
            self._data_json         = deepcopy(df.astype(str))
            self._data_json.columns = ['y', 'x']
            df["sy"]                = 0.
            df["sx"]                = 0.
        elif self._mode == 0:
            self._has_sy            = not self._has_sy
            self._has_sx            = not self._has_sx
            self._data_json.columns = ['x', 'y']
            df["sy"]                = 0.
            df["sx"]                = 0.
        elif self._mode == 1:
            self._has_sx            = not self._has_sx
            self._data_json.columns = ['x', 'y', 'sy']
            df["sx"]                = 0.
        else:
            try:
                self._data_json.columns = ['x', 'y', 'sy', 'sx']
                uniqueSi = self._data_json["sy"].unique().astype(float)
                if 0. in uniqueSi:
                    if len(uniqueSi) > 1:
                        self._msgHandler.raiseWarn("Um valor nulo foi encontrado nas incertezas em y, removendo coluna de sy.")
                    self._has_sy = False
                uniqueSi = self._data_json["sx"].unique().astype(float)
                if 0. in uniqueSi:
                    if len(uniqueSi) > 1:
                        self._msgHandler.raiseWarn("Um valor nulo foi encontrado nas incertezas em x, removendo coluna de sx.")
                    self._has_sx = False
            except ValueError:
                self._msgHandler.raiseError("Há mais do que 4 colunas. Rever entrada de dados.")
                return None

        # df.columns     = ['x', 'y', 'sy', 'sx']
        self._data     = deepcopy(df)
        self._has_data = True

        self.uploadData.emit(self._data_json.to_dict(orient='list'), fileName)

        # print(self._data_json.to_dict())
        
        # if df_array is None:
        #     if self._has_sx and self._has_sy:
        #         for i in self._data_json.index:
        #             self.fillDataTable.emit(self._data_json["x"][i], self._data_json["y"][i], self._data_json["sy"][i], self._data_json["sx"][i], '1', fileName)
        #     elif self._has_sx:
        #         for i in self._data_json.index:
        #             self.fillDataTable.emit(self._data_json["x"][i], self._data_json["y"][i], str(0), self._data_json["sx"][i], '1', fileName)
        #     elif self._has_sy:
        #         for i in self._data_json.index:
        #             self.fillDataTable.emit(self._data_json["x"][i], self._data_json["y"][i], self._data_json["sy"][i], str(0), '1', fileName)
        #     else:
        #         for i in self._data_json.index:
        #             self.fillDataTable.emit(self._data_json["x"][i], self._data_json["y"][i], str(0), str(0), '1', fileName)
        # else:
        #     if self._has_sx and self._has_sy:
        #         for i in self._data_json.index:
        #             self.fillDataTable.emit(self._data_json["x"][i], self._data_json["y"][i], self._data_json["sy"][i], self._data_json["sx"][i], bools[i], fileName)
        #     elif self._has_sx:
        #         for i in self._data_json.index:
        #             self.fillDataTable.emit(self._data_json["x"][i], self._data_json["y"][i], str(0), self._data_json["sx"][i], bools[i], fileName)
        #     elif self._has_sy:
        #         for i in self._data_json.index:
        #             self.fillDataTable.emit(self._data_json["x"][i], self._data_json["y"][i], self._data_json["sy"][i], str(0), bools[i], fileName)
        #     else:
        #         for i in self._data_json.index:
        #             self.fillDataTable.emit(self._data_json["x"][i], self._data_json["y"][i], str(0), str(0), bools[i], fileName)

    def set_p0(self, p0):
        ''' Coloca os chutes iniciais. '''
        self._p0 = p0.split(",")
        
    def set_expression(self, exp = "", varInd = "x"):
        """ Set new expression to model. """
        self._exp_model = exp
        self._indVar    = varInd
        
    def fit(self, **kargs):
        ''' Interpretador de qual ajuste deve ser feito. '''
        wsx = kargs.pop("wsx", True)
        wsy = kargs.pop("wsy", True)
        # Getting Model
        try:
            self._model = ExpressionModel(self._exp_model + " + 0*%s"%self._indVar, independent_vars=[self._indVar])
        except ValueError:
            self._msgHandler.raiseError("Expressão de ajuste escrita de forma errada. Rever função de ajuste.")
            return None
        except SyntaxError:
            self._msgHandler.raiseError("Expressão de ajuste escrita de forma errada. Rever função de ajuste.")
            return None
        # Getting coefficients
        self._coef = [i for i in self._model.param_names]
        # If there's no p0, everything is set to 1.0
        pi = [0]*len(self._model.param_names)   # Inital values
        if self._p0 is None:
            for i in range(len(self._model.param_names)):
                pi[i] = 1.0
        else:
            for i in range(len(self._model.param_names)):
                try:
                    pi[i] = float(self._p0[i])
                except:
                    pi[i] = 1.0
        # Data
        x, y, sy, sx = self.data
        indices = np.arange(len(self._data.index))
        if self.xmin != self.xmax:
            indices = np.where((self.xmin <= self._data["x"]) & (self.xmax >= self._data["x"]))[0]
        x, y, sy, sx = x.iloc[indices], y.iloc[indices], sy.iloc[indices], sx.iloc[indices]
        data = None
        if self._has_sy and self._has_sx: # Caso com as duas incs
            if wsx == True and wsy == True:
                self.__fit_lm_wy(x, y, pi)
                if (self._result is None) == False:
                    self.__set_param_values_lm_special(x)
                    self.__set_report_lm_special(x)
                else:
                    return None
            elif wsx:
                self.__fit_lm(x, y, sy, pi)
                if (self._result is None) == False:
                    self.__set_param_values_lm(x)
                    self.__set_report_lm(x)
                else:
                    return None
            elif wsy:
                self.__fit_ODR_special(x, y, sx, pi)
                if (self._result is None) == False:
                    self.__set_param_values_ODR(x)
                    self.__set_report_ODR(x)
                else:
                    return None
            else:
                data = RealData(x, y, sx = sx, sy = sy)
                self.__fit_ODR(data, pi)
                if (self._result is None) == False:
                    self.__set_param_values_ODR(x)
                    self.__set_report_ODR(x)
                else:
                    return None
        elif self._has_sy:  # Caso com a incerteza só em y
            if wsy:
                self.__fit_lm_wy(x, y, pi)
                if (self._result is None) == False:
                    self.__set_param_values_lm_special(x)
                    self.__set_report_lm_special(x)
                else:
                    return None
            else:
                self.__fit_lm(x, y, sy, pi)
                if (self._result is None) == False:
                    self.__set_param_values_lm(x)
                    self.__set_report_lm(x)
                else:
                    return None
        elif self._has_sx:  # Caso com a incerteza só em x
            if wsx:
                self.__fit_lm_wy(x, y, pi)
                if (self._result is None) == False:
                    self.__set_param_values_lm_special(x)
                    self.__set_report_lm_special(x)
                else:
                    return None
            else:
                self.__fit_ODR_special(x, y, sx, pi)
                if (self._result is None) == False:
                    self.__set_param_values_ODR(x)
                    self.__set_report_ODR(x)
                else:
                    return None
        else: # Caso sem incertezas
            self.__fit_lm_wy(x, y, pi)
            if (self._result is None) == False:
                self.__set_param_values_lm_special(x)
                self.__set_report_lm_special(x)
            else:
                return None
        params = self.get_params()
        keys   = list(params.keys())
        for i in range(len(keys)):
            self.fillParamsTable.emit(keys[i], params[keys[i]][0], params[keys[i]][1])
        self.writeInfos.emit(self._report_fit)

    def __fit_ODR(self, data, pi):
        '''Fit com ODR.'''
        def f(a, x):
            param = Parameters()
            for i in range(len(a)):
                param.add(self._model.param_names[i], value=a[i])
            return eval("self._model.eval(%s=x, params=param)"%self._indVar,
             {'x': x, 'param': param, 'self': self})
        model = SciPyModel(f)
        try:
            myodr = ODR(data, model, beta0 = pi, maxit = 250)
            self._result = myodr.run()
        except TypeError:
            self._msgHandler.raiseError("Expressão de ajuste escrita de forma errada. Rever função de ajuste.")
            return None
    
    def __fit_ODR_special(self, x_orig, y, sx, pi):
        '''Fit com ODR quando só há incertezas em x.'''
        # x = np.copy(x_orig)
        def f(a, x):
            param = Parameters()
            for i in range(len(a)):
                param.add(self._model.param_names[i], value=a[i])
            return eval("self._model.eval(%s=x, params=param)"%self._indVar, None,
                {'x': x, 'param': param, 'self': self})
        # data  = RealData(x, y, sx = sx)
        # model = SciPyModel(f)
        # try:
        #     myodr = ODR(data, model, beta0 = pi, maxit = 40)
        #     self._result = myodr.run()
        # except TypeError:
        #     self._msgHandler.raiseError("Expressão de ajuste escrita de forma errada. Rever função de ajuste.")
        #     self._result = None
        #     return None
        # self._params = Parameters()
        # for i in range(len(self._coef)):
        #     self._params.add(self._coef[i], self._result.beta[i])
        # sy = np.zeros(len(self._data["x"]), dtype = float)
        # for i, x in enumerate(self._data["x"]):
        #     x_var = np.array([x + self._data["sx"].iloc[i], x - self._data["sx"].iloc[i]])
        #     y_prd = eval("self._model.eval(%s = x, params = self._params)"%self._indVar, None,
        # {'x': x, 'self': self})
        #     y_var = eval("self._model.eval(%s = x_var, params = self._params)"%self._indVar, None,
        # {'x_var': x_var, 'self': self})
        #     sy[i] = np.abs(y_var - y_prd).mean()
        # sy = sy.astype(float)/1000
        x  = np.copy(x_orig)
        sy = np.array([1e-50]*len(x))
        data  = RealData(x, y, sx = sx, sy = sy)
        model = SciPyModel(f)
        try:
            myodr = ODR(data, model, beta0 = pi, maxit = 100)
            self._result = myodr.run()
        except TypeError:
            self._msgHandler.raiseError("Expressão de ajuste escrita de forma errada. Rever função de ajuste.")
            self._result = None
            return None
        sy = np.zeros(len(self._data["x"]), dtype = float)
        for i, x in enumerate(self._data["x"]):
            x_var = np.array([x + self._data["sx"].iloc[i], x - self._data["sx"].iloc[i]])
            y_prd = eval("self._model.eval(%s = x, params = self._params)"%self._indVar, None,
        {'x': x, 'self': self})
            y_var = eval("self._model.eval(%s = x_var, params = self._params)"%self._indVar, None,
        {'x_var': x_var, 'self': self})
            sy[i] = np.abs(y_var - y_prd).mean()
        x_var = self._data["x"]
        self._result.sum_square = np.sum(((eval("self._model.eval(%s = x_var, params = self._params)"%self._indVar, None,
        {'x_var': x_var, 'self': self}) - self._data["y"].to_numpy())/sy)**2)

    def __fit_lm(self, x, y, sy, pi):
        '''Fit com MMQ.'''
        params = Parameters()
        for i in range(len(self._coef)):
            params.add(self._coef[i], pi[i])
        try:
            self._result = eval("self._model.fit(data = y, %s = x, weights = 1/sy, params = params, scale_covar=False, max_nfev = 250)"%self._indVar, None,
            {'y': y, 'x': x, 'params': params, 'self': self, 'sy': sy})
        except ValueError:
            self._msgHandler.raiseError("A função ajustada gera valores não numéricos, rever ajuste.")
            # A função ajustada gera valores não numéricos, rever ajuste.
            return None
        except TypeError:
            self._msgHandler.raiseError("A função ajustada possui algum termo inválido, rever ajuste.")
            # A função ajustada possui algum termo inválido, rever ajuste.
            return None
        if self._result.covar is None:
            self._msgHandler.raiseError("A função ajustada não convergiu, rever ajuste.")
            self._result = None
            return None
    
    def __fit_lm_wy(self, x, y, pi):
        '''Fit com MMQ quando não há incertezas.'''
        params = Parameters()
        for i in range(len(self._coef)):
            params.add(self._coef[i], pi[i])
        try:
            self._result = eval("self._model.fit(data = y, %s = x, params = params, scale_covar=False, max_nfev = 250)"%self._indVar, None,
            {'y': y, 'x': x, 'params': params, 'self': self})
        except ValueError:
            self._msgHandler.raiseError("A função ajustada gera valores não numéricos, rever ajuste.")
            # A função ajustada gera valores não numéricos, rever ajuste.
            return None
        except TypeError:
            self._msgHandler.raiseError("A função ajustada possui algum termo inválido, rever ajuste.")
            # A função ajustada possui algum termo inválido, rever ajuste.
            return None
        if self._result.covar is None:
            self._msgHandler.raiseError("A função ajustada não convergiu, rever ajuste.")
            self._result = None
            return None
        
    def get_params(self):
        '''Retorna um dicionário onde as keys são os parâmetros e que retornam uma lista com [valor, incerteza].'''
        return self._dict

    def __set_param_values_lm(self, x):
        '''Constrói o dicionário e o Parameters dos valores do ajuste.'''
        self._dict.clear()
        self._params = Parameters()
        # inc_cons     = np.sqrt(self._result.chisqr/ngl)
        # inc_cons_q   = inc_cons**2
        for i in range(len(self._coef)):
            self._params.add(self._coef[i], self._result.values[self._coef[i]])
            # self._dict.update({self._coef[i]: [self._result.values[self._coef[i]], np.sqrt(self._result.covar[i, i])*inc_cons]})
            self._dict.update({self._coef[i]: [self._result.values[self._coef[i]], np.sqrt(self._result.covar[i, i])]})

    def __set_param_values_lm_special(self, x):
        '''Constrói o dicionário e o Parameters dos valores do ajuste, quando não há incertezas.'''
        self._dict.clear()
        self._dict2.clear()
        self._params = Parameters()
        ngl          = len(x) - len(self._coef)
        inc_cons     = np.sqrt(self._result.chisqr/ngl)
        for i in range(len(self._coef)):
            self._params.add(self._coef[i], self._result.values[self._coef[i]])
            self._dict.update({self._coef[i]: [self._result.values[self._coef[i]], np.sqrt(self._result.covar[i, i])*inc_cons]})
            self._dict2.update({self._coef[i]: [self._result.values[self._coef[i]], np.sqrt(self._result.covar[i, i])]})

    def __set_param_values_ODR(self, x):
        '''Constrói o dicionário e o Parameters dos valores do ajuste.'''
        self._dict.clear()
        self._params = Parameters()
        for i in range(len(self._coef)):
            self._params.add(self._coef[i], self._result.beta[i])
            self._dict.update({self._coef[i]: [self._result.beta[i], np.sqrt(self._result.cov_beta[i, i])]})

    def __set_report_lm(self, x):
        '''Constrói a string com os resultados.'''
        self._report_fit  = ""
        self._report_fit += "\nAjuste: y = %s\n"%self._exp_model
        self._report_fit += "\nNGL  = %d"%(len(x) - len(self._coef))
        self._report_fit += "\nChi² = %f\n"%self._result.chisqr
        self._report_fit += "\nMatriz de covariância:\n\n" + self.matprint(self._result.covar) + "\n"
        lista             = list(self._params.keys())
        matriz_corr       = np.zeros((len(self._result.covar), len(self._result.covar)))
        z                 = range(len(matriz_corr))
        for i in z:
            for j in z:
                matriz_corr[i, j] = self._result.covar[i, j]/(self._dict[lista[i]][1]*self._dict[lista[j]][1])
        matriz_corr       = matriz_corr.round(3)
        self._report_fit += "\nMatriz de correlação:\n\n" + self.matprint(matriz_corr, ".3f") + "\n\n"
        self._report_fit += self.paramsPrint()
        self._report_fit += "\n"
        self._isvalid     = True
    
    def __set_report_lm_special(self, x):
        '''Constrói a string com os resultados, neste caso quando não há incertezas.'''
        ngl               = len(x) - len(self._coef)
        inc_considerada   = np.sqrt(self._result.chisqr/ngl)
        inc_considerada_q = inc_considerada**2
        self._report_fit  = ""
        self._report_fit += "\nAjuste: y = %s\n"%self._exp_model
        self._report_fit += "\nNGL  = %d"%(ngl)
        self._report_fit += "\nSomatória dos resíduos absolutos ao quadrado = %f\n"%self._result.chisqr
        self._report_fit += "Incerteza considerada = %f\n"%inc_considerada
        try:
            self._report_fit += "\nMatriz de covariância:\n\n" + self.matprint(self._result.covar*inc_considerada_q) + "\n"
            lista             = list(self._params.keys())
            matriz_corr       = np.zeros((len(self._result.covar), len(self._result.covar)))
            z                 = range(len(matriz_corr))
            for i in z:
                for j in z:
                    matriz_corr[i, j] = self._result.covar[i, j]/(self._dict2[lista[i]][1]*self._dict2[lista[j]][1])
            matriz_corr       = matriz_corr.round(3)
            self._report_fit += "\nMatriz de correlação:\n\n" + self.matprint(matriz_corr, ".3f") + "\n\n"
            self._report_fit += self.paramsPrint2(inc_considerada)
            self._report_fit += "\n"
            self._isvalid     = True
        except TypeError:
            self._msgHandler.raiseError("A função ajustada provavelmente não possui parâmetros para serem ajustados. Rever ajuste.")
            return None

    def __set_report_ODR(self, x):
        ''' Constrói a string com os resultados. '''
        self._report_fit  = ""
        self._report_fit += "\nAjuste: y = %s\n"%self._exp_model
        self._report_fit += "\nNGL  = %d"%(len(x) - len(self._coef))
        self._report_fit += "\nChi² = %f\n"%self._result.sum_square
        self._report_fit += "\nMatriz de covariância:\n\n" + self.matprint(self._result.cov_beta) + "\n"
        lista             = list(self._params.keys())
        matriz_corr       = np.zeros((len(self._result.cov_beta), len(self._result.cov_beta)))
        z                 = range(len(matriz_corr))
        for i in z:
            for j in z:
                matriz_corr[i, j] = self._result.cov_beta[i, j]/(self._dict[lista[i]][1]*self._dict[lista[j]][1])
        matriz_corr       = matriz_corr.round(3)
        self._report_fit += "\nMatriz de correlação:\n\n" + self.matprint(matriz_corr, ".3f") + "\n\n"
        self._report_fit += self.paramsPrint()
        self._report_fit += "\n"
        self._isvalid     = True

    def __set_report_ODR_special(self, x):
        '''Constrói a string com os resultados, neste caso quando só há incertezas em x.'''
        self._report_fit  = ""
        self._report_fit += "\nAjuste: y = %s\n"%self._exp_model
        self._report_fit += "\nNGL  = %d"%(len(x) - len(self._coef))
        self._report_fit += "\nChi² = %f\n"%self._result.sum_square
        self._report_fit += "\nMatriz de covariância:\n\n" + self.matprint(self._result.cov_beta) + "\n"
        lista             = list(self._params.keys())
        matriz_corr       = np.zeros((len(self._result.cov_beta), len(self._result.cov_beta)))
        z                 = range(len(matriz_corr))
        for i in z:
            for j in z:
                matriz_corr[i, j] = self._result.cov_beta[i, j]/(self._dict[lista[i]][1]*self._dict[lista[j]][1])
        matriz_corr       = matriz_corr.round(3)
        self._report_fit += "\nMatriz de correlação:\n\n" + self.matprint(matriz_corr, ".3f") + "\n\n"
        self._report_fit += self.paramsPrint()
        self._report_fit += "\n"
        self._isvalid     = True
        

    @property
    def coefficients(self):
        '''Retorna uma lista com os nomes dos coeficientes.'''
        return self._coef

    @property
    def xmin(self):
        return self.xmin_adj
    
    @xmin.setter
    def xmin(self, valor):
        self.xmin_adj = valor
    
    @property
    def xmax(self):
        return self.xmax_adj
    
    @xmax.setter
    def xmax(self, valor):
        self.xmax_adj = valor
    
    @property
    def data(self, *args):
        '''Retorna x, y, sx e sy.'''
        return self._data["x"], self._data["y"], self._data["sy"], self._data["sx"]
        
    @property
    def residuo(self):
        '''Retorna os valores de y_i - f(x_i).'''
        return self._data["y"] - self._model.eval(x = self._data["x"])

    @property
    def residuoDummy(self):
        '''Retorna os valores de y_i - f(x_i).'''
        # self._coef = [i for i in self._model.param_names]
        # If there's no p0, everything is set to 1.0
        # pi = [0]*len(self._model.param_names)   # Inital values
        # if self._p0 is None:
        #     for i in range(len(self._model.param_names)):
        #         pi[i] = 1.0
        # else:
        #     for i in range(len(self._model.param_names)):
        #         try:
        #             pi[i] = float(self._p0[i])
        #         except:
        #             pi[i] = 1.0
        # paramss = Parameters()
        # for i in range(len(self._coef)):
        #     paramss.add(self._coef[i], pi[i])
        return self._data["y"] - eval("self._model.eval(%s = self._data['x'], params = self._params)"%self._indVar, None,
        {'self': self})

    def get_predict(self, fig, x_min = None, x_max = None):
        '''Retorna a previsão do modelo.'''
        x_plot = np.linspace(x_min, x_max, int(fig.get_size_inches()[0]*fig.dpi*1.75))
        return x_plot, eval("self._model.eval(%s = x_plot, params = self._params)"%self._indVar, None,
        {'x_plot': x_plot, 'self': self})
    
    def get_predict_log(self, fig, x_min = None, x_max = None):
        '''Retorna a previsão do modelo.'''
        x_plot = np.logspace(np.log10(x_min), np.log10(x_max), int(fig.get_size_inches()[0]*fig.dpi*2.1))
        return x_plot, eval("self._model.eval(%s = x_plot, params = self._params)"%self._indVar, None,
        {'x_plot': x_plot, 'self': self})
    
    def predictInc(self, wsx):
        if self._has_sx and (wsx == False) and self._has_sy:
            sy = np.zeros(len(self._data["x"]), dtype = float)
            for i, x in enumerate(self._data["x"]):
                x_var = np.array([x + self._data["sx"].iloc[i], x - self._data["sx"].iloc[i]])
                y_prd = eval("self._model.eval(%s = x, params = self._params)"%self._indVar, None,
            {'x': x, 'self': self})
                y_var = eval("self._model.eval(%s = x_var, params = self._params)"%self._indVar, None,
            {'x_var': x_var, 'self': self})
                sy[i] = np.abs(y_var - y_prd).mean()
                sy[i] = np.sqrt(self._data["sy"].iloc[i]**2 + sy[i]**2)
            return sy
        elif self._has_sx and (wsx == False) and self._has_sy == False:
            sy = np.zeros(len(self._data["x"]), dtype = float)
            for i, x in enumerate(self._data["x"]):
                x_var = np.array([x + self._data["sx"].iloc[i], x - self._data["sx"].iloc[i]])
                y_prd = eval("self._model.eval(%s = x, params = self._params)"%self._indVar, None,
            {'x': x, 'self': self})
                y_var = eval("self._model.eval(%s = x_var, params = self._params)"%self._indVar, None,
            {'x_var': x_var, 'self': self})
                sy[i] = np.abs(y_var - y_prd).mean()
            return sy
        return self._data["sy"]
    
    def createDummyModel(self):
        try:
            self._model = ExpressionModel(self._exp_model + " + 0*%s"%self._indVar, independent_vars=[self._indVar])
        except ValueError:
            self._msgHandler.raiseError("Expressão de ajuste escrita de forma errada. Rever função de ajuste.")
            return None
        except SyntaxError:
            self._msgHandler.raiseError("Expressão de ajuste escrita de forma errada. Rever função de ajuste.")
            return None
        pi = [0]*len(self._model.param_names)
        if self._p0 is None:
            for i in range(len(self._model.param_names)):
                pi[i] = 1.0
        else:
            for i in range(len(self._model.param_names)):
                try:
                    pi[i] = float(self._p0[i])
                except:
                    pi[i] = 1.0
        self._params = Parameters()
        self._coef = [i for i in self._model.param_names]
        for i in range(len(self._coef)):
            self._params.add(self._coef[i], pi[i])
        self._isvalid = True


    def matprint(self, mat, fmt="f"):
        col_maxes = [max([len(("{:"+fmt+"}").format(x)) for x in col]) for col in mat.T]
        matrix    = ""
        for x in mat:
            for i, y in enumerate(x):
                matrix += ("{:"+str(col_maxes[i])+fmt+"}").format(y) + "  "
            matrix += "\n"
        return matrix
    
    def paramsPrint(self):
        df         = pd.DataFrame(self._dict)
        df         = df.transpose()
        df.columns = ["Valor", "|    Incerteza"]
        df.index   = self._coef
        return str(df)

    def paramsPrint2(self, inc_considerada):
        df         = pd.DataFrame(self._dict2)
        df         = df.transpose()
        df.columns = ["Valor", "|    Incerteza"]
        df["|    Incerteza"] = df["|    Incerteza"]*inc_considerada
        df.index   = self._coef
        return str(df)

    def reset(self):
        self._data       = None
        self._data_json  = None
        self._exp_model  = ""
        self._model      = None
        self._report_fit = ""
        self._result     = None
        self._coef       = list()
        self._params     = Parameters()
        self._dict       = dict()
        self._dict2      = dict()
        self._p0         = None
        self.xmin_adj    = 0.
        self.xmax_adj    = 0.
        self._mode       = 0
        self._has_data   = False
        self._isvalid    = False
        self._has_sx     = True
        self._has_sy     = True