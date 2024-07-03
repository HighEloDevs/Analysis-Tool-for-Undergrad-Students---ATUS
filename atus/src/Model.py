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
from __future__ import annotations

import numpy as np
import pandas as pd

# from PyQt5.QtGui import QGuiApplication
from PyQt5.QtCore import (
    QObject,
    # QJsonValue,
    # QUrl,
    QVariant,
    pyqtSignal,
    pyqtSlot,
)
from scipy.odr import ODR, Model as SciPyModel, RealData
from lmfit.models import ExpressionModel
from lmfit import Parameters
from .MessageHandler import MessageHandler

# from copy import deepcopy
# from io import StringIO
import re


class Model(QObject):
    """
    Class used for fit.
    """

    # Signals
    # fillDataTable = pyqtSignal(
    #     str,
    #     str,
    #     str,
    #     str,
    #     str,
    #     str,
    #     arguments=["x", "y", "sy", "sx", "filename"],
    # )
    fillParamsTable = pyqtSignal(
        str, float, float, arguments=["param", "value", "uncertainty"]
    )
    writeInfos = pyqtSignal(str, arguments="expr")
    uploadData = pyqtSignal(QVariant, str, arguments=["data", "fileName"])

    def __init__(self, messageHandler):
        super().__init__()
        pd.set_option("display.expand_frame_repr", False)
        self._msg_handler: MessageHandler = messageHandler
        self._data: pd.DataFrame = None
        self._data_json: pd.DataFrame = None
        self._exp_model: str = ""
        self._ind_var: str = "x"
        self._model = None
        self._report_fit = ""
        self._mat_corr = ""
        self._mat_cov = ""
        self._dict_param = {}
        self._result = None
        self._coef: list[str] = []
        self._par_var = []
        self._params = Parameters()
        self._dict = {}
        self._dict2 = {}
        self._p0: list[str] = None
        self.xmin_adj = 0.0
        self.xmax_adj = 0.0
        self._has_data = False
        self._isvalid = False
        self._has_sx = True
        self._has_sy = True
        self._indices = []

    def __str__(self):
        return self._report_fit

    def set_p0(self, p0: str):
        """Coloca os chutes iniciais."""
        self._p0 = p0.replace(" ", "").split(",")

    def set_expression(self, exp="", ind_var="x"):
        """Set new expression to model."""
        self._exp_model = exp
        self._ind_var = ind_var

    @property
    def data(self):
        """Retorna x, y, sx e sy."""
        return (
            self._data["x"],
            self._data["y"],
            self._data["sy"],
            self._data["sx"],
        )

    @data.setter
    def data(self, data):
        self._data = data

    def _create_model(self) -> bool:
        """Cria o modelo de ajuste."""
        try:
            self._model = ExpressionModel(
                self._exp_model + f" + 0*{self._ind_var}",
                independent_vars=[self._ind_var],
            )
            return True
        except ValueError:
            self._msg_handler.raise_error(
                "Função de ajuste escrita de forma errada. Rever função de ajuste."
            )
            return False
        except SyntaxError:
            self._msg_handler.raise_error("Erro de sintaxe. Rever função de ajuste.")
            return False

    def fit(self, **kargs):
        """Interpretador de qual ajuste deve ser feito."""
        wsx = kargs.pop("wsx", True)
        wsy = kargs.pop("wsy", True)

        # Getting Model
        if not self._create_model():
            return None
        # Getting coefficients
        self._coef = [i for i in self._model.param_names]
        # Data
        x, y, sy, sx = self.data
        self._indices = np.arange(len(self._data.index))
        if self.xmin != self.xmax:
            self._indices = np.where(
                (self.xmin <= self._data["x"]) & (self.xmax >= self._data["x"])
            )[0]
        x, y, sy, sx = (
            x.iloc[self._indices].to_numpy(),
            y.iloc[self._indices].to_numpy(),
            sy.iloc[self._indices].to_numpy(),
            sx.iloc[self._indices].to_numpy(),
        )
        data = None
        if self._has_sy and self._has_sx:  # Caso com as duas incs
            if (wsx is True) and (wsy is True):
                self.__fit_lm_wy(x, y)
                if (self._result is None) is False:
                    self.__set_param_values_lm_special(x)
                    self.__set_report_lm_special(x)
                else:
                    return None
            elif wsx:
                self.__fit_lm(x, y, sy)
                if (self._result is None) is False:
                    self.__set_param_values_lm(x)
                    self.__set_report_lm(x)
                else:
                    return None
            elif wsy:
                self.__fit_ODR_special(x, y, sx)
                if (self._result is None) is False:
                    self.__set_param_values_ODR(x)
                    self.__set_report_ODR(x)
                else:
                    return None
            else:
                data = RealData(x, y, sx=sx, sy=sy)
                self.__fit_ODR(data)
                if (self._result is None) is False:
                    self.__set_param_values_ODR(x)
                    self.__set_report_ODR(x)
                else:
                    return None
        elif self._has_sy:  # Caso com a incerteza só em y
            if wsy:
                self.__fit_lm_wy(x, y)
                if (self._result is None) is False:
                    self.__set_param_values_lm_special(x)
                    self.__set_report_lm_special(x)
                else:
                    return None
            else:
                self.__fit_lm(x, y, sy)
                if (self._result is None) is False:
                    self.__set_param_values_lm(x)
                    self.__set_report_lm(x)
                else:
                    return None
        elif self._has_sx:  # Caso com a incerteza só em x
            if wsx:
                self.__fit_lm_wy(x, y)
                if (self._result is None) is False:
                    self.__set_param_values_lm_special(x)
                    self.__set_report_lm_special(x)
                else:
                    return None
            else:
                self.__fit_ODR_special(x, y, sx)
                if (self._result is None) is False:
                    self.__set_param_values_ODR(x)
                    self.__set_report_ODR(x)
                else:
                    return None
        else:  # Caso sem incertezas
            self.__fit_lm_wy(x, y)
            if (self._result is None) is False:
                self.__set_param_values_lm_special(x)
                self.__set_report_lm_special(x)
            else:
                return None
        params = self.get_params()
        keys = list(params.keys())
        for i in range(len(keys)):
            self.fillParamsTable.emit(keys[i], params[keys[i]][0], params[keys[i]][1])
        self.writeInfos.emit(self._report_fit)

    def __make_parameters_lm(self):
        """Constrói os parâmetros para ajuste com o lmfit."""
        self._params = Parameters()
        if self._p0 is None:
            for i in range(len(self._coef)):
                self._params.add(self._coef[i], 1.0)
        else:
            coefs = {c: [1, True, -np.inf, np.inf] for c in self._coef}
            coefs_2 = {
                c: False for c in self._coef
            }  # Para evitar de substituir atribuição de parâmetros
            for i in range(len(self._coef)):
                try:
                    res = re.match(
                        r"((?P<parameter>.+?)=)?(?P<value>[\d\-\.\@]+)(\[((?P<lim_inf>.+?)?;(?P<lim_sup>.+?))\])?",
                        self._p0[i],
                    ).groupdict()
                    if res["parameter"] is not None:
                        valor = res["value"].replace("@", "")
                        var = res["parameter"]
                        if var in coefs:
                            lim_inf = (
                                -np.inf
                                if res["lim_inf"] is None
                                else float(res["lim_inf"])
                            )
                            lim_sup = (
                                np.inf
                                if res["lim_sup"] is None
                                else float(res["lim_sup"])
                            )
                            coefs[var] = [
                                float(valor),
                                "@" not in res["value"],
                                lim_inf,
                                lim_sup,
                            ]
                            coefs_2[var] = True
                    else:
                        if coefs_2[self._coef[i]] is False and self._coef[i] in coefs:
                            lim_inf = (
                                -np.inf
                                if res["lim_inf"] is None
                                else float(res["lim_inf"])
                            )
                            lim_sup = (
                                np.inf
                                if res["lim_sup"] is None
                                else float(res["lim_sup"])
                            )
                            coefs[self._coef[i]] = [
                                float(res["value"].replace("@", "")),
                                "@" not in res["value"],
                                lim_inf,
                                lim_sup,
                            ]
                            coefs_2[self._coef[i]] = True
                except Exception:
                    if coefs_2[self._coef[i]] is False:
                        coefs[self._coef[i]] = [1, True, -np.inf, np.inf]
                        coefs_2[self._coef[i]] = True
            for nome in coefs.keys():
                self._params.add(
                    nome,
                    coefs[nome][0],
                    vary=coefs[nome][1],
                    min=coefs[nome][2],
                    max=coefs[nome][3],
                )

    def __make_parameters_odr(
        self,
    ) -> tuple[list[float], list[bool], list[float], list[float]]:
        """Constrói os parâmetros para ajuste com o ODR."""
        pi: list[float] = [1.0] * len(self._coef)
        fixed: list[bool] = [1] * len(
            self._coef
        )  # TODO: confirmar mudança para boolean
        arr_lim_inf: list[float] = [-np.inf] * len(self._coef)
        arr_lim_sup: list[float] = [np.inf] * len(self._coef)
        aux = {c: i for i, c in enumerate(self._coef)}
        if self._p0 is None:
            pass
        else:
            coefs = {c: [1, True, -np.inf, np.inf] for c in self._coef}
            coefs_2 = {
                c: False for c in self._coef
            }  # Para evitar de substituir atribuição de parâmetros
            for i in range(len(self._coef)):
                try:
                    res = re.match(
                        r"((?P<parameter>.+?)=)?(?P<value>[\d\-\.\@]+)(\[((?P<lim_inf>.+?)?;(?P<lim_sup>.+?))\])?",
                        self._p0[i],
                    ).groupdict()
                    if res["parameter"] is not None:
                        valor = res["value"].replace("@", "")
                        var = res["parameter"]
                        lim_inf = (
                            -np.inf if res["lim_inf"] is None else float(res["lim_inf"])
                        )
                        lim_sup = (
                            np.inf if res["lim_sup"] is None else float(res["lim_sup"])
                        )
                        if var in coefs:
                            coefs[var] = [
                                float(valor),
                                "@" not in res["value"],
                                lim_inf,
                                lim_sup,
                            ]
                            coefs_2[var] = True
                    else:
                        if coefs_2[self._coef[i]] is False and self._coef[i] in coefs:
                            lim_inf = (
                                -np.inf
                                if res["lim_inf"] is None
                                else float(res["lim_inf"])
                            )
                            lim_sup = (
                                np.inf
                                if res["lim_sup"] is None
                                else float(res["lim_sup"])
                            )
                            coefs[self._coef[i]] = [
                                float(res["value"].replace("@", "")),
                                "@" not in res["value"],
                                lim_inf,
                                lim_sup,
                            ]
                            coefs_2[self._coef[i]] = True
                except Exception:
                    if coefs_2[self._coef[i]] is False:
                        coefs[self._coef[i]] = [1, True, -np.inf, np.inf]
                        coefs_2[self._coef[i]] = True
            for i, nome in enumerate(self._coef):
                pi[aux[nome]] = coefs[nome][0]
                fixed[aux[nome]] = coefs[nome][1]
                arr_lim_inf[aux[nome]] = coefs[nome][2]
                arr_lim_sup[aux[nome]] = coefs[nome][3]
        self._par_var = []
        for i, parametro in enumerate(self._coef):
            if fixed[i] > 0:
                self._par_var.append(parametro)
        return pi, fixed, arr_lim_inf, arr_lim_sup

    def __fit_ODR(self, data):
        """Fit com ODR."""
        pi, fixed, lim_inf, lim_sup = self.__make_parameters_odr()

        def f(a, x):
            param = Parameters()
            for i in range(len(a)):
                param.add(
                    self._model.param_names[i],
                    value=a[i],
                    vary=fixed[i],
                    min=lim_inf[i],
                    max=lim_sup[i],
                )
            return eval(
                f"self._model.eval({self._ind_var}=x, params=param)",
                {"x": x, "param": param, "self": self},
            )

        model = SciPyModel(f)
        try:
            myodr = ODR(data, model, beta0=pi, maxit=200, ifixb=fixed)
            self._result = myodr.run()
        except TypeError:
            self._msg_handler.raise_error(
                "Expressão de ajuste escrita de forma errada. Rever função de ajuste."
            )
            return None

    def __fit_ODR_special(self, x_orig, y, sx):
        """Fit com ODR quando só há incertezas em x."""
        pi, fixed, lim_inf, lim_sup = self.__make_parameters_odr()

        def f(a, x):
            param = Parameters()
            for i in range(len(a)):
                param.add(
                    self._model.param_names[i],
                    value=a[i],
                    vary=fixed[i],
                    min=lim_inf[i],
                    max=lim_sup[i],
                )
            return eval(
                f"self._model.eval({self._ind_var}=x, params=param)",
                {"x": x, "param": param, "self": self},
            )

        # data  = RealData(x, y, sx = sx)
        # model = SciPyModel(f)
        # try:
        #     myodr = ODR(data, model, beta0 = pi, maxit = 40)
        #     self._result = myodr.run()
        # except TypeError:
        #     self._msg_handler.raise_error("Expressão de ajuste escrita de forma errada. Rever função de ajuste.")
        #     self._result = None
        #     return None
        # self._params = Parameters()
        # for i in range(len(self._coef)):
        #     self._params.add(self._coef[i], self._result.beta[i])
        # sy = np.zeros(len(self._data["x"]), dtype = float)
        # for i, x in enumerate(self._data["x"]):
        #     x_var = np.array([x + self._data["sx"].iloc[i], x - self._data["sx"].iloc[i]])
        #     y_prd = eval("self._model.eval(%s = x, params = self._params)"%self._ind_var, None,
        # {"x": x, "self": self})
        #     y_var = eval("self._model.eval(%s = x_var, params = self._params)"%self._ind_var, None,
        # {"x_var": x_var, "self": self})
        #     sy[i] = np.abs(y_var - y_prd).mean()
        # sy = sy.astype(float)/1000
        x = np.copy(x_orig)
        sy = np.array([1e-50] * len(x), dtype=float)
        data = RealData(x, y, sx=sx, sy=sy)
        model = SciPyModel(f)
        try:
            myodr = ODR(data, model, beta0=pi, maxit=100, ifixb=fixed)
            self._result = myodr.run()
        except TypeError as e:
            print(e)
            self._msg_handler.raise_error(
                "Expressão de ajuste escrita de forma errada. Rever função de ajuste."
            )
            self._result = None
            return None
        sy = np.zeros(len(self._data["x"]), dtype=float)
        for i, x in enumerate(self._data["x"]):
            x_var = np.array(
                [x + self._data["sx"].iloc[i], x - self._data["sx"].iloc[i]]
            )
            y_prd = eval(
                f"self._model.eval({self._ind_var} = x, params = self._params)",
                None,
                {"x": x, "self": self},
            )
            y_var = eval(
                f"self._model.eval({self._ind_var} = x_var, params = self._params)",
                None,
                {"x_var": x_var, "self": self},
            )
            sy[i] = np.abs(y_var - y_prd).mean()
        x_var = self._data["x"]
        self._result.sum_square = np.sum(
            (
                (
                    eval(
                        f"self._model.eval({self._ind_var} = x_var, params = self._params)",
                        None,
                        {"x_var": x_var.to_numpy(), "self": self},
                    )
                    - self._data["y"].to_numpy()
                )
                / sy
            )
            ** 2
        )

    def __fit_lm(self, x, y, sy):
        """Fit com MMQ."""
        self.__make_parameters_lm()
        try:
            self._result = eval(
                f"self._model.fit(data = y, {self._ind_var} = x, weights = 1/sy, params = params, scale_covar=False, max_nfev = 250)",
                None,
                {
                    "y": y,
                    "x": x,
                    "params": self._params,
                    "self": self,
                    "sy": sy,
                },
            )
        except ValueError:
            self._msg_handler.raise_error(
                "A função ajustada gera valores não numéricos, rever ajuste e/ou parâmetros inciais."
            )
            return None
        except TypeError:
            self._msg_handler.raise_error(
                "A função ajustada possui algum termo inválido, rever ajuste e/ou parâmetros inciais."
            )
            return None
        if self._result.covar is None:
            self._msg_handler.raise_error(
                "A função ajustada não convergiu, rever ajuste e/ou parâmetros inciais."
            )
            self._result = None
            return None

    def __fit_lm_wy(self, x, y):
        """Fit com MMQ quando não há incertezas."""
        self.__make_parameters_lm()
        try:
            self._result = eval(
                f"self._model.fit(data = y, {self._ind_var} = x, params = self._params, scale_covar=False, max_nfev = 250)",
                None,
                {"y": y, "x": x, "self": self},
            )
        except ValueError:
            self._msg_handler.raise_error(
                "A função ajustada gera valores não numéricos, rever ajuste e/ou parâmetros inciais."
            )
            return None
        except TypeError:
            self._msg_handler.raise_error(
                "A função ajustada possui algum termo inválido, rever ajuste e/ou parâmetros inciais."
            )
            return None
        if self._result.covar is None:
            self._msg_handler.raise_error(
                "A função ajustada não convergiu, rever ajuste e/ou parâmetros inciais."
            )
            self._result = None
            return None

    def get_params(self):
        """Retorna um dicionário onde as keys são os parâmetros e que retornam uma lista com [valor, incerteza]."""
        return self._dict

    def __set_param_values_lm(self, x):
        """Constrói o dicionário e o Parameters dos valores do ajuste."""
        self._dict.clear()
        self._dict_param.clear()
        self._params = Parameters()
        self._par_var = []
        for i in list(self._result.params.keys()):
            if self._result.params[i].vary:
                self._par_var.append(i)
        for i, param_name in enumerate(self._par_var):
            self._params.add(param_name, self._result.values[param_name])
            # self._dict.update({self._coef[i]: [self._result.values[self._coef[i]], np.sqrt(self._result.covar[i, i])*inc_cons]})
            self._dict.update(
                {
                    param_name: [
                        self._result.values[param_name],
                        np.sqrt(self._result.covar[i, i]),
                    ]
                }
            )
            self._dict_param.update(
                {
                    param_name: [
                        self._result.values[param_name],
                        np.sqrt(self._result.covar[i, i]),
                    ]
                }
            )

    def __set_param_values_lm_special(self, x):
        """Constrói o dicionário e o Parameters dos valores do ajuste,
        quando não há incertezas."""
        self._dict.clear()
        self._dict2.clear()
        self._dict_param.clear()
        self._params = Parameters()
        ngl = len(x) - self._result.nvarys
        inc_cons = np.sqrt(self._result.chisqr / ngl) if ngl > 0 else 1
        self._par_var = []
        for i in list(self._result.params.keys()):
            if self._result.params[i].vary:
                self._par_var.append(i)
        for i, param_name in enumerate(self._par_var):
            self._params.add(param_name, self._result.values[param_name])
            self._dict.update(
                {
                    param_name: [
                        self._result.values[param_name],
                        np.sqrt(self._result.covar[i, i]) * inc_cons,
                    ]
                }
            )
            self._dict2.update(
                {
                    param_name: [
                        self._result.values[param_name],
                        np.sqrt(self._result.covar[i, i]),
                    ]
                }
            )
            self._dict_param.update(
                {
                    param_name: [
                        self._result.values[param_name],
                        np.sqrt(self._result.covar[i, i]) * inc_cons,
                    ]
                }
            )

    def __set_param_values_ODR(self, x):
        """Constrói o dicionário e o Parameters dos valores do ajuste."""
        self._dict.clear()
        self._dict_param.clear()
        self._params = Parameters()
        for i in range(len(self._coef)):
            self._params.add(self._coef[i], self._result.beta[i])
            self._dict.update(
                {
                    self._coef[i]: [
                        self._result.beta[i],
                        np.sqrt(self._result.cov_beta[i, i]),
                    ]
                }
            )
            self._dict_param.update({self._coef[i]: self._dict[self._coef[i]]})

    def __set_report_lm(self, x):
        """Constrói a string com os resultados."""
        self._report_fit = ""
        self._report_fit += f"\nAjuste: y = {self._exp_model}\n"
        self._report_fit += "\nNGL  = %d" % (len(x) - self._result.nvarys)
        self._report_fit += "\nChi² = %f\n\n" % self._result.chisqr
        self._report_fit += self.params_print()
        self._report_fit += "\n"
        self._mat_cov = self._result.covar
        lista = list(self._params.keys())
        matriz_corr = np.zeros((len(self._result.covar), len(self._result.covar)))
        z = range(len(matriz_corr))
        for i in z:
            for j in z:
                matriz_corr[i, j] = self._result.covar[i, j] / (
                    self._dict[lista[i]][1] * self._dict[lista[j]][1]
                )
        self._mat_corr = matriz_corr.round(3)
        self._report_fit += (
            "\nMatriz de correlação:\n\n" + self.matprint(self._mat_corr, ".3f") + "\n"
        )
        self._report_fit += (
            "Matriz de covariância:\n\n"
            + self.matprint(self._result.covar, fmt=".3e")
            + "\n\n"
        )
        self._isvalid = True

    def __set_report_lm_special(self, x):
        """Constrói a string com os resultados, neste caso quando não há incertezas."""
        ngl = len(x) - self._result.nvarys
        inc_considerada = np.sqrt(self._result.chisqr / ngl) if ngl > 0 else 0
        inc_considerada_q = inc_considerada**2
        self._report_fit = ""
        self._report_fit += f"\nAjuste: y = {self._exp_model}\n"
        self._report_fit += "\nNGL  = %d" % (ngl)
        self._report_fit += (
            "\nSomatória dos resíduos absolutos ao quadrado = %f\n"
            % self._result.chisqr
        )
        self._report_fit += "Incerteza considerada = %f\n\n" % inc_considerada
        try:
            self._mat_cov = self._result.covar * inc_considerada_q
            self._report_fit += self.params_print2(inc_considerada)
            self._report_fit += "\n"
            lista = list(self._params.keys())
            matriz_corr = np.zeros((len(self._result.covar), len(self._result.covar)))
            z = range(len(matriz_corr))
            for i in z:
                for j in z:
                    matriz_corr[i, j] = self._result.covar[i, j] / (
                        self._dict2[lista[i]][1] * self._dict2[lista[j]][1]
                    )
            self._mat_corr = matriz_corr.round(3)
            self._report_fit += (
                "\nMatriz de correlação:\n\n"
                + self.matprint(self._mat_corr, ".3f")
                + "\n"
            )
            self._report_fit += (
                "Matriz de covariância:\n\n"
                + self.matprint(self._mat_cov, fmt=".3e")
                + "\n"
            )
            self._isvalid = True
        except TypeError:
            self._msg_handler.raise_error(
                "A função ajustada provavelmente não possui parâmetros para serem ajustados. Rever ajuste."
            )
            return None

    def __set_report_ODR(self, x):
        """Constrói a string com os resultados."""
        self._report_fit = ""
        self._report_fit += f"\nAjuste: y = {self._exp_model}\n"
        self._report_fit += f"\nNGL  = {len(x) - len(self._par_var)}"
        self._report_fit += "\nChi² = %f\n\n" % self._result.sum_square
        self._report_fit += self.params_print()
        self._report_fit += "\n"
        self._mat_cov = self._result.cov_beta
        lista = list(self._params.keys())
        matriz_corr = np.zeros(
            (len(self._result.cov_beta), len(self._result.cov_beta)),
            dtype=float,
        )
        z = range(len(matriz_corr))
        for i in z:
            for j in z:
                if self._dict[lista[i]][1] == 0 or self._dict[lista[j]][1] == 0:
                    matriz_corr[i, j] = np.nan
                else:
                    matriz_corr[i, j] = self._result.cov_beta[i, j] / (
                        self._dict[lista[i]][1] * self._dict[lista[j]][1]
                    )
        self._mat_corr = matriz_corr.round(3)
        self._report_fit += (
            "\nMatriz de correlação:\n\n" + self.matprint(self._mat_corr, ".3f") + "\n"
        )
        self._report_fit += (
            "Matriz de covariância:\n\n"
            + self.matprint(self._result.cov_beta, fmt=".3e")
            + "\n"
        )
        self._isvalid = True

    @property
    def coefficients(self):
        """Retorna uma lista com os nomes dos coeficientes."""
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
    def residuo(self):
        """Retorna os valores de y_i - f(x_i)."""
        return self._data["y"].to_numpy() - eval(
            f"self._model.eval({self._ind_var}=self._data['x'].to_numpy())",
            None,
            {"self": self},
        )

    @property
    def residuo_dummy(self):
        """Retorna os valores de y_i - f(x_i)."""
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
        #         except Exception:
        #             pi[i] = 1.0
        # paramss = Parameters()
        # for i in range(len(self._coef)):
        #     paramss.add(self._coef[i], pi[i])
        # print("rfs")
        # print("self._model.eval(%s = self._data['x'], params = self._params)"%self._ind_var)
        # print(self._data["x"])
        # print(self._params)
        # print(self._model.eval(x = self._data['x'].to_numpy(), params = self._params))
        y = eval(
            "self._model.eval(%s = self._data['x'].to_numpy(), params = self._params)"
            % self._ind_var,
            None,
            {"self": self},
        )
        # print(y)
        # return self._data["y"].to_numpy() - eval("self._model.eval(%s = self._data['x'], params = self._params)"%self._ind_var, None,
        # {"self": self})
        return self._data["y"].to_numpy() - y

    def get_predict(self, fig, x_min=None, x_max=None):
        """Retorna a previsão do modelo."""
        x_plot = np.linspace(
            x_min, x_max, int(fig.get_size_inches()[0] * fig.dpi * 1.75)
        )
        return x_plot, eval(
            "self._model.eval(%s = x_plot, params = self._params)" % self._ind_var,
            None,
            {"x_plot": x_plot, "self": self},
        )

    @property
    def inliers(self):
        """Retorna os pontos usados no ajuste."""
        return self._indices

    @property
    def outliers(self):
        """Retorna os pontos não usados no ajuste."""
        return np.array(list(set(np.arange(len(self._data))) - set(self._indices)))

    @property
    def exp_model(self):
        """Retorna a expressão do modelo."""
        return self._exp_model

    @exp_model.setter
    def exp_model(self, expression: str):
        self._exp_model = expression

    @property
    def isvalid(self):
        """Retorna se o ajuste é válido."""
        return self._isvalid

    @isvalid.setter
    def isvalid(self, state: bool):
        self._isvalid = state

    @property
    def params(self):
        """Retorna os parâmetros do modelo."""
        return self._params

    def get_predict_log(self, fig, x_min=None, x_max=None):
        """Retorna a previsão do modelo."""
        x_plot = np.logspace(
            np.log10(x_min),
            np.log10(x_max),
            int(fig.get_size_inches()[0] * fig.dpi * 2.1),
        )
        return x_plot, eval(
            f"self._model.eval({self._ind_var} = x_plot, params = self._params)",
            None,
            {"x_plot": x_plot, "self": self},
        )

    def predictInc(self, wsx, wsy: bool = False):
        if wsx is False and wsy is False and self._has_sx and self._has_sy:
            sy = np.zeros(len(self._data["x"]), dtype=float)
            for i, x in enumerate(self._data["x"]):
                x_var = np.array(
                    [x + self._data["sx"].iloc[i], x - self._data["sx"].iloc[i]]
                )
                y_prd = eval(
                    f"self._model.eval({self._ind_var} = x, params = self._params)",
                    None,
                    {"x": x, "self": self},
                )
                y_var = eval(
                    f"self._model.eval({self._ind_var} = x_var, params = self._params)",
                    None,
                    {"x_var": x_var, "self": self},
                )
                sy[i] = np.abs(y_var - y_prd).mean()
                sy[i] = np.sqrt(self._data["sy"].iloc[i] ** 2 + sy[i] ** 2)
            return sy
        elif wsx is False and wsy is False and self._has_sy is False and self._has_sx:
            sy = np.zeros(len(self._data["x"]), dtype=float)
            for i, x in enumerate(self._data["x"]):
                x_var = np.array(
                    [x + self._data["sx"].iloc[i], x - self._data["sx"].iloc[i]]
                )
                y_prd = eval(
                    f"self._model.eval({self._ind_var} = x, params = self._params)",
                    None,
                    {"x": x, "self": self},
                )
                y_var = eval(
                    f"self._model.eval({self._ind_var} = x_var, params = self._params)",
                    None,
                    {"x_var": x_var, "self": self},
                )
                sy[i] = np.abs(y_var - y_prd).mean()
        elif (
            wsx is False
            and wsy is False
            and self._has_sy is False
            and self._has_sx is False
        ):
            return np.zeros(len(self._data["x"]), dtype=float)
        elif wsx is False and wsy and self._has_sy and self._has_sx is False:
            return np.zeros(len(self._data["x"]), dtype=float)
        return self._data["sy"].to_numpy()

    def createDummyModel(self):
        try:
            self._model = ExpressionModel(
                self._exp_model + " + 0*%s" % self._ind_var,
                independent_vars=[self._ind_var],
            )
        except ValueError:
            self._msg_handler.raise_error(
                "Expressão de ajuste escrita de forma errada. Rever função de ajuste."
            )
            return None
        except SyntaxError:
            self._msg_handler.raise_error(
                "Expressão de ajuste escrita de forma errada. Rever função de ajuste."
            )
            return None
        self._coef = [i for i in self._model.param_names]
        self.__make_parameters_lm()
        # if self._p0 is None:
        #     for i in range(len(self._coef)):
        #         self._params.add(self._coef[i], 1.)
        # else:
        #     coefs   = {c : [1, True] for c in self._coef}
        #     coefs_2 = {c : False for c in self._coef} # Para evitar de substituir atribuição de parâmetros
        #     for i in range(len(self._coef)):
        #         try:
        #             res = re.match("\s?(?:(.*[^\s])\s?=\s?(.*)|(@?\d*@?))", self._p0[i])
        #             if res.groups()[2] is None:
        #                     valor = res.groups()[1].replace("@", "")
        #                     var   = res.groups()[0]
        #                     if var in coefs:
        #                         coefs[var]   = [float(valor), not "@" in res.groups()[1]]
        #                         coefs_2[var] = True
        #             else:
        #                 if coefs_2[self._coef[i]] is False and self._coef[i] in coefs:
        #                     coefs[self._coef[i]]   = [float(res.groups()[2].replace("@", "")), not "@" in res.groups()[2]]
        #                     coefs_2[self._coef[i]] = True
        #         except Exception:
        #             if coefs_2[self._coef[i]] is False:
        #                 coefs[self._coef[i]]   = [1, True]
        #                 coefs_2[self._coef[i]] = True
        #     for nome in coefs.keys():
        #         self._params.add(nome, coefs[nome][0], vary = coefs[nome][1])
        # coefs   = {c : [1, True] for c in self._coef}
        # coefs_2 = {c : False for c in self._coef} # Para evitar de substituir atribuição de parâmetros
        # for i in range(len(self._coef)):
        #     try:
        #         p_i = self._p0[i].split("=")
        #         if len(p_i) == 2:
        #             if "@" in p_i[1]:
        #                 p_i[1] = p_i[1].replace("@", "")
        #                 var = p_i[0].strip()
        #                 if var in coefs:
        #                     coefs[var] = [float(p_i[1].strip()), False]
        #                     coefs_2[var] = True
        #             else:
        #                 var = p_i[0].strip()
        #                 if var in coefs:
        #                     coefs[var]   = [float(p_i[1].strip()), True]
        #                     coefs_2[var] = True
        #         else:
        #             if "@" in self._p0[i]:
        #                 if coefs_2[self._coef[i]] is False:
        #                     coefs[self._coef[i]]   = [float(self._p0[i].replace("@", "")), False]
        #                     coefs_2[self._coef[i]] = True
        #             else:
        #                 if self._coef[i] in coefs and coefs_2[self._coef[i]] is False:
        #                     coefs[self._coef[i]]   = [float(self._p0[i]), True]
        #                     coefs_2[self._coef[i]] = True
        #     except Exception:
        #         if coefs_2[self._coef[i]] is False:
        #             coefs[self._coef[i]]   = [1, True]
        #             coefs_2[self._coef[i]] = True
        # for nome in coefs.keys():
        #     self._params.add(nome, coefs[nome][0], vary = coefs[nome][1])
        self._indices = np.arange(len(self._data))
        self._isvalid = True

    def matprint(self, mat, fmt="f"):
        col_maxes = [
            max([len(("{:" + fmt + "}").format(x)) for x in col]) for col in mat.T
        ]
        matrix = ""
        for x in mat:
            for i, y in enumerate(x):
                matrix += ("{:" + str(col_maxes[i]) + fmt + "}").format(y) + "  "
            matrix += "\n"
        return matrix

    def params_print(self):
        df = pd.DataFrame(self._dict)
        df = df.transpose()
        df.columns = ["Valor", "|    Incerteza"]
        try:
            df.index = self._coef
        except Exception:
            df.index = self._par_var
        return str(df)

    def params_print2(self, inc_considerada):
        df = pd.DataFrame(self._dict2).transpose()
        df.columns = ["Valor", "|    Incerteza"]
        df["|    Incerteza"] = df["|    Incerteza"] * inc_considerada
        df.index = self._par_var
        return str(df)

    def params_print3(self):
        df = pd.DataFrame(self._dict, columns=["Valor", "Incerteza"]).transpose()
        try:
            df.index = self._coef
        except Exception:
            df.index = self._par_var
        return df

    @pyqtSlot(str, str, bool)
    def copyParamsClipboard(self, sep, decimal, header):
        """Copy parameters to the clipboard."""
        sep_decimal = {
            "Ponto": ".",
            "Vírgula": ",",
        }
        sep_columns = {
            "Tabulação": "\t",
            "Espaço": " ",
            ",": ",",
            "|": "|",
            ";": ";",
        }
        try:
            if header:
                pd.DataFrame(
                    self._dict_param, index=["Valor", "Incerteza"]
                ).transpose().to_clipboard(
                    sep=sep_columns[sep], decimal=sep_decimal[decimal]
                )
            else:
                df = pd.DataFrame.from_dict(self._dict_param, orient="index")
                df.columns = [""] * len(df.columns)
                df.to_clipboard(sep=sep_columns[sep], decimal=sep_decimal[decimal])

        except Exception:
            self._msg_handler.raise_error(
                "Não foi possível copiar para a área de transferência."
            )

    @pyqtSlot(str, str)
    def copyCovarianceClipboard(self, sep, decimal):
        """Copy parameters to the clipboard."""
        sep_decimal = {
            "Ponto": ".",
            "Vírgula": ",",
        }
        sep_columns = {
            "Tabulação": "\t",
            "Espaço": " ",
            ",": ",",
            "|": "|",
            ";": ";",
        }
        try:
            pd.DataFrame(self._mat_cov).to_clipboard(
                sep=sep_columns[sep],
                decimal=sep_decimal[decimal],
                index=False,
                header=False,
            )
        except Exception:
            self._msg_handler.raise_error(
                "Não foi possível copiar para a área de transferência."
            )

    @pyqtSlot(str, str)
    def copyCorrelationClipboard(self, sep, decimal):
        """Copy parameters to the clipboard."""
        sep_decimal = {
            "Ponto": ".",
            "Vírgula": ",",
        }
        sep_columns = {
            "Tabulação": "\t",
            "Espaço": " ",
            ",": ",",
            "|": "|",
            ";": ";",
        }
        try:
            pd.DataFrame(self._mat_corr).to_clipboard(
                sep=sep_columns[sep],
                decimal=sep_decimal[decimal],
                index=False,
                header=False,
            )
        except Exception:
            self._msg_handler.raise_error(
                "Não foi possível copiar para a área de transferência."
            )

    def reset(self):
        self._data = None

        self._exp_model = ""
        self._model = None
        self._report_fit = ""
        self._result = None
        self._coef = []
        self._params = Parameters()
        self._dict = {}
        self._dict2 = {}
        self._p0 = None
        self.xmin_adj = 0.0
        self.xmax_adj = 0.0
        self._has_data = False
        self._isvalid = False
        self._has_sx = True
        self._has_sy = True
        self._indices = []
