from __future__ import annotations

import numpy as np
import pandas as pd
from PyQt5.QtCore import (
    QObject,
    QJsonValue,
    QUrl,
    QVariant,
    pyqtSignal,
    pyqtSlot,
)
from copy import deepcopy
from .MessageHandler import MessageHandler
from PyQt5.QtGui import QGuiApplication
from io import StringIO


# data_handler
class DataHandler(QObject):
    # Signals plot
    uploadData = pyqtSignal(QVariant, str, arguments=["data", "fileName"])

    def __init__(self, messageHandler) -> None:
        super().__init__()
        self._msg_handler: MessageHandler = messageHandler
        self._df: pd.DataFrame = None
        self._data_json: pd.DataFrame = None
        self._has_sx: bool = True
        self._has_sy: bool = True
        self._has_data: bool = False
        self._data: pd.DataFrame = None

    def reset(self) -> None:
        self._data = None
        self._data_json = None
        self._has_data = False
        self._has_sx = True
        self._has_sy = True

    def _is_number(self, s: any) -> bool:
        if isinstance(s, str):
            s = s.replace(",", ".")
        try:
            float(s)
            return True
        except Exception:
            return False

    def _read_csv(self, data_path: str) -> None:
        try:
            self._df = (
                pd.read_csv(data_path, sep=",", header=None, dtype=str)
                .dropna(how="all")
                .replace(np.nan, "0")
            )
            self._df = self._df.rename({0: "x", 1: "y", 2: "sy", 3: "sx"}, axis=1)
            return None
        except pd.errors.ParserError:
            self._msg_handler.raise_error(
                "Separação de colunas de arquivos csv são com vírgula (','). Rever dados de entrada."
            )
            return None

        except UnicodeDecodeError:
            self._msg_handler.raise_error(
                "O encoding do arquivo é inválido. Use o utf-8."
            )
            return None

    def _read_tsv_txt(self, data_path: str) -> None:
        try:
            self._df = (
                pd.read_csv(
                    data_path,
                    sep=r"\t|\s",
                    header=None,
                    dtype=str,
                    engine="python",
                )
                .dropna(how="all")
                .replace(np.nan, "0")
            )
            self._df = self._df.rename({0: "x", 1: "y", 2: "sy", 3: "sx"}, axis=1)
        except pd.errors.ParserError:
            self._msg_handler.raise_error(
                "Separação de colunas de arquivos txt e tsv são com tab ou espaço. Rever dados de entrada."
            )
            return None

    def _fill_df_with_array(
        self, df_array: list[list[str, str, str, str, bool]] | None
    ) -> None:
        self._df = pd.DataFrame.from_records(
            df_array, columns=["x", "y", "sy", "sx", "bool"]
        )
        del self._df["bool"]
        uniqueSi = self._df["sy"].astype(float).unique()
        if 0.0 in uniqueSi:
            if len(uniqueSi) > 1:
                self._msg_handler.raise_warn(
                    "Um valor nulo foi encontrado nas incertezas em y, removendo coluna de sy."
                )
            self._has_sy = False
        uniqueSi = self._df["sx"].astype(float).unique()
        if 0.0 in uniqueSi:
            if len(uniqueSi) > 1:
                self._msg_handler.raise_warn(
                    "Um valor nulo foi encontrado nas incertezas em x, removendo coluna de sx."
                )
            self._has_sx = False

    def _to_float(self, df: pd.DataFrame) -> tuple[pd.DataFrame, None]:
        for col in df.columns:
            df[col] = df[col].astype(float)
            # try:
            #     df[col] = df[col].astype(float)
            # except ValueError:
            #     self._msg_handler.raise_error(
            #         "A entrada de dados só permite entrada de números. Rever arquivo de entrada."
            #     )
            #     return None
        return df

    def _comma_to_dot(self, df: pd.DataFrame) -> pd.DataFrame:
        # df = self._drop_header(df)
        for col in df.columns:
            if df[col].dtype != np.float64:
                df[col] = [x.replace(",", ".") for x in df[col]]
                df[col] = df[col].astype(str)
        return df

    def _to_check_columns(
        self, df: pd.DataFrame, df_json: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame] | tuple[None, None]:
        number_of_cols = len(df.columns)
        if number_of_cols == 1:
            self._has_sy = not self._has_sy
            self._has_sx = not self._has_sx
            df["y"] = df["x"].copy()
            df["x"] = np.arange(len(df), dtype=float)
            df_json = deepcopy(df.astype(str))
            df_json.columns = ["x", "y"]
            df.insert(2, "sy", 0.0)
            df.insert(3, "sx", 0.0)
            return df, df_json
        elif number_of_cols == 2:
            self._has_sy = not self._has_sy
            self._has_sx = not self._has_sx
            df_json = deepcopy(df.astype(str))
            df_json.columns = ["x", "y"]
            df.insert(2, "sy", 0.0)
            df.insert(3, "sx", 0.0)
            return df, df_json
        elif number_of_cols == 3:
            self._has_sx = not self._has_sx
            df_json = deepcopy(df.astype(str))
            df_json.columns = ["x", "y", "sy"]
            df.insert(3, "sx", 0.0)
            unique_sy = df_json["sy"].astype(float).unique()
            if 0.0 in unique_sy:
                if len(unique_sy) > 1:
                    self._msg_handler.raise_warn(
                        "Um valor nulo foi encontrado nas incertezas em y, removendo coluna de sy."
                    )
                self._has_sy = False
            return df, df_json
        elif number_of_cols == 4:
            df_json = deepcopy(df.astype(str))
            df_json.columns = ["x", "y", "sy", "sx"]
            unique_sy = df_json["sy"].astype(float).unique()
            if 0.0 in unique_sy:
                if len(unique_sy) > 1:
                    self._msg_handler.raise_warn(
                        "Um valor nulo foi encontrado nas incertezas em y, removendo coluna de sy."
                    )
                self._has_sy = False
            unique_sx = df_json["sx"].astype(float).unique()
            if 0.0 in unique_sx:
                if len(unique_sx) > 1:
                    self._msg_handler.raise_warn(
                        "Um valor nulo foi encontrado nas incertezas em x, removendo coluna de sx."
                    )
                self._has_sx = False
            return df, df_json
        else:
            self._msg_handler.raise_error(
                "Há mais do que 4 colunas. Rever entrada de dados."
            )
            return None, None

    def _load_by_data_path(self, data_path: str) -> None:
        if data_path[-3:] == "csv":
            self._read_csv(data_path)
        else:
            self._read_tsv_txt(data_path)

    def _fill_df_with_clipboardText(self, clipboardText):
        try:
            df = (
                pd.read_csv(
                    StringIO(clipboardText),
                    sep=r"\t|\s",
                    header=None,
                    engine="python",
                    dtype=str,
                )
                .dropna(how="all")
                .replace(np.nan, "0")
            )
        except pd.errors.ParserError:
            self._msg_handler.raise_error(
                "Separação de colunas de arquivos txt e tsv são com tab ou espaço. Rever dados de entrada."
            )
            return None
        # Only consider number of columns less than 4
        df = df.rename({0: "x", 1: "y", 2: "sy", 3: "sx"}, axis=1)
        # Replacing all commas for dots
        self._df = df

    def _filter_string_rows(self, df: pd.DataFrame) -> pd.DataFrame:
        cond = [
            pd.to_numeric(df[column], errors="coerce").notnull()
            for column in df.columns
        ]
        cond = np.all(cond, axis=0)
        if False in cond:
            self._msg_handler.raise_warn("Linhas com valores não numéricos removidas.")
        df = df[cond]
        df.reset_index(drop=True, inplace=True)
        return df

    @pyqtSlot(str)
    def load_data(
        self,
        data_path: str = "",
        df_array: list[list[str, str, str, str, bool]] = None,
        clipboardText="",
    ) -> None:
        fileName = "Dados Carregados do Projeto"
        if len(data_path) > 0:
            # Loading from .csv or (.txt and .tsv)
            data_path = QUrl(data_path).toLocalFile()
            self._load_by_data_path(data_path)
            fileName = data_path.split("/")[-1]
        elif df_array is not None:
            self._fill_df_with_array(df_array)
        elif clipboardText != "":
            self._fill_df_with_clipboardText(clipboardText)

        if isinstance(self._df, pd.DataFrame) and (self._df.empty is False):
            self._df = self._comma_to_dot(self._df)
            self._df = self._filter_string_rows(self._df)
            self._df, self._data_json = self._to_check_columns(
                self._df, self._data_json
            )
            if self._df is None:
                return None
            self._df = self._to_float(self._df)
            self._data = deepcopy(self._df)
            self._has_data = True
            self.uploadData.emit(self._data_json.to_dict(orient="list"), fileName)

    @pyqtSlot(str)
    def _load_data_bottom(self, clipboardText_bottom) -> None:
        try:
            df = (
                pd.read_csv(
                    StringIO(clipboardText_bottom),
                    sep=r"\t|\s",
                    engine="python",
                    header=None,
                    dtype=str,
                )
                .dropna(how="all")
                .replace(np.nan, "0")
            )
        except pd.errors.ParserError:
            self._msg_handler.raise_error(
                "Separação de colunas de arquivos txt e tsv são com tab ou espaço. Rever dados de entrada."
            )
            return None

        if isinstance(df, pd.DataFrame):
            df = self._comma_to_dot(df)
            if len(df.columns) > 1:
                df = df.rename({0: "x", 1: "y", 2: "sy", 3: "sx"}, axis=1)
                df = self._filter_string_rows(df)
                self._df = pd.concat(
                    [self._data_json, df], axis=0, ignore_index=True
                ).fillna("0")
                self._df, self._data_json = self._to_check_columns(
                    self._df, self._data_json
                )
                if self._df is None:
                    return None
                self._df = self._to_float(self._df)
            else:
                self._msg_handler.raise_warn(
                    "Para inserir novos dados, o número de colunas tem que ser maior do que 1."
                )

            fileName = "Dados Carregados do Projeto"
            self._data = deepcopy(self._df)
            self._has_data = True
            self.uploadData.emit(self._data_json.to_dict(orient="list"), fileName)

    @pyqtSlot(QJsonValue)
    def loadDataTable(
        self, data: list[list[str, str, str, str, bool]] | None = None
    ) -> None:
        """Getting data from table."""
        self._df = pd.DataFrame.from_records(
            data, columns=["x", "y", "sy", "sx", "bool"]
        ).replace("", "0")
        # Removing not chosen rows
        self._df = self._df[self._df["bool"] == 1]
        del self._df["bool"]
        uniqueSi = self._df["sy"].astype(float).unique()
        if 0.0 in uniqueSi:
            if len(uniqueSi) > 1:
                self._msg_handler.raise_warn(
                    "Um valor nulo foi encontrado nas incertezas em y, removendo coluna de sy."
                )
            self._has_sy = False
        uniqueSi = self._df["sx"].astype(float).unique()  # TODO: Verificar check
        if 0.0 in uniqueSi:
            if len(uniqueSi) > 1:
                self._msg_handler.raise_warn(
                    "Um valor nulo foi encontrado nas incertezas em x, removendo coluna de sx."
                )
            self._has_sx = False
        self._data_json = deepcopy(self._df)

        # Turn everything into number (str -> number)
        self._df = self._df.astype(float)
        self._data = deepcopy(self._df)
        self._has_data = True

    @pyqtSlot()
    def loadDataClipboard(self) -> None:
        """Pega a tabela de dados do Clipboard."""
        # Instantiating clipboard
        clipboard = QGuiApplication.clipboard()
        clipboardText = clipboard.mimeData().text()
        self.load_data(clipboardText=clipboardText)

    @pyqtSlot()
    def loadDataClipboard_bottom(self) -> None:
        """Pega a tabela de dados do Clipboard."""
        # Instantiating clipboard
        clipboard = QGuiApplication.clipboard()
        clipboardText_bottom = clipboard.mimeData().text()
        self._load_data_bottom(clipboardText_bottom)

    @property
    def data(self) -> pd.DataFrame:
        return self._data

    @property
    def separated_data(
        self,
    ) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
        """Retorna x, y, sx e sy."""
        return (
            self._data["x"],
            self._data["y"],
            self._data["sy"],
            self._data["sx"],
        )

    @property
    def has_sx(self) -> bool:
        return self._has_sx

    @property
    def has_sy(self) -> bool:
        return self._has_sy
