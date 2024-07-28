from __future__ import annotations

from atus.src.DataHandler import DataHandler
import pytest
from unittest.mock import patch, MagicMock
import pandas as pd
from pandas.testing import assert_frame_equal
from copy import deepcopy
from io import StringIO
from PyQt5.QtCore import QUrl
import os
import tempfile
from atus.src.MessageHandler import MessageHandler


@pytest.mark.data_handler
class TestDataHandler:
    @pytest.fixture
    def data_handler(self):
        messageHandler = MessageHandler()
        return DataHandler(messageHandler)

    def test_reset(self, data_handler: DataHandler):
        data_handler._data = pd.DataFrame([1])
        data_handler._data_json = pd.DataFrame(["1"])
        data_handler._has_data = True
        data_handler._has_sx = False
        data_handler._has_sy = False
        data_handler.reset()
        assert data_handler._data == None
        assert data_handler._data_json == None
        assert data_handler._has_data == False
        assert data_handler.has_sx == True
        assert data_handler.has_sy == True

    @pytest.mark.parametrize(
        "test_input, expected",
        [(1, True), (1.0, True), ("1", True), ("1,1", True), ("a", False)],
    )
    def test_is_number(self, test_input, expected, data_handler: DataHandler):
        assert data_handler._is_number(test_input) == expected

    def test_read_csv(self, data_handler: DataHandler):
        with tempfile.TemporaryDirectory() as tmpdir:
            data_path = os.path.join(tmpdir, "test.csv")
            with open(data_path, "w") as f:
                f.write("1,2,3,4\n5,6,7,8\n9,10,11,12")
            data_handler._read_csv(data_path)
        df = pd.DataFrame(
            [
                ["1", "2", "3", "4"],
                ["5", "6", "7", "8"],
                ["9", "10", "11", "12"],
            ],
            columns=["x", "y", "sy", "sx"],
        )
        pd.testing.assert_frame_equal(data_handler._df, df)

        data_handler._msg_handler.raise_error = MagicMock()
        with tempfile.TemporaryDirectory() as tmpdir:
            data_path = os.path.join(tmpdir, "test.csv")
            test_data = "1,2,3\n4,5,6,7\n8,9,1"
            message = "Separação de colunas de arquivos csv são com vírgula (','). Rever dados de entrada."
            with open(data_path, "w") as f:
                f.write(test_data)
            data_handler._read_csv(data_path)
        data_handler._msg_handler.raise_error.assert_called_once_with(message)

        # with tempfile.TemporaryDirectory() as tmpdir:
        #     data_path = os.path.join(tmpdir, "test.csv")
        #     test_data = test_data.encode("utf-8").decode("ISO-8859-1", "ignore")
        #     message = "O encoding do arquivo é inválido. Use o utf-8."
        #     with open(data_path, "w") as f:
        #         f.write(test_data)
        #     data_handler._read_csv(data_path)
        # data_handler._msg_handler.raise_error.assert_called_with(message)

    def test_read_tsv(self, data_handler: DataHandler):
        with tempfile.TemporaryDirectory() as tmpdir:
            data_path = os.path.join(tmpdir, "test.tsv")
            with open(data_path, "w") as f:
                f.write("1\t2\t3\t4\n5\t6\t7\t8\n9\t10\t11\t12")
            data_handler._read_tsv_txt(data_path)
        df = pd.DataFrame(
            [
                ["1", "2", "3", "4"],
                ["5", "6", "7", "8"],
                ["9", "10", "11", "12"],
            ],
            columns=["x", "y", "sy", "sx"],
        )
        pd.testing.assert_frame_equal(data_handler._df, df)

        data_handler._msg_handler.raise_error = MagicMock()
        with tempfile.TemporaryDirectory() as tmpdir:
            data_path = os.path.join(tmpdir, "test.csv")
            test_data = "1\t2\t3\n4\t5\t6\t7\n8\t9\t1"
            message = "Separação de colunas de arquivos txt e tsv são com tab ou espaço. Rever dados de entrada."
            with open(data_path, "w") as f:
                f.write(test_data)
            data_handler._read_tsv_txt(data_path)
        data_handler._msg_handler.raise_error.assert_called_once_with(message)

    def test_read_txt(self, data_handler: DataHandler):
        with tempfile.TemporaryDirectory() as tmpdir:
            data_path = os.path.join(tmpdir, "test.txt")
            with open(data_path, "w") as f:
                f.write("1 2 3 4\n5 6 7 8\n9 10 11 12")
            data_handler._read_tsv_txt(data_path)
        df = pd.DataFrame(
            [
                ["1", "2", "3", "4"],
                ["5", "6", "7", "8"],
                ["9", "10", "11", "12"],
            ],
            columns=["x", "y", "sy", "sx"],
        )
        pd.testing.assert_frame_equal(data_handler._df, df)

        data_handler._msg_handler.raise_error = MagicMock()
        with tempfile.TemporaryDirectory() as tmpdir:
            data_path = os.path.join(tmpdir, "test.csv")
            # Temos que mudar a mensagem deste erro
            # O Atus deveria mesmo da erro nessa situação?
            test_data = "1 2 3\n4 5 6 7\n8 9 1"
            message = "Separação de colunas de arquivos txt e tsv são com tab ou espaço. Rever dados de entrada."
            with open(data_path, "w") as f:
                f.write(test_data)
            data_handler._read_tsv_txt(data_path)
        data_handler._msg_handler.raise_error.assert_called_once_with(message)

    def test_fill_df_with_array(self, data_handler: DataHandler):
        df_array = [["1", "2", "3", "4", True], ["5", "6", "7", "8", True]]
        columns = ["x", "y", "sy", "sx"]
        expected_data = [["1", "2", "3", "4"], ["5", "6", "7", "8"]]
        df = pd.DataFrame(expected_data, columns=columns)
        data_handler._fill_df_with_array(df_array)
        pd.testing.assert_frame_equal(data_handler._df, df)

    @pytest.mark.parametrize(
        "df_array,message",
        [
            (
                [["1", "2", "0", "4", True], ["5", "6", "7", "8", True]],
                "Um valor nulo foi encontrado nas incertezas em y, removendo coluna de sy.",
            ),
            (
                [["1", "2", "3", "4", True], ["5", "6", "7", "0", True]],
                "Um valor nulo foi encontrado nas incertezas em x, removendo coluna de sx.",
            ),
        ],
    )
    def test_fill_df_with_array_warns(
        self, data_handler: DataHandler, df_array, message
    ):
        data_handler._msg_handler.raise_warn = MagicMock()
        data_handler._fill_df_with_array(df_array)
        data_handler._msg_handler.raise_warn.assert_called_once_with(message)

    def test_comma_to_dot(self, data_handler: DataHandler):
        columns = ["x", "y", "sy", "sx"]
        test_data = [["1,1", "2,2", "3,3", "4,4"], ["5,5", "6,6", "7,7", "8,8"]]
        expected_data = [
            ["1.1", "2.2", "3.3", "4.4"],
            ["5.5", "6.6", "7.7", "8.8"],
        ]
        test_df = pd.DataFrame(test_data, columns=columns)
        expected = pd.DataFrame(expected_data, columns=columns)
        result = data_handler._comma_to_dot(test_df)
        pd.testing.assert_frame_equal(result, expected)

    def test_to_float(self, data_handler: DataHandler):
        columns = ["x", "y", "sy", "sx"]
        test_data = [["1.1", "2.2", "3.3", "4.4"], ["5.5", "6.6", "7.7", "8.8"]]
        expected_data = [[1.1, 2.2, 3.3, 4.4], [5.5, 6.6, 7.7, 8.8]]
        test_df = pd.DataFrame(test_data, columns=columns)
        expected = pd.DataFrame(expected_data, columns=columns)
        result = data_handler._to_float(test_df)
        pd.testing.assert_frame_equal(result, expected)

    # def test_to_float_error(self, data_handler: DataHandler):
    #     columns = ["x"]
    #     test_data = [["NEYMAR"], ["NEYMAR"]]
    #     test_df = pd.DataFrame(test_data, columns=columns)
    #     data_handler._msg_handler.raise_error = MagicMock()
    #     data_handler._to_float(test_df)
    #     message = "A entrada de dados só permite entrada de números. Rever arquivo de entrada."
    #     data_handler._msg_handler.raise_error.assert_called_once_with(message)

    @pytest.fixture
    def four_columns_df(self, data_handler: DataHandler):
        data = [[1, 2, 3, 4], [5, 6, 7, 8]]
        columns = ["x", "y", "sy", "sx"]
        return pd.DataFrame(data, columns=columns)

    @pytest.mark.parametrize(
        "input_columns,other_columns,has_sy,has_sx",
        [
            (
                ["x"],
                {
                    "x": [0.0, 1.0],
                    "y": [1, 5],
                    "sy": [0.0, 0.0],
                    "sx": [0.0, 0.0],
                },
                False,
                False,
            ),
            (["x", "y"], {"sy": [0.0, 0.0], "sx": [0.0, 0.0]}, False, False),
            (["x", "y", "sy"], {"sx": [0.0, 0.0]}, True, False),
            (["x", "y", "sy", "sx"], {}, True, True),
        ],
    )
    def test_to_check_columns(
        self,
        data_handler: DataHandler,
        four_columns_df,
        input_columns,
        other_columns,
        has_sx,
        has_sy,
    ):
        data_handler._df = four_columns_df[input_columns]
        data_handler._data_json = deepcopy(data_handler._df)
        (
            data_handler._df,
            data_handler._data_json,
        ) = data_handler._to_check_columns(data_handler._df, data_handler._data_json)
        expected_df = deepcopy(four_columns_df)

        for col in other_columns.keys():
            expected_df[col] = other_columns[col]

        result_df = data_handler._df

        pd.testing.assert_frame_equal(result_df, expected_df)
        assert data_handler.has_sx == has_sx
        assert data_handler.has_sy == has_sy

    @pytest.mark.parametrize(
        "data,columns,message",
        [
            (
                [["1", "2", "0", "4"], ["5", "6", "7", "8"]],
                ["x", "y", "sy", "sx"],
                "Um valor nulo foi encontrado nas incertezas em y, removendo coluna de sy.",
            ),
            (
                [["1", "2", "0"], ["5", "6", "7"]],
                ["x", "y", "sy"],
                "Um valor nulo foi encontrado nas incertezas em y, removendo coluna de sy.",
            ),
            (
                [["1", "2", "3", "4"], ["5", "6", "7", "0"]],
                ["x", "y", "sy", "sx"],
                "Um valor nulo foi encontrado nas incertezas em x, removendo coluna de sx.",
            ),
        ],
    )
    def test_to_check_columns_warns(
        self, data_handler: DataHandler, data, columns, message
    ):
        df = pd.DataFrame(data, columns=columns)
        data_handler._df = df
        data_handler._data_json = deepcopy(data_handler._df)
        data_handler._msg_handler.raise_warn = MagicMock()
        data_handler._to_check_columns(data_handler._df, data_handler._data_json)
        data_handler._msg_handler.raise_warn.assert_called_once_with(message)

    @pytest.mark.parametrize(
        "data,columns,message",
        [
            (
                [["1", "2", "3", "4", "4"], ["5", "6", "7", "8", "9"]],
                ["x", "y", "sy", "sx", "extra"],
                "Há mais do que 4 colunas. Rever entrada de dados.",
            ),
        ],
    )
    def test_to_check_columns_error(
        self, data_handler: DataHandler, data, columns, message
    ):
        df = pd.DataFrame(data, columns=columns)
        data_handler._df = df
        data_handler._data_json = deepcopy(data_handler._df)
        data_handler._msg_handler.raise_error = MagicMock()
        df2, data_json = data_handler._to_check_columns(
            data_handler._df, data_handler._data_json
        )
        data_handler._msg_handler.raise_error.assert_called_once_with(message)
        assert (df2, data_json) == (None, None)

    @patch("atus.src.DataHandler.DataHandler._read_csv")
    @patch("atus.src.DataHandler.DataHandler._read_tsv_txt")
    def test_load_by_data_path(
        self,
        mock_tsv: MagicMock,
        mock_csv: MagicMock,
        data_handler: DataHandler,
    ):
        test_string = "arquivo.csv"
        data_handler._load_by_data_path(test_string)
        mock_csv.assert_called_once()
        mock_tsv.assert_not_called()
        test_string = "arquivo.tsv"
        data_handler._load_by_data_path(test_string)
        mock_tsv.assert_called_once()
        mock_csv.assert_called_once()

    def test_fill_df_with_clipboardText(self, data_handler: DataHandler):
        clipboardText = "1\t2\t3\t4\n5\t6\t7\t8"
        df = pd.read_csv(StringIO(clipboardText), sep="\t", header=None, dtype=str)
        df = df.rename(columns={0: "x", 1: "y", 2: "sy", 3: "sx"})
        data_handler._fill_df_with_clipboardText(clipboardText)
        pd.testing.assert_frame_equal(df, data_handler._df)

    def test_check_if_load_data_use_right_method(self, data_handler: DataHandler):
        data_handler._to_float = MagicMock()

        clipboardText = "1\t2\t3\t4\n5\t6\t7\t8"
        data_handler._fill_df_with_clipboardText = MagicMock()
        data_handler.load_data(clipboardText=clipboardText)
        data_handler._fill_df_with_clipboardText.assert_called_once_with(clipboardText)

        data_path = r"file:///C:/User"
        data_handler._load_by_data_path = MagicMock()
        data_handler.load_data(data_path=data_path)
        data_handler._load_by_data_path.assert_called_once_with(
            QUrl(data_path).toLocalFile()
        )

        df_array = [["1", "2", "3", "4", True], ["5", "6", "7", "8", True]]
        data_handler._fill_df_with_array = MagicMock()
        data_handler.load_data(df_array=df_array)
        data_handler._fill_df_with_array.assert_called_once_with(df_array)

    @pytest.mark.parametrize(
        "data_test, data_expected",
        [
            (
                [["a", "b", "c", "d"], ["1", "2", "3", "4"]],
                [[1.0, 2.0, 3.0, 4.0]],
            ),
            (
                [
                    ["a", "b", "c", "d"],
                    ["5", "6", "7", "8"],
                    ["5", "6", "7", "d"],
                ],
                [[5.0, 6.0, 7.0, 8.0]],
            ),
            (
                [
                    ["a", "b", "c", "d"],
                    ["5", "6", "7", "8"],
                    ["a", "b", "c", "d"],
                ],
                [[5.0, 6.0, 7.0, 8.0]],
            ),
            (
                [["a", "b", "c", "d"], ["", "", "", ""], ["5", "6", "7", "8"]],
                [[5.0, 6.0, 7.0, 8.0]],
            ),
        ],
    )
    def test_load_data(self, data_handler: DataHandler, data_test, data_expected):
        df_test = pd.DataFrame(data_test)
        df_test = df_test.rename(columns={0: "x", 1: "y", 2: "sy", 3: "sx"})
        data_handler._df = df_test
        df_expected = pd.DataFrame(data_expected)
        df_expected = df_expected.rename(columns={0: "x", 1: "y", 2: "sy", 3: "sx"})
        data_handler.load_data()
        pd.testing.assert_frame_equal(data_handler._df, df_expected)

    @pytest.mark.parametrize(
        "data_test",
        [
            ([["1", "2", "3", "4", "5"]]),
        ],
    )
    def test_load_data_when_df_is_none(self, data_handler: DataHandler, data_test):
        df_test = pd.DataFrame(data_test)
        df_test = df_test.rename(columns={0: "x", 1: "y", 2: "sy", 3: "sx"})
        data_handler._df = df_test
        data_handler.load_data()
        assert data_handler._df == None

    @pytest.mark.parametrize(
        "data_test",
        [
            ([["Neymar"]]),
        ],
    )
    def test_load_data_when_df_is_empty(self, data_handler: DataHandler, data_test):
        df_test = pd.DataFrame(data_test)
        df_test = df_test.rename(columns={0: "x", 1: "y", 2: "sy", 3: "sx"})
        data_handler._df = df_test
        data_handler.load_data()
        assert data_handler._df.empty == True

    # @pytest.mark.parametrize(
    #     "clipboardText_bottom",
    #     [
    #         (
    #             "1\t2\t3\t4\n5\t6\t7\t8\t9",
    #             "1\t2\t3\t4\t4\n5\t6\t7\t8\t9",
    #         ),
    #     ],
    # )

    @pytest.mark.parametrize(
        "clipboardText_bottom, data_expected",
        [
            (
                "a\tb\tc\td\n5\t6\t7\t8",
                [["1", "2", "3", "4"], ["5", "6", "7", "8"]],
            ),
            (
                "1\tb\tc\t1\n5\t6\t7\t8",
                [["1", "2", "3", "4"], ["5", "6", "7", "8"]],
            ),
            (
                "1\tb\tc\t1\n5\t6\t7\t8",
                [["1", "2", "3", "4"], ["5", "6", "7", "8"]],
            ),
            (
                "1\tb\tc\t1\n5\t6\t7",
                [["1", "2", "3", "4"], ["5", "6", "7", "0"]],
            ),
            ("Neymar", [["1", "2", "3", "4"]]),
        ],
    )
    def test_load_data_bottom_data_json(
        self, data_handler: DataHandler, clipboardText_bottom, data_expected
    ):
        df_top = pd.DataFrame([["1", "2", "3", "4"]])
        df_top = df_top.rename(columns={0: "x", 1: "y", 2: "sy", 3: "sx"})
        data_handler._data_json = df_top
        data_handler.load_data()
        data_handler._load_data_bottom(clipboardText_bottom)

        df_expected = pd.DataFrame(data_expected)
        df_expected = df_expected.rename(columns={0: "x", 1: "y", 2: "sy", 3: "sx"})
        pd.testing.assert_frame_equal(data_handler._data_json, df_expected)

    @pytest.mark.parametrize(
        "clipboardText_bottom, data_expected",
        [
            (
                "a\tb\tc\td\n5,1\t6,1\t7,1\t8,1",
                [[1.0, 2.0, 3.0, 4.0], [5.1, 6.1, 7.1, 8.1]],
            ),
            (
                "1\tb\tc\t1\n5,1\t6,1\t7,1\t8,1",
                [[1.0, 2.0, 3.0, 4.0], [5.1, 6.1, 7.1, 8.1]],
            ),
            (
                "1\tb\tc\t1\n\n5,1\t6,1\t7,1\t8,1",
                [[1.0, 2.0, 3.0, 4.0], [5.1, 6.1, 7.1, 8.1]],
            ),
            (
                "1\tb\tc\t1\n\n5,1\t6,1\t7,1",
                [[1.0, 2.0, 3.0, 4.0], [5.1, 6.1, 7.1, 0.0]],
            ),
            ("Neymar", [[1.0, 2.0, 3.0, 4.0]]),
        ],
    )
    def test_load_data_bottom_data_json(
        self, data_handler: DataHandler, clipboardText_bottom, data_expected
    ):
        df_top = pd.DataFrame([["1", "2", "3", "4"]])
        df_top = df_top.rename(columns={0: "x", 1: "y", 2: "sy", 3: "sx"})
        data_handler._df = df_top
        data_handler.load_data()
        data_handler._load_data_bottom(clipboardText_bottom)

        df_expected = pd.DataFrame(data_expected)
        df_expected = df_expected.rename(columns={0: "x", 1: "y", 2: "sy", 3: "sx"})
        pd.testing.assert_frame_equal(data_handler._df, df_expected)

    @pytest.mark.parametrize(
        "clipboardText_bottom",
        [
            ("1\t2\t3\t4\t5"),
        ],
    )
    def test_load_data_bottom_when_df_is_none(
        self, data_handler: DataHandler, clipboardText_bottom
    ):
        df_top = pd.DataFrame([["1", "2", "3", "4"]])
        df_top = df_top.rename(columns={0: "x", 1: "y", 2: "sy", 3: "sx"})
        data_handler._df = df_top
        data_handler.load_data()
        data_handler._load_data_bottom(clipboardText_bottom)
        assert data_handler._df == None

    @pytest.mark.parametrize(
        "data_top, clipboardText_bottom, has_sy, has_sx",
        [
            (
                [["1", "2", "3", "4", True], ["5", "6", "7", "8", True]],
                "9\t10\t11\t12\n13\t14\t15\t16",
                True,
                True,
            ),
            (
                [["1", "2", "3", "0.0", True], ["5", "6", "7", "8", True]],
                "9\t10\t11\t12\n13\t14\t15\t16",
                True,
                False,
            ),
            (
                [["1", "2", "3", "4", True], ["5", "6", "7", "8", True]],
                "9\t10\t11\t12\n13\t14\t0.0\t16",
                False,
                True,
            ),
            (
                [["1", "2", "3", "4", True], ["5", "6", "7", "8", True]],
                "9\t10\t11\t12\n13\t14\t0.0\t0.0",
                False,
                False,
            ),
        ],
    )
    def test_load_data_bottom_uncertainty_with_zeros(
        self,
        data_top,
        clipboardText_bottom,
        has_sy,
        has_sx,
        data_handler: DataHandler,
    ):
        data_handler.load_data(df_array=data_top)
        data_handler._load_data_bottom(clipboardText_bottom)
        assert has_sx == data_handler.has_sx
        assert has_sy == data_handler.has_sy

    def test_load_data_bottom_warn(self, data_handler: DataHandler):
        data_top = [["1", "2", "3", "4", True], ["5", "6", "7", "8", True]]
        data_handler.load_data(df_array=data_top)
        clipboardText_bottom = "9\n10\n11\n12"
        data_handler._msg_handler.raise_warn = MagicMock()
        data_handler._load_data_bottom(clipboardText_bottom)
        message = (
            "Para inserir novos dados, o número de colunas tem que ser maior do que 1."
        )
        data_handler._msg_handler.raise_warn.assert_called_once_with(message)

    @pytest.mark.parametrize(
        "data, has_sy, has_sx, message",
        [
            (
                [["1", "2", "3", "0.0", True], ["5", "6", "7", "8", True]],
                True,
                False,
                "Um valor nulo foi encontrado nas incertezas em x, removendo coluna de sx.",
            ),
            (
                [["1", "2", "0.0", "4", True], ["5", "6", "7", "8", True]],
                False,
                True,
                "Um valor nulo foi encontrado nas incertezas em y, removendo coluna de sy.",
            ),
        ],
    )
    def test_loadDataTable_warns(
        self, data_handler: DataHandler, data, has_sy, has_sx, message
    ):
        data_handler._msg_handler.raise_warn = MagicMock()
        data_handler.loadDataTable(data)
        data_handler._msg_handler.raise_warn.assert_called_once_with(message)
        assert has_sx == data_handler.has_sx
        assert has_sy == data_handler.has_sy

    def test_data(self, data_handler: DataHandler):
        data = [[1, 2, 3, 4], [5, 6, 7, 8]]
        df = pd.DataFrame(data, columns=["x", "y", "sy", "sx"])
        data_handler._data = df
        x, y, sy, sx = data_handler.separated_data
        pd.testing.assert_series_equal(x, df["x"])
        pd.testing.assert_series_equal(y, df["y"])
        pd.testing.assert_series_equal(sy, df["sy"])
        pd.testing.assert_series_equal(sx, df["sx"])
        pd.testing.assert_frame_equal(df, data_handler.data)
