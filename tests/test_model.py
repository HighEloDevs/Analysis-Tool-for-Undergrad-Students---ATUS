from __future__ import annotations

from atus.src.Model import Model
from atus.src.DataHandler import DataHandler
from atus.src.MessageHandler import MessageHandler
import pytest
from unittest.mock import patch, MagicMock
import pandas as pd


@pytest.mark.model
class TestModel:
    def test_model_init(self):
        model = Model("dummy_str")
        assert isinstance(model, Model)

    @pytest.mark.parametrize(
        "guesses,expected",
        [
            ("1,2,3,4", ["1", "2", "3", "4"]),
            ("a =      1, 2", ["a=1", "2"]),
            ("5.5, b = 2", ["5.5", "b=2"]),
            ("5.5, b = @2", ["5.5", "b=@2"]),
            ("a=15, b = 2e10", ["a=15", "b=2e10"]),
            ("a=15, b = 2e10, , ", ["a=15", "b=2e10", "", ""]),
        ],
    )
    def test_set_initial_guesses(self, guesses: str, expected: list[str]):
        model = Model("")
        model.set_p0(p0=guesses)
        assert model._p0 == expected

    @pytest.mark.parametrize(
        "expression, ind_var, expected_exp, expected_ind_var",
        [
            ("a", "x", "a", "x"),
            ("a*x + b", "x", "a*x + b", "x"),
            ("(a/pi)*t + b", "t", "(a/pi)*t + b", "t"),
        ],
    )
    def test_set_expression(
        self,
        expression: str,
        ind_var: str,
        expected_exp: str,
        expected_ind_var: str,
    ):
        model = Model("")
        model.set_expression(exp=expression, ind_var=ind_var)
        assert model._exp_model == expected_exp
        assert model._ind_var == expected_ind_var

    @pytest.mark.parametrize(
        "expression, expected_exp, expected_ind_var",
        [
            ("a", "a", "x"),
            ("a*x + b", "a*x + b", "x"),
            ("(a/pi)*t + b", "(a/pi)*t + b", "x"),
        ],
    )
    def test_set_expression_without_ind_var(
        self,
        expression: str,
        expected_exp: str,
        expected_ind_var: str,
    ):
        model = Model("")
        model.set_expression(exp=expression)
        assert model._exp_model == expected_exp
        assert model._ind_var == expected_ind_var

    @pytest.mark.parametrize(
        "expression, ind_var, expected_expression_model, expected_coefficients",
        [
            ("a", "x", "a + 0*x", ["a"]),
            ("a*x + b", "x", "a*x + b + 0*x", ["a", "b"]),
            ("(a/pi)*t + b", "t", "(a/pi)*t + b + 0*t", ["a", "b"]),
            ("(a/pi)*x + b", "t", "(a/pi)*x + b + 0*t", ["a", "b", "x"]),
            ("1e5*x + b", "x", "1e5*x + b + 0*x", ["b"]),
        ],
    )
    def test_create_model_expression(
        self,
        expression: str,
        ind_var: str,
        expected_expression_model: str,
        expected_coefficients: list[str],
    ):
        model = Model("")
        model.set_expression(exp=expression, ind_var=ind_var)
        assert model._create_model()
        assert model._model.expr == expected_expression_model
        parameters: list[str] = model._model.param_names
        parameters.sort()
        assert parameters == expected_coefficients
        assert model._model.independent_vars == [ind_var]

    @pytest.mark.parametrize(
        "expression, ind_var, expected_error_msg",
        [
            ("3a", "x", "Erro de sintaxe. Rever função de ajuste."),
            ("(a///pi)*t + b", "t", "Erro de sintaxe. Rever função de ajuste."),
            ("a***t + b", "t", "Erro de sintaxe. Rever função de ajuste."),
        ],
    )
    def test_create_model_false(
        self,
        expression: str,
        ind_var: str,
        expected_error_msg: str,
    ):
        msg_handler_mock = MessageHandler()
        msg_handler_mock.raise_error = MagicMock()
        model = Model(messageHandler=msg_handler_mock)
        model.set_expression(exp=expression, ind_var=ind_var)
        assert model._create_model() is False
        msg_handler_mock.raise_error.assert_called_with(expected_error_msg)
