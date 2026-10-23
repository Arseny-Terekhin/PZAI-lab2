"""Признаки варианта 1 (отток клиентов Telco). Импортируется и ноутбуком, и сервисом.

Перенесено из features.py ЛР1. Модуль ничего не выполняет при импорте:
только константы, функции и классы. Сохранённый пайплайн ссылается на
src.features.add_features, поэтому загружается в любом процессе, где
доступен пакет src.
"""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

RANDOM_STATE = 42

# входные столбцы модели (без customerID и целевой переменной)
INPUT_COLUMNS = ['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure',
                 'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity',
                 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV',
                 'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod',
                 'MonthlyCharges', 'TotalCharges']

SERVICES = ['PhoneService', 'MultipleLines', 'OnlineSecurity', 'OnlineBackup',
            'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']
PROTECTION = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport']
INTERNET_OPTIONS = PROTECTION + ['StreamingTV', 'StreamingMovies']

# все новые признаки ЛР1; по итогам абляции в финальной модели оставлен только n_services
NEW_FEATURES = ['avg_monthly_charge', 'charge_increase', 'n_services',
                'contract_payment', 'fiber_no_protection', 'auto_payment',
                'tenure_bucket']
KEPT_FEATURES = ['n_services']
DROP_FEATURES = [f for f in NEW_FEATURES if f not in KEPT_FEATURES]


def clean(X: pd.DataFrame) -> pd.DataFrame:
    """Очистка сырых данных: приведение типов, удаление идентификатора."""
    X = X.copy()
    X = X.drop(columns=['customerID', 'Churn'], errors='ignore')
    # у новых клиентов (tenure = 0) TotalCharges — пустая строка или пропуск
    X['TotalCharges'] = pd.to_numeric(X['TotalCharges'], errors='coerce')
    X.loc[X['TotalCharges'].isna() & (X['tenure'] == 0), 'TotalCharges'] = 0.0
    X['SeniorCitizen'] = X['SeniorCitizen'].map({0: 'No', 1: 'Yes'}).fillna(
        X['SeniorCitizen'].astype(str))
    return X


def add_features(X: pd.DataFrame, drop=()) -> pd.DataFrame:
    """Новые признаки ЛР1; drop — имена признаков, которые не создавать."""
    X = clean(X)
    tenure_safe = X['tenure'].replace(0, np.nan)
    # средний фактический платёж за месяц за всё время обслуживания
    X['avg_monthly_charge'] = (X['TotalCharges'] / tenure_safe).fillna(X['MonthlyCharges'])
    # насколько текущий тариф выше исторического среднего (рост цены)
    X['charge_increase'] = X['MonthlyCharges'] - X['avg_monthly_charge']
    # число подключённых услуг
    X['n_services'] = (X[SERVICES] == 'Yes').sum(axis=1) + (X['InternetService'] != 'No')
    # сочетание типа договора и способа оплаты
    X['contract_payment'] = X['Contract'] + ' | ' + X['PaymentMethod']
    # оптоволокно без единой услуги защиты/поддержки
    X['fiber_no_protection'] = ((X['InternetService'] == 'Fiber optic')
                                & (X[PROTECTION] != 'Yes').all(axis=1)).astype(int)
    # автоматический платёж
    X['auto_payment'] = X['PaymentMethod'].str.contains('automatic').astype(int)
    # стаж в годах (0–6)
    X['tenure_bucket'] = np.minimum(X['tenure'] // 12, 6).astype(int)
    return X.drop(columns=list(drop))


def to_category(X: pd.DataFrame) -> pd.DataFrame:
    """Строковые столбцы — в тип category."""
    X = X.copy()
    cols = X.select_dtypes(exclude='number').columns
    X[cols] = X[cols].astype(str).astype('category')
    return X


def cat_columns(X: pd.DataFrame) -> list:
    return X.select_dtypes(exclude='number').columns.tolist()


class CategoryAligner(TransformerMixin, BaseEstimator):
    """Переводит строковые столбцы в category с категориями, запомненными на fit.

    Одинаковое кодирование категорий на обучении и на новых данных;
    неизвестные значения становятся NaN (LightGBM обрабатывает пропуски сам).
    """

    def fit(self, X, y=None):
        self.categories_ = {c: sorted(X[c].dropna().astype(str).unique())
                            for c in cat_columns(X)}
        return self

    def transform(self, X):
        X = X.copy()
        for c, cats in self.categories_.items():
            X[c] = pd.Categorical(X[c].astype(str), categories=cats)
        return X
