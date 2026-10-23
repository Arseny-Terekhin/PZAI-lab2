"""Загрузка данных, разбиение, обучение и оценка моделей (ЛР2, вариант 1).

Функции принимают данные и параметры и возвращают результат; трекинг в
MLflow выполняется в ноутбуке. При импорте ничего не выполняется.
"""
import hashlib

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from src.features import (DROP_FEATURES, INPUT_COLUMNS, RANDOM_STATE, CategoryAligner,
                          add_features, clean)

C_FN, C_FP = 6000.0, 800.0      # пропуск уходящего клиента / скидка лояльному, ₽ (из ЛР1)
THRESHOLD = 0.10                # порог ЛР1, выбран по минимуму стоимости ошибок на OOF

LGB_BASE = dict(random_state=RANDOM_STATE, verbose=-1, n_jobs=4)
# лучшие параметры LightGBM из Optuna в ЛР1 (60 испытаний, 5-кратная CV)
LGB_TUNED = dict(n_estimators=350, learning_rate=0.012719593273367318, num_leaves=7,
                 min_child_samples=49, subsample=0.642798783336023,
                 colsample_bytree=0.7057136719967809, reg_lambda=0.00231741384605258,
                 cat_smooth=2.988684085052872, subsample_freq=1, **LGB_BASE)


def file_md5(path: str) -> str:
    """Версия данных — MD5 файла."""
    with open(path, 'rb') as f:
        return hashlib.md5(f.read()).hexdigest()


def load_data(path: str):
    """Сырые данные Telco -> (X, y); X содержит customerID и 19 входных признаков."""
    raw = pd.read_csv(path)
    y = (raw['Churn'] == 'Yes').astype(int)
    X = raw[['customerID'] + INPUT_COLUMNS]
    return X, y


def split(X, y, test_size=0.2):
    """То же разбиение, что в ЛР1: стратифицированное, random_state = 42."""
    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=RANDOM_STATE)


def make_model(kind: str, params: dict | None = None, drop=DROP_FEATURES):
    """Пайплайн «сырые столбцы -> вероятность»; признаки строятся внутри пайплайна."""
    params = params or {}
    if kind == 'lightgbm':
        return make_pipeline(FunctionTransformer(add_features, kw_args={'drop': list(drop)}),
                             CategoryAligner(), lgb.LGBMClassifier(**{**LGB_BASE, **params}))
    if kind == 'logreg':
        preprocess = ColumnTransformer([
            ('num', make_pipeline(SimpleImputer(strategy='median'), StandardScaler()),
             make_column_selector(dtype_include='number')),
            ('cat', make_pipeline(SimpleImputer(strategy='most_frequent'),
                                  OneHotEncoder(handle_unknown='infrequent_if_exist',
                                                min_frequency=0.01)),
             make_column_selector(dtype_exclude='number')),
        ])
        return make_pipeline(FunctionTransformer(clean), preprocess,
                             LogisticRegression(max_iter=2000, **params))
    raise ValueError(f'неизвестный тип модели: {kind}')


def train_model(X_train, y_train, params: dict | None = None, kind: str = 'lightgbm',
                drop=DROP_FEATURES):
    """Обучает пайплайн на сырых данных и возвращает его."""
    return make_model(kind, params, drop).fit(X_train, y_train)


def cv_roc_auc(X_train, y_train, params=None, kind='lightgbm', drop=DROP_FEATURES, n_splits=5):
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    return cross_val_score(make_model(kind, params, drop), X_train, y_train, cv=cv,
                           scoring='roc_auc')


def total_cost(y_true, proba, threshold, c_fn=C_FN, c_fp=C_FP) -> float:
    """Стоимость ошибок, ₽: пропущенные уходящие и скидки лояльным."""
    y_true, pred = np.asarray(y_true), np.asarray(proba) >= threshold
    return float(c_fn * ((y_true == 1) & ~pred).sum() + c_fp * ((y_true == 0) & pred).sum())


def evaluate(y_true, proba, threshold=THRESHOLD) -> dict:
    return {'roc_auc_test': float(roc_auc_score(y_true, proba)),
            'pr_auc_test': float(average_precision_score(y_true, proba)),
            'brier_test': float(brier_score_loss(y_true, proba)),
            'cost_per_client_test': total_cost(y_true, proba, threshold) / len(y_true)}


def make_input_schema(X_train, threshold=THRESHOLD) -> dict:
    """Описание входа (как input_schema.json ЛР1): тип, диапазон или допустимые значения."""
    X = X_train[INPUT_COLUMNS]
    features = {}
    for col in INPUT_COLUMNS:
        s = X[col]
        if pd.api.types.is_numeric_dtype(s):
            features[col] = {'type': 'integer' if pd.api.types.is_integer_dtype(s) else 'number',
                             'min': float(s.min()), 'max': float(s.max())}
        else:
            features[col] = {'type': 'category',
                             'values': sorted(s.dropna().astype(str).unique().tolist())}
    features['TotalCharges'] = {'type': 'number', 'nullable': True, 'min': 0.0,
                                'max': float(clean(X_train)['TotalCharges'].max())}
    return {'description': 'Вход модели оттока Telco. Одна строка — один клиент на дату среза.',
            'target': {'name': 'Churn', 'positive': 'Yes'},
            'threshold': threshold, 'optional_columns': ['customerID'], 'features': features}
