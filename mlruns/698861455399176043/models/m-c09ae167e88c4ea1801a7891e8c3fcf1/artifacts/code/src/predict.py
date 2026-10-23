"""Загрузка модели из реестра MLflow и предсказание по списку словарей.

При импорте ничего не выполняется; mlflow импортируется только при загрузке.
"""
import os

import numpy as np
import pandas as pd

from src.features import INPUT_COLUMNS

MODEL_NAME = 'lab2_terekhin'
MODEL_ALIAS = 'champion'


def tracking_uri() -> str:
    """Хранилище MLflow: переменная окружения или каталог mlruns в текущем каталоге проекта."""
    os.environ.setdefault('MLFLOW_ALLOW_FILE_STORE', 'true')   # файловый режим (MLflow 3)
    return os.environ.get('MLFLOW_TRACKING_URI') or f"file:{os.path.abspath('mlruns')}"


def load_registered(name: str = MODEL_NAME, alias: str = MODEL_ALIAS) -> dict:
    """Загружает версию модели по имени и псевдониму реестра; порог берётся из тегов версии."""
    import mlflow
    import mlflow.sklearn
    from mlflow import MlflowClient

    mlflow.set_tracking_uri(tracking_uri())
    client = MlflowClient()
    mv = client.get_model_version_by_alias(name, alias)
    run = client.get_run(mv.run_id)
    pipeline = mlflow.sklearn.load_model(f'models:/{name}/{mv.version}')
    return {'pipeline': pipeline,
            'threshold': float(mv.tags['threshold']),
            'name': name, 'version': int(mv.version), 'alias': alias, 'run_id': mv.run_id,
            'model_type': run.data.tags.get('model_type', type(pipeline[-1]).__name__),
            'roc_auc_test': run.data.metrics.get('roc_auc_test'),
            'data_version': run.data.tags.get('data_version'),
            'features': list(INPUT_COLUMNS)}


def records_to_frame(records: list[dict]) -> pd.DataFrame:
    """Список словарей -> DataFrame с входными столбцами в порядке обучения."""
    return pd.DataFrame.from_records(records, columns=INPUT_COLUMNS)


def predict_proba(pipeline, records: list[dict]) -> np.ndarray:
    """Вероятности ухода для списка клиентов."""
    return pipeline.predict_proba(records_to_frame(records))[:, 1]


def frame_to_records(X: pd.DataFrame) -> list[dict]:
    """Сырые строки набора -> JSON-совместимые объекты запроса (пустой TotalCharges -> None)."""
    X = X[[c for c in ['customerID'] + INPUT_COLUMNS if c in X.columns]].copy()
    total = pd.to_numeric(X['TotalCharges'], errors='coerce').astype(object)
    X['TotalCharges'] = total.where(total.notna(), None)
    records = X.to_dict(orient='records')
    for r in records:
        r['SeniorCitizen'] = int(r['SeniorCitizen'])
        r['tenure'] = int(r['tenure'])
        r['MonthlyCharges'] = float(r['MonthlyCharges'])
    return records
