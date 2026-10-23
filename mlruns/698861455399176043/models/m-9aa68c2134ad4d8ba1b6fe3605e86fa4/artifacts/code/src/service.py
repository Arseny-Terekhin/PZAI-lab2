"""REST-сервис оценки риска оттока (ЛР2, вариант 1).

Эндпоинты: /health, /model-info, /predict (один клиент), /predict-batch (пакет).
Модель загружается один раз при старте приложения из реестра MLflow по имени
и псевдониму. Каждый запрос записывается в журнал прогнозов (доп. задание).
"""
import os
import time
import uuid
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI, HTTPException, Request
from fastapi.exception_handlers import request_validation_exception_handler
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel, ConfigDict, Field, model_validator

from src.features import INTERNET_OPTIONS
from src.monitoring import PredictionLogger
from src.predict import MODEL_ALIAS, MODEL_NAME, load_registered, predict_proba

MAX_BATCH = 20_000          # вариант 1: суточный пакет 10 000 клиентов, запас ×2
STATE: dict = {}

YesNo = Literal['No', 'Yes']
InternetOption = Literal['No', 'No internet service', 'Yes']


class Item(BaseModel):
    """Один клиент на дату среза. Типы и границы — из input_schema.json ЛР1."""
    model_config = ConfigDict(extra='forbid')

    customerID: str | None = Field(default=None, max_length=32,
                                   description='необязательный идентификатор, в модель не идёт')
    gender: Literal['Female', 'Male']
    SeniorCitizen: Literal[0, 1] = Field(description='1 — пенсионер')
    Partner: YesNo
    Dependents: YesNo
    tenure: int = Field(ge=0, le=72, description='стаж, месяцев (в данных 0–72)')
    PhoneService: YesNo
    MultipleLines: Literal['No', 'No phone service', 'Yes']
    InternetService: Literal['DSL', 'Fiber optic', 'No']
    OnlineSecurity: InternetOption
    OnlineBackup: InternetOption
    DeviceProtection: InternetOption
    TechSupport: InternetOption
    StreamingTV: InternetOption
    StreamingMovies: InternetOption
    Contract: Literal['Month-to-month', 'One year', 'Two year']
    PaperlessBilling: YesNo
    PaymentMethod: Literal['Bank transfer (automatic)', 'Credit card (automatic)',
                           'Electronic check', 'Mailed check']
    MonthlyCharges: float = Field(ge=18.0, le=120.0, description='текущий платёж, $ (в данных 18,4–118,75)')
    TotalCharges: float | None = Field(default=None, ge=0, le=9000,
                                       description='сумма платежей, $; пусто у новых клиентов')

    @model_validator(mode='after')
    def consistent_services(self):
        """Согласованность услуг: так устроены данные оператора."""
        if (self.PhoneService == 'No') != (self.MultipleLines == 'No phone service'):
            raise ValueError("MultipleLines = 'No phone service' тогда и только тогда, "
                             "когда PhoneService = 'No'")
        no_inet = self.InternetService == 'No'
        for c in INTERNET_OPTIONS:
            if no_inet != (getattr(self, c) == 'No internet service'):
                raise ValueError(f"{c} = 'No internet service' тогда и только тогда, "
                                 "когда InternetService = 'No'")
        return self


class Prediction(BaseModel):
    customerID: str | None
    proba: float = Field(description='вероятность ухода')
    decision: int = Field(description='1 — предложить скидку (proba >= threshold)')
    threshold: float
    model_version: int


class BatchPrediction(BaseModel):
    n: int
    threshold: float
    model_version: int
    customerID: list[str | None]
    proba: list[float]
    decision: list[int]


@asynccontextmanager
async def lifespan(app: FastAPI):
    # модель загружается один раз при старте, а не в обработчике запроса
    STATE['bundle'] = load_registered(os.environ.get('MODEL_NAME', MODEL_NAME),
                                      os.environ.get('MODEL_ALIAS', MODEL_ALIAS))
    STATE['loaded_at'] = time.strftime('%Y-%m-%dT%H:%M:%S')
    STATE['logger'] = PredictionLogger(os.environ.get('PRED_LOG', 'logs/predictions.jsonl'))
    yield
    STATE['logger'].close()


app = FastAPI(title='Отток клиентов Telco — сервис оценки (ЛР2, вариант 1)',
              version='1.0', lifespan=lifespan)


@app.exception_handler(RequestValidationError)
async def log_validation_error(request: Request, exc: RequestValidationError):
    # ошибка клиента: 422 с описанием поля; в журнал — факт отказа
    if 'logger' in STATE:
        STATE['logger'].log({'endpoint': request.url.path, 'status': 422,
                             'errors': len(exc.errors())})
    return await request_validation_exception_handler(request, exc)


@app.get('/health')
def health():
    return {'status': 'ok', 'model_loaded': 'bundle' in STATE}


@app.get('/model-info')
def model_info():
    b = STATE['bundle']
    return {'name': b['name'], 'version': b['version'], 'alias': b['alias'],
            'run_id': b['run_id'], 'model_type': b['model_type'], 'threshold': b['threshold'],
            'roc_auc_test': b['roc_auc_test'], 'data_version': b['data_version'],
            'features': b['features'], 'loaded_at': STATE['loaded_at'], 'max_batch': MAX_BATCH}


def _score(items: list[Item], endpoint: str):
    b, t0 = STATE['bundle'], time.perf_counter()
    records = [i.model_dump(exclude={'customerID'}) for i in items]
    proba = [round(float(p), 4) for p in predict_proba(b['pipeline'], records)]
    decision = [int(p >= b['threshold']) for p in proba]
    STATE['logger'].log({'endpoint': endpoint, 'status': 200, 'request_id': uuid.uuid4().hex[:12],
                         'n': len(items), 'model_version': b['version'],
                         'threshold': b['threshold'],
                         'latency_ms': round((time.perf_counter() - t0) * 1000, 2),
                         'proba': proba})
    return b, proba, decision


@app.post('/predict', response_model=Prediction)
def predict(item: Item):
    b, proba, decision = _score([item], '/predict')
    return {'customerID': item.customerID, 'proba': proba[0], 'decision': decision[0],
            'threshold': b['threshold'], 'model_version': b['version']}


@app.post('/predict-batch', response_model=BatchPrediction)
def predict_batch(items: list[Item]):
    if not items:
        raise HTTPException(status_code=400, detail='пустой пакет: нужен хотя бы один объект')
    if len(items) > MAX_BATCH:
        raise HTTPException(status_code=413, detail=f'пакет больше {MAX_BATCH} объектов')
    b, proba, decision = _score(items, '/predict-batch')
    return {'n': len(items), 'threshold': b['threshold'], 'model_version': b['version'],
            'customerID': [i.customerID for i in items], 'proba': proba, 'decision': decision}
