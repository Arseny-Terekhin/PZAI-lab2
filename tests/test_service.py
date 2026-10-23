"""Тесты сервиса оттока. TestClient обращается к приложению напрямую, без сервера.

Запуск из корня проекта: python -m pytest tests/ -q
"""
import os

import pytest
from fastapi.testclient import TestClient

os.environ['PRED_LOG'] = 'logs/test_predictions.jsonl'   # не смешивать с журналом рабочих запросов

from src.service import MAX_BATCH, app  # noqa: E402

OK = {'customerID': '0001-TEST', 'gender': 'Female', 'SeniorCitizen': 0, 'Partner': 'Yes',
      'Dependents': 'No', 'tenure': 2, 'PhoneService': 'Yes', 'MultipleLines': 'No',
      'InternetService': 'Fiber optic', 'OnlineSecurity': 'No', 'OnlineBackup': 'No',
      'DeviceProtection': 'No', 'TechSupport': 'No', 'StreamingTV': 'Yes',
      'StreamingMovies': 'No', 'Contract': 'Month-to-month', 'PaperlessBilling': 'Yes',
      'PaymentMethod': 'Electronic check', 'MonthlyCharges': 85.5, 'TotalCharges': 170.0}
LOYAL = {**OK, 'customerID': '0002-TEST', 'tenure': 60, 'Contract': 'Two year',
         'InternetService': 'DSL', 'OnlineSecurity': 'Yes', 'TechSupport': 'Yes',
         'PaymentMethod': 'Credit card (automatic)', 'MonthlyCharges': 60.0, 'TotalCharges': 3600.0}


@pytest.fixture(scope='module')
def client():
    with TestClient(app) as c:      # контекст запускает lifespan: модель загружается один раз
        yield c


def field_errors(r):
    return {e['loc'][-1] for e in r.json()['detail']}


def test_health(client):
    r = client.get('/health')
    assert r.status_code == 200
    assert r.json() == {'status': 'ok', 'model_loaded': True}


def test_model_info(client):
    info = client.get('/model-info').json()
    assert info['name'] == 'lab2_terekhin' and info['version'] >= 1
    assert info['threshold'] == pytest.approx(0.10)
    assert len(info['features']) == 19 and 'customerID' not in info['features']


def test_predict_ok(client):
    r = client.post('/predict', json=OK)
    assert r.status_code == 200
    body = r.json()
    assert 0.0 <= body['proba'] <= 1.0
    assert body['decision'] == int(body['proba'] >= body['threshold'])
    assert body['customerID'] == '0001-TEST'


def test_risk_ordering(client):
    # новый клиент с помесячным договором рискованнее лояльного с двухлетним
    risky = client.post('/predict', json=OK).json()['proba']
    loyal = client.post('/predict', json=LOYAL).json()['proba']
    assert risky > loyal


def test_new_client_without_total_charges(client):
    r = client.post('/predict', json={**OK, 'tenure': 0, 'TotalCharges': None})
    assert r.status_code == 200


def test_wrong_type(client):
    r = client.post('/predict', json={**OK, 'tenure': 'много'})
    assert r.status_code == 422
    assert 'tenure' in field_errors(r)


def test_missing_field(client):
    bad = {k: v for k, v in OK.items() if k != 'Contract'}
    r = client.post('/predict', json=bad)
    assert r.status_code == 422
    assert 'Contract' in field_errors(r)


@pytest.mark.parametrize('field,value', [('tenure', -1), ('tenure', 500),
                                         ('MonthlyCharges', 1000.0), ('TotalCharges', -5.0),
                                         ('SeniorCitizen', 2), ('Contract', 'Weekly')])
def test_out_of_range(client, field, value):
    r = client.post('/predict', json={**OK, field: value})
    assert r.status_code == 422
    assert field in field_errors(r)


def test_unknown_field(client):
    r = client.post('/predict', json={**OK, 'tarif': 'gold'})
    assert r.status_code == 422


def test_inconsistent_services(client):
    r = client.post('/predict', json={**OK, 'InternetService': 'No'})
    assert r.status_code == 422
    assert 'No internet service' in r.json()['detail'][0]['msg']


def test_batch_1000(client):
    r = client.post('/predict-batch', json=[OK, LOYAL] * 500)
    assert r.status_code == 200
    body = r.json()
    assert body['n'] == 1000 and len(body['proba']) == 1000 and len(body['decision']) == 1000
    assert all(0.0 <= p <= 1.0 for p in body['proba'])
    assert body['decision'] == [int(p >= body['threshold']) for p in body['proba']]
    # пакет даёт те же ответы, что одиночный запрос
    single = client.post('/predict', json=OK).json()['proba']
    assert body['proba'][0] == pytest.approx(single, abs=1e-4)


def test_batch_empty(client):
    r = client.post('/predict-batch', json=[])
    assert r.status_code == 400


def test_batch_too_large(client):
    r = client.post('/predict-batch', json=[OK] * (MAX_BATCH + 1))
    assert r.status_code == 413


def test_batch_with_bad_item(client):
    r = client.post('/predict-batch', json=[OK, {**OK, 'tenure': -3}, OK])
    assert r.status_code == 422
    assert r.json()['detail'][0]['loc'][:3] == ['body', 1, 'tenure']   # номер объекта в пакете
