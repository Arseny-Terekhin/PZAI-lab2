"""Нагрузочный сценарий сервиса оттока (вариант 1, пакетный режим).

Клиенты берутся из data/sample_clients.json (реальные записи тестовой выборки).
Смесь: одиночные запросы (оператор колл-центра проверяет клиента) и пакеты
по 100 клиентов (выгрузка части суточного пакета).
"""
import json
import random

from locust import HttpUser, between, task

with open('data/sample_clients.json', encoding='utf-8') as f:
    CLIENTS = json.load(f)


class ScoringUser(HttpUser):
    wait_time = between(0.01, 0.05)

    @task(5)
    def predict(self):
        self.client.post('/predict', json=random.choice(CLIENTS))

    @task(1)
    def batch(self):
        self.client.post('/predict-batch', json=random.sample(CLIENTS, 100))
