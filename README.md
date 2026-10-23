# ЛР2. Вывод модели в эксплуатацию — вариант 1 (отток клиентов Telco)

Терехин А., группа 23ИВ1б. Дисциплина «Прикладные задачи искусственного интеллекта», ПензГТУ.

Модель ЛР1 (LightGBM, порог 0,10, ROC-AUC на тесте 0,849) выведена в сервис: модули, MLflow,
FastAPI, тесты, нагрузка, дрейф, Gradio. Доп. задание — журнал запросов и суточный отчёт о прогнозах.

| Путь | Содержание |
|---|---|
| `lab2_terekhin.ipynb` | Ноутбук Colab: выполняется сверху вниз после сброса среды, сам создаёт модули и все артефакты |
| `src/features.py` | Признаки из ЛР1 (`clean`, `add_features`, `CategoryAligner`) |
| `src/train.py` | Загрузка данных, разбиение, обучение (`train_model`), метрики, стоимость ошибок, схема входа |
| `src/predict.py` | Загрузка модели из реестра MLflow по имени/псевдониму, `predict_proba(список словарей)` |
| `src/service.py` | FastAPI: `/health`, `/model-info`, `/predict`, `/predict-batch`; схема Pydantic |
| `src/monitoring.py` | Журнал запросов (JSONL, фоновый поток), PSI, суточный отчёт |
| `tests/test_service.py` | 19 проверок pytest |
| `locustfile.py` | Сценарий нагрузки Locust |
| `mlruns/` | Журнал экспериментов и реестр моделей MLflow (файловое хранилище) |
| `models/input_schema.json` | Описание входа модели |
| `reports/` | `drift_report.html` (Evidently), `daily_report.html`, таблицы нагрузки и дрейфа, графики, результат pytest |
| `report/report_lab2_terekhin.pdf` | Отчёт |
| `requirements.txt` | `pip freeze` окружения, в котором выполнен ноутбук |

## Запуск в Colab

1. Скопировать репозиторий на Google Диск в `MyDrive/lab2_terekhin` (или открыть только ноутбук — он создаст
   каталог сам и скачает данные с проверкой MD5).
2. Среда выполнения без ускорителя → «Выполнить всё». Единственное ручное действие — доступ к Диску.
   Если установка обновит уже загруженный numpy/pandas, ноутбук сам перезапустит среду — тогда снова «Выполнить всё».

## Запуск вне Colab

```bash
pip install -r requirements.txt
export LAB2_PROJECT=$PWD MLFLOW_ALLOW_FILE_STORE=true
jupyter nbconvert --to notebook --execute lab2_terekhin.ipynb --inplace
# или только сервис (после того как ноутбук зарегистрировал модель):
MLFLOW_TRACKING_URI=file:$PWD/mlruns uvicorn src.service:app --port 8000
python -m pytest tests/ -q
```

Пути в `mlruns/` абсолютные (`/content/drive/MyDrive/lab2_terekhin/...`), поэтому реестр из репозитория
загружается как есть только из этого каталога; в другом каталоге ноутбук создаёт свой журнал заново.
