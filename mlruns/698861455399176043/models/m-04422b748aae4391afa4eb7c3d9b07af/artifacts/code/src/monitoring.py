"""Журнал запросов сервиса и суточный отчёт о распределении прогнозов (доп. задание).

PredictionLogger пишет записи JSONL в фоновом потоке, чтобы запись на диск
не добавлялась к задержке ответа. daily_report строит по журналу отчёт за
сутки: объём запросов, отказы, задержки, распределение вероятностей и его
сдвиг относительно эталона (PSI).
"""
import base64
import io
import json
import os
import queue
import threading
import time

import numpy as np
import pandas as pd


class PredictionLogger:
    def __init__(self, path: str):
        self.path = path
        if path:
            os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        self._q: queue.Queue = queue.Queue()
        self._thread = threading.Thread(target=self._writer, daemon=True)
        self._thread.start()

    def log(self, record: dict):
        if self.path:
            self._q.put({'ts': time.strftime('%Y-%m-%dT%H:%M:%S'), **record})

    def _writer(self):
        while True:
            rec = self._q.get()
            if rec is None:
                self._q.task_done()
                return
            with open(self.path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(rec, ensure_ascii=False) + '\n')
            self._q.task_done()

    def flush(self):
        self._q.join()

    def close(self):
        self._q.put(None)
        self._thread.join(timeout=5)


def read_log(path: str) -> pd.DataFrame:
    with open(path, encoding='utf-8') as f:
        df = pd.DataFrame([json.loads(line) for line in f if line.strip()])
    df['ts'] = pd.to_datetime(df['ts'])
    df['day'] = df['ts'].dt.strftime('%Y-%m-%d')
    return df


def psi(reference, current, bins: int = 10) -> float:
    """Population Stability Index по квантильным корзинам эталона."""
    reference, current = np.asarray(reference), np.asarray(current)
    edges = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    r = np.histogram(reference, edges)[0] / len(reference)
    c = np.histogram(current, edges)[0] / len(current)
    r, c = np.clip(r, 1e-4, None), np.clip(c, 1e-4, None)
    return float(np.sum((c - r) * np.log(c / r)))


def psi_level(value: float) -> str:
    if value < 0.1:
        return 'стабильно'
    return 'умеренный сдвиг' if value < 0.25 else 'значимый сдвиг'


def _hist_png(reference, current) -> str:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.5, 3))
    bins = np.linspace(0, 1, 26)
    ax.hist(reference, bins, density=True, alpha=0.5, label='эталон (тест ЛР1)')
    ax.hist(current, bins, density=True, alpha=0.5, label='прогнозы за сутки')
    ax.set_xlabel('вероятность ухода'); ax.set_ylabel('плотность'); ax.legend()
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=110)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def daily_report(log_path: str, day: str, reference_proba, threshold: float,
                 out_html: str | None = None) -> dict:
    """Суточный отчёт по журналу: возвращает сводку и (по желанию) сохраняет HTML."""
    log = read_log(log_path)
    log = log[log['day'] == day]
    ok = log[log['status'] == 200]
    proba = np.concatenate([np.asarray(p) for p in ok['proba']]) if len(ok) else np.array([])
    reference_proba = np.asarray(reference_proba)

    by_endpoint = log.groupby(['endpoint', 'status']).agg(
        requests=('ts', 'size'), objects=('n', 'sum'),
        latency_p50_ms=('latency_ms', 'median'),
        latency_p95_ms=('latency_ms', lambda s: s.quantile(0.95))).reset_index()

    batches = ok[ok['endpoint'] == '/predict-batch'].copy()
    batches['mean_proba'] = batches['proba'].map(np.mean)
    batches['share_offer'] = batches['proba'].map(lambda p: float(np.mean(np.asarray(p) >= threshold)))
    batches['psi'] = batches['proba'].map(lambda p: psi(reference_proba, p) if len(p) >= 100 else np.nan)
    batches['оценка'] = batches['psi'].map(lambda v: psi_level(v) if v == v else 'мало объектов')
    batches['n'] = batches['n'].astype(int)
    batches = batches[['ts', 'request_id', 'n', 'latency_ms', 'mean_proba', 'share_offer', 'psi', 'оценка']]

    summary = {
        'day': day, 'requests': int(len(log)), 'rejected_422': int((log['status'] == 422).sum()),
        'objects_scored': int(len(proba)),
        'mean_proba': float(proba.mean()) if len(proba) else None,
        'share_offer': float((proba >= threshold).mean()) if len(proba) else None,
        'expected_churners': float(proba.sum()),
        'ref_mean_proba': float(reference_proba.mean()),
        'ref_share_offer': float((reference_proba >= threshold).mean()),
        'psi_day': psi(reference_proba, proba) if len(proba) >= 100 else None,
    }
    summary['psi_level'] = psi_level(summary['psi_day']) if summary['psi_day'] is not None else '—'

    if out_html:
        q = np.quantile(proba, [0.1, 0.25, 0.5, 0.75, 0.9]) if len(proba) else [np.nan] * 5
        quant = pd.DataFrame({'квантиль': ['10%', '25%', '50%', '75%', '90%'],
                              'сутки': np.round(q, 4),
                              'эталон': np.round(np.quantile(reference_proba, [0.1, 0.25, 0.5, 0.75, 0.9]), 4)})
        img = _hist_png(reference_proba, proba) if len(proba) else ''
        html = f"""<html><head><meta charset="utf-8"><title>Суточный отчёт {day}</title>
<style>body{{font-family:sans-serif;margin:24px}}table{{border-collapse:collapse}}
td,th{{border:1px solid #bbb;padding:3px 8px;text-align:right}}</style></head><body>
<h2>Суточный отчёт о прогнозах сервиса оттока — {day}</h2>
<p>Запросов: {summary['requests']}, отклонено (422): {summary['rejected_422']},
оценено клиентов: {summary['objects_scored']}. Порог решения: {threshold}.</p>
<p>Средняя вероятность ухода: {summary['mean_proba']:.4f} (эталон {summary['ref_mean_proba']:.4f});
доля клиентов со скидкой: {summary['share_offer']:.3f} (эталон {summary['ref_share_offer']:.3f});
ожидаемое число уходящих: {summary['expected_churners']:.0f}.</p>
<p><b>PSI прогнозов за сутки: {summary['psi_day'] if summary['psi_day'] is None else round(summary['psi_day'], 4)}
— {summary['psi_level']}</b> (&lt;0,1 — стабильно, 0,1–0,25 — умеренный сдвиг, &gt;0,25 — значимый).</p>
<h3>Запросы по эндпоинтам</h3>{by_endpoint.round(2).to_html(index=False)}
<h3>Пакеты</h3>{batches.round(4).to_html(index=False)}
<h3>Квантили вероятности</h3>{quant.to_html(index=False)}
<img src="data:image/png;base64,{img}"></body></html>"""
        os.makedirs(os.path.dirname(out_html) or '.', exist_ok=True)
        with open(out_html, 'w', encoding='utf-8') as f:
            f.write(html)
    return {'summary': summary, 'by_endpoint': by_endpoint, 'batches': batches}
