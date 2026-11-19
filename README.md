# lambdamart_v11 — Gold Signal Inference Service

A small FastAPI service that runs the **XGBoost LambdaMART ranker (`v11`)** on HSH965 gold-price data, records BUY signals, and tracks each one until it hits take-profit, stop-loss, or the end of its trading session.

> This is the first stand-alone deployment of the model. The full system (live WebSocket candles, LLM agent, dashboards) lives in **[นักขุดทอง — Gold Trading AI Signal Generator](https://github.com/athiphat67/CN240_DATA-SCIENCE-FOR-SIGNAL-PROCESSING_PROJECT)**.

## How it works

```
Supabase (input DB: price ticks)
  ▼ POST /cron/predict   → builds features (synthetic THB gold price, macro momentum,
                           session stats, RSI/Bollinger) → ranker score
                           → record a signal if score ≥ 0.0744 (one per candle)
  ▼ POST /cron/track     → scans later ticks for the first TP / SL hit, or closes the
                           order at session end (Morning 12:00 · Afternoon 18:00 · Night 02:00 BKK)
Supabase (output DB: signals + outcomes)
```

An external scheduler calls both endpoints. `GET /` is a health check.

## Engineering notes

The header of [`app.py`](app.py) logs **8 bugs found and fixed** before deployment. Most were time-series correctness issues that quietly break a backtest-to-live handoff:

- Ticks read newest-first, so the tracker judged outcomes from the wrong end of the series.
- Separate resamples misaligned the high/low index.
- Session timeouts were hard-coded, so orders leaked across sessions.
- Duplicate predictions were written for the same candle.

## Run it

```bash
pip install -r requirements.txt
export SUPABASE_URL_INPUT=... SUPABASE_KEY_INPUT=... SUPABASE_URL_OUTPUT=... SUPABASE_KEY_OUTPUT=...
uvicorn app:app --port 8000
```

The `Procfile` deploys it as-is on Render / Heroku-style hosts.

> Signals are for research only and are **not financial advice**.
