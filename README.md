# LsPythonDj – MT5 + Yahoo (híbrido)

- MT5 é a fonte principal para preço ao vivo/intraday e candles D1 fechados; Yahoo (yfinance) só entra como fallback de histórico (ativos novos ou buracos). Nenhum dado intraday é gravado em `DailyPrice` (apenas D1 `< hoje`).
- Modelos-chave: `mt5api.LiveTick` (tick cru), `mt5api.DailyPrice` (OHLC D1 MT5 ou fallback Yahoo), `cotacoes.QuoteLive` (preço ao vivo por ativo) e `cotacoes.QuoteDaily` (legado/fallback visual).
- Views/serviços: `longshort/services/mt5_provider.py` (bridge HTTP MT5), `longshort/services/quotes.py` (live + histórico MT5/Yahoo), `longshort/services/price_provider.py` (fonte unificada D1), `/api/mt5/push-live-quote/` (entrada de ticks), `/api/mt5/push-daily/` (entrada OHLC D1).

## Live MT5 (entrada principal)
- Endpoint: `POST /api/mt5/push-live-quote/`
- Headers: `X-API-KEY: <MT5_API_KEY>` + IP whitelist (`MT5_ALLOWED_IPS`).
- Payload exemplo: `{"ticker":"PETR4","bid":37.10,"ask":37.12,"last":37.11,"as_of":"2025-01-11T14:30:00-03:00"}`
- Efeitos: grava `LiveTick` e atualiza `QuoteLive` via `apply_live_quote` (sem .SA nem `ticker_yf`).

## Push D1 MT5
- Endpoint: `POST /api/mt5/push-daily/`
- Payload exemplo: `{"ticker":"PETR4","date":"2025-01-11","open":37.0,"high":38.0,"low":36.5,"close":37.5}`
- Somente aceita `date < hoje`; grava/atualiza `mt5api.DailyPrice`.

## Consolidação diária (D1 fechado)
- Comando: `python manage.py mt5sync_daily --date YYYY-MM-DD` (ou `agg_daily_prices` direto) **sempre com data < hoje**. Consolida `LiveTick` em `DailyPrice` (OHLC) por ticker MT5.
- Página “Cotações” e cards usam `get_daily_prices` para D1; intraday é apenas visual (DOM/polling) via `QuoteLive/MT5` e não persiste em D1.

## Yahoo como fallback
- `ensure_min_history_for_asset` e `bulk_update_quotes` recorrem ao Yahoo para preencher histórico fechado que faltar (ativos novos ou buracos), gravando em `DailyPrice` (somente data < hoje).
- Ativos `use_mt5=True` usam MT5 como fonte primária; Yahoo só entra quando o D1 MT5 não cobre o mínimo exigido.

### Teste rápido (Yahoo / CCRO3)
```
python manage.py shell -c "from acoes.models import Asset; from longshort.services.quotes import ensure_daily_history; ensure_daily_history(Asset.objects.get(ticker='CCRO3'))"
```
Depois abra `/cotacoes/` para ver se os candles D1 apareceram.

## Variáveis de ambiente úteis
- `USE_MT5_LIVE`, `MT5_API_KEY`, `MT5_ALLOWED_IPS`, `MT5_HTTP_BASE`
- `DJANGO_TIME_ZONE=America/Sao_Paulo`
- `DJANGO_ALLOWED_HOSTS`, `DJANGO_CSRF_TRUSTED_ORIGINS`, `DATABASE_URL`

## Smoke rápido
```
curl -X POST http://127.0.0.1:8000/api/mt5/push-live-quote/ \
  -H "X-API-KEY: <MT5_API_KEY>" \
  -H "Content-Type: application/json" \
  -d '{"ticker":"PETR4","last":37.11}'

python manage.py agg_daily_prices --date 2025-01-11
python manage.py shell -c "from mt5api.models import LiveTick, DailyPrice; print(LiveTick.objects.count()); print(DailyPrice.objects.count())"
```
