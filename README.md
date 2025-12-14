# LsPythonDj – MT5 + Yahoo (híbrido)

- Yahoo (yfinance) é a fonte única persistida para candles D1 fechados e histórico diário; MT5 permanece só para ticks/live e visualização intraday via `QuoteLive` e `LiveTick`.
- Modelos-chave: `mt5api.LiveTick` (tick cru), `cotacoes.QuoteLive` (preço ao vivo por ativo), `cotacoes.QuoteDaily` (fonte oficial dos candles D1 com dados Yahoo) e `mt5api.DailyPrice` (legado/compatibilidade).
- Views/serviços: `longshort/services/mt5_provider.py` (bridge HTTP MT5), `longshort/services/quotes.py` (live + histórico com Yahoo), `longshort/services/price_provider.py` (fonte unificada D1), `/api/mt5/push-live-quote/` (entrada de ticks) e `/api/mt5/push-daily/` (mantido só por compatibilidade).

## Live MT5 (entrada principal)
- Endpoint: `POST /api/mt5/push-live-quote/`
- Headers: `X-API-KEY: <MT5_API_KEY>` + IP whitelist (`MT5_ALLOWED_IPS`).
- Payload exemplo: `{"ticker":"PETR4","bid":37.10,"ask":37.12,"last":37.11,"as_of":"2025-01-11T14:30:00-03:00"}`
- Efeitos: grava `LiveTick` e atualiza `QuoteLive` via `apply_live_quote` (sem .SA nem `ticker_yf`).

## Push D1 MT5
- Endpoint: `POST /api/mt5/push-daily/`
- Payload exemplo: `{"ticker":"PETR4","date":"2025-01-11","open":37.0,"high":38.0,"low":36.5,"close":37.5}`
- Somente aceita `date < hoje`; atualmente retorna erro informando que o histórico D1 é mantido via Yahoo.

## Consolidação diária (D1 fechado)
- Comando: `python manage.py mt5sync_daily --date YYYY-MM-DD` **sempre com data < hoje**; apenas valida ticks recebidos e alerta, pois a consolidação MT5 foi desativada (`agg_daily_prices` agora só mostra o aviso). `QuoteDaily` vem do Yahoo.
- Página “Cotações” e cards usam `get_daily_prices` para D1 (dados Yahoo); intraday continua apenas visual (DOM/polling) via `QuoteLive/MT5` sem persistência D1.

## Yahoo como fonte oficial
- `ensure_min_history_for_asset` e `bulk_update_quotes` garantem `QuoteDaily` via Yahoo (datas < hoje), cobrindo histórico e buracos.
- Ativos `use_mt5=True` continuam recebendo ticks live via MT5, mas os candles D1 exibidos e armazenados seguem vindo do Yahoo, com logs de faltantes ajudando a detectar falhas.

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

python manage.py agg_daily_prices --date 2025-01-11  # somente informa que a consolidação MT5 foi desativada
python manage.py shell -c "from mt5api.models import LiveTick, DailyPrice; print(LiveTick.objects.count()); print(DailyPrice.objects.count())"
```
