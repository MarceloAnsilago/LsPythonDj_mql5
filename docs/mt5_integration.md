# Integracao MT5 no LongShort

- Yahoo (yfinance) é a fonte oficial para candles D1 fechados (datas < hoje); MT5 continua entregando ticks live/intraday via `QuoteLive` e `LiveTick` sem persistir D1.
- O histórico completo é mantido em `cotacoes.QuoteDaily`; `mt5api.DailyPrice` continua disponível apenas para compatibilidade legada.

## Ativar MT5 para um ativo
- Marque `use_mt5=True` no cadastro de `Asset` (admin ou formulario do app).
- Quando marcado, o ativo passa a receber ticks live via MT5/QuoteLive; o intraday não é persistido e o Yahoo continua abastecendo o histórico D1.
- Pivot, analises e metricas continuam lendo `cotacoes.QuoteDaily` (Yahoo); `mt5api.DailyPrice` fica apenas para compatibilidade administrativa.

## Entrada de ticks (tempo real)
- Endpoint: `POST /api/mt5/push-live-quote/`
- Cabecalho obrigatorio: `X-API-KEY` = `MT5_API_KEY` (variavel de ambiente). IP e validado se `MT5_ALLOWED_IPS` estiver configurado.
- Payload JSON esperado:
  ```json
  { "ticker": "PETR4", "bid": 37.10, "ask": 37.12, "last": 37.11 }
  ```
- A view grava:
  - `QuoteLive` (ultimo preco ao vivo por ativo)
  - `LiveTick` (um registro por tick recebido)

## Consolidacao diaria (OHLC)
- Comando base: `python manage.py agg_daily_prices --date YYYY-MM-DD` (use sempre data < hoje) apenas informa que a consolidação MT5 foi desativada.
- Os ticks MT5 seguem sendo gravados em `LiveTick`, mas os candles D1 são mantidos via Yahoo em `cotacoes.QuoteDaily`.

-## Provedor de precos unificado
- Arquivo: `longshort/services/price_provider.py`
- Funcao: `get_daily_prices(asset, start_date, end_date)` retorna lista ordenada de dicts `{date, open, high, low, close}`.
  - Usa `cotacoes.QuoteDaily` para todos os ativos (todos os candles D1 fechados vêm do Yahoo).
  - `mt5api.DailyPrice` segue disponível apenas para compatibilidade administrativa.
- Toda a camada de metricas, pivots e `/api/mt5/get-signal/` passou a consumir esta funcao.

## Relatorios no admin
- Lista padrao de `DailyPrice` em `/admin/mt5api/dailyprice/`.
- Pivot MT5 em `/admin/mt5api/dailypricepivot/`:
  - Filtros: `ticker`, `data inicial`, `data final`
  - Renderiza tabela dinamica (datas x tickers) com `close` consolidado.

## Historico oficial
- Ativos `use_mt5=True`:
  - D1 principal em `cotacoes.QuoteDaily` via Yahoo (datas < hoje); os ticks MT5 continuam gerando `QuoteLive`/`LiveTick`.
  - Yahoo garante o histórico completo; os logs de faltantes ajudam a detectar falhas pontuais.
  - `PriceHistory` (app_pares) mantém compatibilidade sincronizando com `QuoteDaily`.

## Rotina diaria recomendada
1. Receber ticks via `push-live-quote` ao longo do dia.
2. Consolidar e validar:
   - `python manage.py mt5sync_daily --date YYYY-MM-DD` (sempre data < hoje) apenas verifica ticks e avisa; `agg_daily_prices` hoje só exibe um aviso de que a consolidação MT5 foi desativada.
3. Executar fluxos de pares/estrategia normalmente; as metricas usarao `cotacoes.QuoteDaily` (Yahoo) para todos os ativos.
