# Integracao MT5 no LongShort

- MT5 é a fonte principal (live/intraday + D1 fechado). Intraday não persiste em `QuoteDaily/DailyPrice`.
- Yahoo permanece apenas como fallback para histórico faltante/ativos novos (grava em `QuoteDaily`), sempre com data < hoje.

## Ativar MT5 para um ativo
- Marque `use_mt5=True` no cadastro de `Asset` (admin ou formulario do app).
- Quando marcado, o ativo **nao** usa Yahoo para intraday; Yahoo só é chamado para backfill se faltar histórico fechado.
- Pivot, analises e metricas passam a ler `mt5api.DailyPrice` em vez de `cotacoes.QuoteDaily`.

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
- Comando base: `python manage.py agg_daily_prices --date YYYY-MM-DD` (use sempre data < hoje).
- Consolida todos os `LiveTick` do dia por ticker:
  - `open`: primeiro tick do dia (`last`)
  - `close`: ultimo tick do dia
  - `high`/`low`: maximas e minimas do dia
- Resultado e salvo em `mt5api.DailyPrice` via `update_or_create`.

## Provedor de precos unificado
- Arquivo: `longshort/services/price_provider.py`
- Funcao: `get_daily_prices(asset, start_date, end_date)` retorna lista ordenada de dicts `{date, open, high, low, close}`.
  - Se `asset.use_mt5=True` -> le `mt5api.DailyPrice`.
  - Caso contrario -> usa `cotacoes.QuoteDaily` (replicando OHLC com `close`).
- Toda a camada de metricas, pivots e `/api/mt5/get-signal/` passou a consumir esta funcao.

## Relatorios no admin
- Lista padrao de `DailyPrice` em `/admin/mt5api/dailyprice/`.
- Pivot MT5 em `/admin/mt5api/dailypricepivot/`:
  - Filtros: `ticker`, `data inicial`, `data final`
  - Renderiza tabela dinamica (datas x tickers) com `close` consolidado.

## Historico oficial com MT5
- Ativos `use_mt5=True`:
  - D1 principal em `DailyPrice` via MT5 (ticks agregados ou HTTP).
  - Yahoo só entra como fallback (QuoteDaily) para buracos/ativos novos.
  - `PriceHistory` (app_pares) também é sincronizado a partir de `DailyPrice`.

## Rotina diaria recomendada
1. Receber ticks via `push-live-quote` ao longo do dia.
2. Consolidar e validar:
   - `python manage.py mt5sync_daily --date YYYY-MM-DD` (sempre data < hoje)
   - Esse comando roda `agg_daily_prices`, verifica se todos `use_mt5=True` tem `DailyPrice` e alerta se houver poucos/nenhum tick.
3. Executar fluxos de pares/estrategia normalmente; as metricas usarao `DailyPrice` para ativos MT5.
