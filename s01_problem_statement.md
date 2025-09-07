
## 1) Objective / Research Question

Predict next‑day/next week or next 2 weeks change in 30‑day implied volatility (IV) for BTC (and optionally ETH) and monetize the forecast via long/short volatility trades. Primary target: daily ΔDVOL (Deribit 30‑day IV index). Translate forecasts into positions in DVOL futures or delta‑hedged options structures (long straddles for IV↑, short strangles/iron condors for IV↓).

**Success Criteria (research):** Out‑of‑sample Sharpe ≥ 1.0 (net of realistic costs), max drawdown ≤ 20%, positive skew in daily P&L, and statistically significant directional accuracy (>53%).

## 2) Market & Asset Scope

- Underlyings: BTC (core), ETH (secondary).
    
- Instruments: Deribit DVOL futures (primary expression), BTC/ETH options for delta‑hedged straddles/short‑premium spreads.
    
- Trading cadence: Daily EOD signal; positions sized conservatively with hard risk limits.
    

## 3) Data Inputs & Sources (CryptoDataDownload)

- **IV proxy:** Deribit DVOL daily OHLC (BTC/ETH).
    
- **Options flow/surface:** Deribit options daily summaries (e.g., NetVega, AvgIV, volume, OI proxies).
    
- **Market tape:** Spot/futures OHLCV (e.g., Binance spot BTCUSDT, Deribit perpetuals) and aggregated funding rates (1h/8h → daily).
    
- **On‑chain (macro‑microstructure):** Daily BTC/ETH summaries (tx count, fees, hashrate, block interval).
    
- (Optional later) External sentiment/attention (Google Trends, social), integrated only if timestamped pre‑cutoff.
    

**Time alignment:** All features timestamped and frozen before the daily DVOL close used for prediction (strict leakage control).

## 4) Hypothesis / Signal Assumption

1. IV exhibits short‑horizon mean reversion and regime dependence. 2) Realized volatility dynamics (HAR terms, jump proxies) and options flow (net vega/volume) provide predictive information about next‑day/next week or next 2 weeks IV changes. 3) Funding/basis and on‑chain activity proxy investor risk appetite, improving forecasts.
    

## 5) Modeling Approach

- **Baselines:** HAR‑RV/Lasso/LightGBM on daily features.
    
- **Primary model:** LSTM with 90‑day lookback; regression head for ΔDVOL and optional classification head (up/flat/down) for position gating.
    
- **Loss & regularization:** Huber loss (reg) + 0.2× cross‑entropy (cls), dropout 0.1, early stopping, train‑only scaling.
    
- **Validation:** Daily walk‑forward (expanding window); report RMSE/MAE, Spearman sign correlation, and directional hit rate.
    

## 6) Features (daily, low‑leakage)

- **Price/volatility:** Realized vol RV_1/5/22; HAR terms; overnight gap; jump proxy (RV − bipower variation, clipped at 0).
    
- **IV/surface & flow:** DVOL level (t−1); DVOL OHLC range; 7d→30d term slope (if available); AvgIV − DVOL (rich/cheap); NetVega and its 5‑day z‑score; options volume/OI changes.
    
- **Funding/basis:** Perp funding mean & std (daily), 5‑day change.
    
- **On‑chain:** Tx count, fees, mempool/throughput proxies, and 5‑day changes.
    

## 7) Trade Mapping & Execution Rules

- **Signal (ŷ):** Multi‑horizon forecasts: ΔDVOL for t+1d, t+1w, t+2w; horizon‑specific thresholds (τ₁d, τ₁w, τ₂w) learned on validation to clear costs and mapped to tenor‑aligned trades.
    
- **Positioning:**
    
    - If ŷ ≥ +τ → Long 1 DVOL future (or long ATM straddle, 7–21d, delta‑hedged daily).
        
    - If ŷ ≤ −τ → Short 1 DVOL future (or short strangle/iron condor, risk‑defined, delta‑aware).
        
    - Else → Flat.
        
- **Risk controls:** Max contracts/vega; event guardrails (reduce size on jump/funding spikes); daily stop‑loss; hard kill‑switch.
    

## 8) Backtesting & Evaluation

- **Forecast metrics:** RMSE, MAE, Spearman, hit rate.
    
- **Strategy metrics (net of costs):** Sharpe, Sortino, max drawdown, turnover, tail loss on jump days. DVOL futures P&L uses close‑to‑close changes and contract settlement rules; options P&L is delta‑hedged mark‑to‑market with daily re‑hedge.
    
- **Robustness:** Parameter sensitivity (τ, lookback, hedge frequency), regime subsamples, and out‑of‑time periods.
    

## 9) Deployment Context

- **ETL:** Daily pull from CryptoDataDownload → raw parquet → feature table.
    
- **Modeling service:** Walk‑forward refresh and scoring at fixed EOD schedule; artifacts versioned.
    
- **Signal API:** `/signal/{asset}` returns {ŷ, direction, confidence}.
    
- **Execution:** Paper trade first; then controlled size on Deribit; full audit logs and real‑time monitors (data freshness, feature drift, P&L).