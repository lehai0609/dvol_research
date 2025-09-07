# Research Design Plan — DVOL Forecasting & Volatility Trading

## 1) Project Overview

**Problem Restatement.** We aim to forecast changes in the 30‑day implied volatility index (DVOL) for Bitcoin (primary) and optionally Ether (secondary) over three horizons: next calendar day, next 7 days, and next 14 days. Implied volatility (IV) is the market’s consensus estimate of future volatility backed out from option prices. We will monetize forecasts by taking positions in DVOL futures or by constructing options portfolios whose profit and loss primarily depend on changes in implied volatility (for example, delta‑hedged at‑the‑money straddles for rising IV and risk‑defined short‑premium spreads for falling IV).

**Primary Research Question.** Can we build a predictive model, using only information available before the daily cut‑off, that delivers out‑of‑sample forecasts of DVOL changes that are strong enough to sustain a positive, cost‑adjusted Sharpe ratio when mapped to real trades?

**Scope.**

- **Assets:** BTC (core), ETH (optional extension).
    
- **Instruments:** Deribit DVOL futures (preferred expression) and BTC/ETH listed options to implement delta‑hedged long‑volatility or short‑volatility structures when appropriate.
    
- **Horizons:** 1 day, 1 week, 2 weeks.
    
- **Cadence:** Daily end‑of‑day (EOD) signal with strict timestamp discipline.
    
- **Success criteria:** Out‑of‑sample, net‑of‑costs Sharpe ≥ 1.0; maximum drawdown ≤ 20%; statistically significant directional hit‑rate > 53% over a long, independent test period; positive tail behavior on large volatility shock days.
    

---

## 2) Data Design

### 2.1 Sources (initial wave)

- **Implied volatility index:** Deribit DVOL daily OHLC for BTC (and ETH when available).
    
- **Options market activity:** Daily summaries of options volume, open interest, and net vega (or proxies derivable from exchange summaries). If net vega is not directly available, we will approximate using contract specifications and midpoint IVs.
    
- **Spot and futures tape:** Daily OHLCV for BTCUSDT (large spot venues) and Deribit perpetuals; aggregated funding rates (8‑hour windows aggregated to daily features).
    
- **On‑chain activity:** Daily summaries such as transaction count, fees, and hashrate. These are used as slow‑moving risk‑appetite and congestion proxies.
    
- **Optional external attention data (phase 2):** Google Trends and social sentiment feeds, included only if strictly timestamped before the daily cut‑off.
    

### 2.2 Frequency, cut‑off, and timezone

- **Native frequency:** We work at daily resolution for modeling and backtesting.
    
- **Cut‑off rule:** We define a fixed daily cut‑off at 00:00 UTC. All features must be finalized and “frozen” before the DVOL close used as the target. This avoids look‑ahead bias (also called data leakage).
    
- **Timezone normalization:** All timestamps are converted to UTC during processing and only converted to local time at the reporting layer when needed.
    

### 2.3 Data ingestion and storage

- **ETL:** Pull raw files from the provider’s endpoints once per day after the cut‑off. Persist to partitioned Parquet (by asset and date) with schema validation.
    
- **Staging tables:** `raw_ohlcv`, `raw_dvol`, `raw_options_summary`, `raw_funding`, `raw_onchain`.
    
- **Feature table:** `daily_features` with one row per asset‑date after the cut‑off; includes lagged targets for supervised learning.
    

### 2.4 Data cleaning and validation rules

- **Missing data:** Forward‑fill single‑day gaps where appropriate; drop periods with sustained outages; mark imputed fields with boolean flags.
    
- **Bad ticks/outliers:** Detect and winsorize with robust z‑scores (median and median absolute deviation) at the feature‑engineering stage; keep raw values for audit.
    
- **Cross‑source consistency:** Funding, options, and DVOL dates must align to the same “as‑of date.” We perform as‑of joins keyed by asset and UTC date.
    
- **Audit trails:** Every pipeline run emits row counts, date ranges, and hash checksums for downstream reproducibility.
    

### 2.5 Feature engineering (daily, low‑leakage)

We compute only features that would have been known before the target DVOL close:

- **Realized volatility family:**
    
    - Close‑to‑close log returns over 1, 5, and 22 trading days; their standard deviations form RV(1), RV(5), RV(22).
        
    - Heterogeneous autoregressive (HAR) terms: regressors that embed yesterday’s RV, last week’s average RV, and last month’s average RV to capture persistent volatility dynamics.
        
    - Jump proxy: Positive part of RV minus bipower variation to flag jump days.
        
    - Overnight gap size and sign (from defined close to next open proxy).
        
- **Implied‑volatility level and structure:**
    
    - DVOL level (t‑1), daily change, and intraday range (from DVOL OHLC).
        
    - Term structure slope if 7‑day and 30‑day IV proxies are both available (short‑term IV minus 30‑day DVOL).
        
    - “Rich‑cheap” proxy: average listed IV around at‑the‑money minus DVOL level when derivable.
        
- **Options flow:**
    
    - Total options volume, change in open interest, and net vega (or a proxy). Include 5‑day z‑scores to standardize across regimes.
        
- **Funding/basis:**
    
    - Daily mean and standard deviation of perpetual funding; 5‑day change to capture shifts in positioning pressure.
        
- **On‑chain activity:**
    
    - Daily transaction count and total fees; 5‑day changes as participation proxies.
        

**Transformations:**

- Log transforms where distributions are heavy‑tailed (volume, fees).
    
- Robust scaling (median/MAD) fit on the training set only, then applied to validation/test.
    
- Lags and rolling windows strictly end at the cut‑off.
    

**Leakage control checklist:**

- “As‑of” joins; no forward‑fills across the cut‑off; targets shifted by +1, +7, +14 days as appropriate; scalers and encoders fit only on training windows.
    

---

## 3) Hypotheses and Signal Design

### 3.1 Economic intuition

1. **Short‑horizon mean reversion in IV:** After sharp spikes in implied volatility, market makers over‑hedge and option demand normalizes, causing IV to drift back down. After tranquil periods, unexpected realized volatility tends to pull IV up.
    
2. **Realized → Implied linkage:** Future implied volatility is partly anchored to recently realized volatility, but with regime dependence (trending or jumpy markets versus calm regimes).
    
3. **Options flow as information:** Changes in net vega and volume reveal demand/supply imbalances for volatility that precede changes in DVOL.
    
4. **Funding and basis as risk appetite:** Rising positive funding (longs paying shorts) can precede increased realized volatility and risk of IV repricing.
    
5. **On‑chain activity as participation:** Surges in fees or transactions often accompany narrative‑driven moves and volatility regime shifts.
    

### 3.2 Target variables and labels

- **Regression targets:** ΔDVOL over 1 day, 7 days, and 14 days (index‑point change). We may also use percentage change as a secondary target for scale‑invariance.
    
- **Optional classification labels:** Up / Flat / Down classes created by thresholding ΔDVOL, which can be used to gate positions or to train a joint model (multi‑task learning).
    

### 3.3 Candidate signal variables

All features listed in §2.5, plus engineered interactions (for example, DVOL level × funding change) where justified and regularized to avoid overfitting.

### 3.4 Stationarity and stability checks

- Augmented Dickey–Fuller (ADF) tests on key features; use differencing or percentage changes when needed.
    
- Rolling correlation and feature‑importance stability reports to guard against regime‑specific artifacts.
    

---

## 4) Modeling Approach

### 4.1 Baselines (must‑beat)

- **Naïve reversion:** Predict next ΔDVOL = −k × last ΔDVOL, k calibrated on training data.
    
- **HAR‑RV / linear models:** Lasso or ridge regression on the realized‑volatility family and simple flow features.
    

### 4.2 Primary candidates

- **Gradient‑boosted trees:** LightGBM or XGBoost on tabular daily features; strong baselines with transparent feature importances.
    
- **Sequence model:** Long Short‑Term Memory (LSTM) using a 90‑day lookback window across standardized features; a regression head for ΔDVOL and an optional auxiliary classification head (multi‑task).
    
- **Temporal convolution (TCN) as ablation:** Causal 1‑D convolutions over the same 90‑day window.
    

### 4.3 Losses, regularization, and calibration

- **Loss:** Huber loss for regression (robust to outliers) plus 0.2× cross‑entropy when jointly training classification.
    
- **Regularization:** Dropout (0.1) for LSTM, early stopping on a rolling validation set, l2 weight decay for trees if supported.
    
- **Uncertainty:** Calibrate predictive intervals via quantile regression (for trees) or Monte Carlo dropout (for LSTM) to produce confidence‑aware position sizing.
    

### 4.4 Hyperparameter search

- **Method:** Random search followed by Bayesian optimization over the most sensitive parameters (tree depth, learning rate, lookback length, hidden units, dropout).
    
- **Budget:** Cap trials per walk‑forward slice; persist all trials with seeds and metrics.
    

### 4.5 Cross‑validation and walk‑forward design

- **Expanding‑window walk‑forward:**
    
    - Split the history into sequential folds. For fold _i_, train on start…Tᵢ, validate on (Tᵢ, Vᵢ], test on (Vᵢ, Xᵢ].
        
    - Retrain and rescore at each step to mimic live operation.
        
- **Leakage defenses:** Scalers and encoders re‑fit within each fold; no peeking at future targets or features.
    

### 4.6 Ensembling and model selection

- Blend tree and LSTM forecasts with weights tuned on validation Sharpe (cost‑adjusted) rather than RMSE alone.
    

---

## 5) Evaluation Framework

### 5.1 Backtesting methodology

- **Signal generation:** For each date, produce horizon‑specific forecasts using only information available at the cut‑off.
    
- **Trade mapping:**
    
    - **Primary:** DVOL futures: go long if forecast ≥ positive threshold; go short if ≤ negative threshold; otherwise flat. Thresholds are horizon‑specific and chosen to exceed estimated trading costs.
        
    - **Secondary (safety‑first):** Options structures:
        
        - Rising IV view → buy at‑the‑money straddle with 7–21 days to expiry; delta‑hedge once per day using the underlying perpetual contract so P&L is dominated by IV changes rather than price direction.
            
        - Falling IV view → sell a risk‑defined short strangle or an iron condor centered around at‑the‑money strikes; enforce wings to cap tail risk.
            
- **Re‑hedging policy (options):** Daily at the cut‑off; additional intra‑day hedges can be studied in sensitivity analysis.
    
- **Cost model:** Include maker/taker fees, slippage via half‑spread assumptions (venue‑specific), and funding transfer for perpetual hedges.
    

### 5.2 Datasets and periods

- **Training start:** As early as reliable DVOL and options data allow.
    
- **Hold‑out test:** Reserve the most recent 20–30% of the timeline as a final, untouched test set.
    
- **Regime sub‑tests:** High‑volatility months, low‑volatility months, and event windows (halvings, macro shocks).
    

### 5.3 Metrics

- **Prediction metrics:** Root Mean Squared Error (RMSE), Mean Absolute Error (MAE), Spearman rank correlation (for directionality), and hit‑rate for sign prediction.
    
- **Trading metrics (net of costs):** Annualized return, Sharpe ratio, Sortino ratio, maximum drawdown, turnover, average trade duration, and tail‑loss on jump days.
    
- **Benchmarks:** Flat (no‑trade), and a simple long‑only DVOL future (if investable) or a static options book with daily delta‑hedging.
    

### 5.4 Robustness and stress testing

- **Parameter sensitivity:** Thresholds, lookback windows, and re‑hedge frequency.
    
- **Noise injection:** Price and IV micro‑jitter to emulate slippage.
    
- **Resampling:** Block bootstrap of returns to estimate confidence intervals on Sharpe and drawdown.
    
- **Post‑mortems on worst months:** Attribution by feature regime and by instrument.
    

---

## 6) Risk and Constraint Analysis

- **Market risk:** Volatility regime shifts and clustered jumps that can overwhelm short‑volatility structures; illiquid DVOL contract hours.
    
- **Model risk:** Overfitting to a short sample; feature drift where relationships change; under‑estimated costs.
    
- **Operational risk:** Data lags, API downtime, incomplete options snapshots, time‑zone mishandling.
    
- **Constraints:** Hard caps on vega exposure per asset; maximum leverage and margin utilization; event guardrails that reduce size when jump proxies exceed a threshold; daily stop‑loss and kill‑switch.
    
- **Compliance and governance:** Clear audit logs, reproducible runs, change‑control with approvals before deployment of new models or thresholds.
    

---

## 7) Experiment Tracking and Reproducibility

- **Versioning:** Git for code; Data Version Control (or partitioned Parquet with checksums) for data; MLflow or Weights & Biases for models and metrics.
    
- **Environment:** Frozen `requirements.txt` / container image with pinned library versions; deterministic random seeds where applicable.
    
- **Artifacts:** Persisted scalers, feature lists, and trained model weights per walk‑forward fold.
    
- **Documentation:** A living experiment log: what changed, why it changed, and what we observed.
    

---

## 8) Deployment Plan (Staging → Live)

1. **Data pipeline in staging.** Nightly ETL, data‑quality checks, and feature table materialization.
    
2. **Research loop.** EDA notebooks, baseline models, and first walk‑forward reports.
    
3. **Trading simulation.** Costed backtests for DVOL futures and for the options engine with delta‑hedging.
    
4. **Paper trading.** Live signals with simulated fills; monitor feature freshness, forecast drift, and P&L attribution.
    
5. **Limited production.** Small size on Deribit with strict guardrails; continuous monitoring and weekly review.
    

**Deliverables:** Clean datasets; feature dictionary; modeling notebooks; backtest reports with plots and tables; a small service that returns `{forecast, direction, confidence}` for each horizon; and an execution notebook for each instrument choice.

---

## 9) Glossary (plain‑English)

- **Implied volatility (IV):** The annualized volatility level implied by option prices; it reflects the market’s expectation of future variability.
    
- **DVOL:** Deribit’s 30‑day implied volatility index for an underlying (for example, BTC), designed to summarize the option market’s IV around a 30‑day horizon.
    
- **Delta‑hedged straddle:** Buy one call and one put at the same strike and expiry (usually at‑the‑money) and frequently hedge the net delta with the underlying so P&L depends mainly on IV changes and realized volatility rather than price direction.
    
- **Short‑premium spread (short strangle / iron condor):** Sell options away from the current price to collect premium, usually adding protective wings (buying farther options) to cap downside; these benefit when IV falls or stays contained.
    
- **HAR model:** A regression approach that uses yesterday’s, last week’s, and last month’s realized volatility to predict tomorrow’s realized volatility.
    
- **Bipower variation:** A robust volatility estimator that filters out jumps, helping us identify jump components.
    
- **Net vega:** The net sensitivity of an options portfolio to changes in IV; a proxy for market demand for volatility.
    

---

## 10) Open Questions to Resolve Early

- Exact availability and definition of options summary fields (for example, whether net vega can be derived reliably each day).
    
- Contract specifications and liquidity patterns for DVOL futures across time; finalize a conservative cost and slippage model.
    
- Whether to prefer percentage or absolute changes in DVOL for each horizon, balancing interpretability and stationarity.
    
- Inclusion criteria and lags for on‑chain metrics to guarantee they are known before the cut‑off.
    

---

## 11) Milestones and Timeline (indicative)

**Week 1–2:** Build and validate the daily ETL and feature table; produce data‑quality dashboards.

**Week 3–4:** Implement baselines (HAR, linear) and tree models; first expanding‑window backtests; preliminary threshold calibration for DVOL futures mapping.

**Week 5–6:** Implement the LSTM sequence model; add uncertainty calibration; run robustness tests and ablations.

**Week 7:** Options engine with delta‑hedging simulator; add cost curves and stress tests; compare DVOL futures vs. options expressions.

**Week 8:** Paper‑trade launch with monitoring; weekly governance review; go/no‑go for limited production size.