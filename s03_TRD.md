Below is a complete, hand‑off‑ready **technical implementation plan** for an algorithmic trading strategy project built with **Notebook‑Driven Development (NDD)**. It integrates **Erik Gartner’s one‑notebook methodology** (code + tests + docs in a single artifact) with **Markus Borg’s Buttresses** (reinforcements around notebooks for quality, reproducibility, and maintainability) while following literate programming principles. The plan assumes a **DVOL (30‑day implied volatility) forecasting** concept on BTC (and optionally ETH) expressed via **DVOL futures** or **delta‑hedged options**, as scoped in the supplied planning documents.

---

## 1) Project Overview (Hypothesis, Scope, Success Metrics)

**Trading hypothesis.** Predict near‑term changes in 30‑day implied volatility (DVOL) for BTC (and optionally ETH) using only information known at a strict daily cut‑off. Monetize forecasts through DVOL futures (preferred) or delta‑hedged options structures (long straddles for IV↑, risk‑defined short‑premium for IV↓).

**Target market/instruments.**

- **Assets:** BTC (core), ETH (extension).
    
- **Instruments:** Deribit **DVOL futures**; BTC/ETH listed options (delta‑hedged).
    
- **Cadence:** Daily EOD signal, conservative sizing, strict timestamp policy.
    

**Primary modeling target.** Daily **ΔDVOL** (index‑point change), plus optional 1‑week and 2‑week horizons for tenor‑aware trade mapping.

**Success metrics (research → strategy).**

|Category|Metric|Target / Guardrail|
|---|---|---|
|Prediction|RMSE / MAE|Competitive vs. HAR/linear baselines|
|Prediction|Spearman sign corr., Hit‑rate|Sign hit‑rate > **53%** out‑of‑sample|
|Strategy (net of costs)|**Sharpe**|**≥ 1.0** OOS|
|Strategy|**Max Drawdown**|**≤ 20%**|
|Risk|Tail loss on jump days|Controlled vs. baseline|
|Ops|Data freshness SLA|All features “frozen” pre cut‑off|

Targets follow the Problem Statement and Research Design Plan.

---

## 2) Notebook Structure (Gartner + Buttresses)

**One notebook to cover everything**: _`01_dvol_research.ipynb`_. The notebook is the product during research: code, tests, documentation, EDA, backtests, and decisions are co‑located. Gartner’s approach is realized via explicit **Buttress cells** (Borg) that harden the workflow.

> **Cell Order & Tags** (use Jupyter cell tags; `# %%` headers if paired with `.py` via Jupytext)

1. **Title & Context** _(markdown)_ — problem summary, assumptions, success metrics (**doc buttress**).
    
2. **Config & Repro** _(code; tag: `parameters`, `buttress`)_ — fixed **random seeds**, `Config` dataclass, `pydantic` validation, print library versions (**env buttress**).
    
3. **Data Contracts** _(code; tag: `buttress`)_ — schema definitions (`pandera`), time‑zone normalization, cut‑off rules; assert “as‑of” joins (**data buttress**).
    
4. **Data Acquisition** _(code)_ — pull/load historical datasets; cache & persist to Parquet.
    
5. **Preprocessing** _(code)_ — cleaning, alignment, missing‑data policy, outlier handling.
    
6. **EDA & Sanity Checks** _(mixed)_ — plots, summary tables, leak checks; auto‑generated report cells.
    
7. **Feature Engineering** _(code + markdown)_ — RV/HAR, jump proxies, options flow/funding/on‑chain features, standardization.
    
8. **Baselines** _(code)_ — HAR/linear, LightGBM; cross‑validated results tables.
    
9. **Primary Models** _(code)_ — LSTM (and optional TCN), uncertainty calibration; walk‑forward scoring.
    
10. **Backtesting Layer** _(code + markdown)_ — DVOL futures P&L, options engine (delta‑hedged) with costs; plots, tear‑sheet.
    
11. **Scenario & Robustness** _(code)_ — regime tests, parameter sweeps, noise injection.
    
12. **Results & Decisions** _(markdown)_ — interpretability, ablations, go/no‑go notes (**decision buttress**).
    
13. **Inline Tests** _(code; tag: `test`, `buttress`)_ — `pytest`‑style assertions (`nbval/nbmake` compatible).
    
14. **Changelog & Next Steps** _(markdown)_ — what changed, why, and what to try next (**governance buttress**).
    

**Reinforced notebook standards.**

- Pair with **Jupytext** (`.ipynb` ↔ `.py`) to enable readable diffs (**review buttress**).
    
- Enforce style with `ruff` + `black` (`nbQA` for notebooks) (**quality buttress**).
    
- Execute end‑to‑end with `papermill`/`nbclient` for CI (**CI buttress**).
    
- Track experiments with **MLflow**/**Weights & Biases**; capture params, metrics, and artifacts (**experiment buttress**).
    

---

## 3) Data Pipeline (Source → Clean → Store)

**Data backbone:** **CryptoDataDownload (CDD)** for DVOL OHLC, Deribit options summaries (e.g., NetVega/AvgIV/Volumes), futures funding, and on‑chain BTC/ETH daily summaries. Persist daily EOD in partitioned **Parquet**.

**Acquisition & staging.**

- **Sources & frequency:**
    
    - DVOL daily OHLC (BTC/ETH) → target & IV‑level features.
        
    - Options daily summaries (NetVega, AvgIV, volume, OI proxies) → flow features.
        
    - Perp funding (1h/8h aggregated to daily mean/stdev) → risk‑appetite proxies.
        
    - Spot/futures OHLCV (Binance/Deribit) → RV/HAR/jump proxies.
        
    - On‑chain daily (tx count, fees, hashrate) → participation proxies.
        
- **Staging tables:** `raw_dvol`, `raw_options_summary`, `raw_funding`, `raw_ohlcv`, `raw_onchain`.
    
- **Feature table:** `features_daily` (one row per asset‑date): `dt, asset, y_{1d,1w,2w}, X1...Xk, masks, data_flags`.
    

**Cleaning & validation.**

- **Cut‑off discipline:** define a daily UTC cut‑off; all features must be **known before** DVOL close used as the target (**no leakage**).
    
- **Schema checks:** `pandera` schema; monotonic dates, duplicates, tz=UTC, no NaNs in target rows.
    
- **Outliers:** robust z‑score winsorization; preserve raw columns for audit.
    
- **Provenance:** compute per‑file SHA256; log row counts/date ranges; emit a **data quality report** cell.
    

**Storage & speed.**

- **Parquet** with compression (`snappy`/`zstd`), partitioning by `asset/dt`; index small metadata in SQLite/Postgres to track gaps.
    
- **Caching:** `joblib.Memory`/`diskcache` for intermediate feature frames to keep iteration fast.
    

---

## 4) Strategy Prototyping (Fast in‑notebook)

**Start simple → earn complexity.**

1. **Baselines:**
    
    - **Naïve mean‑reversion** on ΔDVOL.
        
    - **HAR/Linear** and **LightGBM** on tabular features.
        
2. **Primary model:** **LSTM** with 90‑day window across standardized features; optional multi‑task head for direction gating.
    
3. **Feature set (minimal, low‑leakage):** RV(1/5/22), HAR terms, jump proxy (RV − bipower variation)+, DVOL(t‑1), DVOL OHLC range, NetVega (and z‑score), AvgIV − DVOL, funding mean/std and 5‑day change, on‑chain level/Δ5d.
    
4. **Label(s):** ΔDVOL for 1d / 1w / 2w (index‑point changes).
    

**Inline testing while prototyping.**

- **Leakage tests:** assert all `feature_ts <= target_ts - horizon`.
    
- **Determinism:** fixed seeds; assert identical metrics on rerun.
    
- **Shape/NA tests:** unit tests on loaders, featurizers, backtester math.
    
- **Cost model test:** sanity check P&L sensitivity to fees/slippage.
    

**Recommended libraries.**

- **Data/validation:** `pandas`/`polars`, `pandera`, `pydantic`, `numpy`.
    
- **Models:** `scikit‑learn`, `lightgbm`, **`pytorch`** or **`tensorflow` (Keras)** for LSTM.
    
- **Tuning:** **`Optuna`** (pruners + time‑series CV).
    
- **Backtesting:** **`vectorbt`** (fast vectorized), plus a small custom layer for DVOL futures and delta‑hedged options specifics.
    
- **Visualization:** `matplotlib`, `plotly` for interactive tear‑sheets.
    

---

## 5) Backtesting & Metrics (Inline, Cost‑aware)

**Execution mapping.**

- **DVOL futures (primary):** signal `ŷ` (forecast ΔDVOL). Threshold **τ** so expected edge clears costs.
    
    - `ŷ ≥ +τ` → **Long** 1 contract; `ŷ ≤ −τ` → **Short** 1; else flat.
        
    - P&L: `pos_t * (F_{t+1} − F_t) − fees − slippage`.
        
- **Options (secondary):**
    
    - IV↑ → buy **ATM straddle** (7–21D), daily **delta‑hedge** with perp;
        
    - IV↓ → **short strangle/iron condor** (risk‑defined), daily hedge;
        
    - Include bid‑ask haircuts, maker/taker fees.
        

**Risk controls.** Hard caps (contracts/vega), event guardrails (reduce size on jump proxy/funding spikes), daily stop (e.g., −1.5× expected edge), kill‑switch.

**Evaluation (inline charts & tables).**

- **Prediction:** RMSE, MAE, Spearman, sign hit‑rate.
    
- **Trading (net):** Annualized return, **Sharpe**, **Sortino**, **max drawdown**, turnover, exposure time, tail loss on jump days; per‑horizon breakdown.
    
- **Regimes:** subsample results (high/low vol months, event windows).
    
- **Diagnostics:** equity curve, drawdown curve, forecast vs. realized ΔDVOL scatter, calibration plots.
    

**Framework hooks.**

- Provide a **single `backtest()` API** with dependency injection for model and cost settings.
    
- Export a **tear‑sheet** (HTML) per run; register artifact via MLflow/W&B.
    

---

## 6) Rapid Iteration Practices (Speed without Overfit)

- **Tight inner loop:** cache feature table; parameterize notebook via **Papermill** cells (`Config`) to re‑run only compute‑heavy sections.
    
- **Search budget discipline:** random search → **Optuna** Bayesian refinement on the **most sensitive** params (τ thresholds, LSTM hidden units/seq length, LightGBM depth/eta).
    
- **Walk‑forward only:** expanding‑window with **TimeSeriesSplit**; **no shuffles**.
    
- **Purged/embargoed CV** (if using overlapping labels) to avoid leakage.
    
- **Model gating:** promotion requires improving **net Sharpe and drawdown**, not just RMSE.
    
- **Ablations:** remove feature groups (flow, funding, on‑chain) to confirm incremental value.
    
- **Stability checks:** rolling feature importance; robustness to small noise in DVOL and funding; parameter sensitivity plots.
    
- **Keep feature count modest (≤ ~25)**; prefer simple, interpretable features first.
    

---

## 7) Versioning & Reproducibility (Buttresses in Practice)

**Code & notebooks.**

- **Git** + **pre‑commit** (`ruff`, `black`, `nbQA`).
    
- **Jupytext pairing** (`.py` for diffs; `.ipynb` for execution).
    
- **nbdime** for notebook‑specific diffs.
    

**Data & artifacts.**

- Partitioned **Parquet** with checksums; optionally **DVC** or **lakeFS** for data versioning.
    
- **MLflow/W&B** to track params, metrics, figures, and trained weights.
    
- Persist scalers, feature lists, thresholds per fold.
    

**Environment.**

- `pyproject.toml` (or `requirements.txt`) with **pinned versions**; lockfile via `pip‑tools`/`uv`/`poetry`.
    
- **Docker** image for CI; `Makefile`/`nox` tasks for repeatable runs.
    
- **Secrets** via environment variables (no tokens in notebook).
    

**Repro cell (top).**

- Print OS, CPU, Python, lib versions; assert GPU availability if used.
    
- Set seeds (NumPy/PyTorch/TF); configure determinism flags.
    

**CI.**

- GitHub Actions: run **`papermill`** to execute the notebook on a small rolling subset; **`pytest` + `nbval/nbmake`** for cell tests and modules.
    

---

## 8) Transition to Production (Criteria & Path)

**Promotion criteria.**

- **Data quality:** no leakage; green schema checks; stable ETL for ≥ 4 weeks.
    
- **Research hurdles met:** OOS net **Sharpe ≥ 1.0**, **MDD ≤ 20%**, sign hit‑rate > 53% on final hold‑out.
    
- **Robustness:** parameter sensitivity acceptable; regime performance not dominated by a few months.
    
- **Ops:** alerting in place; backtest results reproducible from clean checkout.
    

**Refactor plan.**

1. **Module extraction** from the notebook into `/src` (loaders, features, models, backtester). Use **nbdev** or Jupytext export to keep code and docs aligned.
    
2. **Unit tests** for modules (`pytest`, coverage targets on core logic).
    
3. **Service layer:**
    
    - **Signal API** with **FastAPI**: `/signal/{asset}` → `{yhat_1d, yhat_1w, yhat_2w, direction, confidence}`.
        
    - **Scheduler** (Prefect/Cron) for EOD ETL → feature build → score.
        
4. **Paper trading** (≥ 2–4 weeks): simulate fills & hedges; reconcile logs vs. forecasts.
    
5. **Limited live trading:** Deribit execution adapter (REST/WebSocket); strict position/vega caps; real‑time monitoring (data freshness, drift, P&L).
    
6. **Governance:** weekly review; change‑control on thresholds/models; rollback plan.
    

---

## 9) References (Research & Workflow)

- **Planning documents (project‑specific foundations):**
    
    - _Initial Idea_ — single‑vendor data backbone via CDD; DVOL OHLC as label; options flow/funding/on‑chain as features; LSTM primary model; DVOL futures & options expression.
        
    - _Problem Statement_ — objectives, success criteria (Sharpe ≥ 1.0, MDD ≤ 20%, sign hit‑rate > 53%), instruments and cadence.
        
    - _Research Design Plan_ — detailed data design, features (HAR/jump/flow/funding/on‑chain), walk‑forward evaluation, risk & deployment plan.
        
- **Workflow & tooling:** Gartner’s single‑notebook methodology (code+tests+docs in one), Borg’s **Buttresses** (reinforced standards: environment, data contracts, testing, CI, governance), **nbdev/Jupytext/nbQA/nbval** for literate, reinforced notebooks.
    
- **Methodological canon (well‑known in quant vol research):** HAR‑RV models, leakage‑safe time‑series CV, cost‑aware backtesting, delta‑hedged options P&L accounting.
    

---

### Appendix A — Library/Tool Suggestions (at a glance)

|Area|Tools|
|---|---|
|Data|`pandas` / `polars`, `pyarrow`, `pandera`, `pydantic`|
|Modeling|`scikit‑learn`, `lightgbm`, `xgboost`, `pytorch`/**`tensorflow`**|
|Tuning|**`Optuna`**|
|Backtesting|**`vectorbt`**, custom DVOL/option layer, `numpy` P&L kernels|
|Tracking|**`MLflow`**/**`Weights & Biases`**|
|Notebook QA|**`nbQA`**, **`nbdime`**, **`nbval`/`nbmake`**|
|Packaging|`poetry`/`uv`/`pip‑tools`, **Docker**|
|Orchestration|**Prefect** / cron|
|API|**FastAPI**|
|Style/CI|`ruff`, `black`, **GitHub Actions**|

---

### Appendix B — Minimal “contract” code sketches (to drop into the notebook)

> **Config & repro cell**

```python
from dataclasses import dataclass
from pydantic import BaseModel, Field, ValidationError
import numpy as np, random, os, platform, sys

SEED = 7
np.random.seed(SEED); random.seed(SEED)
try:
    import torch; torch.manual_seed(SEED); torch.use_deterministic_algorithms(True)
except Exception: pass

print("Python", sys.version, "| OS:", platform.platform())

class Config(BaseModel):
    asset: str = Field("BTC")
    horizons: tuple[int, ...] = (1, 7, 14)
    cutoff_utc: str = "00:00"
    feature_lag_days: int = 1
    seq_len: int = 90
    backtest_fee_bps: float = 1.0
CFG = Config()
```

> **Data schema buttress**

```python
import pandera as pa
from pandera.typing import Series

class DVolSchema(pa.SchemaModel):
    dt: Series[pd.DatetimeTZDtype] = pa.Field(coerce=True)
    close: Series[float]
    class Config: coerce = True

# Example check (raises on failure)
DVolSchema.validate(dvol_df)
```

These codelets illustrate the **buttress cells** (env, config, schema) that should appear early in the notebook to enforce standards.

---

## What the team does next (day‑1 checklist)

1. Scaffold repo with **`01_dvol_research.ipynb`** + **Jupytext** pairing.
    
2. Add **buttress cells** (env, config, schema, tests) and enable **pre‑commit** (`ruff`, `black`, `nbQA`).
    
3. Implement **CDD loaders** for DVOL, options summaries, funding, on‑chain; persist to **Parquet**.
    
4. Build minimal **feature table** and **HAR/LightGBM baselines**; establish walk‑forward harness.
    
5. Add **vectorized DVOL futures backtester** with fees and slippage; plot tear‑sheet.
    
6. Implement **LSTM** (90‑day lookback); calibrate thresholds **τ** via validation **net Sharpe**.
    
7. Run **robustness suite** (regimes, noise injection, ablations).
    
8. Wire **MLflow/W&B** tracking; export an HTML report per run.
    
9. Start **paper trading** once OOS hurdles are met; schedule EOD ETL→score→signal pipeline; prepare **FastAPI** endpoint for signals.
    

This plan balances rapid, single‑notebook experimentation with reinforced engineering practices so the work is reproducible, reviewable, and production‑ready when the results justify promotion. 