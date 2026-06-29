# Maia v2 — Comprehensive Development Plan

> A prediction system built on top of a chess engine. We use the chess-engine
> **"search + learned evaluation"** paradigm as a proving ground, reproduce the
> **Maia** behavioral-cloning idea (predict the move a *human* plays, not the best
> move), then generalize that idea to forecast real-world **market / business /
> political trends** — using **prediction markets** as the labeled "human choice"
> signal.

This document is the master plan. It is intentionally staged: each phase has an
objective, concrete tasks, deliverables, and an explicit **gate** that must pass
before the next phase starts. The single most important discipline in the whole
project is **honest out-of-sample evaluation** — see §7.

---

## 0. Why this lives in a Stockfish fork

This repository is the [Stockfish](../README.md) C++ source. We do **not** modify
the engine. Stockfish is the *feature source* for Layer B: a decades-stable UCI
engine we drive from Python to extract calibrated evaluation signals. All Maia v2
code is a self-contained Python project under `maia-v2/`, isolated from the C++
build. The engine is a dependency, not a thing we change.

The name "Maia v2" reflects the lineage: Maia (Microsoft Research + University of
Toronto, KDD 2020) took the Leela/AlphaZero architecture and retrained it on human
games to predict human moves. We take that *philosophy* — model the likely human
choice, errors and all — and point it at everyday forecasting.

---

## 1. Goals, scope, and non-goals

### 1.1 Primary goal
Build a working, honestly-evaluated pipeline that predicts the **most likely
outcome** of a real-world situation (market/business/political trend) by learning
from millions of historical (situation → outcome) and (situation → crowd-belief)
data points, using techniques lifted from chess engines.

### 1.2 The distinctive bet (the centerpiece)
We train **two** models, not one:

- **Behavioral model** (the Maia analog): predicts *what the crowd believes/does*.
  Its training target is the **prediction-market price itself** — actual aggregated
  human choices, biases and errors baked in.
- **Outcome model** (the Stockfish analog): predicts *what actually happened* —
  trained on realized results.

The **gap between them is the signal**: places where the crowd is *predictably*
wrong (favorite-longshot bias, chronic overpricing of rare events). Prediction
markets are the ideal substrate because the price *literally is* the aggregated
human choice, and it is miscalibrated in well-documented, systematic ways.

### 1.3 Secondary goals
- Reproduce Maia move-matching (~50% top-1 on its rating band) as a credibility
  anchor and to validate the Layer B → Layer C pipeline.
- Provide a reusable MDP / "search + learned evaluation" abstraction (Layer A) with
  worked **logistics** and **finance** blueprints.

### 1.4 Non-goals (explicit)
- **Not** a true "world model" simulator (predicting next-state given state+action).
  Ours is a policy/value *predictor* — simpler and the right tool here.
- **Not** high-frequency trading, market making, or anything latency-sensitive.
- **Not** financial advice or an autotrader. Outputs are research signals only.
- **Not** modifying Stockfish's search or NNUE. We consume it via UCI.

### 1.5 Project-level success criteria
1. Layer B reproduces Lichess win% within a few points on a held-out game sample.
2. A tabular outcome model beats a material-only baseline on AUC/calibration.
3. Pretrained Maia top-1 move-match reproduced within ~2pp of published numbers.
4. The market-trend predictor shows a **positive, honest out-of-sample edge** over a
   naive persistence/last-value baseline on a chronological test split.
5. The two-model divergence is shown to be **calibrated** against realized outcomes
   on data the models never saw — and is *not* merely re-deriving the current price.

---

## 2. System architecture

Three layers (B → C → A), plus the two-model design that is the product.

```
                          ┌─────────────────────────────────────────────┐
                          │  LAYER A — Generalization (MDP framework)     │
                          │  state encoder · model/sim · value · planner  │
                          │  → logistics (OR-Tools/MPC) · finance (RL/MPC)│
                          └───────────────▲─────────────────────────────┘
                                          │ abstracts
   ┌──────────────────────────┐   ┌───────┴───────────────────────────────┐
   │ LAYER B — Engine features │   │ LAYER C — Human/outcome prediction     │
   │ Stockfish/Lc0 via UCI     │──▶│ • outcome model (GBT, fast baseline)   │
   │ cp · WDL · MultiPV · PV   │   │ • Maia reproduction (CNN policy/value) │
   └──────────────────────────┘   └────────────────────────────────────────┘
                                          │ same recipe, retargeted
                                          ▼
                 ┌───────────────────────────────────────────────────────┐
                 │ PRODUCT — Two-model trend engine (markets/politics)     │
                 │  BEHAVIORAL model  →  P(crowd belief)  (target: price)  │
                 │  OUTCOME    model  →  P(reality)       (target: result) │
                 │  EDGE = divergence(outcome − behavioral)                │
                 └───────────────────────────────────────────────────────┘
```

**What transfers from chess and what breaks** (the honest caveat that governs
design choices):

| Concept | Transfers? | Notes |
|---|---|---|
| Learned value/eval function `V(s)` | ✅ | This is the heart — NNUE/value-net ≈ RL value fn |
| Policy/value decomposition | ✅ | Maia loss = cross-entropy on human move + MSE on outcome |
| MCTS / PUCT planning | ✅ | General MDP method (logistics rollouts; limited in finance) |
| Search + eval = MPC / model-based RL | ✅ | Receding-horizon planning over a sim |
| Behavioral cloning (Maia) | ✅ | The core idea we generalize |
| Minimax / alpha-beta | ❌ | Adversarial two-player only — *not* our problem |
| Exact tree search over futures | ❌ | Futures are unknown stochastic processes — plan over **distributions/expectations**, never enumerated price paths |
| Known deterministic transition | ❌ | False in markets, partly false in logistics |
| Cheap perfect simulator | ❌ | Chess rules are free; market/demand sims are approximations whose error compounds |

---

## 3. Proposed repository layout

All new code lives under `maia-v2/`. Four-layer separation; the eval harness is the
**only** code allowed to report accuracy.

```
maia-v2/
├── README.md                  # orientation (entry point)
├── DEVELOPMENT_PLAN.md        # this document
├── pyproject.toml             # uv-managed; deps pinned
├── config/
│   └── settings.example.toml  # paths, API keys (never commit real keys)
├── data/                      # gitignored; parquet snapshots live here
├── maia_v2/
│   ├── ingest/                # one module per source → parquet
│   │   ├── lichess.py         # PGN → tabular (pgn-extract + zstd streaming)
│   │   ├── prices.py          # yfinance → EODHD bulk
│   │   ├── prediction_markets.py  # Polymarket + Kalshi
│   │   ├── fred.py            # macro series
│   │   ├── gdelt.py           # (Tier 2) event features
│   │   └── usaspending.py     # (Tier 2) state↔private link
│   ├── engine/                # LAYER B
│   │   ├── uci.py             # python-chess driver, reproducible limits
│   │   ├── features.py        # cp/WDL/MultiPV → feature dicts
│   │   └── winprob.py         # cp↔win% models (sf / lichess), pinned
│   ├── chess_models/          # LAYER C
│   │   ├── outcome_gbt.py     # tabular GBT baseline
│   │   ├── policy_value_net.py# Maia-style CNN
│   │   └── maia_loader.py     # load pretrained Maia nets (UofTCSSLab)
│   ├── trend/                 # PRODUCT — tiny market/trend predictor
│   │   ├── features.py        # rolling-window feature engineering
│   │   ├── tiny_net.py        # 1D-conv policy(3-way)+value net
│   │   └── two_model.py       # behavioral vs outcome + divergence
│   ├── generalize/            # LAYER A
│   │   ├── mdp.py             # state/action/transition/value interfaces
│   │   ├── planners.py        # MPC / MCTS-with-chance-nodes / CEM
│   │   ├── logistics/         # newsvendor, VRP envs + OR-Tools baseline
│   │   └── finance/           # execution env, Almgren-Chriss baseline
│   ├── entity/                # name↔ticker resolution (Tier 2 join key)
│   └── eval/                  # THE ONLY place accuracy is reported
│       ├── splits.py          # chronological / walk-forward / purged-embargoed
│       ├── metrics.py         # top-1, calibration, Sharpe, Deflated Sharpe, PBO
│       └── report.py
├── notebooks/                 # exploration only; never the source of truth
└── tests/
```

Add a `maia-v2/.gitignore` for `data/`, `*.parquet`, model checkpoints, and
`config/settings.toml`. (The repo root `.gitignore` already ignores
`__pycache__/` and `*.nnue`.)

---

## 4. Tech stack & environment

| Concern | Choice | Rationale |
|---|---|---|
| Language | **Python 3.11+** | Where python-chess, PyTorch, and every data API live |
| Env / deps | **uv** (lockfile committed) | Fast, reproducible; conda acceptable alt |
| Engine driver | **python-chess** (`chess.engine`, UCI) | Decades-stable protocol; use `depth`/`nodes` limits (not `time`) for reproducibility |
| Deep models | **PyTorch** | CNN policy/value (Maia-style), tiny 1D-conv trend net |
| Tabular models | **LightGBM / XGBoost** | Strong, fast, well-calibrated baselines |
| Data wrangling | **pandas**, then **polars/DuckDB** | DuckDB queries parquet off-disk without loading to RAM |
| Storage | **Parquet**, partitioned by date | Format the Polymarket/Kalshi archives already ship in |
| HTTP | **httpx** + on-disk cache | Avoid re-hitting rate limits |
| RL / envs | **gymnasium** + **stable-baselines3** | PPO/SAC/DQN for Layer A |
| Logistics baseline | **OR-Tools** (routing + CP-SAT), **OR-Gym** | The classical baseline RL must beat |
| Chess engine binary | **Stockfish** (this repo, NNUE/CPU); optional **Lc0** (GPU, policy+WDL) | Build Stockfish via `src/Makefile`; Maia nets are Lc0-format |

**Coexistence with the C++ build:** Maia v2 never touches `src/`. CI for the engine
and CI for `maia-v2/` are separate. A helper script builds Stockfish once and
records the binary path into `config/settings.toml` for the Python layer.

---

## 5. Datasets (tiered) & data pipeline

**Discipline: ship the entire pipeline on Tier 1 before touching Tier 2.** Every
source added multiplies overfitting risk on the same noisy target. The join key
across everything is the **date/timestamp**.

### Tier 1 — build with *only* these
| Source | What it gives | Access | Size |
|---|---|---|---|
| **Lichess open DB** | ~7.8B rated games, Glicko-2 ratings, clocks, ~6% `%eval` | monthly PGN + zstd | large; stream, don't decompress fully |
| **yfinance** (→ EODHD later) | OHLCV across thousands of tickers | free / freemium bulk endpoint | small |
| **Polymarket + Kalshi** | event→price history (the "human choice" layer) | free public endpoints | outcomes+15-min history ≈ few hundred MB |
| **FRED** | macro backdrop (rates, inflation, employment) | one free key | small |

> **Prediction-market sizing** (decided): we use **outcomes + 15-minute price
> history**, *not* tick/order-book archives. That tier is a few hundred MB to low
> single-digit GB for the full history of both platforms — fits on a laptop. The
> 107GB Polymarket / 36GB Kalshi full-trade archives and TB-scale order-book tiers
> are explicitly **out of scope**.

### Tier 2 — only after Tier 1 trains and evaluates honestly
- **GDELT** — political/economic/diplomatic events (since 1979, 15-min updates),
  joined on date. Queryable in BigQuery for scale.
- **USAspending.gov** — the literal state↔private link (who got which federal
  contract/grant). No key. Requires the **entity-resolution** step first
  (company name → ticker), which is the real cost.

### Tier 3 — only if a specific hypothesis demands it
- **ACLED** (political violence/protests), **Congress.gov** (bill/vote timestamps).

### Pipeline shape (four stages, strictly layered)
```
ingest/  (one script per source → parquet)
   └─▶ features/  (join into one dated table; engineer rolling features)
          └─▶ model/  (GBT, Maia CNN, tiny trend net, two-model)
                 └─▶ eval/  (walk-forward; the ONLY accuracy reporter)
```
**Freeze a data snapshot per experiment** so results reproduce. Entity resolution
gets its own module and a cached mapping table built once.

---

## 6. Phased roadmap

Effort estimates assume one focused developer; treat as relative sizing, not
commitments.

### Phase 0 — Setup & scaffolding  *(~2–3 days)*
- **Tasks:** `uv` project + pinned deps; build Stockfish from `src/`; smoke-test
  UCI from python-chess; create the directory skeleton (§3); add `maia-v2/.gitignore`;
  wire a trivial CI job that lints + runs a placeholder test.
- **Deliverable:** `import maia_v2`, engine handshake works, `uv run pytest` green.
- **Gate:** A Python script can run `engine.analyse(board, Limit(depth=12))` and
  read back a score.

### Phase 1 — Layer B: engine feature extraction  *(~1–2 weeks)*
- **Tasks:** UCI driver with reproducible `depth`/`nodes` limits and `multipv`;
  extract `cp`, `wdl`, `pv`, `depth`, `multipv_spread` (sharpness), legal-move count,
  phase; implement cp→win% with **both** the Stockfish (`sf`) and Lichess sigmoids,
  **pinned by name** for reproducibility; per-move centipawn-loss / win%-drop;
  eval-volatility feature.
- **Deliverable:** `engine/features.py` producing a tidy per-position feature row;
  batch runner over a few thousand Lichess games → parquet.
- **Gate (project criterion #1):** extracted win% reproduces Lichess `%eval` within
  a few points on annotated games. *Don't silently mix the sf and lichess curves.*

### Phase 2 — Layer C: outcome model, then Maia reproduction  *(~2–4 weeks)*
- **2a — Tabular outcome model first (faster to validate):** engine features +
  position features + ratings → **LightGBM/XGBoost** predicting W/D/L; report AUC and
  a reliability diagram.
  - **Gate (criterion #2):** outcome AUC clearly beats a material-only baseline.
    *Only then* invest in deep move prediction.
- **2b — Reproduce/adopt Maia (don't train from scratch):** load pretrained Maia
  nets (Hugging Face `UofTCSSLab`); measure **top-1 move-match** on the matching
  rating band, excluding the first ~10 ply and low-clock moves (as the papers do).
  - **Gate (criterion #3):** within ~2pp of published (~50% on band).
- **2c — Custom Maia-style net (optional):** PGN → 13-plane (or 112-plane) encoder →
  residual CNN → policy (cross-entropy on human move) + value (MSE on outcome).
  Condition on rating via embedding to mirror Maia bands. Budget data wrangling time
  (the Maia team's full conversion took ~4 days on an 80-core box).
- **Deliverable:** `chess_models/` with GBT baseline, Maia loader + move-match
  report, optional trainable net.

### Phase 3 — Layer A: the generalization framework  *(~1–2 weeks)*
- **Tasks:** define the MDP interfaces (`state encoder`, `model/sim`, `value`,
  `planner`); implement planners as pluggable strategies — **MPC/CEM**,
  **MCTS-with-chance-nodes**, and a learned-policy path; encode the decision rule for
  *which* planner fits *which* problem (see §2 table).
- **Deliverable:** `generalize/mdp.py` + `planners.py` with one toy MDP end-to-end.
- **Gate:** the same planner interface drives both a deterministic and a stochastic
  toy problem; minimax is *deliberately absent*.

### Phase 4 — Product: the tiny market/trend predictor  *(~1–2 weeks)*
- **Tasks:** rolling-window features (recent returns, rolling vol, MA ratio); buckets
  **DOWN / FLAT / UP** over a horizon (mirrors W/D/L) + a value head for expected
  magnitude; tiny 1D-conv policy/value net (tens of thousands of params, CPU). Get to
  "millions of data points" by **stacking many tickers/series** and overlapping
  windows.
- **Deliverable:** `trend/` predictor trained on stacked Tier-1 price series; a single
  forward pass at inference (no search), à la Maia.
- **Gate (criterion #4):** **chronological** (never shuffled) out-of-sample accuracy
  beats a persistence/last-value baseline. In-sample accuracy is ignored entirely.

### Phase 5 — Product: the two-model behavioral/outcome engine  *(~2–4 weeks)*
This is the project's reason for being.
- **Tasks:**
  1. **Behavioral model** — target = the prediction-market **price** (Polymarket/Kalshi).
     Optionally condition on a *sophistication proxy* (liquidity, retail-vs-institutional
     venue) to model sharper vs. less-sophisticated crowds — the Maia "rating band" idea.
  2. **Outcome model** — target = realized resolution of the same events.
  3. **Divergence engine** — `edge = outcome_prob − behavioral_prob`; surface events
     where the crowd is *predictably* miscalibrated.
- **Deliverable:** `trend/two_model.py` + an eval report quantifying divergence vs.
  realized outcomes on unseen events.
- **Gate (criterion #5):** divergence is **calibrated** out-of-sample **and** the
  behavioral model is demonstrably *not* just re-deriving the current price (the
  circularity trap — value lives entirely in divergence from outcomes). If the
  behavioral model only reproduces price, the phase has produced nothing.

### Phase 6 — Layer A worked blueprints (logistics, then finance)  *(advanced/optional)*
- **Logistics (the forgiving target):** newsvendor + VRP gym envs; an MPC/value-greedy
  reorder policy ("search + eval" with Monte-Carlo *expectation* over demand, not
  minimax) and a PPO agent; **benchmark against OR-Tools** — ship only if we beat it on
  cost *or* latency. Switch to MPC/OR-Tools when the sim is accurate and the action
  space is combinatorial.
- **Finance (the unforgiving target):** Almgren-Chriss / TWAP execution baseline; a
  gym execution env where the price step is **drawn from a distribution** (no tree of
  enumerated futures); PPO agent that adapts to realized moves.
- **Gate / kill criteria:** logistics — beat OR-Tools. Finance — purged+embargoed CV,
  report **Deflated Sharpe** and **Probability of Backtest Overfitting**; **kill if
  PBO > 0.5 or out-of-sample Sharpe ≤ 0.**

---

## 7. Evaluation & validation discipline (non-negotiable)

The fastest way to fool yourself here is in-sample accuracy. Rules:

1. **Never shuffle time series.** All splits are **chronological**. The eval harness
   (`eval/splits.py`) is the only code that constructs splits, and the only code that
   prints accuracy.
2. **Walk-forward** for trend models; **purged + embargoed cross-validation** for any
   finance backtest (prevents leakage across the horizon boundary).
3. **Calibration over raw accuracy** for probabilistic targets — reliability diagrams,
   Brier score. A "90% accurate" model that's miscalibrated is worse than useless.
4. **Move-match metrics** for Maia: top-1 (primary), top-3, perplexity, with the
   paper's preprocessing (drop first ~10 ply, low-clock moves).
5. **Finance only:** **Deflated Sharpe Ratio** (corrects for selection bias / multiple
   testing) and **Probability of Backtest Overfitting** (CSCV). Pre-register the
   strategy; cap the number of backtests; treat any backtested edge as guilty until
   proven innocent out-of-sample.
6. **Snapshot + seed everything.** One frozen data snapshot per experiment; record
   git SHA, config, and seeds in every report.

**Kill criteria, stated up front:** discard a finance strategy if `PBO > 0.5` or
`out-of-sample Sharpe ≤ 0`. Discard a logistics policy that doesn't beat OR-Tools on
cost or latency. Stop a two-model phase that only re-derives market price.

---

## 8. Risks & mitigations

| Risk | Severity | Mitigation |
|---|---|---|
| **Overfitting / data snooping** | High | Chronological splits, capped backtests, DSR/PBO, Tier-1-first discipline |
| **Non-stationarity** (markets drift) | High | Walk-forward; regime-shift held-out tests; never assume stationarity |
| **Circularity** (behavioral model = price) | High | Score *only* on divergence from realized outcomes, not price reproduction |
| **Entity resolution cost** (name→ticker) | Medium | Isolate in `entity/`; build a cached mapping once; defer to Tier 2 |
| **Data volume / wrangling** | Medium | Stream zstd; DuckDB-on-parquet; partition by date; budget Maia-scale time |
| **Mixing win% curves** (sf vs lichess) | Medium | Pin model name; never mix silently |
| **Sim model bias** (Layer A) | Medium | Benchmark vs classical baselines; prefer MPC when sim is trusted |
| **Legal / ToS / ethics** | Medium | Respect data-provider ToS; outputs are research signals, **not financial advice**; no autotrading; keep API keys out of git |
| **Scope creep into a "world model"** | Low | Explicit non-goal; we build a policy/value predictor |

---

## 9. Milestone summary

| Phase | Outcome | Gate to advance |
|---|---|---|
| 0 | Project builds; engine handshake | `analyse(depth=12)` returns a score |
| 1 | Layer B features | win% ≈ Lichess `%eval` within a few pts |
| 2 | Outcome GBT + Maia reproduction | GBT beats material-only; Maia within ~2pp |
| 3 | MDP framework | one planner drives det. + stochastic toy MDPs |
| 4 | Tiny trend predictor | OOS accuracy beats persistence baseline |
| 5 | **Two-model divergence engine** | divergence calibrated OOS, not circular |
| 6 | Logistics + finance blueprints | beat OR-Tools; finance passes DSR/PBO or is killed |

---

## 10. Immediate next steps (the first build PR after this plan)

1. Land this plan (this PR).
2. Phase 0: scaffold `maia-v2/` per §3, add `pyproject.toml` (uv) with pinned deps,
   `maia-v2/.gitignore`, and `config/settings.example.toml`.
3. Build Stockfish from `src/` and record the binary path; commit a `scripts/` helper.
4. Write `engine/uci.py` + a smoke test (`analyse` returns a score) to satisfy the
   Phase 0 gate.
5. Begin Phase 1 Layer B feature extraction on a small Lichess sample.

---

## Appendix A — Key references

- **AlphaZero** — Silver et al., *Science* 362:1140 (2018), arXiv:1712.01815.
- **MuZero** — Schrittwieser et al., *Nature* 588:604 (2020).
- **Maia** — McIlroy-Young, Sen, Kleinberg, Anderson, KDD 2020, arXiv:2006.01855;
  **Maia-2** NeurIPS 2024, arXiv:2409.20553; **Maia-3 / "Chessformer"** ICLR 2026
  (79M params, 57.1% move-match).
- **KL-Regularized Search** — Jacob et al. (2021), arXiv:2112.07544.
- **MCTS** — Kocsis & Szepesvári (2006, UCT); Browne et al. (2012, survey).
- **NNUE** — Yu Nasu (2018), computer shogi; ported to Stockfish 12 (Aug 2020).
- **RL for VRP** — Nazari et al. (NeurIPS 2018); Kool et al. (2019); survey arXiv:2205.02453.
- **Beer Game DQN** — Oroojlooyjadid et al. (*M&SOM* 2021), arXiv:1708.05924.
- **DiDi dispatch** — Qin et al. (*INFORMS J. Applied Analytics* 50(5):272–286).
- **Optimal execution** — Almgren & Chriss (2001); Nevmyvaka, Feng & Kearns (ICML 2006);
  Hendricks & Wilcox (2014, IEEE CIFEr, arXiv:1403.2229).
- **Overfitting discipline** — Bailey & López de Prado, Deflated Sharpe Ratio (2014);
  Probability of Backtest Overfitting (CSCV).

## Appendix B — Data source quick reference

| Source | Tier | Key/Auth | Notes |
|---|---|---|---|
| Lichess open DB | 1 | none | monthly PGN+zstd; ratings, clocks, ~6% evals |
| yfinance | 1 | none | bulk OHLCV; move to EODHD for 30-yr depth |
| EODHD | 1+ | key (paid) | bulk endpoint: full exchange in seconds |
| Polymarket | 1 | none | official price-history endpoint |
| Kalshi | 1 | none | public market-data endpoints |
| FRED | 1 | one free key | 845k macro series |
| GDELT | 2 | none | events since 1979; BigQuery for scale |
| USAspending | 2 | none | federal contracts/grants; needs entity resolution |
| ACLED | 3 | free reg. | political violence/protests |
| Congress.gov | 3 | key via api.data.gov | bill/vote timestamps |

> Reproducibility note: pin the python-chess WDL model name (`sf` tracks latest
> Stockfish; use `sf12`/`lichess` for stable results). Re-verify live dataset sizes
> (Lichess game counts grow monthly) before publishing any figure.
