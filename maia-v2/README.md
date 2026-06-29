# Maia v2

**A prediction system built on top of a chess engine.**

Maia v2 uses the chess-engine *"search + learned evaluation"* paradigm as a proving
ground, reproduces the **Maia** idea of predicting the move a *human* would actually
play (not the objectively best move), and generalizes that behavioral-cloning idea to
forecast real-world **market / business / political trends** — using **prediction
markets** (Polymarket, Kalshi) as the labeled "human choice" signal.

This subproject lives inside a [Stockfish](../README.md) fork. **It does not modify the
engine.** Stockfish is consumed via the UCI protocol as a calibrated feature source.

## The core idea

Train **two** models instead of one:

- **Behavioral model** (the *Maia* analog) — predicts what the crowd *believes*.
  Trained on the prediction-market **price** itself (human choices, biases included).
- **Outcome model** (the *Stockfish* analog) — predicts what *actually happened*.
  Trained on realized results.

The **divergence** between them — where the crowd is *predictably* wrong — is the
signal the whole project is built to find.

## Where to start

📖 **[DEVELOPMENT_PLAN.md](./DEVELOPMENT_PLAN.md)** — the comprehensive, phased plan:
architecture, repo layout, tech stack, tiered datasets, the staged roadmap (with gates),
and the evaluation discipline that keeps the project honest.

## The three layers (see the plan for detail)

- **Layer B** — extract engine signals (centipawn eval, WDL, MultiPV, PV) from
  Stockfish/Lc0 via `python-chess`.
- **Layer C** — predict human moves (reproduce Maia) and game outcomes.
- **Layer A** — generalize the MDP "search + learned evaluation" abstraction to
  logistics and finance.

## Status

📋 Planning. No implementation code yet — Phase 0 (scaffolding) is the next step.
