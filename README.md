# Polymarket Long-Shot Bias Decay — Analysis (2022–2026)

A four-phase quantitative autopsy of the long-shot bias anomaly in Polymarket between November 2022 and February 2026, over 593 million on-chain trades.

**Full write-up:** (https://medium.com/p/f325f4a3d1b7?postPublishedType=initial)

## What this repo is

Four self-contained Python scripts that ingest the full Polymarket trade history and progressively validate (or falsify) the hypothesis that buying moderate underdogs at 30–40¢ is structurally profitable. The phases escalate: each one asks a sharper question than the previous, and the conclusions of each phase are used as filter criteria for the next.

| phase | script | question answered | runtime |
|---|---|---|---|
| 0 | `fase0_analysis_v3.py` | Does a long-shot bias exist in the aggregate dataset? | ~10 min |
| 0.5 | `fase05_validation.py` | Does the edge survive four independent sanity checks (temporal, walk-forward, volume, wallets)? | ~15 min |
| 0.6 | `fase06_category_survival.py` | Does any category preserve edge in Q1 2026 after the general decay? | ~6 min |
| 0.7 | `fase07_quarterly_volatility.py` | Is the Q1 2026 signal in surviving categories statistically distinguishable from quarterly noise? | ~5 min |

The final answer, if you want to skip to it: **no operable edge as of 2026 Q1**. See Fase 0.7 output or the Medium post for the numbers.

## Key findings

1. **Bucket 0.30–0.40 showed +2.19% EV per dollar globally** across 10.2 million trades (2022–2026).
2. The edge **decayed monotonically by year**: +9.77% (2023) → +4.16% (2024) → +1.50% (2025) → −4.96% (2026 Q1). Consistent with alpha decay in a competitive market.
3. The mirror bucket 0.60–0.70 showed symmetric opposite flip. Since binary prediction markets have mathematically paired buckets, this is a single structural mispricing with two sides.
4. In Q1 2026, three large categories (Politics, Sports, Other) appeared to flip to positive EV in the 0.60–0.70 bucket. **Quarterly z-score testing revealed this was noise**: all three z-scores below 2σ.
5. Generalizable rule: *When a market's quarterly σ exceeds its aggregate μ, there is no edge — there is a distribution.*

## Dataset

The scripts operate on two CSV inputs you need to produce before running anything:

- `~/poly_data/processed/trades.csv` — Polymarket on-chain trade history. 593 M rows, ~144 GB. Schema: `timestamp, market_id, maker, taker, nonusdc_side, maker_direction, taker_direction, price, usd_amount, token_amount, transactionHash`.
- `~/poly_data/markets.csv` — Polymarket market metadata. ~125 K rows, a few MB. Schema: `createdAt, id, question, answer1, answer2, neg_risk, market_slug, token1, token2, condition_id, volume, ticker, closedTime`.

### How to obtain the trades dataset

Use the public goldsky orderbook subgraph for Polymarket: `project_cl6mb8i9h0003e201j6li0diw/subgraphs/orderbook-subgraph/0.0.1/gn`. A basic GraphQL paginator that dumps the `orderFilled` entity to CSV will produce the full history in ~10 hours on a decent connection. A reference ingestion script is at `tools/ingest_goldsky.py` (not included here; see the Medium write-up for pointers to the warproxxx/poly_data community repo which has a working version that needs minor patching as of April 2026).

### How to obtain the markets dataset

Use Polymarket's gamma-api with pagination: `https://gamma-api.polymarket.com/markets?closed=true&order=createdAt&ascending=true&limit=500&offset=<N>`. Note: this endpoint returns HTTP 500 on large offsets during US trading hours; run the ingestion at night (CET) in batches.

## Requirements

- Python 3.11 or higher
- Polars 1.25 or higher (streaming engine API changed around this version)
- ~200 GB free disk space (144 GB dataset + scratch + outputs)
- A machine with at least 16 GB RAM. Everything uses streaming aggregations so no full dataset is loaded into memory.

Install with:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install polars
```

That's the only dependency.

## How to reproduce the four-phase analysis

After you have `trades.csv` and `markets.csv` in place:

```bash
# Phase 0: discover the U-shape, compute Q1/Q2/Q3
python3 fase0_analysis_v3.py

# Phase 0.5: four validation tests
python3 fase05_validation.py

# Phase 0.6: category × year × bucket survival
python3 fase06_category_survival.py

# Phase 0.7: quarterly volatility z-score test
python3 fase07_quarterly_volatility.py
```

Each phase reads outputs from the previous one under `~/poly_data/`. Total end-to-end runtime is about 45 minutes on an M4 Pro MacBook. Outputs are CSVs + a `veredict.json` per phase containing automated PASS/FAIL/MARGINAL verdicts and aggregated statistics.

## Design notes

### Why Polars streaming and not pandas or DuckDB

At 144 GB the dataset is too large for pandas on a laptop. DuckDB would also have worked well — I picked Polars for the expression API because the analysis has many derived columns (bucket labels, market lifetime position, UNIX-to-quarter conversion) where Polars' chained `.with_columns()` reads cleanly. The streaming engine handles the >memory aggregations without any tuning.

### Two gotchas with Polars lazy frame operations

- Polars' regex is Rust-based and does **not** support lookahead/lookbehind. Keyword matching like `\bfed\b(?!\s?ex)` had to be rewritten to whole phrases.
- `str.to_datetime` with `strict=False` works in eager but fails in lazy expression context when the data contains timezones. Use `str.strptime` with an explicit format string and `pl.Datetime(time_zone="UTC")`. Normalize short offsets like `+00` to `+00:00` before parsing.

### Statistical decision thresholds

The automated verdict logic in each script uses fixed thresholds. Tuning guidance:

- Phase 0.5, threshold for "A Temporal PASS": EV at cutoff 0.50 ≥ +1.0%. Rationale: if the edge vanishes below 1% when excluding late trades, it wasn't structural.
- Phase 0.5, threshold for "D Wallets PASS": top-50 wallet share < 30%. Rationale: above this, edge is captured by sharps not replicable by retail sizing.
- Phase 0.7, threshold for SIGNAL: |z| > 2σ against the baseline of 2024–2025 quarterly EVs. Standard normal cutoff for rejection of the null.

## Repository layout

```
.
├── fase0_analysis_v3.py
├── fase05_validation.py
├── fase06_category_survival.py
├── fase07_quarterly_volatility.py
├── outputs_example/         # the CSV/JSON results from the April 2026 run
├── figures/                 # the charts used in the Medium post
└── README.md
```

## License

MIT. Use the scripts, fork them, adapt them to your markets. If you find a similar long-shot-bias-decay pattern in another prediction market (Kalshi, Manifold, Metaculus), I'd like to hear about it.

## Citation

If you reference this analysis in work of your own:

> sergiofernandezcar (2026). *The +9.77% edge that decayed to −4.96%: a Polymarket post-mortem.* Retrieved from (https://medium.com/p/f325f4a3d1b7?postPublishedType=initial) and https://github.com/sergiofernandezcar-coder/bot_polymarket.

## Contact

- @SFCNene86
- sergiofernandezcar@gmail.com

Open to discussions on validation methodology for systematic trading, polars streaming patterns for large on-chain datasets, and quantitative post-mortems.
