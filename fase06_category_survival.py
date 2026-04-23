#!/usr/bin/env python3
"""
Fase 0.6 — Supervivencia del edge por nicho
=============================================

El veredicto de Fase 0.5: el edge del bucket 0.30-0.40 decayó monótonamente
de +9.77% (2023) a -4.96% (2026 Q1) globalmente. La pregunta que cierra el
proyecto: ¿alguna categoría específica mantiene edge positivo HOY?

Reutiliza:
  - ~/poly_data/fase0_output_v3/winners_resolved.csv  (market_id, category, winner)
  - ~/poly_data/processed/trades.csv                  (trades crudos)

Criterios de clasificación (por combo categoría×bucket):
  ALIVE_STABLE: EV_2025 >= +1.0% y EV_2026 >= +1.0% con n_2026 >= 10K
  ALIVE:        EV_2026 >= +1.0% con n_2026 >= 10K (pero 2025 no necesariamente +)
  FADING:       0 <= EV_2026 < +1.0% con n_2026 >= 10K
  DEAD:         EV_2026 < 0 con n_2026 >= 10K
  INSUFFICIENT: n_2026 < 10K (no se puede concluir)

Veredicto global:
  GO_NICHE      — al menos 1 combo ALIVE_STABLE con n_2026 >= 20K
  MARGINAL_GO   — solo hay combos ALIVE (no stable)
  CONFIRMED_DEAD — todos los combos con datos suficientes son FADING/DEAD

Output: ~/poly_data/fase06_output/
  - matrix_cat_year_bucket.csv
  - classification_by_combo.csv
  - alive_combos.csv   (si los hay)
  - veredict.json

Tiempo esperado: 6-12 min (un solo pass).
"""

import json
import sys
import traceback
from datetime import datetime
from pathlib import Path

import polars as pl

# ─── Config ──────────────────────────────────────────────────────────────────
ROOT = Path.home() / "poly_data"
TRADES_CSV = ROOT / "processed" / "trades.csv"
V3_DIR = ROOT / "fase0_output_v3"
OUT_DIR = ROOT / "fase06_output"
OUT_DIR.mkdir(exist_ok=True)

BUCKET_BREAKS = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]
BUCKET_LABELS = [
    "0.00-0.10", "0.10-0.20", "0.20-0.30", "0.30-0.40", "0.40-0.50",
    "0.50-0.60", "0.60-0.70", "0.70-0.80", "0.80-0.90", "0.90-1.00",
]
FOCUS_BUCKETS = ["0.30-0.40", "0.60-0.70"]
MIN_N_2026 = 10_000      # n mínimo por combo en 2026 para clasificar
STABLE_N_2026 = 20_000   # n mínimo para GO_NICHE fuerte
EV_ALIVE = 0.010         # +1.0% EV threshold para "vivo"


def ts() -> str:
    return datetime.now().isoformat(timespec="seconds")


def header(title: str) -> None:
    print(f"\n{'═' * 72}\n  {title}\n  [{ts()}]\n{'═' * 72}")


def stream_collect(lf):
    try:
        return lf.collect(engine="streaming")
    except TypeError:
        return lf.collect(streaming=True)


# ─── Base: winners_resolved → (market_id, category, winner) ──────────────────
def load_base():
    wpath = V3_DIR / "winners_resolved.csv"
    if not wpath.exists():
        raise FileNotFoundError(
            f"No encuentro {wpath}. Ejecuta antes fase0_analysis_v3.py."
        )
    w = pl.read_csv(str(wpath), infer_schema_length=0)
    w = (
        w.filter(pl.col("winner").is_not_null())
        .select(["market_id", "category", "winner"])
    )
    print(f"  Mercados resueltos con winner + categoría: {w.height:,}")
    return w


# ─── Enriched lazy: trades + join + year + bucket + trade_won ────────────────
def build_enriched(base: pl.DataFrame) -> pl.LazyFrame:
    t = pl.scan_csv(str(TRADES_CSV), infer_schema_length=0)
    t = (
        t.filter(pl.col("market_id").is_not_null())
        .filter(pl.col("nonusdc_side").is_in(["token1", "token2"]))
        .with_columns(
            [
                pl.col("timestamp").cast(pl.Int64, strict=False).alias("ts_int"),
                pl.col("price").cast(pl.Float64, strict=False).alias("entry_price"),
            ]
        )
        .filter(pl.col("ts_int").is_not_null())
        .filter((pl.col("entry_price") > 0.0) & (pl.col("entry_price") < 1.0))
        .select(["market_id", "nonusdc_side", "ts_int", "entry_price"])
    )
    enriched = (
        t.join(base.lazy(), on="market_id", how="inner")
        .with_columns(
            [
                pl.from_epoch(pl.col("ts_int"), time_unit="s").dt.year().alias("year"),
                (pl.col("nonusdc_side") == pl.col("winner")).alias("trade_won"),
                pl.col("entry_price")
                .cut(breaks=BUCKET_BREAKS, labels=BUCKET_LABELS)
                .alias("bucket"),
            ]
        )
    )
    return enriched


# ─── Matrix cat × year × bucket ──────────────────────────────────────────────
def compute_matrix(enriched):
    header("Agregando matrix (cat × year × bucket)")
    matrix = (
        enriched.filter(pl.col("bucket").is_in(FOCUS_BUCKETS))
        .group_by(["category", "year", "bucket"])
        .agg(
            [
                pl.len().alias("n_trades"),
                pl.col("trade_won").mean().alias("win_rate"),
                pl.col("entry_price").mean().alias("avg_entry"),
            ]
        )
        .with_columns(
            (
                pl.col("win_rate") * (1.0 - pl.col("avg_entry"))
                - (1.0 - pl.col("win_rate")) * pl.col("avg_entry")
            ).alias("ev_per_dollar")
        )
        .sort(["category", "bucket", "year"])
        .pipe(stream_collect)
    )
    print(f"  Matrix rows: {matrix.height}")
    matrix.write_csv(OUT_DIR / "matrix_cat_year_bucket.csv")
    return matrix


# ─── Clasificar combos ───────────────────────────────────────────────────────
def classify(matrix: pl.DataFrame):
    header("Clasificando combos (cat × bucket)")
    rows = []
    for cat in matrix["category"].unique().sort().to_list():
        for bucket in FOCUS_BUCKETS:
            subset = matrix.filter(
                (pl.col("category") == cat) & (pl.col("bucket") == bucket)
            ).sort("year")
            if subset.height == 0:
                continue
            yr_to_ev = {
                int(r["year"]): (float(r["ev_per_dollar"]), int(r["n_trades"]))
                for r in subset.iter_rows(named=True)
            }
            ev_2023, n_2023 = yr_to_ev.get(2023, (None, 0))
            ev_2024, n_2024 = yr_to_ev.get(2024, (None, 0))
            ev_2025, n_2025 = yr_to_ev.get(2025, (None, 0))
            ev_2026, n_2026 = yr_to_ev.get(2026, (None, 0))

            if n_2026 < MIN_N_2026:
                verdict = "INSUFFICIENT"
            elif ev_2026 is None:
                verdict = "INSUFFICIENT"
            elif ev_2026 >= EV_ALIVE and ev_2025 is not None and ev_2025 >= EV_ALIVE:
                verdict = "ALIVE_STABLE"
            elif ev_2026 >= EV_ALIVE:
                verdict = "ALIVE"
            elif ev_2026 >= 0:
                verdict = "FADING"
            else:
                verdict = "DEAD"

            # trayectoria
            trend = [ev_2023, ev_2024, ev_2025, ev_2026]
            ns = [n_2023, n_2024, n_2025, n_2026]

            rows.append(
                {
                    "category": cat,
                    "bucket": bucket,
                    "verdict": verdict,
                    "ev_2023": ev_2023,
                    "ev_2024": ev_2024,
                    "ev_2025": ev_2025,
                    "ev_2026": ev_2026,
                    "n_2023": n_2023,
                    "n_2024": n_2024,
                    "n_2025": n_2025,
                    "n_2026": n_2026,
                }
            )

    df = pl.DataFrame(rows)
    df = df.sort(["verdict", "ev_2026"], descending=[False, True], nulls_last=True)
    df.write_csv(OUT_DIR / "classification_by_combo.csv")
    print(df)
    return df


# ─── Resumen ejecutivo + veredicto ──────────────────────────────────────────
def build_verdict(classified: pl.DataFrame):
    header("Veredicto")
    counts = (
        classified.group_by("verdict")
        .agg(pl.len().alias("n"))
        .sort("n", descending=True)
    )
    print("Distribución de combos:")
    print(counts)

    alive_stable = classified.filter(pl.col("verdict") == "ALIVE_STABLE")
    alive = classified.filter(pl.col("verdict") == "ALIVE")
    fading = classified.filter(pl.col("verdict") == "FADING")
    dead = classified.filter(pl.col("verdict") == "DEAD")

    # GO_NICHE: al menos 1 ALIVE_STABLE con n_2026 >= 20K
    strong = alive_stable.filter(pl.col("n_2026") >= STABLE_N_2026)
    strong.write_csv(OUT_DIR / "alive_combos.csv")

    if strong.height >= 1:
        decision = "GO_NICHE"
    elif alive.height >= 1 or alive_stable.height >= 1:
        decision = "MARGINAL_GO"
    elif fading.height >= 1 and dead.height == 0:
        decision = "MARGINAL_PAUSE"
    else:
        decision = "CONFIRMED_DEAD"

    summary = {
        "n_combos_total": classified.height,
        "counts": {r["verdict"]: int(r["n"]) for r in counts.iter_rows(named=True)},
        "n_alive_stable_strong": strong.height,
        "alive_stable_combos": alive_stable.to_dicts(),
        "alive_combos": alive.to_dicts(),
        "top_fading": fading.head(5).to_dicts(),
        "top_dead": dead.sort("ev_2026").head(5).to_dicts(),
        "decision": decision,
    }

    print(f"\n🎯 DECISIÓN: {decision}")
    if strong.height:
        print(f"   Combos ALIVE_STABLE fuertes (n_2026 >= {STABLE_N_2026}):")
        print(strong)
    return summary


# ─── Main ────────────────────────────────────────────────────────────────────
def main():
    summary = {"started": ts(), "polars": pl.__version__, "errors": []}
    try:
        base = load_base()
        summary["n_markets_base"] = base.height
    except Exception as e:
        summary["errors"].append({"step": "load_base", "err": str(e), "tb": traceback.format_exc()})
        (OUT_DIR / "veredict.json").write_text(json.dumps(summary, indent=2, default=str))
        print(f"❌ load_base: {e}")
        sys.exit(1)

    enriched = build_enriched(base)

    try:
        matrix = compute_matrix(enriched)
        summary["matrix_rows"] = matrix.height
    except Exception as e:
        summary["errors"].append({"step": "matrix", "err": str(e), "tb": traceback.format_exc()})
        (OUT_DIR / "veredict.json").write_text(json.dumps(summary, indent=2, default=str))
        print(f"❌ matrix: {e}")
        sys.exit(1)

    try:
        classified = classify(matrix)
        summary["classification"] = classified.to_dicts()
    except Exception as e:
        summary["errors"].append({"step": "classify", "err": str(e), "tb": traceback.format_exc()})
        print(f"⚠ classify: {e}")

    try:
        verdict = build_verdict(classified)
        summary["verdict"] = verdict
    except Exception as e:
        summary["errors"].append({"step": "verdict", "err": str(e), "tb": traceback.format_exc()})
        print(f"⚠ verdict: {e}")

    summary["finished"] = ts()
    (OUT_DIR / "veredict.json").write_text(json.dumps(summary, indent=2, default=str))
    header("✓ FASE 0.6 COMPLETA")
    print(f"Outputs en: {OUT_DIR}")
    print("\nPegar en el próximo chat:")
    print("  1. veredict.json")
    print("  2. classification_by_combo.csv")
    print("  3. matrix_cat_year_bucket.csv")
    print("  4. alive_combos.csv  (si existe)")


if __name__ == "__main__":
    main()
