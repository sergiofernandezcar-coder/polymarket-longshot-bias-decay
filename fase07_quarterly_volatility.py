#!/usr/bin/env python3
"""
Fase 0.7 — Volatilidad trimestral del edge en bucket 0.60-0.70
===============================================================

Fase 0.6 arrojó 3 combos "ALIVE" en bucket 0.60-0.70 con EV positivo en
2026 Q1, pero con trayectoria anual OSCILANTE (POLITICS osciló entre -29%
y +17% entre años). La pregunta: ¿es Q1 2026 señal o ruido trimestral?

Método:
  1. Particionar trades en trimestres (2023 Q1 … 2026 Q1)
  2. Para las 3 categorías ALIVE (POLITICS, SPORTS, OTHER) en bucket 0.60-0.70,
     calcular EV por trimestre
  3. Calcular media μ y desviación σ del EV trimestral para 2024-2025
     (ignorando 2023 por n bajo)
  4. Clasificar Q1 2026:
       SIGNAL   si Q1_2026 > μ + 2σ  (outlier positivo claro)
       NOISE    si |Q1_2026 - μ| ≤ 2σ (dentro de rango normal)
       FLIP     si Q1_2026 < μ - 2σ  (outlier negativo, reversión)

Criterio de decisión final:
  PROCEED_PAPER  — Q1 2026 es SIGNAL en ≥2 de 3 categorías
  KILL_PROJECT   — Q1 2026 es NOISE en ≥2 de 3 categorías
  AMBIGUOUS      — mezcla

Output: ~/poly_data/fase07_output/
  - matrix_cat_quarter.csv
  - quarterly_stats.csv
  - veredict.json

Tiempo esperado: 5-10 min (un pass streaming con agregación por trimestre).
"""

import json
import math
import sys
import traceback
from datetime import datetime
from pathlib import Path

import polars as pl

# ─── Config ──────────────────────────────────────────────────────────────────
ROOT = Path.home() / "poly_data"
TRADES_CSV = ROOT / "processed" / "trades.csv"
V3_DIR = ROOT / "fase0_output_v3"
OUT_DIR = ROOT / "fase07_output"
OUT_DIR.mkdir(exist_ok=True)

BUCKET_BREAKS = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]
BUCKET_LABELS = [
    "0.00-0.10", "0.10-0.20", "0.20-0.30", "0.30-0.40", "0.40-0.50",
    "0.50-0.60", "0.60-0.70", "0.70-0.80", "0.80-0.90", "0.90-1.00",
]
FOCUS_BUCKET = "0.60-0.70"
FOCUS_CATEGORIES = ["POLITICS", "SPORTS", "OTHER"]
MIN_N_PER_QUARTER = 5000   # cualquier trimestre con n menor se excluye de μ/σ
BASELINE_PERIOD = ("2024Q1", "2025Q4")  # rango del baseline para μ,σ
Q1_2026 = "2026Q1"


def ts() -> str:
    return datetime.now().isoformat(timespec="seconds")


def header(title: str) -> None:
    print(f"\n{'═' * 72}\n  {title}\n  [{ts()}]\n{'═' * 72}")


def stream_collect(lf):
    try:
        return lf.collect(engine="streaming")
    except TypeError:
        return lf.collect(streaming=True)


# ─── Base ────────────────────────────────────────────────────────────────────
def load_base():
    wpath = V3_DIR / "winners_resolved.csv"
    if not wpath.exists():
        raise FileNotFoundError(f"No encuentro {wpath}")
    w = pl.read_csv(str(wpath), infer_schema_length=0)
    w = (
        w.filter(pl.col("winner").is_not_null())
        .filter(pl.col("category").is_in(FOCUS_CATEGORIES))
        .select(["market_id", "category", "winner"])
    )
    print(f"  Mercados resueltos en categorías focus: {w.height:,}")
    return w


# ─── Enriched con quarter + bucket ───────────────────────────────────────────
def build_enriched(base):
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
                pl.from_epoch(pl.col("ts_int"), time_unit="s").alias("trade_dt"),
                (pl.col("nonusdc_side") == pl.col("winner")).alias("trade_won"),
                pl.col("entry_price")
                .cut(breaks=BUCKET_BREAKS, labels=BUCKET_LABELS)
                .alias("bucket"),
            ]
        )
        .filter(pl.col("bucket") == FOCUS_BUCKET)
        .with_columns(
            (
                pl.col("trade_dt").dt.year().cast(pl.Utf8)
                + "Q"
                + ((pl.col("trade_dt").dt.month() + 2) // 3).cast(pl.Utf8)
            ).alias("quarter")
        )
    )
    return enriched


# ─── Matrix cat × quarter ────────────────────────────────────────────────────
def compute_matrix(enriched):
    header("Agregando matrix (cat × quarter × bucket 0.60-0.70)")
    matrix = (
        enriched.group_by(["category", "quarter"])
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
        .sort(["category", "quarter"])
        .pipe(stream_collect)
    )
    matrix.write_csv(OUT_DIR / "matrix_cat_quarter.csv")
    print(f"Matrix filas: {matrix.height}")
    print(matrix)
    return matrix


# ─── Estadística por categoría ───────────────────────────────────────────────
def compute_quarterly_stats(matrix: pl.DataFrame):
    header("Computando μ, σ del EV trimestral por categoría (baseline 2024-2025)")
    rows = []
    for cat in FOCUS_CATEGORIES:
        sub = matrix.filter(pl.col("category") == cat).sort("quarter")
        if sub.height == 0:
            continue

        # Baseline: trimestres 2024Q1 .. 2025Q4 con n >= MIN_N_PER_QUARTER
        baseline = sub.filter(
            (pl.col("quarter") >= BASELINE_PERIOD[0])
            & (pl.col("quarter") <= BASELINE_PERIOD[1])
            & (pl.col("n_trades") >= MIN_N_PER_QUARTER)
        )

        # Q1 2026
        q1_row = sub.filter(pl.col("quarter") == Q1_2026)
        ev_q1_2026 = float(q1_row["ev_per_dollar"][0]) if q1_row.height else None
        n_q1_2026 = int(q1_row["n_trades"][0]) if q1_row.height else 0

        if baseline.height < 3:
            # Necesitamos al menos 3 trimestres de baseline para estadística
            verdict = "INSUFFICIENT_BASELINE"
            mu = None
            sigma = None
            z = None
        else:
            evs_baseline = baseline["ev_per_dollar"].to_list()
            mu = float(sum(evs_baseline) / len(evs_baseline))
            if len(evs_baseline) > 1:
                var = sum((x - mu) ** 2 for x in evs_baseline) / (
                    len(evs_baseline) - 1
                )
                sigma = float(math.sqrt(var))
            else:
                sigma = 0.0

            if ev_q1_2026 is None:
                verdict = "NO_Q1_2026_DATA"
                z = None
            elif sigma == 0:
                z = None
                verdict = "ZERO_VARIANCE"
            else:
                z = (ev_q1_2026 - mu) / sigma
                if z > 2.0:
                    verdict = "SIGNAL"  # outlier positivo
                elif z < -2.0:
                    verdict = "FLIP_NEGATIVE"  # outlier negativo
                else:
                    verdict = "NOISE"  # dentro del rango baseline

        rows.append(
            {
                "category": cat,
                "n_baseline_quarters": baseline.height,
                "baseline_mu_ev": mu,
                "baseline_sigma_ev": sigma,
                "ev_q1_2026": ev_q1_2026,
                "n_q1_2026": n_q1_2026,
                "z_score": z,
                "verdict": verdict,
            }
        )

    stats = pl.DataFrame(rows)
    stats.write_csv(OUT_DIR / "quarterly_stats.csv")
    print(stats)
    return stats


# ─── Decisión final ──────────────────────────────────────────────────────────
def build_final_verdict(stats: pl.DataFrame):
    header("DECISIÓN FINAL")
    verdicts = stats["verdict"].to_list()
    n_signal = sum(1 for v in verdicts if v == "SIGNAL")
    n_noise = sum(1 for v in verdicts if v == "NOISE")
    n_flip_neg = sum(1 for v in verdicts if v == "FLIP_NEGATIVE")

    if n_signal >= 2:
        decision = "PROCEED_PAPER"
        reason = f"{n_signal}/3 categorías con Q1 2026 como outlier positivo (>2σ)"
    elif n_noise >= 2:
        decision = "KILL_PROJECT"
        reason = f"{n_noise}/3 categorías muestran que Q1 2026 está dentro del ruido trimestral"
    elif n_flip_neg >= 2:
        decision = "KILL_PROJECT"
        reason = f"{n_flip_neg}/3 categorías muestran reversión negativa extrema en Q1 2026"
    else:
        decision = "AMBIGUOUS"
        reason = f"signal={n_signal}, noise={n_noise}, flip_neg={n_flip_neg}"

    print(f"🎯 {decision}")
    print(f"   {reason}")

    out = {
        "decision": decision,
        "reason": reason,
        "per_category": stats.to_dicts(),
    }
    return out


# ─── Main ────────────────────────────────────────────────────────────────────
def main():
    summary = {"started": ts(), "polars": pl.__version__, "errors": []}
    try:
        base = load_base()
    except Exception as e:
        summary["errors"].append({"step": "base", "err": str(e), "tb": traceback.format_exc()})
        (OUT_DIR / "veredict.json").write_text(json.dumps(summary, indent=2, default=str))
        print(f"❌ {e}")
        sys.exit(1)

    enriched = build_enriched(base)

    matrix = None
    try:
        matrix = compute_matrix(enriched)
    except Exception as e:
        summary["errors"].append({"step": "matrix", "err": str(e), "tb": traceback.format_exc()})
        print(f"❌ matrix: {e}")
        (OUT_DIR / "veredict.json").write_text(json.dumps(summary, indent=2, default=str))
        sys.exit(1)

    try:
        stats = compute_quarterly_stats(matrix)
        summary["stats"] = stats.to_dicts()
    except Exception as e:
        summary["errors"].append({"step": "stats", "err": str(e), "tb": traceback.format_exc()})
        print(f"❌ stats: {e}")
        stats = None

    if stats is not None:
        try:
            final = build_final_verdict(stats)
            summary["final"] = final
        except Exception as e:
            summary["errors"].append({"step": "final", "err": str(e), "tb": traceback.format_exc()})
            print(f"❌ final: {e}")

    summary["finished"] = ts()
    (OUT_DIR / "veredict.json").write_text(json.dumps(summary, indent=2, default=str))
    header("✓ FASE 0.7 COMPLETA")
    print(f"Outputs en: {OUT_DIR}")
    print("\nPegar en el próximo chat:")
    print("  1. veredict.json")
    print("  2. matrix_cat_quarter.csv")
    print("  3. quarterly_stats.csv")


if __name__ == "__main__":
    main()
