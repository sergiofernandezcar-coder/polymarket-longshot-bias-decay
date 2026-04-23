#!/usr/bin/env python3
"""
Fase 0.5 — Validaciones críticas del edge del bucket 0.30-0.40
===============================================================

Reutiliza los outputs de Fase 0 v3:
  - ~/poly_data/fase0_output_v3/winners_resolved.csv
  - ~/poly_data/markets.csv      (para createdAt, necesario para lifetime)
  - ~/poly_data/processed/trades.csv

Responde 4 preguntas:

  A. TEMPORAL:   ¿sobrevive el edge del bucket 0.30-0.40 cuando excluimos
                 trades cerca del cierre? (cutoffs: 0.50, 0.80, 0.95, 1.00)
                 → PASS si EV @ cutoff 0.50 > +1.0%
                 → FAIL si EV @ cutoff 0.50 < +0.5%

  B. WALK-FWD:   ¿es el edge estable en el tiempo (2023/2024/2025/2026)?
                 → PASS si std del EV < 1pp entre años con n>500K
                 → FAIL si decae monótonamente de >+3% a <+0.5%

  C. VOLUME:     ¿sobrevive en mercados líquidos (volumen >= 10K)?
                 → PASS si EV en tier >10K es > +0.5%
                 → FAIL si el edge solo existe en mercados ilíquidos

  D. WALLETS:    ¿es smart money concentrado o bias estructural distribuido?
                 → PASS si top-50 wallets < 30% del volumen del bucket
                 → FAIL si top-50 > 50% (smart money dominante, no replicable)

Output: ~/poly_data/fase05_output/
  - fase05_a_temporal.csv
  - fase05_b_walkforward.csv
  - fase05_c_volume.csv
  - fase05_d_top100_wallets.csv
  - veredict.json          ← PASS/FAIL por validación + decisión final

Tiempo esperado: 15-25 min (4 passes streaming sobre ~40M trades enriched)
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
MARKETS_CSV = ROOT / "markets.csv"
V3_DIR = ROOT / "fase0_output_v3"
OUT_DIR = ROOT / "fase05_output"
OUT_DIR.mkdir(exist_ok=True)

BUCKET_BREAKS = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]
BUCKET_LABELS = [
    "0.00-0.10", "0.10-0.20", "0.20-0.30", "0.30-0.40", "0.40-0.50",
    "0.50-0.60", "0.60-0.70", "0.70-0.80", "0.80-0.90", "0.90-1.00",
]
FOCUS_BUCKETS = ["0.30-0.40", "0.60-0.70"]
POSITION_CUTOFFS = [0.50, 0.80, 0.95, 1.00]
VOLUME_TIERS_BREAKS = [1000.0, 10000.0, 100000.0]
VOLUME_TIERS_LABELS = ["<1K", "1K-10K", "10K-100K", ">100K"]
MIN_LIFETIME_SECONDS = 3600  # >1h, excluye BTC 5-min markets

# Criterios PASS/FAIL
THRESHOLD_A_PASS = 0.010   # EV >= +1.0% al cutoff 0.50 → edge temprano real
THRESHOLD_A_FAIL = 0.005   # EV < +0.5% → timing bias
THRESHOLD_C_PASS = 0.005   # EV >= +0.5% en tier volumen ≥ 10K
THRESHOLD_D_PASS = 0.30    # top-50 wallets < 30% del volumen → bias distribuido
THRESHOLD_D_FAIL = 0.50    # top-50 > 50% → smart money concentrado


def ts() -> str:
    return datetime.now().isoformat(timespec="seconds")


def header(title: str) -> None:
    print(f"\n{'═' * 72}\n  {title}\n  [{ts()}]\n{'═' * 72}")


def stream_collect(lf):
    try:
        return lf.collect(engine="streaming")
    except TypeError:
        return lf.collect(streaming=True)


# ─── Preparar markets_base: join winners + lifetime + vol_tier ───────────────
def build_markets_base():
    header("Preparando markets_base (eager)")
    # markets.csv (125K filas, pequeño → eager)
    m = pl.read_csv(str(MARKETS_CSV), infer_schema_length=0)
    print(f"  markets.csv: {m.height:,} filas")

    # Parse dates: createdAt ISO8601 con Z (2023-01-01T10:00:00.000Z),
    # closedTime custom con +00 corto (2023-01-01 10:00:00+00).
    # Polars lazy requiere formato explícito con timezone → normalizamos ambos.
    m = m.with_columns(
        [
            pl.col("createdAt")
            .str.replace(r"Z$", "+0000")
            .str.strptime(
                pl.Datetime(time_zone="UTC"),
                format="%Y-%m-%dT%H:%M:%S%.f%z",
                strict=False,
            )
            .alias("created_dt"),
            pl.col("closedTime")
            .str.replace(r"([+-]\d{2})$", r"${1}:00")
            .str.strptime(
                pl.Datetime(time_zone="UTC"),
                format="%Y-%m-%d %H:%M:%S%z",
                strict=False,
            )
            .alias("closed_dt"),
            pl.col("volume").cast(pl.Float64, strict=False).alias("volume_f"),
        ]
    )
    m = m.filter(
        pl.col("created_dt").is_not_null() & pl.col("closed_dt").is_not_null()
    )
    print(f"  con created_dt y closed_dt válidos: {m.height:,}")

    # winners_resolved.csv — output de Fase 0 v3
    wpath = V3_DIR / "winners_resolved.csv"
    if not wpath.exists():
        raise FileNotFoundError(
            f"No encuentro {wpath}. Ejecuta antes fase0_analysis_v3.py."
        )
    w = pl.read_csv(str(wpath), infer_schema_length=0)
    w = w.filter(pl.col("winner").is_not_null()).select(
        ["market_id", "winner", "category"]
    )
    print(f"  winners_resolved con winner determinado: {w.height:,}")

    # Join + derivar lifetime_s y created_ts (unix segundos)
    base = m.join(w, left_on="id", right_on="market_id", how="inner").with_columns(
        [
            (pl.col("closed_dt") - pl.col("created_dt"))
            .dt.total_seconds()
            .cast(pl.Int64)
            .alias("lifetime_s"),
            pl.col("created_dt").dt.timestamp("ms").floordiv(1000).cast(pl.Int64)
            .alias("created_ts"),
            pl.col("volume_f")
            .cut(breaks=VOLUME_TIERS_BREAKS, labels=VOLUME_TIERS_LABELS)
            .alias("vol_tier"),
        ]
    )
    base = base.filter(pl.col("lifetime_s") > MIN_LIFETIME_SECONDS)
    print(f"  mercados con lifetime > {MIN_LIFETIME_SECONDS}s: {base.height:,}")

    # Columnas finales que necesitamos
    base = base.select(
        ["id", "winner", "category", "created_ts", "lifetime_s",
         "volume_f", "vol_tier"]
    )
    return base


# ─── Construir LazyFrame enriquecido ─────────────────────────────────────────
def build_enriched_lazy(base_df: pl.DataFrame) -> pl.LazyFrame:
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
        .select(
            [
                "market_id",
                "nonusdc_side",
                "maker",
                "taker",
                "maker_direction",
                "ts_int",
                "entry_price",
            ]
        )
    )
    enriched = (
        t.join(base_df.lazy(), left_on="market_id", right_on="id", how="inner")
        .with_columns(
            [
                (
                    (pl.col("ts_int") - pl.col("created_ts")).cast(pl.Float64)
                    / pl.col("lifetime_s").cast(pl.Float64)
                ).alias("position"),
                pl.from_epoch(pl.col("ts_int"), time_unit="s").dt.year().alias("year"),
                pl.when(pl.col("maker_direction") == "BUY")
                .then(pl.col("maker"))
                .otherwise(pl.col("taker"))
                .alias("buyer_wallet"),
                (pl.col("nonusdc_side") == pl.col("winner")).alias("trade_won"),
                pl.col("entry_price")
                .cut(breaks=BUCKET_BREAKS, labels=BUCKET_LABELS)
                .alias("bucket"),
            ]
        )
        .filter(pl.col("position") >= 0.0)
        .filter(pl.col("position") <= 1.0)
    )
    return enriched


def _agg_bucket_ev(lf, group_cols):
    """Agg estándar: n, wr, avg_p, ev_per_dollar."""
    return (
        lf.group_by(group_cols)
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
    )


# ─── Fase A: temporal filter ─────────────────────────────────────────────────
def fase_a(enriched):
    header("A · Temporal filter — cutoffs 0.50, 0.80, 0.95, 1.00")
    results = []
    for cutoff in POSITION_CUTOFFS:
        print(f"  computando cutoff pos<{cutoff:.2f} ...")
        q = (
            _agg_bucket_ev(enriched.filter(pl.col("position") < cutoff), ["bucket"])
            .sort("bucket")
            .pipe(stream_collect)
            .with_columns(pl.lit(cutoff).alias("position_cutoff"))
        )
        results.append(q)
    out = pl.concat(results, how="vertical").select(
        ["position_cutoff", "bucket", "n_trades", "win_rate", "avg_entry", "ev_per_dollar"]
    )
    out.write_csv(OUT_DIR / "fase05_a_temporal.csv")
    print("\nResultados Fase A (todos los cutoffs):")
    print(out)

    # Comprobación para bucket 0.30-0.40 a cutoff 0.50 y 0.80
    focus = out.filter(pl.col("bucket") == "0.30-0.40")
    print("\nFoco bucket 0.30-0.40 por cutoff:")
    print(focus)
    return out, focus


# ─── Fase B: walk-forward por año ────────────────────────────────────────────
def fase_b(enriched):
    header("B · Walk-forward por año — focus buckets 0.30-0.40 y 0.60-0.70")
    q = (
        _agg_bucket_ev(
            enriched.filter(pl.col("bucket").is_in(FOCUS_BUCKETS)),
            ["year", "bucket"],
        )
        .sort(["year", "bucket"])
        .pipe(stream_collect)
    )
    q.write_csv(OUT_DIR / "fase05_b_walkforward.csv")
    print(q)
    return q


# ─── Fase C: volume tier ─────────────────────────────────────────────────────
def fase_c(enriched):
    header("C · Volume tier segmentation")
    q = (
        _agg_bucket_ev(
            enriched.filter(pl.col("bucket").is_in(FOCUS_BUCKETS)),
            ["vol_tier", "bucket"],
        )
        .sort(["vol_tier", "bucket"])
        .pipe(stream_collect)
    )
    q.write_csv(OUT_DIR / "fase05_c_volume.csv")
    print(q)
    return q


# ─── Fase D: wallet concentration ────────────────────────────────────────────
def fase_d(enriched):
    header("D · Wallet concentration en bucket 0.30-0.40")
    w_stats = (
        enriched.filter(pl.col("bucket") == "0.30-0.40")
        .group_by("buyer_wallet")
        .agg(
            [
                pl.len().alias("n_trades"),
                pl.col("trade_won").mean().alias("win_rate"),
                pl.col("entry_price").mean().alias("avg_entry"),
            ]
        )
        .sort("n_trades", descending=True)
        .pipe(stream_collect)
    )
    total = int(w_stats["n_trades"].sum())
    n_unique = w_stats.height

    if total == 0:
        stats = {
            "total_trades": 0,
            "n_unique_wallets": 0,
            "top10_share": None,
            "top50_share": None,
            "top100_share": None,
        }
    else:
        top10 = float(w_stats.head(10)["n_trades"].sum()) / total
        top50 = float(w_stats.head(50)["n_trades"].sum()) / total
        top100 = float(w_stats.head(100)["n_trades"].sum()) / total
        top50_df = w_stats.head(50)
        rest_df = w_stats.slice(50, None) if w_stats.height > 50 else None

        # WR ponderado (trades) para comparar grupos
        def weighted_wr(df):
            if df is None or df.height == 0:
                return None
            n = df["n_trades"].sum()
            if n == 0:
                return None
            return float((df["win_rate"] * df["n_trades"]).sum()) / n

        stats = {
            "total_trades": total,
            "n_unique_wallets": n_unique,
            "top10_share": top10,
            "top50_share": top50,
            "top100_share": top100,
            "top50_weighted_wr": weighted_wr(top50_df),
            "rest_weighted_wr": weighted_wr(rest_df),
        }

    # Guardar top 100 wallets
    w_stats.head(100).write_csv(OUT_DIR / "fase05_d_top100_wallets.csv")

    print(f"\nTotal trades en bucket 0.30-0.40: {total:,}")
    print(f"Wallets únicas: {n_unique:,}")
    if total > 0:
        print(f"Top-10 share:  {stats['top10_share']:.4f} ({100*stats['top10_share']:.2f}%)")
        print(f"Top-50 share:  {stats['top50_share']:.4f} ({100*stats['top50_share']:.2f}%)")
        print(f"Top-100 share: {stats['top100_share']:.4f} ({100*stats['top100_share']:.2f}%)")
        if stats["top50_weighted_wr"] is not None:
            print(f"Top-50 WR (ponderado): {stats['top50_weighted_wr']:.4f}")
        if stats["rest_weighted_wr"] is not None:
            print(f"Resto WR (ponderado):  {stats['rest_weighted_wr']:.4f}")
    return w_stats, stats


# ─── Veredicto automático ────────────────────────────────────────────────────
def build_verdict(a_focus, b_df, c_df, d_stats):
    verdict = {"validations": {}}

    # A · Temporal
    a_050 = a_focus.filter(pl.col("position_cutoff") == 0.50)
    ev_050 = float(a_050["ev_per_dollar"][0]) if a_050.height else None
    a_pass = ev_050 is not None and ev_050 >= THRESHOLD_A_PASS
    a_fail = ev_050 is not None and ev_050 < THRESHOLD_A_FAIL
    verdict["validations"]["A_temporal"] = {
        "ev_bucket_0.30-0.40_at_cutoff_0.50": ev_050,
        "threshold_pass": THRESHOLD_A_PASS,
        "threshold_fail": THRESHOLD_A_FAIL,
        "verdict": "PASS" if a_pass else ("FAIL" if a_fail else "MARGINAL"),
    }

    # B · Walk-forward
    b_focus = b_df.filter(pl.col("bucket") == "0.30-0.40").filter(
        pl.col("n_trades") >= 500_000
    )
    if b_focus.height >= 2:
        ev_by_year = b_focus.select(["year", "ev_per_dollar"]).sort("year")
        evs = ev_by_year["ev_per_dollar"].to_list()
        years = ev_by_year["year"].to_list()
        std_ev = float(ev_by_year["ev_per_dollar"].std())
        # ¿Decaimiento monótono? si cada año < año anterior
        monotonic_decay = all(
            evs[i + 1] < evs[i] for i in range(len(evs) - 1)
        ) and evs[0] - evs[-1] > 0.02
        b_pass = std_ev < 0.01 and not monotonic_decay
        verdict["validations"]["B_walkforward"] = {
            "years": years,
            "ev_per_year": evs,
            "std_ev": std_ev,
            "monotonic_decay": monotonic_decay,
            "verdict": "PASS" if b_pass else ("FAIL" if monotonic_decay else "MARGINAL"),
        }
    else:
        verdict["validations"]["B_walkforward"] = {
            "verdict": "INSUFFICIENT_DATA",
            "n_years_with_sufficient_data": b_focus.height,
        }

    # C · Volume
    # Filtrar tier ">10K" o ">100K"
    c_focus = c_df.filter(pl.col("bucket") == "0.30-0.40").filter(
        pl.col("vol_tier").is_in(["10K-100K", ">100K"])
    )
    if c_focus.height:
        ev_liq = float(
            (c_focus["ev_per_dollar"] * c_focus["n_trades"]).sum()
            / c_focus["n_trades"].sum()
        )
        c_pass = ev_liq >= THRESHOLD_C_PASS
        verdict["validations"]["C_volume"] = {
            "ev_weighted_liquid_markets": ev_liq,
            "threshold_pass": THRESHOLD_C_PASS,
            "verdict": "PASS" if c_pass else "FAIL",
        }
    else:
        verdict["validations"]["C_volume"] = {"verdict": "INSUFFICIENT_DATA"}

    # D · Wallets
    if d_stats["total_trades"] > 0:
        top50_share = d_stats["top50_share"]
        d_pass = top50_share < THRESHOLD_D_PASS
        d_fail = top50_share > THRESHOLD_D_FAIL
        verdict["validations"]["D_wallets"] = {
            "top50_share": top50_share,
            "threshold_pass": THRESHOLD_D_PASS,
            "threshold_fail": THRESHOLD_D_FAIL,
            "verdict": "PASS" if d_pass else ("FAIL" if d_fail else "MARGINAL"),
        }
    else:
        verdict["validations"]["D_wallets"] = {"verdict": "NO_DATA"}

    # Decisión final
    verdicts = [v["verdict"] for v in verdict["validations"].values()]
    n_pass = sum(1 for v in verdicts if v == "PASS")
    n_fail = sum(1 for v in verdicts if v == "FAIL")
    if n_fail >= 2:
        final = "ABANDON — edge no sobrevive múltiples validaciones"
    elif n_pass >= 3 and n_fail == 0:
        final = "PROCEED_TO_FASE_1 — edge robusto, viable construir bot minimalista"
    elif n_pass >= 2 and n_fail <= 1:
        final = "MARGINAL_PROCEED — edge existe pero frágil, tamaño mínimo y validar forward"
    else:
        final = "MARGINAL_PAUSE — evidencia mixta, reasignar tiempo a otro proyecto"

    verdict["final_decision"] = final
    verdict["n_pass"] = n_pass
    verdict["n_fail"] = n_fail
    return verdict


# ─── Main ────────────────────────────────────────────────────────────────────
def main():
    summary = {"started": ts(), "polars": pl.__version__, "errors": []}

    try:
        base = build_markets_base()
        summary["markets_base_rows"] = base.height
    except Exception as e:
        summary["errors"].append({"phase": "base", "err": str(e), "tb": traceback.format_exc()})
        (OUT_DIR / "veredict.json").write_text(json.dumps(summary, indent=2, default=str))
        print(f"❌ build_markets_base falló: {e}")
        sys.exit(1)

    enriched = build_enriched_lazy(base)

    a_focus = None
    try:
        _, a_focus = fase_a(enriched)
    except Exception as e:
        summary["errors"].append({"phase": "A", "err": str(e), "tb": traceback.format_exc()})
        print(f"⚠ Fase A falló: {e}")

    b_df = None
    try:
        b_df = fase_b(enriched)
    except Exception as e:
        summary["errors"].append({"phase": "B", "err": str(e), "tb": traceback.format_exc()})
        print(f"⚠ Fase B falló: {e}")

    c_df = None
    try:
        c_df = fase_c(enriched)
    except Exception as e:
        summary["errors"].append({"phase": "C", "err": str(e), "tb": traceback.format_exc()})
        print(f"⚠ Fase C falló: {e}")

    d_stats = None
    try:
        _, d_stats = fase_d(enriched)
    except Exception as e:
        summary["errors"].append({"phase": "D", "err": str(e), "tb": traceback.format_exc()})
        print(f"⚠ Fase D falló: {e}")

    # Veredicto
    if a_focus is not None and b_df is not None and c_df is not None and d_stats is not None:
        try:
            verdict = build_verdict(a_focus, b_df, c_df, d_stats)
            summary["verdict"] = verdict
            header("VEREDICTO")
            print(json.dumps(verdict, indent=2, default=str))
        except Exception as e:
            summary["errors"].append({"phase": "verdict", "err": str(e), "tb": traceback.format_exc()})
            print(f"⚠ Veredicto falló: {e}")

    summary["finished"] = ts()
    (OUT_DIR / "veredict.json").write_text(json.dumps(summary, indent=2, default=str))
    header("✓ FASE 0.5 COMPLETA")
    print(f"Outputs en: {OUT_DIR}")
    print("\nPegar en el próximo chat:")
    print("  1. veredict.json            ← PASS/FAIL por validación + decisión")
    print("  2. fase05_a_temporal.csv")
    print("  3. fase05_b_walkforward.csv")
    print("  4. fase05_c_volume.csv")
    print("  5. fase05_d_top100_wallets.csv (si lo quieres)")


if __name__ == "__main__":
    main()
