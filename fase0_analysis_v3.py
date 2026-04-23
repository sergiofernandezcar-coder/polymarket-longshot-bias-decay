#!/usr/bin/env python3
"""
Fase 0 v3 — Q1/Q2/Q3 definitivos
=================================

Schema confirmado de exploración v2:
  trades.csv:
    nonusdc_side ∈ {null, USDC, token1, token2}
    maker_direction / taker_direction ∈ {BUY, SELL} (complementarios)
    Cuando nonusdc_side=token1 o token2: `price` es el precio DE ESE TOKEN.
    En cualquier trade hay un BUYER (maker si maker_direction=BUY, si no taker).

Plan:
  Fase A — markets con categorización REFINADA (prioridades claras, menos falsos
           positivos; separa BUSINESS, HEALTH de OTHER).
  Fase B — determinar WINNER por market: pivot last_price sobre
           (market_id, nonusdc_side). Regla:
             last_p_token1 > last_p_token2 → token1 ganó
             y cross-check: min≤0.10 & max≥0.90
  Fase C — Q1: NO rate real por categoría + global (valida contra 73% oficial).
  Fase D — Q2: WR/EV por bucket de precio (buys identificados vía nonusdc_side).
  Fase E — Q3: matriz categoría × bucket, filtro WR≥70% n≥500, deal-flow.

Output: ~/poly_data/fase0_output_v3/
  - markets_with_category_v3.csv
  - winners_resolved.csv
  - q1_no_rate.csv + q1_global.json
  - q2_ev_by_bucket.csv
  - q3_cat_x_bucket.csv
  - q3_winning_combos.csv (WR≥70%, n≥500)
  - summary.json

Tiempo esperado: 8-15 min.
"""

import json
import re
import sys
import traceback
from datetime import datetime
from pathlib import Path

import polars as pl

# ─── Config ──────────────────────────────────────────────────────────────────
ROOT = Path.home() / "poly_data"
TRADES_CSV = ROOT / "processed" / "trades.csv"
MARKETS_CSV = ROOT / "markets.csv"
OUT_DIR = ROOT / "fase0_output_v3"
OUT_DIR.mkdir(exist_ok=True)

FEES_TAKER = 0.000  # Polymarket histórico taker fee ≈ 0 fuera de deportes US
BUCKET_BREAKS = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]
BUCKET_LABELS = [
    "0.00-0.10", "0.10-0.20", "0.20-0.30", "0.30-0.40", "0.40-0.50",
    "0.50-0.60", "0.60-0.70", "0.70-0.80", "0.80-0.90", "0.90-1.00",
]
MIN_N_Q3 = 500
WR_TARGET_Q3 = 0.70

# Prioridad: aplica en ORDEN, primer match gana (no override).
# Keywords específicos, con palabras-frontera cuando aplique.
# Uso listas de regex para word boundaries.
CATEGORY_PATTERNS = [
    ("CRYPTO", [
        r"\bbitcoin\b", r"\bbtc\b", r"\bethereum\b", r"\beth\b", r"\bsolana\b",
        r"\bsol\b", r"\bxrp\b", r"\bdoge(coin)?\b", r"\bcardano\b", r"\bada\b",
        r"\bavalanche\b", r"\bavax\b", r"\bpolygon\b", r"\bmatic\b",
        r"\bfilecoin\b", r"\bfil\b", r"\bchainlink\b", r"\blink\b",
        r"\btether\b", r"\busdt\b", r"\busdc\b", r"\bpepe\b", r"\bshib\b",
        r"\btoken\b", r"\bcrypto\b", r"\bcoinbase\b", r"\bbinance\b",
        r"\bdefi\b", r"\bnft\b", r"\bairdrop\b", r"\bhalving\b", r"\bweb3\b",
        r"\btvl\b", r"\bstable\s*coin\b", r"\bhyperliquid\b", r"\bpolymarket\b",
        r"\b5[-\s]?minute\b", r"\b5\s?min\b", r"\b15\s?min\b",
        r"\bup\s?or\s?down\b",
    ]),
    ("POLITICS", [
        r"\bpresident(ial)?\b", r"\belection\b", r"\bsenator\b", r"\bcongress\b",
        r"\bbiden\b", r"\btrump\b", r"\bharris\b", r"\bkamala\b",
        r"\bdemocrat\b", r"\brepublican\b", r"\bgovernor\b", r"\bparliament\b",
        r"\bprime minister\b", r"\bcabinet\b", r"\bnominee\b", r"\bimpeach\b",
        r"\bvote\b", r"\bballot\b", r"\bprimary\b", r"\bcaucus\b",
        r"\binauguration\b", r"\bsupreme court\b", r"\bscotus\b",
        r"\bnato\b", r"\bsanction\b", r"\bputin\b", r"\bzelensky\b",
        r"\brussia\b", r"\bukraine\b", r"\bisrael\b", r"\bgaza\b",
        r"\bchina\b", r"\bxi jinping\b", r"\bkim jong\b", r"\bnorth korea\b",
        r"\bdems\b", r"\breps\b", r"\bhouse seat\b", r"\bswing state\b",
        r"\bhunter biden\b", r"\bdesantis\b", r"\bron desantis\b",
        r"\bmitch mcconnell\b", r"\bpelosi\b", r"\bschumer\b", r"\bclinton\b",
        r"\bcampaign\b", r"\bgop\b", r"\bfed chair\b", r"\bpowell\b",
    ]),
    ("SPORTS", [
        r"\bnba\b", r"\bnfl\b", r"\bnhl\b", r"\bmlb\b", r"\bncaa\b",
        r"\bepl\b", r"\bpremier league\b", r"\bla liga\b", r"\bserie a\b",
        r"\bbundesliga\b", r"\bchampions league\b", r"\bworld cup\b",
        r"\bsuper bowl\b", r"\bplayoff\b", r"\btouchdown\b", r"\bufc\b",
        r"\bmma\b", r"\bboxing\b", r"\bformula 1\b", r"\bgrand prix\b",
        r"\btennis\b", r"\batp\b", r"\bwta\b", r"\bgrand slam\b",
        r"\bwimbledon\b", r"\bfrench open\b", r"\bopen final\b",
        # teams (NBA/NFL/NHL/MLB most recognizable)
        r"\blakers\b", r"\bceltics\b", r"\bwarriors\b", r"\bknicks\b",
        r"\bnets\b", r"\bheat\b", r"\bbucks\b", r"\bnuggets\b",
        r"\byankees\b", r"\bred sox\b", r"\bdodgers\b", r"\bmets\b",
        r"\bchiefs\b", r"\beagles\b", r"\bcowboys\b", r"\bpatriots\b",
        r"\b49ers\b", r"\bseahawks\b", r"\brams\b",
        r"\brangers\b", r"\bbruins\b", r"\boilers\b", r"\bmaple leafs\b",
        r"\breal madrid\b", r"\bbarcelona\b", r"\bbarça\b",
        r"\bmanchester\b", r"\bliverpool\b", r"\barsenal\b", r"\bchelsea\b",
        r"\bpsg\b", r"\bbayern\b", r"\bjuventus\b", r"\binter milan\b",
        r"\bkhabib\b", r"\bmcgregor\b", r"\blebron\b", r"\bmessi\b",
        r"\bronaldo\b", r"\bmahomes\b", r"\bjordan\b",
        r"\bwin\s+the\s+(game|match|series|championship|title|finals|cup)\b",
        r"\bbeat\s+the\b",
    ]),
    ("FINANCE", [
        r"\bfomc\b", r"\binterest rate\b", r"\bcpi\b",
        r"\binflation\b", r"\bgdp\b", r"\bunemployment\b",
        r"\bs&p\b", r"\bnasdaq\b", r"\bdow jones\b", r"\bdjia\b",
        r"\bnvidia\b", r"\bapple\b", r"\btesla\b", r"\bmicrosoft\b",
        r"\bgoogle\b", r"\bamazon\b", r"\bmeta\b", r"\bfacebook\b",
        r"\bearnings\b", r"\brecession\b", r"\btreasury\b", r"\byield\b",
        r"\bmortgage rate\b", r"\bppi\b", r"\bnon[-\s]farm\b",
        r"\bjobless claims\b", r"\bretail sales\b",
        r"\bfederal reserve\b", r"\bfed cut\b", r"\bfed hike\b",
        r"\bfed rate\b", r"\bfed meeting\b", r"\bfed decision\b",
    ]),
    ("BUSINESS", [
        r"\bipo\b", r"\bpublicly trad", r"\bspac\b", r"\bacquisition\b",
        r"\bmerger\b", r"\bbuyout\b", r"\bbankruptcy\b",
        r"\btwitter\b", r"\bx\.com\b", r"\bairbnb\b", r"\buber\b",
        r"\blyft\b", r"\bstarbucks\b", r"\bdisney\b", r"\bwalmart\b",
        r"\bceo\b", r"\bfounder\b", r"\blayoffs?\b", r"\bfire[ds]?\b",
    ]),
    ("HEALTH", [
        r"\bcovid\b", r"\bcoronavirus\b", r"\bpandemic\b", r"\bvaccine\b",
        r"\bfda\b", r"\bemergency use\b", r"\beua\b", r"\bcancer\b",
        r"\boutbreak\b", r"\bdisease\b", r"\bmedical\b",
        r"\bwho declare\b", r"\bworld health\b",
        r"\bmonkeypox\b", r"\bebola\b", r"\bflu\b", r"\binfection\b",
    ]),
    ("CULTURE", [
        r"\boscar\b", r"\bgrammy\b", r"\bemmy\b", r"\bgolden globe\b",
        r"\bcannes\b", r"\bmet gala\b", r"\btiktok\b", r"\byoutube\b",
        r"\balbum\b", r"\bsong\b", r"\bmovie\b", r"\bfilm\b",
        r"\bbox office\b", r"\bcelebrity\b", r"\bkardashian\b",
        r"\btaylor swift\b", r"\bdrake\b", r"\bkanye\b", r"\beminem\b",
        r"\brihanna\b", r"\bbeyonc", r"\bmarvel\b", r"\bstar wars\b",
    ]),
    ("WEATHER", [
        r"\bhurricane\b", r"\btornado\b", r"\bearthquake\b",
        r"\btemperature\b", r"\bsnow\b", r"\brainfall\b", r"\bweather\b",
        r"\bwildfire\b", r"\bheat wave\b", r"\bcold snap\b",
    ]),
    ("SPACE", [
        r"\bspacex\b", r"\bnasa\b", r"\bmars\b", r"\bmoon landing\b",
        r"\brocket launch\b", r"\bspace launch\b", r"\bstarship\b",
        r"\binternational space station\b", r"\biss\b",
    ]),
]


def ts() -> str:
    return datetime.now().isoformat(timespec="seconds")


def header(title: str) -> None:
    print(f"\n{'═' * 72}\n  {title}\n  [{ts()}]\n{'═' * 72}")


def stream_collect(lf):
    try:
        return lf.collect(engine="streaming")
    except TypeError:
        return lf.collect(streaming=True)


# ─── Fase A: Categorización refinada ─────────────────────────────────────────
def fase_a_categorize():
    header("A · Categorización refinada de mercados")
    lf_m = pl.scan_csv(str(MARKETS_CSV), infer_schema_length=0)

    m = lf_m.with_columns(
        (
            pl.col("question").fill_null("")
            + " | "
            + pl.col("market_slug").fill_null("")
            + " | "
            + pl.col("ticker").fill_null("")
        )
        .str.to_lowercase()
        .alias("text_blob")
    )

    # Prioridad EN ORDEN: el primer match gana. Acumulamos con when-otherwise.
    cat_expr = pl.lit("OTHER")
    for cat, patterns in reversed(CATEGORY_PATTERNS):
        combined = "|".join(f"(?:{p})" for p in patterns)
        cat_expr = (
            pl.when(pl.col("text_blob").str.contains(f"(?i){combined}"))
            .then(pl.lit(cat))
            .otherwise(cat_expr)
        )

    m = m.with_columns(cat_expr.alias("category"))
    m_df = m.select(
        ["id", "question", "answer1", "answer2", "token1", "token2",
         "category", "closedTime", "volume"]
    ).collect()

    # Distribución
    dist = (
        m_df.group_by("category")
        .agg(pl.len().alias("n"))
        .sort("n", descending=True)
    )
    print("Distribución categoría:")
    print(dist)

    # Ejemplos por categoría (3 por cat)
    print("\nEjemplos por categoría (3 por cat):")
    for row in dist.iter_rows(named=True):
        ex = (
            m_df.filter(pl.col("category") == row["category"])
            .select(["question", "answer1", "answer2"])
            .head(3)
        )
        print(f"\n-- {row['category']} ({row['n']:,}) --")
        print(ex)

    m_df.write_csv(OUT_DIR / "markets_with_category_v3.csv")
    print(f"\n✓ markets_with_category_v3.csv escrito ({m_df.height:,} filas)")
    return m_df


# ─── Fase B: Resolver winner por market vía pivot nonusdc_side ───────────────
def fase_b_resolve_winners(markets_df):
    header("B · Resolver winner via last_price por (market_id, nonusdc_side)")

    lf_t = pl.scan_csv(str(TRADES_CSV), infer_schema_length=0)

    # Filtrar a trades con market_id y nonusdc_side informativo
    t = (
        lf_t.filter(pl.col("market_id").is_not_null())
        .filter(pl.col("nonusdc_side").is_in(["token1", "token2"]))
        .with_columns(
            [
                pl.col("timestamp").cast(pl.Int64, strict=False).alias("ts_int"),
                pl.col("price").cast(pl.Float64, strict=False).alias("price_f"),
            ]
        )
        .filter(pl.col("ts_int").is_not_null())
        .filter((pl.col("price_f") >= 0.0) & (pl.col("price_f") <= 1.0))
        .select(["market_id", "nonusdc_side", "ts_int", "price_f"])
    )

    print("Computando last_price por (market_id, nonusdc_side)... (2-5 min)")
    # Para cada (market, side): precio del último timestamp
    last_per_side = (
        t.sort(["market_id", "nonusdc_side", "ts_int"])
        .group_by(["market_id", "nonusdc_side"])
        .agg(
            [
                pl.col("price_f").last().alias("last_price"),
                pl.col("price_f").max().alias("max_price"),
                pl.col("price_f").min().alias("min_price"),
                pl.col("ts_int").last().alias("last_ts"),
                pl.len().alias("n_trades"),
            ]
        )
        .pipe(stream_collect)
    )
    print(f"  {last_per_side.height:,} (market, side) pairs")

    # Pivot: una fila por market con columnas token1 y token2
    pivot = last_per_side.pivot(
        values="last_price",
        index="market_id",
        on="nonusdc_side",
        aggregate_function="first",
    ).rename({"token1": "last_token1", "token2": "last_token2"})

    # Cruzar con markets para añadir metadata
    winners = (
        markets_df.select(["id", "category", "answer1", "answer2", "closedTime", "volume"])
        .rename({"id": "market_id"})
        .join(pivot, on="market_id", how="inner")
    )

    # Determinar winner
    winners = winners.with_columns(
        pl.when(pl.col("last_token1").is_null() | pl.col("last_token2").is_null())
        .then(None)
        .when((pl.col("last_token1") >= 0.90) & (pl.col("last_token2") <= 0.10))
        .then(pl.lit("token1"))
        .when((pl.col("last_token2") >= 0.90) & (pl.col("last_token1") <= 0.10))
        .then(pl.lit("token2"))
        .otherwise(None)
        .alias("winner")
    )

    n_total = winners.height
    n_resolved = winners.filter(pl.col("winner").is_not_null()).height
    n_both_sides = winners.filter(
        pl.col("last_token1").is_not_null() & pl.col("last_token2").is_not_null()
    ).height
    n_one_side = winners.filter(
        pl.col("last_token1").is_null() ^ pl.col("last_token2").is_null()
    ).height

    print(f"\nCobertura:")
    print(f"  Markets con trades válidos: {n_total:,}")
    print(f"  Con trades en AMBOS tokens: {n_both_sides:,}")
    print(f"  Solo trades en UN token:   {n_one_side:,}")
    print(f"  Winner determinado:         {n_resolved:,} ({100*n_resolved/n_total:.1f}%)")

    # Distribución winner
    wdist = (
        winners.filter(pl.col("winner").is_not_null())
        .group_by("winner")
        .agg(pl.len().alias("n"))
    )
    print(f"\nDistribución winner (sobre resueltos):")
    print(wdist)

    winners.write_csv(OUT_DIR / "winners_resolved.csv")
    print(f"\n✓ winners_resolved.csv escrito")
    return winners


# ─── Fase C: Q1 — NO rate real ───────────────────────────────────────────────
def fase_c_q1(winners_df):
    header("C · Q1 — NO resolution rate real")

    # Solo binarios Yes/No puros
    yn = winners_df.filter(
        (pl.col("answer1").str.to_lowercase() == "yes")
        & (pl.col("answer2").str.to_lowercase() == "no")
        & pl.col("winner").is_not_null()
    ).with_columns(
        # Por convención Polymarket: token1 corresponde a answer1 (Yes); token2 a answer2 (No)
        # Validable porque en mercados Yes/No puros, si token1 ganó, ganó YES.
        pl.when(pl.col("winner") == "token1").then(pl.lit("YES"))
        .when(pl.col("winner") == "token2").then(pl.lit("NO"))
        .otherwise(None)
        .alias("outcome_yesno")
    )

    print(f"Binarios Yes/No resueltos: {yn.height:,}")

    # Q1 global
    global_no_rate = yn.filter(pl.col("outcome_yesno") == "NO").height / yn.height
    print(f"\n🎯 Q1 GLOBAL — NO rate real: {global_no_rate:.4f} ({global_no_rate*100:.2f}%)")
    print(f"   Benchmark oficial Polymarket:   0.7330 (73.30%)")
    gap = abs(global_no_rate - 0.733)
    if gap < 0.03:
        print(f"   ✓ Coincide con el benchmark (gap={gap:.4f}). Método válido.")
    else:
        print(f"   ⚠ Desvía del benchmark (gap={gap:.4f}). Revisar mapping token1→answer1.")

    # Q1 por categoría
    by_cat = (
        yn.group_by("category")
        .agg(
            [
                pl.len().alias("n_markets"),
                (pl.col("outcome_yesno") == "NO").sum().alias("n_no_wins"),
                (pl.col("outcome_yesno") == "NO").mean().alias("no_rate"),
                pl.col("volume").cast(pl.Float64, strict=False).median().alias("median_volume"),
            ]
        )
        .sort("n_markets", descending=True)
    )
    print("\nQ1 por categoría:")
    print(by_cat)

    # Categorías con NO rate ≥ 80% y n ≥ 500
    strong = by_cat.filter(pl.col("no_rate") >= 0.80).filter(pl.col("n_markets") >= 500)
    print(f"\n🎯 Categorías con NO rate ≥ 80% y n ≥ 500: {strong.height}")
    if strong.height:
        print(strong)

    by_cat.write_csv(OUT_DIR / "q1_no_rate.csv")
    with (OUT_DIR / "q1_global.json").open("w") as f:
        json.dump(
            {
                "n_binary_yn": yn.height,
                "global_no_rate": global_no_rate,
                "benchmark_polymarket": 0.733,
                "gap_abs": gap,
                "method_valid": gap < 0.03,
            },
            f,
            indent=2,
        )
    return yn


# ─── Fase D+E: Q2 (EV por bucket) + Q3 (cat × bucket) ────────────────────────
def fase_de_q2_q3(markets_df, winners_df):
    header("D+E · Q2 (EV/bucket) + Q3 (cat × bucket)")

    # Resuelto → pasamos a lazy con solo lo que necesitamos
    resolved = winners_df.filter(pl.col("winner").is_not_null()).select(
        ["market_id", "category", "winner"]
    )

    print(f"Universo mercados resueltos: {resolved.height:,}")

    # Trades BUY-able: filas con nonusdc_side in {token1,token2}
    # Para cada fila, el comprador existe siempre (maker o taker).
    # price es el precio del nonusdc_side.
    # trade_won = (nonusdc_side == winner)
    lf_t = pl.scan_csv(str(TRADES_CSV), infer_schema_length=0)
    trades = (
        lf_t.filter(pl.col("market_id").is_not_null())
        .filter(pl.col("nonusdc_side").is_in(["token1", "token2"]))
        .with_columns(
            pl.col("price").cast(pl.Float64, strict=False).alias("entry_price")
        )
        .filter((pl.col("entry_price") > 0.0) & (pl.col("entry_price") < 1.0))
        .select(["market_id", "nonusdc_side", "entry_price"])
    )

    joined = trades.join(resolved.lazy(), on="market_id", how="inner").with_columns(
        (pl.col("nonusdc_side") == pl.col("winner")).alias("trade_won")
    )

    # Bucket de entry_price
    joined = joined.with_columns(
        pl.col("entry_price")
        .cut(breaks=BUCKET_BREAKS, labels=BUCKET_LABELS)
        .alias("bucket")
    )

    # Q2 global
    print("Computando Q2 (WR+EV por bucket)... (3-8 min)")
    q2 = (
        joined.group_by("bucket")
        .agg(
            [
                pl.len().alias("n_trades"),
                pl.col("trade_won").mean().alias("win_rate"),
                pl.col("entry_price").mean().alias("avg_entry"),
            ]
        )
        .sort("bucket")
        .pipe(stream_collect)
    )
    q2 = q2.with_columns(
        (
            pl.col("win_rate") * (1.0 - pl.col("avg_entry"))
            - (1.0 - pl.col("win_rate")) * pl.col("avg_entry")
            - FEES_TAKER * pl.col("avg_entry")
        ).alias("ev_per_dollar")
    )
    print("\nQ2 — WR y EV por bucket (global):")
    print(q2)
    q2.write_csv(OUT_DIR / "q2_ev_by_bucket.csv")

    # Q3 cat × bucket
    print("\nComputando Q3 (cat × bucket)... (3-8 min)")
    q3_all = (
        joined.group_by(["category", "bucket"])
        .agg(
            [
                pl.len().alias("n_trades"),
                pl.col("trade_won").mean().alias("win_rate"),
                pl.col("entry_price").mean().alias("avg_entry"),
            ]
        )
        .pipe(stream_collect)
    )
    q3_all = q3_all.with_columns(
        (
            pl.col("win_rate") * (1.0 - pl.col("avg_entry"))
            - (1.0 - pl.col("win_rate")) * pl.col("avg_entry")
            - FEES_TAKER * pl.col("avg_entry")
        ).alias("ev_per_dollar")
    )
    q3_all = q3_all.sort(["category", "bucket"])
    q3_all.write_csv(OUT_DIR / "q3_cat_x_bucket.csv")
    print(f"\nQ3 total filas: {q3_all.height}")
    print(q3_all)

    # Q3 filtrado: ganadores según umbrales Fase 0
    q3_win = (
        q3_all.filter(pl.col("n_trades") >= MIN_N_Q3)
        .filter(pl.col("win_rate") >= WR_TARGET_Q3)
        .sort(["ev_per_dollar", "n_trades"], descending=[True, True])
    )
    q3_win = q3_win.with_columns(
        # Deal flow aprox: 38 meses (Nov22 → Ene26)
        (pl.col("n_trades") / 38.0).round(1).alias("approx_trades_per_month")
    )
    q3_win.write_csv(OUT_DIR / "q3_winning_combos.csv")
    print(f"\n🎯 Q3 combinaciones con WR≥{WR_TARGET_Q3} y n≥{MIN_N_Q3}: {q3_win.height}")
    if q3_win.height:
        print(q3_win)
    else:
        print("  (ninguna combinación pasa ambos umbrales)")
        # Fallback: top 10 por EV con n≥500 (aunque WR<70%)
        top_ev = (
            q3_all.filter(pl.col("n_trades") >= MIN_N_Q3)
            .sort("ev_per_dollar", descending=True)
            .head(10)
        )
        print("\nFallback — Top 10 por EV con n≥500 (sin umbral WR):")
        print(top_ev)
    return q2, q3_all, q3_win


# ─── Main ────────────────────────────────────────────────────────────────────
def main():
    summary = {"started": ts(), "polars": pl.__version__, "errors": []}

    try:
        markets_df = fase_a_categorize()
        summary["phase_a"] = f"ok — {markets_df.height:,} mercados categorizados"
    except Exception as e:
        summary["errors"].append({"phase": "A", "err": str(e), "tb": traceback.format_exc()})
        (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
        print(f"❌ Phase A falló: {e}")
        sys.exit(1)

    try:
        winners_df = fase_b_resolve_winners(markets_df)
        summary["phase_b"] = {
            "n_markets_total": winners_df.height,
            "n_resolved": winners_df.filter(pl.col("winner").is_not_null()).height,
        }
    except Exception as e:
        summary["errors"].append({"phase": "B", "err": str(e), "tb": traceback.format_exc()})
        (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
        print(f"❌ Phase B falló: {e}")
        sys.exit(1)

    try:
        yn = fase_c_q1(winners_df)
        summary["phase_c"] = {"n_yn_binary": yn.height}
    except Exception as e:
        summary["errors"].append({"phase": "C", "err": str(e), "tb": traceback.format_exc()})
        print(f"⚠ Phase C falló: {e}")

    try:
        q2, q3_all, q3_win = fase_de_q2_q3(markets_df, winners_df)
        summary["phase_d"] = {"q2_buckets": q2.to_dicts()}
        summary["phase_e"] = {
            "q3_total_combos": q3_all.height,
            "q3_winning_combos": q3_win.height,
            "q3_top5": q3_win.head(5).to_dicts() if q3_win.height else [],
        }
    except Exception as e:
        summary["errors"].append({"phase": "D+E", "err": str(e), "tb": traceback.format_exc()})
        print(f"⚠ Phase D+E falló: {e}")

    summary["finished"] = ts()
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2, default=str))

    header("✓ FASE 0 v3 COMPLETA")
    print(f"Outputs en: {OUT_DIR}")
    print("\nPegar en el próximo chat:")
    print("  1. q1_global.json           ← valida método (gap vs 73.3%)")
    print("  2. q1_no_rate.csv           ← NO rate por categoría")
    print("  3. q2_ev_by_bucket.csv      ← WR/EV por precio de entrada")
    print("  4. q3_cat_x_bucket.csv      ← matriz completa")
    print("  5. q3_winning_combos.csv    ← combos ganadores (puede ir vacío)")
    print("  6. summary.json")


if __name__ == "__main__":
    main()
