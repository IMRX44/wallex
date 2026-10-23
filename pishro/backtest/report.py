"""Performance metrics and the HTML/JSON backtest report (Persian)."""
from __future__ import annotations

import html
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


def max_drawdown(equity: pd.Series) -> float:
    if len(equity) == 0:
        return 0.0
    peak = equity.cummax()
    return float((1 - equity / peak).max())


def metrics(trades: pd.DataFrame, equity: pd.Series, capital: float) -> dict:
    out = {"final_equity": float(equity.iloc[-1]) if len(equity) else capital}
    out["return_pct"] = (out["final_equity"] / capital - 1) * 100
    out["max_drawdown_pct"] = max_drawdown(equity) * 100
    days = max((equity.index[-1] - equity.index[0]).total_seconds() / 86400, 1e-9) if len(equity) > 1 else 0
    out["days"] = days
    n = len(trades)
    out["trades"] = n
    if n:
        wins = trades[trades.pnl > 0]
        losses = trades[trades.pnl <= 0]
        out["win_rate_pct"] = len(wins) / n * 100
        out["avg_trade_ret_pct"] = float(trades.ret.mean() * 100)
        out["profit_factor"] = float(wins.pnl.sum() / -losses.pnl.sum()) if len(losses) and losses.pnl.sum() < 0 else float("inf")
        out["volume_tmn"] = float(trades.cost.sum() + trades.proceeds.sum())
        out["avg_trade_tmn"] = float(trades.cost.mean())
        out["avg_hold_min"] = float(trades.hold_min.mean())
        out["trades_per_day"] = n / days if days else 0.0
        out["exit_reasons"] = trades.reason.value_counts().to_dict()
        out["entry_modes"] = trades["mode"].value_counts().to_dict() if "mode" in trades else {}
        per = trades.groupby("symbol").agg(n=("pnl", "size"), pnl=("pnl", "sum"), win=("pnl", lambda x: (x > 0).mean()))
        out["per_market"] = per.sort_values("pnl", ascending=False).round(3).reset_index().to_dict("records")
    if len(equity) > 1:
        daily = equity.resample("1D").last().pct_change().dropna()
        out["sharpe_daily"] = float(daily.mean() / daily.std() * math.sqrt(365)) if len(daily) > 2 and daily.std() > 0 else None
        out["worst_day_pct"] = float(daily.min() * 100) if len(daily) else None
    return out


def monte_carlo_3day(trades: pd.DataFrame, capital: float, days: float, sims: int = 20000, seed: int = 7) -> dict:
    """Bootstrap 3-day outcomes (the challenge length) from the backtest's trade P&L."""
    if len(trades) < 5 or days <= 0:
        return {}
    rng = np.random.default_rng(seed)
    pnl = trades.pnl.to_numpy() / capital
    lam = len(trades) / days * 3
    counts = rng.poisson(lam, sims)
    res = np.array([pnl[rng.integers(0, len(pnl), c)].sum() if c else 0.0 for c in counts]) * 100
    return {"trades_per_3d": lam, "p5": float(np.percentile(res, 5)), "p25": float(np.percentile(res, 25)),
            "median": float(np.median(res)), "p75": float(np.percentile(res, 75)), "p95": float(np.percentile(res, 95)),
            "prob_positive": float((res > 0).mean() * 100)}


# --------------------------------------------------------------------------- HTML
def _svg_lines(series: dict[str, pd.Series], w: int = 860, h: int = 260) -> str:
    series = {k: v for k, v in series.items() if len(v) > 1}
    if not series:
        return ""
    colors = ["#2563eb", "#16a34a", "#f59e0b", "#dc2626", "#7c3aed"]
    allv = pd.concat([v / v.iloc[0] for v in series.values()])
    lo, hi = float(allv.min()), float(allv.max())
    pad = (hi - lo) * 0.05 or 0.01
    lo, hi = lo - pad, hi + pad
    parts = [f'<svg viewBox="0 0 {w} {h}" class="chart" role="img">']
    for k, y in enumerate(np.linspace(lo, hi, 5)):
        py = h - 20 - (y - lo) / (hi - lo) * (h - 40)
        parts.append(f'<line x1="40" x2="{w}" y1="{py:.1f}" y2="{py:.1f}" class="grid"/>'
                     f'<text x="0" y="{py + 4:.1f}" class="axis">{(y - 1) * 100:+.1f}%</text>')
    for idx, (name, s) in enumerate(series.items()):
        v = (s / s.iloc[0]).resample("15min").last().dropna() if isinstance(s.index, pd.DatetimeIndex) else s / s.iloc[0]
        xs = np.linspace(40, w - 5, len(v))
        ys = h - 20 - (v.to_numpy() - lo) / (hi - lo) * (h - 40)
        pts = " ".join(f"{x:.1f},{y:.1f}" for x, y in zip(xs, ys))
        parts.append(f'<polyline points="{pts}" fill="none" stroke="{colors[idx % 5]}" stroke-width="2"/>')
    parts.append("</svg>")
    legend = " ".join(f'<span class="lg"><i style="background:{colors[i % 5]}"></i>{html.escape(n)}</span>'
                      for i, n in enumerate(series))
    return "".join(parts) + f'<div class="legend">{legend}</div>'


def _kv_table(rows: list[tuple[str, str]]) -> str:
    return "<table>" + "".join(f"<tr><th>{html.escape(k)}</th><td>{v}</td></tr>" for k, v in rows) + "</table>"


def _fmt(m: dict) -> list[tuple[str, str]]:
    def f(k, fmt="{:.2f}", suf=""):
        v = m.get(k)
        return "—" if v is None else (fmt.format(v) + suf)
    return [("بازده خالص", f("return_pct", "{:+.2f}", "٪")), ("بیشترین افت سرمایه", f("max_drawdown_pct", "{:.2f}", "٪")),
            ("تعداد معاملات", f("trades", "{:d}")), ("نرخ برد", f("win_rate_pct", "{:.1f}", "٪")),
            ("میانگین بازده هر معامله", f("avg_trade_ret_pct", "{:+.3f}", "٪")), ("Profit factor", f("profit_factor")),
            ("حجم کل معاملات (تومان)", f("volume_tmn", "{:,.0f}")), ("میانگین حجم هر معامله", f("avg_trade_tmn", "{:,.0f}")),
            ("میانگین مدت نگه‌داری (دقیقه)", f("avg_hold_min", "{:.0f}")), ("معامله در روز", f("trades_per_day", "{:.1f}")),
            ("Sharpe روزانه (سالانه‌شده)", f("sharpe_daily")), ("بدترین روز", f("worst_day_pct", "{:+.2f}", "٪"))]


CSS = """
:root{--bg:#f8fafc;--fg:#0f172a;--card:#fff;--muted:#64748b;--line:#e2e8f0}
@media (prefers-color-scheme:dark){:root{--bg:#0b1220;--fg:#e2e8f0;--card:#111a2e;--muted:#94a3b8;--line:#1e293b}}
body{font-family:Vazirmatn,Tahoma,system-ui,sans-serif;background:var(--bg);color:var(--fg);margin:0;padding:16px;line-height:1.7}
.wrap{max-width:980px;margin:auto}.card{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:16px;margin:14px 0}
h1{font-size:1.5rem}h2{font-size:1.15rem;margin-top:0}table{border-collapse:collapse;width:100%;font-size:.92rem}
th,td{border-bottom:1px solid var(--line);padding:6px 8px;text-align:right}th{color:var(--muted);font-weight:500}
.chart{width:100%;height:auto}.grid{stroke:var(--line)}.axis{fill:var(--muted);font-size:11px}
.legend{font-size:.85rem;color:var(--muted)}.lg{margin-left:14px}.lg i{display:inline-block;width:10px;height:10px;border-radius:2px;margin-left:5px}
.note{color:var(--muted);font-size:.88rem}.grid2{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:12px}
code{direction:ltr;unicode-bidi:embed}
"""


def write_report(results: list[dict], out_dir: Path, meta: dict) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    body = [f"<h1>گزارش بک‌تست ربات پیشرو</h1><p class='note'>{html.escape(meta.get('subtitle', ''))}</p>"]
    summary = []
    for r in results:
        title = f"کارمزد میکر {r['maker'] * 100:.2f}٪ / تیکر {r['taker'] * 100:.2f}٪"
        m_rules, m_ml, m_full = r["m_rules_test"], r["m_ml_test"], r["m_rules_full"]
        body.append(f"<div class='card'><h2>{title}</h2>")
        body.append(_svg_lines({"دوره‌ی تست — فقط قواعد": r["eq_rules_test"], "دوره‌ی تست — قواعد + ML": r["eq_ml_test"]}))
        body.append("<div class='grid2'>")
        body.append("<div><h3>تست خارج از نمونه، فقط قواعد</h3>" + _kv_table(_fmt(m_rules)) + "</div>")
        body.append("<div><h3>تست خارج از نمونه، قواعد + مدل ML</h3>" + _kv_table(_fmt(m_ml)) + "</div>")
        body.append("</div>")
        body.append("<h3>کل دوره (فقط قواعد، برای مرجع)</h3>" + _kv_table(_fmt(m_full)))
        mc = r.get("mc_ml_test") or r.get("mc_rules_test")
        if mc:
            body.append("<h3>شبیه‌سازی مونت‌کارلوی ۳ روزه (طول چالش)</h3>" + _kv_table([
                ("احتمال بازده مثبت در ۳ روز", f"{mc['prob_positive']:.1f}٪"), ("میانه بازده ۳ روزه", f"{mc['median']:+.2f}٪"),
                ("بازه‌ی ۵ تا ۹۵ درصد", f"{mc['p5']:+.2f}٪ تا {mc['p95']:+.2f}٪"), ("معامله در ۳ روز", f"{mc['trades_per_3d']:.0f}")]))
        rep = r.get("model_report") or {}
        if rep:
            pa = rep.get("precision_at", {})
            body.append("<h3>مدل ML (LightGBM کالیبره)</h3>" + _kv_table([
                ("نمونه‌های آموزش", str(r.get("train_events"))), ("AUC روی داده‌ی تست مدل", f"{rep.get('auc', float('nan')):.3f}"),
                ("Brier", f"{rep.get('brier', float('nan')):.4f}"), ("نرخ پایه‌ی موفقیت", f"{rep.get('base_rate', 0) * 100:.1f}٪"),
                ("دقت در p≥0.55", html.escape(json.dumps(pa.get('0.55'), ensure_ascii=False))),
                ("مهم‌ترین ویژگی‌ها", html.escape("، ".join(list((rep.get("feature_importance") or {}).keys())[:6])))]))
        per = (m_ml if m_ml.get("trades") else m_rules).get("per_market", [])[:12]
        if per:
            body.append("<h3>سود به تفکیک بازار (تست)</h3><table><tr><th>بازار</th><th>معامله</th><th>سود (تومان)</th><th>نرخ برد</th></tr>" +
                        "".join(f"<tr><td>{p['symbol']}</td><td>{p['n']}</td><td>{p['pnl']:,.0f}</td><td>{p['win'] * 100:.0f}٪</td></tr>" for p in per) +
                        "</table>")
        body.append("</div>")
        summary.append({"maker": r["maker"], "taker": r["taker"], "rules_test": m_rules, "ml_test": m_ml,
                        "rules_full": m_full, "mc": mc, "model": rep, "train_events": r.get("train_events")})
    body.append("<div class='card note'><h2>فرض‌های بک‌تست</h2><ul>"
                "<li>سفارش هر دقیقه فقط با داده‌ی تا پایان دقیقه‌ی قبل ساخته می‌شود (بدون نگاه به آینده).</li>"
                "<li>خرید منتظر فقط وقتی پر می‌شود که قیمت ۰.۰۵٪ از آن عبور کند و حجم دقیقه کافی باشد.</li>"
                "<li>خرید فوری (اسنایپر) با قیمت بسته‌شدن + لغزش ۰.۰۸٪؛ حد ضرر و خروج زمانی با لغزش.</li>"
                "<li>اگر حد ضرر و هدف در یک دقیقه لمس شوند، حد ضرر فرض می‌شود.</li>"
                "<li>مدیر ریسک همان کلاس ربات زنده است (افت روزانه، افت کل، ضررهای پیاپی، سقف پوزیشن).</li>"
                "<li>در پایان دوره همه‌چیز به تومان فروخته می‌شود؛ بازده همان بازده محقق‌شده است.</li></ul></div>")
    page = (f"<!doctype html><html lang='fa' dir='rtl'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>"
            f"<title>Pishro Backtest</title><style>{CSS}</style></head><body><div class='wrap'>{''.join(body)}</div></body></html>")
    path = out_dir / "backtest.html"
    path.write_text(page, encoding="utf-8")
    (out_dir / "backtest.json").write_text(json.dumps({"meta": meta, "results": summary}, ensure_ascii=False, indent=2, default=str))
    return path
