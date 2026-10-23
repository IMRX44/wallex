import math

from pishro.backtest import synthetic, tune
from pishro.config import load_config


def test_sample_covers_ranges_and_respects_risk_limits():
    confs = tune.sample(12, seed=1)
    assert len(confs) == 12
    for k, (lo, hi, _) in tune.SPACE.items():
        vals = [c[k] for c in confs]
        assert min(vals) >= lo - 1e-9 and max(vals) <= hi + 1e-9
    assert max(c["stop_pct"] for c in confs) <= 0.025 and max(c["position_frac"] for c in confs) <= 0.25


def test_score_and_robust_prefers_plateau():
    m = {"return_pct": 4.0, "max_drawdown_pct": 1.0, "trades": 30}
    assert tune.score(m, 3.0, 20) == 2.0
    assert tune.score(dict(m, max_drawdown_pct=3.5), 3.0, 20) == -math.inf
    assert tune.score(dict(m, trades=5), 3.0, 20) == -math.inf
    base = {k: (lo + hi) / 2 for k, (lo, hi, _) in tune.SPACE.items()}
    # a spike whose neighbours are poor vs. a plateau of uniformly good settings
    spike = dict(base, edge_vol_mult=3.2, stop_pct=0.015)
    around = [dict(spike, edge_vol_mult=3.1 - 0.02 * i) for i in range(3)]
    plateau = [dict(base, edge_vol_mult=1.9 + 0.01 * i) for i in range(4)]
    confs = [spike] + around + plateau
    rob = tune.robust_scores(confs, [10.0, 0.0, 0.0, 0.0, 3.0, 3.0, 3.0, 3.0], k=3)
    assert rob.index(max(rob)) >= 4          # the plateau wins over the lonely spike


def test_rolling_windows():
    tr = {"exit_ts": [0, 100, 5 * 86400 + 1], "pnl": [1000, -500, 2000]}
    w = tune.rolling(tr, 0, 10 * 86400, 1_000_000)
    assert [x["trades"] for x in w] == [2, 1] and w[0]["ret_pct"] == 0.05


def test_evaluate_runs_on_synthetic_history(tmp_path):
    cfg = load_config(None, overrides={"data_dir": str(tmp_path)})
    synthetic.generate(cfg.data_dir / "history", days=3, markets=3)
    from pishro.backtest import data, sim
    prep = sim.prepare(data.load_history(cfg))
    tune._init(prep, cfg)
    g = prep["grid"]
    r = tune.evaluate((tune.defaults(cfg), 0.0, 0.0, int(g[0]), int(g[-1]), 5e6, ()))
    assert set(r["metrics"]) >= {"return_pct", "max_drawdown_pct", "trades"}
