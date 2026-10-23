PY ?= python3

install:            ## install dependencies
	$(PY) -m pip install -r requirements-dev.txt

check:              ## connectivity check: Wallex, Binance, Nobitex, account, LLM
	$(PY) -m pishro check

fetch:              ## download 30 days of 1-minute history (run from an Iranian IP)
	$(PY) -m pishro fetch --days 30

backtest:           ## walk-forward backtest + ablation + report + train ML models
	$(PY) -m pishro backtest

demo:               ## offline smoke test of the whole pipeline on synthetic data
	$(PY) -m pishro demo

shadow:             ## shadow test (paper money, live prices) + dashboard on :8000
	PISHRO_MODE=paper $(PY) -m pishro serve

live:               ## REAL orders (needs WALLEX_API_KEY and LIVE_CONFIRM=I_UNDERSTAND)
	PISHRO_MODE=live $(PY) -m pishro serve

test:               ## unit + end-to-end tests
	$(PY) -m pytest -q

export:             ## copy decision log, trades and equity to ./deliverables
	$(PY) -m pishro export

video:              ## render the 5-minute demo video from ./data (needs ffmpeg + playwright)
	$(PY) -m video.render

deploy:             ## deploy to Liara (needs LIARA_API_TOKEN, LIARA_TEAM_ID)
	bash scripts/deploy_liara.sh

.PHONY: install check fetch backtest demo shadow live test export video deploy
