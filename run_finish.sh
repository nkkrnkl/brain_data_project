set -euo pipefail
cd "$(dirname "$0")"

echo "[finish] launching Phase 1c (FD sweep metrics)…"
python3 compute_metrics.py fd > /tmp/metrics_fd.log 2>&1
echo "[finish] FD sweep metrics done."

echo "[finish] running Phase 3 sweep stats (κ + FD)…"
python3 analyze_age.py all > /tmp/stats_all.log 2>&1
echo "[finish] sweep stats done."

echo "[finish] writing Phase 5 full tables…"
python3 make_tables.py > /tmp/tables.log 2>&1
echo "[finish] tables done."

echo "[finish] making fig6 κ sensitivity + refreshing all figures…"
python3 make_figures.py > /tmp/figures.log 2>&1
echo "[finish] figures done."

echo "[finish] ALL DONE."
ls -la results/figures/ results/*.csv | tail -40
