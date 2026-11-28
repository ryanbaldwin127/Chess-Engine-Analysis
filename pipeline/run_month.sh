#!/usr/bin/env bash
# Run the data pipeline for one Lichess month on our standalone Spark cluster.
#
# Usage, from the repo root on the Spark master (charleston), inside tmux:
#   bash pipeline/run_month.sh filter     # chunks   -> data/filtered_<month>/
#   bash pipeline/run_month.sh rows       # filtered -> data/rows_<month>/{train,val}
#   bash pipeline/run_month.sh all        # both, one after the other
#
# Set MONTH to process another month (default 2026-10).
# Progress: http://charleston.cs.colostate.edu:30117 (from the CS network or VPN).
#
# If tasks fail with FileNotFoundException on files that exist, the Spark worker
# daemons were started before you joined the cs435a group and can't read other
# members' files; restart the cluster ($SPARK_HOME/sbin/stop-all.sh, then start-all.sh).

set -euo pipefail

MONTH=${MONTH:-2026-10}
DATA=/s/bach/a/class/cs435/cs435a/chess-zero/data
PY=/usr/local/python-env/py313/bin/python3
cd "$(dirname "${BASH_SOURCE[0]}")/.."

export PYSPARK_PYTHON=$PY
export PYSPARK_DRIVER_PYTHON=$PY

# --executor-memory 2g: the cluster's workers have 2 GB; Spark's 1 GB default is
#   too small to unpack a chunk.
# PYTHONPATH: lets the workers' Python find python-chess and zstandard in ~/.local.
# umask-mode=007: Spark sets file permissions itself, ignoring the shared folder's
#   default ACL; this keeps its output group-writable (department rule: g+rwx).
OPTS=(
    --executor-memory 2g
    --driver-memory 8g
    --conf "spark.pyspark.python=$PY"
    --conf "spark.executorEnv.PYTHONPATH=${PYTHONPATH:-}"
    --conf spark.hadoop.fs.permissions.umask-mode=007
)

run_filter() {
    echo "=== filter_games: chunks_$MONTH -> filtered_$MONTH ($(date))"
    spark-submit "${OPTS[@]}" pipeline/filter_games.py \
        "file://$DATA/chunks_$MONTH" "file://$DATA/filtered_$MONTH"
    echo "=== filter done ($(date)); counts:"
    cat "$DATA/filtered_$MONTH/_stats.json"
}

run_rows() {
    echo "=== make_rows: filtered_$MONTH -> rows_$MONTH ($(date))"
    spark-submit "${OPTS[@]}" pipeline/make_rows.py \
        "file://$DATA/filtered_$MONTH" "file://$DATA/rows_$MONTH"
    echo "=== rows done ($(date)); counts:"
    cat "$DATA/rows_$MONTH/stats.json"
}

case ${1:-} in
    filter) run_filter ;;
    rows)   run_rows ;;
    all)    run_filter; run_rows ;;
    *)      sed -n '4,8p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 1 ;;
esac
