#!/usr/bin/env bash
# Phase 1 queue for the A100 pod. Safe to rerun after a preemption: finished runs are
# skipped and unfinished ones resume from their checkpoint.
#   bash scripts/phase1.sh            # everything
#   TS_JOBS=4 OCR_JOBS=2 bash scripts/phase1.sh
set -uo pipefail
cd "$(dirname "$0")/.."
export PATH=/home/data/.local/bin:/home/data/.cargo/bin:$PATH
export CARGO_HOME=${CARGO_HOME:-/home/data/.cargo} RUSTUP_HOME=${RUSTUP_HOME:-/home/data/.rustup}
export TORCH_HOME=${TORCH_HOME:-/home/data/.cache/torch}
TS_JOBS=${TS_JOBS:-4}
OCR_JOBS=${OCR_JOBS:-2}
LOG=results/phase1/logs
mkdir -p "$LOG"

for t in 1 2; do
  uv run python scripts/run_baselines.py --task "$t" > "$LOG/baselines_task$t.log" 2>&1 || echo "baselines task$t failed"
done

ts_jobs() {
  for seed in 0 1 2; do
    for m in bigru_attention bigru; do echo "1 $m $seed"; done
    for m in seq2seq_attention seq2seq; do echo "2 $m $seed"; done
  done
}
ocr_jobs() {
  for seed in 0 1 2; do
    for m in cnn_attention_lstm cnn_pool_lstm crnn_ctc; do echo "$m $seed"; done
  done
}
export LOG
ts_jobs | xargs -P "$TS_JOBS" -L 1 bash -c \
  'uv run python scripts/run_ts.py --task $0 --model $1 --seed $2 > $LOG/task$0_$1_seed$2.log 2>&1 || echo "FAILED task$0 $1 seed$2"' &
ocr_jobs | xargs -P "$OCR_JOBS" -L 1 bash -c \
  'uv run python scripts/run_ocr.py --model $0 --seed $1 > $LOG/task3_$0_seed$1.log 2>&1 || echo "FAILED task3 $0 seed$1"' &
wait
echo "PHASE1_QUEUE_DONE $(date)"
