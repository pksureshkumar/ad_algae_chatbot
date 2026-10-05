#!/usr/bin/env bash
# LLM-judged RAGAS across every arm, sequentially.
# Sequential on purpose: concurrent judge runs thrash the GPU.
# Judge is phi4:14b, not qwen3:14b (the generator), so the self-preference
# objection stays answered. gemma3:27b was the first choice but another workload
# on this shared box left only ~10GB VRAM free, and 17.4GB will not fit.
set -u
PY=".venv-ragas/Scripts/python.exe"
ARMS="multimodal text_only no_retrieval gemma3_no_retrieval gpt_strict gpt_websearch"
for a in $ARMS; do
  echo ""
  echo "################ $a  ($(date +%H:%M:%S)) ################"
  PYTHONIOENCODING=utf-8 $PY harness/ragas_eval.py --run "runs/$a.jsonl" --judge phi4:14b 2>&1 \
    | tr '\r' '\n' | grep -vE "^Evaluating: *[0-9]+%" | tail -14
  echo "---- $a done at $(date +%H:%M:%S) ----"
done
echo ""
echo "ALL ARMS COMPLETE"
