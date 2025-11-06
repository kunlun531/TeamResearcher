#!/bin/bash
cd /share/project/kunluo/Projects/TeamResearcher/evaluation

INPUT_FOLDER=../inference-team-researcher/outputs/Qwen3-8B_sglang/eval_data/browsecomp-plus-part.jsonl
DATASET=browsecomp_en_full

python evaluate_local.py \
  --input_folder ${INPUT_FOLDER} \
  --dataset ${DATASET} 