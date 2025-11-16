#!/bin/bash
cd /share/project/kunluo/Projects/TeamResearcher/evaluation

INPUT_FOLDER=../inference-team-researcher-v1110/outputs/Qwen3-30B-A3B-Thinking-2507_TeamResearcher/eval_data/browsecomp-plus-part.jsonl
DATASET=browsecomp_en_full

python evaluate_local.py \
  --input_folder ${INPUT_FOLDER} \
  --dataset ${DATASET} 