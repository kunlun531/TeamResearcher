#!/bin/bash
cd /share/project/kunluo/Projects/TeamResearcher/evaluation

INPUT_FOLDER=../inference-team-researcher-v1110/outputs/Qwen3-14B_TeamResearcher/eval_data/browsecomp-plus-part.jsonl
DATASET=browsecomp_en_full

python evaluate_teamresearcher.py \
  --input_folder ${INPUT_FOLDER} \
  --dataset ${DATASET} 