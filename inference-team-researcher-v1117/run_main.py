# run_main.py

import argparse
import json
import os
import pdb
import math
import threading
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

from team_researcher_agent import TeamResearcherAgent

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="")
    parser.add_argument("--output", type=str, default="")
    parser.add_argument("--dataset", type=str, default="gaia")
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--presence_penalty", type=float, default=1.1)
    parser.add_argument("--max_workers", type=int, default=8) # This is for question-level parallelism
    parser.add_argument("--roll_out_count", type=int, default=1) # TeamResearcher is one integrated process, so rollout is 1 unless you want to run the whole team multiple times
    parser.add_argument("--total_splits", type=int, default=1)
    parser.add_argument("--worker_split", type=int, default=1)
    args = parser.parse_args()

    # --- Setup ---
    model_name = os.path.basename(args.model.rstrip('/'))
    output_dir = os.path.join(args.output, f"{model_name}_TeamResearcher", args.dataset)
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Model: {model_name}")
    print(f"Dataset: {args.dataset}")
    print(f"Output Directory: {output_dir}")
    print(f"Data Splitting: {args.worker_split}/{args.total_splits}")

    # --- Load Data ---
    try:
        with open(args.dataset, "r", encoding="utf-8") as f:
            items = [json.loads(line) for line in f] if args.dataset.endswith(".jsonl") else json.load(f)
    except Exception as e:
        print(f"Error reading or parsing dataset file {args.dataset}: {e}")
        exit(1)

    # --- Data Splitting ---
    total_items = len(items)
    items_per_split = math.ceil(total_items / args.total_splits)
    start_idx = (args.worker_split - 1) * items_per_split
    end_idx = min(args.worker_split * items_per_split, total_items)
    items_to_process = items[start_idx:end_idx]
    print(f"Processing items {start_idx} to {end_idx-1} ({len(items_to_process)} items)")

    # --- Resume Logic ---
    output_file = os.path.join(output_dir, f"split_{args.worker_split}_of_{args.total_splits}.jsonl")
    processed_questions = set()
    if os.path.exists(output_file):
        with open(output_file, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    data = json.loads(line)
                    if "question" in data and "error" not in data:
                        processed_questions.add(data["question"].strip())
                except json.JSONDecodeError:
                    continue
    print(f"Found {len(processed_questions)} already processed questions. Skipping them.")

    tasks_to_run = []
    for item in items_to_process:
        question = item.get("question", "").strip()
        if not question: continue
        if question not in processed_questions:
            tasks_to_run.append({"item": item})
    
    if not tasks_to_run:
        print("All questions in this split have been processed. Exiting.")
        return

    # --- Agent and Port Configuration ---
    # Central Agent on the first port, Sub-researchers on the rest.
    central_port = 6001
    sub_researcher_ports = [6002, 6003, 6004, 6005, 6006, 6007, 6008]
    print(f"Central Agent Port: {central_port}, Sub-researcher Ports: {sub_researcher_ports}")

    llm_cfg = {
        'model': args.model,
        'generate_cfg': {
            'max_input_tokens': 320000,
            'max_retries': 10,
            'temperature': args.temperature,
            'top_p': args.top_p,
            'presence_penalty': args.presence_penalty
        }
    }
    
    team_agent = TeamResearcherAgent(
        llm_config=llm_cfg,
        central_port=central_port,
        sub_researcher_ports=sub_researcher_ports
    )
    
    write_lock = threading.Lock()

    # --- Execute Tasks ---
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        future_to_task = {
            executor.submit(team_agent._run, task, args.model): task 
            for task in tasks_to_run
        }

        for future in tqdm(as_completed(future_to_task), total=len(tasks_to_run), desc="Processing Queries with TeamResearcher"):
            task_info = future_to_task[future]
            try:
                result = future.result()
            except Exception as exc:
                question = task_info["item"].get("question", "")
                print(f'Task for question "{question}" generated an exception: {exc}')
                result = {
                    "question": question,
                    "answer": task_info["item"].get("answer", ""),
                    "error": f"Future resolution failed: {exc}",
                    "prediction": "[Failed]",
                }
            
            with write_lock:
                with open(output_file, "a", encoding="utf-8") as f:
                    f.write(json.dumps(result, ensure_ascii=False) + "\n")

    print("\nAll tasks completed!")

if __name__ == "__main__":
    main()