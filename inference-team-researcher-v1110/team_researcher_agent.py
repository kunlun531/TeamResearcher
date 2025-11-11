# team_researcher_agent.py

import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List

from react_agent import MultiTurnReactAgent
from prompt import CENTRAL_PLANNING_PROMPT, CENTRAL_SYNTHESIS_PROMPT, SUB_RESEARCHER_PROMPT

class TeamResearcherAgent:
    def __init__(self, llm_config: Dict, central_port: int, sub_researcher_ports: List[int]):
        self.llm_config = llm_config
        self.central_port = central_port
        self.sub_researcher_ports = sub_researcher_ports
        
        # The central agent caller is essentially a MultiTurnReactAgent instance
        # that we use only for its `call_server` method, pointed at the central port.
        self.central_agent_caller = MultiTurnReactAgent(llm=llm_config)

    def _run(self, data: Dict, model: str, **kwargs) -> Dict:
        original_question = data['item']['question']
        start_time = time.time()

        # === 1. 规划与拆解 (Planning & Decomposition) ===
        print(f"\n--- [Q: {original_question[:50]}...] Phase 1: Planning & Decomposition ---")
        sub_tasks = self._plan_and_decompose(original_question, model)
        if not sub_tasks:
            print("Failed to decompose task. Aborting.")
            return {"error": "Decomposition failed", "question": original_question}
        print(f"Decomposed into {len(sub_tasks)} sub-tasks:")
        for task in sub_tasks:
            print(f"  - Task {task['task_id']}: {task['question']}")

        # === 2. 并行研究 (Parallel Research) ===
        print(f"--- [Q: {original_question[:50]}...] Phase 2: Parallel Research ---")
        sub_researcher_reports = self._execute_parallel_research(sub_tasks, model)
        print("All sub-researchers have completed their tasks.")

        # === 3. 聚合与综合 (Aggregation & Synthesis) ===
        print(f"--- [Q: {original_question[:50]}...] Phase 3: Aggregation & Synthesis ---")
        final_answer = self._aggregate_and_synthesize(original_question, sub_researcher_reports, model)
        print(f"--- [Q: {original_question[:50]}...] Final Answer Generated ---")

        end_time = time.time()
        
        result = {
            "question": original_question,
            "answer": data['item'].get('answer', ''),
            "prediction": final_answer,
            "sub_tasks": sub_tasks,
            "sub_researcher_reports": sub_researcher_reports,
            "total_duration": end_time - start_time,
            "termination": "success"
        }
        return result

    def _plan_and_decompose(self, question: str, model: str) -> List[Dict]:
        self.central_agent_caller.model = model
        prompt = CENTRAL_PLANNING_PROMPT.format(question=question)
        messages = [{"role": "system", "content": "You are a master research planner."}, {"role": "user", "content": prompt}]
        
        response_str = self.central_agent_caller.call_server(messages, self.central_port)
        
        try:
            json_block = response_str.split("```json")[1].split("```")[0]
            tasks = json.loads(json_block)
            return tasks["sub_tasks"]
        except (IndexError, json.JSONDecodeError, KeyError) as e:
            print(f"Error parsing decomposition response: {e}\nResponse: {response_str}")
            return []

    def _execute_parallel_research(self, sub_tasks: List[Dict], model: str) -> List[Dict]:
        reports = []
        with ThreadPoolExecutor(max_workers=len(self.sub_researcher_ports)) as executor:
            future_to_task = {}
            for i, task in enumerate(sub_tasks):
                sub_agent = MultiTurnReactAgent(llm=self.llm_config)
                port = self.sub_researcher_ports[i % len(self.sub_researcher_ports)]
                
                sub_task_data = {
                    'item': {'question': task['question']},
                    'planning_port': port,
                }
                
                future = executor.submit(sub_agent._run_sub_task, sub_task_data, model)
                future_to_task[future] = task

            for future in as_completed(future_to_task):
                task = future_to_task[future]
                try:
                    result = future.result()
                    print(f"Sub-task '{task['question'][:50]}...' completed.")
                    reports.append({
                        "task_id": task["task_id"],
                        "task_question": task["question"],
                        "report": result["prediction"],
                        "messages": result["messages"] # For debugging
                    })
                except Exception as exc:
                    print(f"Sub-task '{task['question']}' generated an exception: {exc}")
                    reports.append({
                        "task_id": task["task_id"],
                        "task_question": task["question"],
                        "report": f"Error: Failed to complete research due to {exc}"
                    })
        # Sort reports by task_id for consistent synthesis
        reports.sort(key=lambda r: r['task_id'])
        return reports

    def _aggregate_and_synthesize(self, original_question: str, reports: List[Dict], model: str) -> str:
        self.central_agent_caller.model = model
        reports_str = "\n\n".join(
            [f"### Report for Sub-Task {r['task_id']}: {r['task_question']}\n{r['report']}" for r in reports]
        )
        prompt = CENTRAL_SYNTHESIS_PROMPT.format(original_question=original_question, reports_str=reports_str)
        messages = [{"role": "system", "content": "You are a lead researcher synthesizing team findings."}, {"role": "user", "content": prompt}]
        
        final_response = self.central_agent_caller.call_server(messages, self.central_port)

        if '<answer>' in final_response and '</answer>' in final_response:
            return final_response.split('<answer>')[1].split('</answer>')[0].strip()
        return final_response