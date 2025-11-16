# team_researcher_agent.py

import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Tuple

from react_agent import MultiTurnReactAgent
from prompt import CENTRAL_PLANNING_PROMPT, CENTRAL_SYNTHESIS_PROMPT

class TeamResearcherAgent:
    def __init__(self, llm_config: Dict, central_port: int, sub_researcher_ports: List[int]):
        self.llm_config = llm_config
        self.central_port = central_port
        self.sub_researcher_ports = sub_researcher_ports
        
        self.central_agent_caller = MultiTurnReactAgent(llm=llm_config)

    def _run(self, data: Dict, model: str, **kwargs) -> Dict:
        original_question = data['item']['question']
        start_time = time.time()
        total_token_consumption = 0
        unified_messages = []

        # === 1. 规划与拆解 (Planning & Decomposition) ===
        print(f"\n--- [Q: {original_question[:50]}...] Phase 1: Planning & Decomposition ---")
        sub_tasks, planning_messages, planning_tokens = self._plan_and_decompose(original_question, model)
        total_token_consumption += planning_tokens
        unified_messages.extend(planning_messages)

        if not sub_tasks:
            print("Failed to decompose task. Aborting.")
            return {"error": "Decomposition failed", "question": original_question, "messages": unified_messages}
        print(f"Decomposed into {len(sub_tasks)} sub-tasks:")
        for task in sub_tasks:
            print(f"  - Task {task['task_id']}: {task['question']}")

        # === 2. 并行研究 (Parallel Research) ===
        print(f"--- [Q: {original_question[:50]}...] Phase 2: Parallel Research ---")
        sub_researcher_reports, research_tokens = self._execute_parallel_research(sub_tasks, model)
        total_token_consumption += research_tokens
        # MODIFICATION: Add sub-researcher messages to the unified list
        for report in sub_researcher_reports:
            unified_messages.extend(report.get("messages", []))
        print("All sub-researchers have completed their tasks.")

        # === 3. 聚合与综合 (Aggregation & Synthesis) ===
        print(f"--- [Q: {original_question[:50]}...] Phase 3: Aggregation & Synthesis ---")
        final_answer, synthesis_messages, synthesis_tokens = self._aggregate_and_synthesize(original_question, sub_researcher_reports, model)
        total_token_consumption += synthesis_tokens
        unified_messages.extend(synthesis_messages)
        print(f"--- [Q: {original_question[:50]}...] Final Answer Generated ---")

        end_time = time.time()
        
        # MODIFICATION: Add 'messages' and 'total_token_consumption' to the final result
        result = {
            "question": original_question,
            "answer": data['item'].get('answer', ''),
            "prediction": final_answer,
            "messages": unified_messages, # FIX for KeyError: 'messages'
            "total_token_consumption": total_token_consumption, # NEW metric
            "sub_tasks": sub_tasks,
            "sub_researcher_reports": sub_researcher_reports,
            "total_duration": end_time - start_time,
            "termination": "success"
        }
        return result

    # MODIFICATION: Return messages and token usage
    def _plan_and_decompose(self, question: str, model: str) -> Tuple[List[Dict], List[Dict], int]:
        self.central_agent_caller.model = model
        prompt = CENTRAL_PLANNING_PROMPT.format(question=question)
        messages = [{"role": "system", "content": "You are a master research planner."}, {"role": "user", "content": prompt}]
        
        response_str, usage = self.central_agent_caller.call_server(messages, self.central_port)
        messages.append({"role": "assistant", "content": response_str})
        
        try:
            json_block = response_str.split("```json")[1].split("```")[0]
            tasks = json.loads(json_block)
            return tasks["sub_tasks"], messages, usage.get('total_tokens', 0)
        except (IndexError, json.JSONDecodeError, KeyError) as e:
            print(f"Error parsing decomposition response: {e}\nResponse: {response_str}")
            return [], messages, usage.get('total_tokens', 0)

    # MODIFICATION: Return total token usage from all sub-tasks
    def _execute_parallel_research(self, sub_tasks: List[Dict], model: str) -> Tuple[List[Dict], int]:
        reports = []
        total_sub_task_tokens = 0
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
                    sub_task_tokens = result.get("total_tokens", 0)
                    total_sub_task_tokens += sub_task_tokens
                    print(f"Sub-task '{task['question'][:50]}...' completed. Tokens used: {sub_task_tokens}")
                    
                    reports.append({
                        "task_id": task["task_id"],
                        "task_question": task["question"],
                        "report": result["prediction"],
                        "messages": result["messages"],
                        "tokens": sub_task_tokens
                    })
                except Exception as exc:
                    print(f"Sub-task '{task['question']}' generated an exception: {exc}")
                    reports.append({
                        "task_id": task["task_id"],
                        "task_question": task["question"],
                        "report": f"Error: Failed to complete research due to {exc}",
                        "messages": [],
                        "tokens": 0
                    })
        
        reports.sort(key=lambda r: r['task_id'])
        return reports, total_sub_task_tokens

    # MODIFICATION: Return messages and token usage
    def _aggregate_and_synthesize(self, original_question: str, reports: List[Dict], model: str) -> Tuple[str, List[Dict], int]:
        self.central_agent_caller.model = model
        reports_str = "\n\n".join(
            [f"### Report for Sub-Task {r['task_id']}: {r['task_question']}\n{r['report']}" for r in reports]
        )
        prompt = CENTRAL_SYNTHESIS_PROMPT.format(original_question=original_question, reports_str=reports_str)
        messages = [{"role": "system", "content": "You are a lead researcher synthesizing team findings."}, {"role": "user", "content": prompt}]
        
        final_response, usage = self.central_agent_caller.call_server(messages, self.central_port)
        messages.append({"role": "assistant", "content": final_response})

        final_answer = final_response
        if '<answer>' in final_response and '</answer>' in final_response:
            final_answer = final_response.split('<answer>')[1].split('</answer>')[0].strip()
        
        return final_answer, messages, usage.get('total_tokens', 0)