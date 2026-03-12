import pandas as pd
import time
import os
from graph import agent_app

# --- 1. The Golden Dataset Metadata ---
# We store the expected answers and the filename to read from
EVAL_METADATA = [
    {"filename": "01_perfect_baseline.txt", "expected_actions": 2, "expected_decisions": 1},
    {"filename": "02_the_brainstorm.txt", "expected_actions": 0, "expected_decisions": 0},
    {"filename": "03_the_pivot.txt", "expected_actions": 0, "expected_decisions": 1},
    {"filename": "04_insufficient_context.txt", "expected_actions": 0, "expected_decisions": 0},
    {"filename": "05_task_salad.txt", "expected_actions": 4, "expected_decisions": 0},
    {"filename": "06_status_update_past.txt", "expected_actions": 0, "expected_decisions": 0},
    {"filename": "07_vague_directive.txt", "expected_actions": 1, "expected_decisions": 0},
    {"filename": "08_missing_owner.txt", "expected_actions": 1, "expected_decisions": 0},
    {"filename": "09_formatting_nightmare.txt", "expected_actions": 1, "expected_decisions": 1},
    {"filename": "10_dense_technical.txt", "expected_actions": 1, "expected_decisions": 1}
]

# --- 2. The Evaluation Engine ---
def run_evaluation():
    print(f"🚀 Starting evaluation of {len(EVAL_METADATA)} real transcript files...")
    results = []
    
    data_dir = "eval_data"

    for idx, test in enumerate(EVAL_METADATA):
        file_path = os.path.join(data_dir, test["filename"])
        print(f"Testing [{idx+1}/{len(EVAL_METADATA)}]: {test['filename']}")
        
        # Read the actual content from the .txt file
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                input_text = f.read()
        except FileNotFoundError:
            print(f"❌ Error: Could not find {file_path}. Did you run setup_test_data.py?")
            continue
        
        start_time = time.time()
        
        try:
            # Send the file content to your LangGraph backend
            output = agent_app.invoke({"original_text": input_text})
            latency = round(time.time() - start_time, 2)
            
            actual_actions = output.get("action_items", [])
            actual_decisions = output.get("key_decisions", [])
            
            actions_match = len(actual_actions) == test["expected_actions"]
            decisions_match = len(actual_decisions) == test["expected_decisions"]
            
            if actions_match and decisions_match:
                grade = "PASS"
            elif actions_match or decisions_match:
                grade = "PARTIAL"
            else:
                grade = "FAIL"
                
            results.append({
                "Test File": test["filename"],
                "Expected Actions": test["expected_actions"],
                "Actual Actions": len(actual_actions),
                "Expected Decisions": test["expected_decisions"],
                "Actual Decisions": len(actual_decisions),
                "Grade": grade,
                "Latency (s)": latency,
                "LLM Summary": output.get("summary", ""),
                "LLM Extracted Actions": "\n".join(actual_actions) if actual_actions else "None",
                "LLM Extracted Decisions": "\n".join(actual_decisions) if actual_decisions else "None",
                "Error": "None"
            })
            
        except Exception as e:
            latency = round(time.time() - start_time, 2)
            results.append({
                "Test File": test["filename"],
                "Expected Actions": test["expected_actions"],
                "Actual Actions": 0,
                "Expected Decisions": test["expected_decisions"],
                "Actual Decisions": 0,
                "Grade": "CRASH",
                "Latency (s)": latency,
                "LLM Summary": "N/A",
                "LLM Extracted Actions": "N/A",
                "LLM Extracted Decisions": "N/A",
                "Error": str(e)
            })

    # --- 3. Export to Excel/CSV ---
    print("\n✅ Evaluation complete! Generating report...")
    df = pd.DataFrame(results)
    
    csv_filename = "eval_results_report.csv"
    df.to_csv(csv_filename, index=False)
    print(f"📊 Report successfully saved to: {csv_filename}")
    
    pass_count = len(df[df['Grade'] == 'PASS'])
    print(f"🎯 Final Score: {pass_count}/{len(EVAL_METADATA)} Passed")

if __name__ == "__main__":
    run_evaluation()