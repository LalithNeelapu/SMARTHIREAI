import sys
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from honeypot_detector import honeypot_penalty
from candidate_filter import load_candidates

def main():
    submission_path = ROOT / "output" / "final_submission.csv"
    candidates_path = (
        ROOT / "data" / "[PUB] India_runs_data_and_ai_challenge"
        / "[PUB] India_runs_data_and_ai_challenge"
        / "India_runs_data_and_ai_challenge" / "candidates.jsonl"
    )
    
    if not submission_path.exists():
        print(f"Error: {submission_path} does not exist.")
        return

    print("Loading candidates...")
    candidates = load_candidates(candidates_path)
    candidate_lookup = {c["candidate_id"]: c for c in candidates}
    
    submission = pd.read_csv(submission_path)
    count = 0
    for cid in submission["candidate_id"]:
        candidate = candidate_lookup.get(cid)
        if candidate:
            penalty = honeypot_penalty(candidate)
            if penalty >= 0.4:
                count += 1
                print(f"Honeypot candidate in Top 100: {cid} (penalty={penalty:.2f})")
        else:
            print(f"Warning: Candidate ID {cid} from submission not found in candidate pool.")
            
    print(f"Honeypots in Top100: {count}")

if __name__ == "__main__":
    main()
