import json
from pathlib import Path

# Import components from other modules
from feature_engineering import (
    title_alignment_score,
    retrieval_score,
    ai_skill_depth_score,
)
from scoring import (
    skill_score,
    experience_score,
    behavioral_score,
)
from honeypot_detector import honeypot_penalty

def heuristic_filter_score(candidate):
    # If experience is less than 4 years, filter out
    years = candidate["profile"].get("years_of_experience", 0)
    if years < 4:
        return -1.0
        
    # If high-probability honeypot (penalty >= 0.4), filter out
    if honeypot_penalty(candidate) >= 0.4:
        return -1.0
        
    # Calculate weighted heuristic alignment score
    score = (
        0.25 * title_alignment_score(candidate)
        + 0.20 * retrieval_score(candidate)
        + 0.15 * skill_score(candidate)
        + 0.15 * experience_score(candidate)
        + 0.15 * behavioral_score(candidate)
        + 0.10 * ai_skill_depth_score(candidate)
    )
    return score

def filter_candidates(candidates, top_n=2500):
    scored_candidates = []
    for candidate in candidates:
        score = heuristic_filter_score(candidate)
        if score >= 0.0:
            scored_candidates.append((score, candidate))
            
    # Sort candidates by heuristic score descending
    scored_candidates.sort(key=lambda x: x[0], reverse=True)
    
    # Return top N candidates
    return [candidate for score, candidate in scored_candidates[:top_n]]

def load_candidates(path):
    candidates = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            candidates.append(json.loads(line))
    return candidates

def write_filtered_candidates(candidates, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as handle:
        for candidate in candidates:
            handle.write(json.dumps(candidate) + "\n")

    return len(candidates)
