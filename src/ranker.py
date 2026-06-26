import json
from pathlib import Path

import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from candidate_filter import filter_candidates, load_candidates
from feature_engineering import (
    ai_skill_depth_score,
    build_corpus_text,
    consulting_penalty,
    evaluation_score,
    open_source_score,
    product_company_score,
    retrieval_score,
    title_alignment_score,
    vector_db_score,
)
from honeypot_detector import honeypot_penalty
from jd_parser import load_jd_for_embedding
from scoring import (
    behavioral_score,
    build_reasoning,
    compute_final_score,
    experience_score,
    skill_score,
)
from submission_generator import save_submission

EMBED_MODEL = "BAAI/bge-small-en-v1.5"
FAST_EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

# Module-level model cache to avoid reloading on every run
_model_cache = {}


def get_model(model_name):
    """Load model once and cache it for subsequent calls."""
    if model_name not in _model_cache:
        print(f"Loading embedding model ({model_name})...")
        _model_cache[model_name] = SentenceTransformer(model_name)
    else:
        print(f"Using cached embedding model ({model_name})")
    return _model_cache[model_name]


def build_candidate_text(candidate):
    profile = candidate.get("profile", {})
    parts = [
        profile.get("headline", ""),
        profile.get("summary", ""),
        profile.get("current_title", ""),
    ]

    for skill in candidate.get("skills", []):
        parts.append(skill.get("name", ""))

    for job in candidate.get("career_history", []):
        parts.extend([
            job.get("title", ""),
            job.get("company", ""),
            job.get("description", ""),
        ])

    return " ".join(parts)


def run_ranking(
    candidates_path,
    output_path,
    filtered_cache=None,
    show_progress=True,
    model_name=FAST_EMBED_MODEL,
):
    candidates_path = Path(candidates_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("Loading candidates...")
    all_candidates = load_candidates(candidates_path)
    print(f"Loaded {len(all_candidates)} candidates")

    print("Filtering...")
    candidates = filter_candidates(all_candidates)
    print(f"Filtered pool: {len(candidates)}")

    if filtered_cache:
        filtered_cache = Path(filtered_cache)
        filtered_cache.parent.mkdir(parents=True, exist_ok=True)
        with open(filtered_cache, "w", encoding="utf-8") as handle:
            for candidate in candidates:
                handle.write(json.dumps(candidate) + "\n")

    # Use cached model instead of loading fresh each time
    model = get_model(model_name)
    job_text = load_jd_for_embedding()
    candidate_texts = [build_candidate_text(candidate) for candidate in candidates]

    print("Generating embeddings...")
    if model_name == EMBED_MODEL:
        job_input = QUERY_PREFIX + job_text
    else:
        job_input = job_text

    job_embedding = model.encode(
        job_input,
        normalize_embeddings=True,
    )
    candidate_embeddings = model.encode(
        candidate_texts,
        batch_size=256,
        show_progress_bar=show_progress,
        normalize_embeddings=True,
    )

    # Vectorized cosine similarity — single call instead of per-candidate loop
    print("Computing similarities...")
    all_similarities = cosine_similarity(
        job_embedding.reshape(1, -1),
        candidate_embeddings,
    )[0]

    print("Scoring...")
    results = []

    for idx, candidate in enumerate(candidates):
        semantic = float(all_similarities[idx])

        # Pre-compute corpus text once (used by retrieval_score & evaluation_score)
        corpus_text = build_corpus_text(candidate)

        skill = skill_score(candidate)
        experience = experience_score(candidate)
        behavior = behavioral_score(candidate)
        retrieval = retrieval_score(candidate, corpus_text=corpus_text)
        vector_db = vector_db_score(candidate)
        company = product_company_score(candidate)
        consulting = consulting_penalty(candidate)
        evaluation = evaluation_score(candidate, corpus_text=corpus_text)
        open_source = open_source_score(candidate)
        title_alignment = title_alignment_score(candidate)
        ai_depth = ai_skill_depth_score(candidate)
        honeypot = honeypot_penalty(candidate)

        final_score = compute_final_score(
            semantic,
            skill,
            experience,
            behavior,
            retrieval,
            vector_db,
            company,
            evaluation,
            open_source,
            title_alignment,
            ai_depth,
            consulting,
            honeypot,
        )

        results.append({
            "candidate_id": candidate["candidate_id"],
            "score": final_score,
            "reasoning": build_reasoning(
                candidate,
                semantic,
                skill,
                retrieval,
                behavior,
                title_alignment,
                company,
                evaluation,
                honeypot,
            ),
        })

    submission = save_submission(pd.DataFrame(results), output_path)

    print("\nTOP 20\n")
    print(submission[["candidate_id", "rank", "score", "reasoning"]].head(20))
    print(f"\nSaved: {output_path}")

    return submission
