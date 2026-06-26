import pandas as pd


REQUIRED_COLUMNS = ["candidate_id", "rank", "score", "reasoning"]


def finalize_submission(df):
    ranked = df.sort_values(
        by=["score", "candidate_id"],
        ascending=[False, True],
    ).head(100).copy()

    ranked["rank"] = range(1, len(ranked) + 1)

    adjusted_scores = []
    previous_score = None
    for raw_score in ranked["score"].astype(float):
        score = round(raw_score, 4)
        if previous_score is not None and score >= previous_score:
            score = round(previous_score - 0.0001, 4)
        adjusted_scores.append(score)
        previous_score = score

    ranked["score"] = adjusted_scores
    return ranked[REQUIRED_COLUMNS]


def save_submission(df, output_path):
    submission = finalize_submission(df)
    submission.to_csv(output_path, index=False)
    return submission
