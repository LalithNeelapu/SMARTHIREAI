from datetime import date, datetime

REFERENCE_DATE = date(2026, 6, 1)

REQUIRED_SKILL_TERMS = {
    "python", "nlp", "rag", "retrieval", "ranking", "embedding", "embeddings",
    "information retrieval", "milvus", "pinecone", "qdrant", "weaviate", "faiss",
    "llm", "fine-tuning llms", "langchain", "recommendation", "machine learning",
    "sentence transformers", "vector database", "pytorch", "transformers",
}


def skill_names(candidate):
    return {
        s.get("name", "").lower()
        for s in candidate.get("skills", [])
    }


def skill_score(candidate):
    names = skill_names(candidate)
    matched = 0

    for term in REQUIRED_SKILL_TERMS:
        if any(term in skill or skill in term for skill in names):
            matched += 1

    return matched / len(REQUIRED_SKILL_TERMS)


def experience_score(candidate):
    years = candidate["profile"].get("years_of_experience", 0)

    if 5 <= years <= 9:
        return 1.0
    if 4 <= years < 5:
        return 0.85
    if 9 < years <= 12:
        return 0.8
    if 12 < years <= 15:
        return 0.55
    return 0.35


def activity_score(last_active):
    if not last_active:
        return 0.2

    active_date = datetime.strptime(last_active, "%Y-%m-%d").date()
    days_inactive = (REFERENCE_DATE - active_date).days

    if days_inactive <= 30:
        return 1.0
    if days_inactive <= 90:
        return 0.85
    if days_inactive <= 180:
        return 0.55
    return 0.2


def assessment_score(signals):
    assessments = signals.get("skill_assessment_scores", {})
    if not assessments:
        return 0.0

    relevant = []
    for name, score in assessments.items():
        lowered = name.lower()
        if any(
            token in lowered
            for token in (
                "python", "ml", "machine", "nlp", "retrieval", "faiss",
                "pinecone", "embedding", "rank", "llm", "transformer",
            )
        ):
            relevant.append(score)

    values = relevant or list(assessments.values())
    return sum(values) / (100 * len(values))


def behavioral_score(candidate):
    signals = candidate.get("redrob_signals", {})

    components = {
        "assessment": (assessment_score(signals), 0.25),
        "response": (signals.get("recruiter_response_rate", 0.0), 0.20),
        "activity": (activity_score(signals.get("last_active_date")), 0.20),
        "interview": (signals.get("interview_completion_rate", 0.0), 0.15),
        "saved": (min(signals.get("saved_by_recruiters_30d", 0) / 10, 1.0), 0.10),
        "completeness": (signals.get("profile_completeness_score", 0) / 100, 0.10),
    }

    return sum(value * weight for value, weight in components.values())


def compute_final_score(
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
):
    if honeypot >= 0.4:
        return 0.0

    score = (
        0.17 * semantic
        + 0.07 * skill
        + 0.11 * experience
        + 0.14 * behavior
        + 0.11 * retrieval
        + 0.06 * vector_db
        + 0.08 * company
        + 0.06 * evaluation
        + 0.05 * open_source
        + 0.11 * title_alignment
        + 0.04 * ai_depth
        - 0.08 * consulting
        - 0.15 * honeypot
    )
    return max(score, 0.0)


def build_reasoning(
    candidate,
    semantic,
    skill,
    retrieval,
    behavior,
    title_alignment,
    company,
    evaluation,
    honeypot,
):
    profile = candidate.get("profile", {})
    title = profile.get("current_title", "Unknown")
    years = profile.get("years_of_experience", 0)
    signals = candidate.get("redrob_signals", {})
    assessment = assessment_score(signals)

    return (
        f"{title} with {years:.1f} years experience. "
        f"Title fit ({title_alignment:.2f}), semantic ({semantic:.2f}), "
        f"skills ({skill:.2f}), retrieval ({retrieval:.2f}), "
        f"evaluation ({evaluation:.2f}), product co ({company:.2f}), "
        f"behavior ({behavior:.2f}), assessments ({assessment:.2f}), "
        f"trap penalty ({honeypot:.2f})."
    )
