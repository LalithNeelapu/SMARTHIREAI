from feature_engineering import normalize_title, build_corpus_text

NON_TECH_TITLES = {
    "marketing manager", "operations manager", "hr manager", "graphic designer",
    "business analyst", "content writer", "sales manager", "account manager",
    "mechanical engineer", "civil engineer", "electrical engineer",
    "project manager", "product manager", "consultant", "recruiter",
}

AI_SKILL_MARKERS = {
    "nlp", "llm", "rag", "milvus", "pinecone", "langchain", "fine-tuning",
    "embedding", "retrieval", "deep learning", "machine learning",
}

HONEYPOT_SUMMARY_PHRASES = (
    "lately i've been curious about how ai tools could augment my work",
    "experimented with chatgpt and a few other tools for productivity",
    "open to roles where i can apply my domain expertise alongside emerging ai",
    "my professional background is in marketing manager",
    "i'm a marketing manager with substantial experience",
)


def _ai_skill_hits(candidate):
    skill_names = {
        s.get("name", "").lower()
        for s in candidate.get("skills", [])
    }
    return sum(
        1 for marker in AI_SKILL_MARKERS
        if any(marker in skill for skill in skill_names)
    )


def honeypot_penalty(candidate):
    profile = candidate.get("profile", {})
    years = profile.get("years_of_experience", 0)
    skills = candidate.get("skills", [])
    title = normalize_title(profile.get("current_title", ""))
    summary = profile.get("summary", "").lower()

    penalty = 0.0

    if len(skills) > 50:
        penalty += 0.35

    if len(skills) > 30:
        penalty += 0.15

    for skill in skills:
        duration_years = skill.get("duration_months", 0) / 12
        if duration_years > years + 2:
            penalty += 0.08

    total_endorsements = sum(s.get("endorsements", 0) for s in skills)
    if total_endorsements > 500 and years < 3:
        penalty += 0.25

    ai_hits = _ai_skill_hits(candidate)
    if any(non_tech in title for non_tech in NON_TECH_TITLES) and ai_hits >= 4:
        penalty += 0.55

    if any(phrase in summary for phrase in HONEYPOT_SUMMARY_PHRASES):
        penalty += 0.45

    if title_alignment_trap(title, ai_hits):
        penalty += 0.35

    if keyword_stuffer_trap(candidate):
        penalty += 0.25

    return min(penalty, 1.0)


def title_alignment_trap(title, ai_hits):
    if ai_hits < 6:
        return False

    tech_title = any(
        keyword in title
        for keyword in ("engineer", "scientist", "researcher", "architect")
    )
    return not tech_title


def keyword_stuffer_trap(candidate):
    skills = candidate.get("skills", [])
    if len(skills) < 20:
        return False

    expert_count = sum(
        1 for skill in skills
        if skill.get("proficiency") == "expert"
        and skill.get("duration_months", 0) < 12
    )

    return expert_count >= 8
