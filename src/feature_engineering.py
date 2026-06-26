import re

PRODUCT_COMPANIES = {
    "amazon", "google", "microsoft", "meta", "facebook",
    "flipkart", "swiggy", "zomato", "uber", "ola", "paytm", "razorpay",
    "netflix", "apple", "spotify", "linkedin", "twitter", "x corp",
}

CONSULTING_COMPANIES = {
    "tcs", "infosys", "wipro", "cognizant", "capgemini",
    "accenture", "hcl", "tech mahindra", "mindtree", "ltimindtree",
    "lti", "mphasis", "persistent", "cyient",
}

VECTOR_DB = {
    "milvus", "pinecone", "qdrant", "weaviate", "faiss",
    "elasticsearch", "opensearch", "chroma", "vectordb", "vector database",
}

RETRIEVAL_TERMS = {
    "retrieval", "ranking", "recommendation", "recommendation system",
    "recommendation systems", "search", "relevance", "embedding",
    "embeddings", "semantic search", "information retrieval",
    "vector search", "hybrid retrieval", "hybrid search", "dense retrieval",
    "bm25", "learning to rank", "rerank", "reranking", "two-tower",
    "embedding-based retrieval", "candidate generation", "personalization",
    "feed ranking", "search engine", "matching system",
}

EVALUATION_PATTERNS = [
    r"\bndcg\b",
    r"\bmrr\b",
    r"\bmean average precision\b",
    r"\bmap@\d",
    r"\bab testing\b",
    r"\ba/b testing\b",
    r"\boffline evaluation\b",
    r"\bonline evaluation\b",
    r"\branking metrics\b",
    r"\boffline-to-online\b",
    r"\ba/b test\b",
]

TITLE_PREFIXES = (
    "senior ", "staff ", "lead ", "principal ", "sr. ", "junior ", "associate ",
)

AI_TITLES = {
    "ai engineer", "machine learning engineer", "ml engineer",
    "nlp engineer", "search engineer", "recommendation systems engineer",
    "applied scientist", "data scientist", "applied ml engineer",
    "applied machine learning engineer", "research engineer",
    "staff machine learning engineer", "lead ai engineer",
}

TECH_TITLE_KEYWORDS = (
    "engineer", "scientist", "researcher", "architect",
)


def _company_matches(company, aliases):
    company_lower = company.lower()
    return any(alias in company_lower for alias in aliases)


def normalize_title(title):
    normalized = title.lower().strip()
    for prefix in TITLE_PREFIXES:
        if normalized.startswith(prefix):
            normalized = normalized[len(prefix):]
    return normalized


def build_corpus_text(candidate):
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
            job.get("description", ""),
            job.get("company", ""),
        ])

    return " ".join(parts).lower()


def product_company_score(candidate):
    score = 0

    for job in candidate.get("career_history", []):
        if _company_matches(job.get("company", ""), PRODUCT_COMPANIES):
            score += 1

    return min(score / 3, 1.0)


def consulting_penalty(candidate):
    jobs = candidate.get("career_history", [])

    if not jobs:
        return 0.0

    consulting_count = sum(
        1 for job in jobs
        if _company_matches(job.get("company", ""), CONSULTING_COMPANIES)
    )

    return consulting_count / len(jobs)


def retrieval_score(candidate, corpus_text=None):
    text = corpus_text if corpus_text is not None else build_corpus_text(candidate)
    matched = sum(1 for term in RETRIEVAL_TERMS if term in text)
    return matched / len(RETRIEVAL_TERMS)


def vector_db_score(candidate):
    skill_names = [
        s.get("name", "").lower()
        for s in candidate.get("skills", [])
    ]

    score = 0
    for db in VECTOR_DB:
        if any(db in skill for skill in skill_names):
            score += 1

    return min(score / 3, 1.0)


def evaluation_score(candidate, corpus_text=None):
    text = corpus_text if corpus_text is not None else build_corpus_text(candidate)
    matched = sum(
        1 for pattern in EVALUATION_PATTERNS
        if re.search(pattern, text)
    )
    return matched / len(EVALUATION_PATTERNS)


def open_source_score(candidate):
    signals = candidate.get("redrob_signals", {})
    github_score = signals.get("github_activity_score", 0)

    if github_score < 0:
        return 0.0

    return min(github_score / 100, 1.0)


def title_alignment_score(candidate):
    title = candidate["profile"].get("current_title", "")
    normalized = normalize_title(title)

    if normalized in AI_TITLES:
        return 1.0

    if any(key in normalized for key in (
        "ai engineer", "ml engineer", "machine learning engineer",
        "nlp engineer", "search engineer", "recommendation",
        "applied scientist", "applied ml",
    )):
        return 1.0

    if any(keyword in normalized for keyword in TECH_TITLE_KEYWORDS):
        return 0.5

    return 0.0


def ai_skill_depth_score(candidate):
    skill_names = {
        s.get("name", "").lower()
        for s in candidate.get("skills", [])
    }

    ai_terms = VECTOR_DB | RETRIEVAL_TERMS | {
        "python", "nlp", "machine learning", "deep learning", "llm",
        "fine-tuning llms", "rag", "langchain", "lora", "qlora", "peft",
        "pytorch", "tensorflow", "transformers", "sentence transformers",
    }

    matched = sum(
        1 for term in ai_terms
        if any(term in skill for skill in skill_names)
    )

    return min(matched / 12, 1.0)
