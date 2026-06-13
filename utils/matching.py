"""
Job Matching Utility — TF-IDF keyword scoring against candidate resume
"""
import re
import math
from collections import Counter


def _tokenize(text: str) -> list[str]:
    """Lowercase, strip punctuation, split into tokens."""
    text = re.sub(r"[^\w\s+#]", " ", text.lower())
    return [t for t in text.split() if len(t) > 2]


def _compute_tfidf_score(candidate_tokens: list[str], job_tokens: list[str]) -> float:
    """Simple TF-IDF overlap score between candidate profile and job description."""
    if not candidate_tokens or not job_tokens:
        return 0.0

    # Term frequency in candidate
    cand_freq = Counter(candidate_tokens)
    job_freq = Counter(job_tokens)

    # Shared vocabulary
    shared = set(cand_freq.keys()) & set(job_freq.keys())
    if not shared:
        return 0.0

    # Weighted overlap: higher for rarer terms (simple IDF approximation)
    total_terms = len(job_freq)
    score = 0.0
    for term in shared:
        tf_cand = cand_freq[term] / len(candidate_tokens)
        tf_job = job_freq[term] / len(job_tokens)
        # Boost important tech keywords
        idf_boost = 2.0 if _is_tech_keyword(term) else 1.0
        score += (tf_cand * tf_job * idf_boost)

    # Normalize to 0-100
    max_possible = sum(
        (job_freq[t] / len(job_tokens)) ** 2 for t in job_freq
    )
    if max_possible == 0:
        return 0.0

    normalized = min((score / math.sqrt(max_possible)) * 100, 100)
    return round(normalized, 1)


# Common tech/skill keywords that should have extra weight
TECH_KEYWORDS = {
    "python", "javascript", "typescript", "react", "nextjs", "node", "nodejs",
    "fastapi", "django", "flask", "sql", "postgresql", "mysql", "mongodb",
    "redis", "docker", "kubernetes", "aws", "gcp", "azure", "git", "graphql",
    "rest", "api", "machine", "learning", "tensorflow", "pytorch", "nlp",
    "llm", "openai", "langchain", "java", "kotlin", "swift", "flutter",
    "golang", "rust", "c++", "devops", "cicd", "terraform", "linux",
    "figma", "product", "management", "agile", "scrum", "leadership",
}


def _is_tech_keyword(term: str) -> bool:
    return term.lower() in TECH_KEYWORDS


def compute_match_score(
    resume_data: dict,
    job_description: str,
    job_title: str = "",
) -> int:
    """
    Compute a 0-100 match score between a resume and a job posting.

    resume_data: {
        skills: list[str],
        experiences: list[{jobTitle, companyName, description}],
        summary: str
    }
    """
    # Build candidate text corpus
    candidate_parts = []

    # Skills are highest signal
    skills = resume_data.get("skills", [])
    candidate_parts.extend(skills * 3)  # Weight skills 3x

    # Experience descriptions
    for exp in resume_data.get("experiences", []):
        if exp.get("jobTitle"):
            candidate_parts.extend([exp["jobTitle"]] * 2)
        if exp.get("description"):
            candidate_parts.append(exp["description"])

    # Summary
    if resume_data.get("summary"):
        candidate_parts.append(resume_data["summary"])

    candidate_text = " ".join(candidate_parts)
    job_text = f"{job_title} {job_description}"

    candidate_tokens = _tokenize(candidate_text)
    job_tokens = _tokenize(job_text)

    score = _compute_tfidf_score(candidate_tokens, job_tokens)

    # Bonus: if job title directly matches a skill or experience title
    jt_lower = job_title.lower()
    for skill in skills:
        if skill.lower() in jt_lower or jt_lower in skill.lower():
            score = min(score + 10, 100)
            break

    return int(score)
