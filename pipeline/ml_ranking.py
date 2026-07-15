"""
pipeline/ml_ranking.py
----------------------
Layer 1 (ML): TF-IDF + cosine similarity resume-to-job ranking.

This is the same logic developed and validated in notebooks/data.ipynb,
extracted into reusable functions so both the notebook and the Streamlit
app call one shared, tested implementation instead of two copies that
could drift apart.
"""

import re

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def clean_text(text) -> str:
    """Lowercases and strips emails/phone numbers/punctuation so the
    vectorizer only sees meaningful words (skills, tools, experience terms)."""
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = re.sub(r"\S+@\S+", " ", text)
    text = re.sub(r"\+?\d[\d\-\s]{7,}\d", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def prepare_job_text(job_row: dict) -> str:
    """Combines a job's description + required/preferred skills into one
    cleaned text blob for vectorization."""
    combined = (
        str(job_row.get("job_description", "") or "")
        + " "
        + str(job_row.get("required_skills", "") or "")
        + " "
        + str(job_row.get("preferred_skills", "") or "")
    )
    return clean_text(combined)


def rank_candidates_for_job(job_row: dict, resumes_df: pd.DataFrame, job_text: str = None) -> pd.DataFrame:
    """Scores every same-category resume against one job via TF-IDF +
    cosine similarity. Returns a copy of the matching resumes sorted by
    match_score descending, with match_score / match_percent / job_role added."""
    candidates = resumes_df[resumes_df["category"] == job_row["category"]].copy()
    if candidates.empty:
        return candidates

    if "clean_text" not in candidates.columns:
        candidates["clean_text"] = candidates["resume_text"].apply(clean_text)

    if job_text is None:
        job_text = prepare_job_text(job_row)

    corpus = [job_text] + candidates["clean_text"].tolist()
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(corpus)

    job_vector = tfidf_matrix[0:1]
    resume_vectors = tfidf_matrix[1:]
    scores = cosine_similarity(job_vector, resume_vectors).flatten()

    candidates["match_score"] = scores
    candidates["match_percent"] = (scores * 100).round(2)
    candidates["job_role"] = job_row["role"]

    return candidates.sort_values("match_score", ascending=False).reset_index(drop=True)


def rank_all_jobs(job_descriptions: pd.DataFrame, resumes: pd.DataFrame) -> pd.DataFrame:
    """Runs rank_candidates_for_job for every job posting and returns one
    combined, ranked dataframe (adds a per-job 'rank' column)."""
    resumes = resumes.copy()
    resumes["clean_text"] = resumes["resume_text"].apply(clean_text)

    all_rankings = []
    for _, job_row in job_descriptions.iterrows():
        ranked = rank_candidates_for_job(job_row, resumes)
        if not ranked.empty:
            all_rankings.append(ranked)

    if not all_rankings:
        return pd.DataFrame()

    all_rankings_df = pd.concat(all_rankings, ignore_index=True)
    all_rankings_df["rank"] = (
        all_rankings_df.groupby("job_role")["match_score"]
        .rank(ascending=False, method="first")
        .astype(int)
    )
    return all_rankings_df


def score_single_resume(resume_text: str, job_row: dict, resumes_df: pd.DataFrame) -> float:
    """Scores ONE new resume's text (e.g. pasted or extracted from an
    uploaded PDF) against one job, using the same job-category peer group
    as context for the vectorizer. Returns a match_percent (0-100)."""
    peers = resumes_df[resumes_df["category"] == job_row["category"]].copy()
    peers["clean_text"] = peers["resume_text"].apply(clean_text)
    job_text = prepare_job_text(job_row)
    new_text = clean_text(resume_text)

    corpus = [job_text, new_text] + peers["clean_text"].tolist()
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(corpus)

    job_vector = tfidf_matrix[0:1]
    new_resume_vector = tfidf_matrix[1:2]
    score = cosine_similarity(job_vector, new_resume_vector).flatten()[0]
    return round(float(score) * 100, 2)
