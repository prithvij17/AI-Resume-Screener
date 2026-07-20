"""
pipeline/evaluation.py
-----------------------
Evaluation metrics comparing the pipeline's outputs against the bundled
dataset's ground-truth `fit_level` labels (high/medium/low).

Primary metrics (from the project proposal):
    - precision_at_k()        -> Precision@5, Scoring Layer, target > 0.60
    - judge_feedback_quality() -> LLM-as-Judge Score, Agent Layer, target > 3.5/5

Supplementary metrics (extra depth beyond the proposal, useful for Q&A):
    - evaluate_ranking_vs_fit_level(), evaluate_extraction_completeness(),
      evaluate_extraction_accuracy(), compare_ml_vs_agent()

Scope note: these metrics are only meaningful on data that has a fit_level
column — i.e. the bundled resumes.csv dataset. A custom PDF batch (real
resumes you upload) has no ground truth to check against.
"""

import json
import re

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, f1_score

FIT_LEVEL_ORDER = ["low", "medium", "high"]
FIT_LEVEL_RANK = {"low": 1, "medium": 2, "high": 3}


# ---------------------------------------------------------------------------
# Metric 1 (proposal): Precision@5 — Scoring Layer, target > 0.60
# ---------------------------------------------------------------------------
def precision_at_k(df: pd.DataFrame, k: int = 5, score_col: str = "match_score",
                    group_col: str = "job_role", positive_label: str = "high") -> dict:
    """
    Precision@K, as defined in the project proposal: "Of the top K ranked
    candidates returned by the system, the fraction that are actually
    high-fit candidates based on ground truth fit labels."

    Computed per job (top-K within each job's own ranked list, since scores
    aren't comparable across different jobs), then averaged across jobs.
    """
    if "fit_level" not in df.columns:
        raise ValueError("This dataframe has no 'fit_level' ground-truth column to evaluate against.")

    per_job = {}
    for job, g in df.groupby(group_col):
        top_k = g.sort_values(score_col, ascending=False).head(k)
        if len(top_k) == 0:
            continue
        hits = int((top_k["fit_level"] == positive_label).sum())
        per_job[job] = hits / len(top_k)

    overall = float(np.mean(list(per_job.values()))) if per_job else None
    return {
        "metric": f"Precision@{k}",
        "per_job_precision": per_job,
        "overall_precision": overall,
        "target": 0.60,
        "meets_target": (overall is not None and round(overall, 6) > 0.60),
        "k": k,
        "positive_label": positive_label,
    }


# ---------------------------------------------------------------------------
# Metric 2 (proposal): LLM-as-Judge Score — Agent Layer, target > 3.5/5
# ---------------------------------------------------------------------------
JUDGE_PROMPT_TEMPLATE = """You are evaluating the quality of AI-generated recruiting feedback. \
You did not write this feedback — you are an independent reviewer.

Candidate: {candidate_name}
Job role: {job_role}
Feedback to evaluate:
\"\"\"{feedback_text}\"\"\"

Rate the feedback on each dimension below, from 1 (poor) to 5 (excellent):
- Relevance: does it address this specific candidate and role, rather than being generic boilerplate?
- Accuracy: is it internally consistent, with no fabricated or contradictory claims?
- Clarity: is it clearly written and easy to understand?
- Actionability: would this actually help the candidate understand what to improve, or help a recruiter decide?

Respond ONLY with valid JSON in this exact shape, no other text:
{{"relevance": <1-5 int>, "accuracy": <1-5 int>, "clarity": <1-5 int>, "actionability": <1-5 int>, "rationale": "<1-2 sentences>"}}
"""


def _safe_json_parse(text: str):
    text = text.strip().strip("`")
    if text.lower().startswith("json"):
        text = text[4:].strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            pass
    return None


def judge_feedback_quality(scored_df: pd.DataFrame, feedback_col: str = "internal_feedback",
                            sample_size: int | None = None, temperature: float = 0.0) -> dict:
    """
    LLM-as-Judge, as defined in the project proposal: "A second LLM call
    rates generated feedback on four dimensions: Relevance, Accuracy,
    Clarity, and Actionability (each scored 1-5)."

    Uses a SEPARATE call from the one that generated the feedback (the
    Feedback Agent in agents/crew_agents.py) — same Groq setup, but acting
    as an independent reviewer rather than the original author. Costs real
    Groq API credits: one call per row evaluated (use sample_size to cap it).
    """
    import os
    import sys

    agents_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "agents")
    if agents_dir not in sys.path:
        sys.path.insert(0, agents_dir)
    import crew_agents  # local import — keeps crewai an optional dependency for the rest of this module

    llm = crew_agents.get_llm(temperature=temperature)

    df = scored_df.dropna(subset=[feedback_col]).copy()
    if sample_size is not None and len(df) > sample_size:
        df = df.sample(n=sample_size, random_state=42)

    rows = []
    for _, row in df.iterrows():
        prompt = JUDGE_PROMPT_TEMPLATE.format(
            candidate_name=row["candidate_name"], job_role=row["job_role"], feedback_text=row[feedback_col]
        )
        try:
            response = llm.call(messages=[{"role": "user", "content": prompt}])
            parsed = _safe_json_parse(response)
        except Exception as e:  # noqa: BLE001 - log and continue, one bad row shouldn't kill the batch
            parsed = None
            rows.append({
                "candidate_name": row["candidate_name"], "job_role": row["job_role"],
                "relevance": None, "accuracy": None, "clarity": None, "actionability": None,
                "rationale": None, "judge_error": str(e),
            })
            continue

        if parsed:
            rows.append({
                "candidate_name": row["candidate_name"], "job_role": row["job_role"],
                "relevance": parsed.get("relevance"), "accuracy": parsed.get("accuracy"),
                "clarity": parsed.get("clarity"), "actionability": parsed.get("actionability"),
                "rationale": parsed.get("rationale"), "judge_error": None,
            })
        else:
            rows.append({
                "candidate_name": row["candidate_name"], "job_role": row["job_role"],
                "relevance": None, "accuracy": None, "clarity": None, "actionability": None,
                "rationale": None, "judge_error": "could not parse judge response as JSON",
            })

    results_df = pd.DataFrame(rows)
    dims = ["relevance", "accuracy", "clarity", "actionability"]
    valid = results_df.dropna(subset=dims)
    if valid.empty:
        return {"metric": "LLM-as-Judge Score", "results": results_df, "average_score": None, "target": 3.5, "meets_target": False}

    results_df.loc[valid.index, "overall"] = valid[dims].mean(axis=1)
    avg_score = float(valid[dims].mean(axis=1).mean())

    return {
        "metric": "LLM-as-Judge Score",
        "results": results_df,
        "average_score": avg_score,
        "average_by_dimension": {d: float(valid[d].mean()) for d in dims},
        "n_evaluated": len(valid),
        "n_failed": len(results_df) - len(valid),
        "target": 3.5,
        "meets_target": round(avg_score, 6) > 3.5,
    }


# ---------------------------------------------------------------------------
# Supplementary metrics (extra depth beyond the proposal)
# ---------------------------------------------------------------------------
def evaluate_ranking_vs_fit_level(df: pd.DataFrame, score_col: str, group_col: str = "job_role") -> dict:
    """
    Two complementary metrics, computed per group (job) then combined —
    scores from different jobs aren't directly comparable, so correlation
    and bucketing are both done within each job first:

    1. Spearman rank correlation between `score_col` and fit_level, per job,
       then averaged. Answers: "does a higher score reliably mean a better
       labeled candidate, within each job's applicant pool?"
    2. A percentile-bucketed confusion matrix: within each job, candidates
       are split into predicted low/medium/high tiers using the SAME group
       sizes as the true fit_level distribution, then compared against the
       true labels. Answers: "if I only trusted the score to sort candidates
       into tiers, how often would it agree with the ground truth?"
    """
    if "fit_level" not in df.columns:
        raise ValueError("This dataframe has no 'fit_level' ground-truth column to evaluate against.")

    df = df.dropna(subset=[score_col, "fit_level"]).copy()
    df["fit_rank"] = df["fit_level"].map(FIT_LEVEL_RANK)

    correlations = {}
    predicted_tiers, true_tiers = [], []

    for group, g in df.groupby(group_col):
        if g["fit_rank"].nunique() >= 2 and g[score_col].nunique() >= 2:
            corr, _ = spearmanr(g[score_col], g["fit_rank"])
            correlations[group] = corr

        true_counts = g["fit_level"].value_counts()
        n = len(g)
        n_high = int(true_counts.get("high", 0))
        n_medium = int(true_counts.get("medium", 0))
        n_low = n - n_high - n_medium

        ranked = g.sort_values(score_col, ascending=False)
        tiers = (["high"] * n_high) + (["medium"] * n_medium) + (["low"] * n_low)
        predicted_tiers.extend(tiers[: len(ranked)])
        true_tiers.extend(ranked["fit_level"].tolist())

    avg_corr = float(np.mean(list(correlations.values()))) if correlations else None
    acc = accuracy_score(true_tiers, predicted_tiers) if true_tiers else None
    f1 = f1_score(true_tiers, predicted_tiers, average="macro", labels=FIT_LEVEL_ORDER, zero_division=0) if true_tiers else None
    cm = confusion_matrix(true_tiers, predicted_tiers, labels=FIT_LEVEL_ORDER) if true_tiers else None
    report = (
        classification_report(true_tiers, predicted_tiers, labels=FIT_LEVEL_ORDER, zero_division=0)
        if true_tiers else None
    )

    return {
        "per_job_correlation": correlations,
        "average_correlation": avg_corr,
        "tier_accuracy": acc,
        "tier_macro_f1": f1,
        "confusion_matrix": cm,
        "confusion_matrix_labels": FIT_LEVEL_ORDER,
        "classification_report": report,
        "n_evaluated": len(true_tiers),
    }


def evaluate_extraction_completeness(enriched_df: pd.DataFrame) -> pd.DataFrame:
    """For each key NLP-extracted field, what % of rows have a non-null /
    non-empty value. A basic 'did the extractor produce anything at all'
    check — see evaluate_extraction_accuracy for 'was it actually correct'."""
    fields = [
        "email", "phone", "location", "degree_extracted", "university_extracted",
        "grad_year_extracted", "years_exp_extracted", "skills_extracted",
    ]
    rows = []
    n = len(enriched_df)
    for f in fields:
        if f not in enriched_df.columns:
            continue
        col = enriched_df[f]
        if col.apply(lambda v: isinstance(v, list)).any():
            filled = col.apply(lambda v: isinstance(v, list) and len(v) > 0).sum()
        else:
            filled = col.notna().sum()
        rows.append({"field": f, "filled": int(filled), "total": n, "completeness_pct": round(100 * filled / n, 1)})
    return pd.DataFrame(rows)


def evaluate_extraction_accuracy(enriched_df: pd.DataFrame) -> pd.DataFrame:
    """Only meaningful on the bundled dataset, which has known-correct
    columns (experience_years, skills) to check extracted values against.
    A real uploaded resume has no such ground truth to compare to."""
    rows = []
    if {"experience_years", "years_exp_extracted"}.issubset(enriched_df.columns):
        match = (enriched_df["years_exp_extracted"] == enriched_df["experience_years"]).mean()
        rows.append({"check": "years_exp_extracted matches experience_years (ground truth)", "score_pct": round(100 * match, 1)})

    if {"skills", "skills_extracted"}.issubset(enriched_df.columns):
        def recall(row):
            if pd.isna(row["skills"]):
                return None
            listed = {s.strip().lower() for s in str(row["skills"]).split(",") if s.strip()}
            found = {s.lower() for s in row["skills_extracted"]} if isinstance(row["skills_extracted"], list) else set()
            return len(listed & found) / len(listed) if listed else None

        recalls = enriched_df.apply(recall, axis=1).dropna()
        rows.append({"check": "skill recall (listed skills also found by extractor)", "score_pct": round(100 * recalls.mean(), 1)})

    return pd.DataFrame(rows)


def compare_ml_vs_agent(scored_df: pd.DataFrame) -> dict:
    """Correlation between the TF-IDF match_percent and the agent's
    fit_score. High agreement suggests the agent's scoring is consistent
    with the cheaper ML signal rather than arbitrary — it does NOT by
    itself prove either one is 'more correct'; the fit_level ground-truth
    comparison is what actually validates correctness."""
    if not {"fit_score", "match_percent"}.issubset(scored_df.columns):
        return {"correlation": None, "n": 0}
    df = scored_df.dropna(subset=["fit_score", "match_percent"])
    if df.empty or df["fit_score"].nunique() < 2:
        return {"correlation": None, "n": len(df)}
    corr, _ = spearmanr(df["match_percent"], df["fit_score"])
    return {"correlation": float(corr), "n": len(df)}
