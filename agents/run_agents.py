"""
run_agents.py
--------------
Loads enriched_candidates.csv + job_descriptions.csv, runs the CrewAI
pipeline (Screening -> Scoring -> Feedback -> Bias-Check) for each
candidate against their matching job, and saves scored_candidates.csv.

This is the ONLY script that spends LLM API credits. It is deliberately
kept separate from the notebook and can be re-run safely: it skips any
candidate/job pair already present in an existing scored_candidates.csv
(resume-safe), so a crash or rate limit halfway through does not force a
full re-run.

Usage:
    python run_agents.py                 # process MAX_PER_JOB candidates per job (see below)
    python run_agents.py --all           # process every candidate (slower, costs more)
    python run_agents.py --limit 3       # override MAX_PER_JOB for this run

Requires a .env file (copy .env.example) with GROQ_API_KEY set.
"""

import os
import sys
import json
import time
import argparse
import ast

import pandas as pd
from dotenv import load_dotenv
import crew_agents
from crew_agents import get_llm, build_agents, build_crew_for_candidate

# ---------------------------------------------------------------------------
# Config — tune these before a big run
# ---------------------------------------------------------------------------
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
ENRICHED_CSV = os.path.join(DATA_DIR, "enriched_candidates.csv")
JOBS_CSV = os.path.join(DATA_DIR, "job_descriptions.csv")
OUTPUT_CSV = os.path.join(DATA_DIR, "scored_candidates.csv")

MAX_PER_JOB_DEFAULT = 5      # candidates per job role processed by default (cost control)
SLEEP_BETWEEN_CALLS = 2.0    # seconds, be polite to free-tier rate limits
SAVE_EVERY = 5                # write partial progress to disk every N candidates
REJECTION_THRESHOLD = 50      # fit_score below this -> status = "rejected"

LIST_COLUMNS = [
    "skills_extracted",
    "bonus_skills_found_in_text",
    "organizations_ner",
    "certifications_extracted",
    "languages_extracted",
]


def parse_list_column(value):
    """enriched_candidates.csv stores list columns as Python-repr strings
    (e.g. "['Python', 'SQL']") because pandas wrote them with to_csv.
    Convert them back to real lists."""
    if isinstance(value, list):
        return value
    if pd.isna(value):
        return []
    try:
        parsed = ast.literal_eval(value)
        return parsed if isinstance(parsed, list) else []
    except (ValueError, SyntaxError):
        return []


def safe_json_parse(text: str):
    """LLMs sometimes wrap JSON in ```json fences or add stray text despite
    instructions. Try a direct parse first, then fall back to extracting the
    first {...} block."""
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    text = text.strip("`")
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


def load_data():
    if not os.path.exists(ENRICHED_CSV):
        sys.exit(
            f"Could not find {ENRICHED_CSV}.\n"
            "Run the ML + NLP layers in the notebook first to generate enriched_candidates.csv, "
            "then copy it into the data/ folder (or update DATA_DIR above)."
        )
    resumes = pd.read_csv(ENRICHED_CSV)
    for col in LIST_COLUMNS:
        if col in resumes.columns:
            resumes[col] = resumes[col].apply(parse_list_column)

    jobs = pd.read_csv(JOBS_CSV)
    jobs_by_category = {row["category"]: row.to_dict() for _, row in jobs.iterrows()}
    return resumes, jobs_by_category


def already_processed(output_path):
    """Returns a set of (candidate_name, job_role) pairs already scored
    SUCCESSFULLY, so re-running the script doesn't redo (and re-pay for)
    finished work — but rows that errored last time (e.g. a rate limit or
    timeout) are left out on purpose, so they get retried automatically."""
    if not os.path.exists(output_path):
        return set()
    existing = pd.read_csv(output_path)
    succeeded = existing[existing["agent_error"].isna()]
    return set(zip(succeeded["candidate_name"], succeeded["job_role"]))


def build_candidate_payload(row) -> dict:
    """Only the structured facts the agents are allowed to see —
    fit_level (the ground-truth label) is deliberately excluded so it
    can't leak into the LLM's scoring; it's reattached afterwards purely
    for our own evaluation."""
    return {
        "candidate_name": row["candidate_name"],
        "degree_extracted": row.get("degree_extracted"),
        "years_exp_extracted": int(row.get("years_exp_extracted", 0) or 0),
        "skills_extracted": row.get("skills_extracted", []),
        "certifications_extracted": row.get("certifications_extracted", []),
        "match_percent": row.get("match_percent"),
    }


def build_job_payload(job: dict) -> dict:
    return {
        "role": job.get("role"),
        "category": job.get("category"),
        "required_skills": job.get("required_skills", ""),
        "preferred_skills": job.get("preferred_skills", ""),
        "experience_required": job.get("experience_required", ""),
        "education_required": job.get("education_required", ""),
        "job_description": job.get("job_description", ""),
    }


def run_pipeline_for_candidate(agents, candidate_payload, job_payload):
    """Runs the 4-agent crew for one candidate and returns a result dict.
    Any failure is caught so one bad candidate doesn't kill the whole batch."""
    try:
        crew = build_crew_for_candidate(agents, candidate_payload, job_payload)
        crew_output = crew_agents.run_crew_with_retry(crew)
        outputs = crew_output.tasks_output  # [screening, scoring, feedback, bias_check]

        screening_summary = outputs[0].raw.strip()
        scoring = safe_json_parse(outputs[1].raw) or {}
        feedback = safe_json_parse(outputs[2].raw) or {}
        bias_check = safe_json_parse(outputs[3].raw) or {}

        fit_score = scoring.get("fit_score")
        try:
            fit_score = int(fit_score)
        except (TypeError, ValueError):
            fit_score = None

        return {
            "screening_summary": screening_summary,
            "fit_score": fit_score,
            "scoring_justification": scoring.get("justification"),
            "internal_feedback": feedback.get("internal_feedback"),
            "final_candidate_feedback": bias_check.get("final_candidate_feedback")
            or feedback.get("candidate_feedback"),
            "bias_was_revised": bias_check.get("was_revised"),
            "bias_revision_reason": bias_check.get("revision_reason"),
            "agent_error": None,
        }
    except Exception as e:  # noqa: BLE001 - we want to log and continue, not crash the batch
        return {
            "screening_summary": None,
            "fit_score": None,
            "scoring_justification": None,
            "internal_feedback": None,
            "final_candidate_feedback": None,
            "bias_was_revised": None,
            "bias_revision_reason": None,
            "agent_error": str(e),
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--all", action="store_true", help="Process every candidate, not just MAX_PER_JOB")
    parser.add_argument("--limit", type=int, default=None, help="Override candidates-per-job for this run")
    args = parser.parse_args()

    load_dotenv()
    llm = get_llm()
    agents = build_agents(llm)

    resumes, jobs_by_category = load_data()
    done = already_processed(OUTPUT_CSV)
    print(f"Already scored: {len(done)} candidate/job pairs (will be skipped)")

    max_per_job = None if args.all else (args.limit or MAX_PER_JOB_DEFAULT)

    # pick candidates: highest TF-IDF match first within each job/category,
    # so a limited run demos the most relevant candidates
    to_process = []
    for category, group in resumes.groupby("category"):
        job = jobs_by_category.get(category)
        if job is None:
            print(f"  no job posting found for category '{category}', skipping {len(group)} resumes")
            continue
        group_sorted = group.sort_values("match_score", ascending=False)
        if max_per_job is not None:
            group_sorted = group_sorted.head(max_per_job)
        for _, row in group_sorted.iterrows():
            if (row["candidate_name"], job["role"]) not in done:
                to_process.append((row, job))

    print(f"Candidates to process this run: {len(to_process)}")
    if not to_process:
        print("Nothing to do.")
        return

    results = []
    for i, (row, job) in enumerate(to_process, start=1):
        candidate_payload = build_candidate_payload(row)
        job_payload = build_job_payload(job)

        print(f"[{i}/{len(to_process)}] {candidate_payload['candidate_name']} -> {job_payload['role']}")
        result = run_pipeline_for_candidate(agents, candidate_payload, job_payload)

        if result["agent_error"]:
            print(f"    ! error: {result['agent_error']}")

        record = {
            "candidate_name": row["candidate_name"],
            "email": row.get("email"),
            "job_role": job["role"],
            "category": row["category"],
            "match_percent": row.get("match_percent"),
            "fit_level": row.get("fit_level"),  # ground truth, for evaluation only
            **result,
        }
        record["status"] = (
            "rejected"
            if (record["fit_score"] is not None and record["fit_score"] < REJECTION_THRESHOLD)
            else "shortlisted"
            if record["fit_score"] is not None
            else "error"
        )
        results.append(record)

        if i % SAVE_EVERY == 0 or i == len(to_process):
            partial_df = pd.DataFrame(results)
            if os.path.exists(OUTPUT_CSV):
                existing = pd.read_csv(OUTPUT_CSV)
                # drop any old rows for the same (candidate, job) we just retried,
                # so a previous error row doesn't stick around alongside the new result
                new_keys = set(zip(partial_df["candidate_name"], partial_df["job_role"]))
                existing = existing[
                    ~existing.apply(lambda r: (r["candidate_name"], r["job_role"]) in new_keys, axis=1)
                ]
                combined = pd.concat([existing, partial_df], ignore_index=True)
            else:
                combined = partial_df
            combined.to_csv(OUTPUT_CSV, index=False)
            results = []  # flushed to disk, clear the in-memory buffer
            print(f"    saved progress -> {OUTPUT_CSV}")

        time.sleep(SLEEP_BETWEEN_CALLS)

    print("Done.")


if __name__ == "__main__":
    main()
