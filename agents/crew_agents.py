"""
crew_agents.py
---------------
Defines the CrewAI multi-agent pipeline used to score and explain each
resume-to-job match: Screening -> Scoring -> Feedback -> Bias-Check.

This file only DEFINES agents/tasks/the crew-building function. It makes
no LLM calls on import and sends no emails. Running `run_agents.py` is
what actually executes the pipeline against real candidates.

Design notes (read before changing thresholds / prompts):
- Hard-requirement checking (experience years, required skills) is done by
  a plain Python function (`check_hard_requirements`), wrapped as a tool.
  This keeps the pass/fail gate 100% deterministic and free to run — the
  LLM is only asked to interpret and explain the result, never to
  recompute it. This avoids the model inventing or misjudging a
  requirement check it could get wrong.
- Education match is informational only (not a hard gate). Real hiring
  is usually flexible here if skills/experience are strong, and the
  `education_required` column is a loose phrase ("B.Tech/M.Tech in CSE,
  Statistics, or related field"), not a clean value to match exactly.
- Every downstream task is given ONLY the structured facts already
  extracted (Steps ML+NLP layers, plus the screening tool's output) —
  never the raw resume_text. This keeps the LLM grounded and reduces
  hallucinated feedback.
"""

import os
import json
import re

from crewai import Agent, Task, Crew, Process, LLM
from crewai.tools import tool


# ---------------------------------------------------------------------------
# LLM setup (Groq)
# ---------------------------------------------------------------------------
# Project spec: Groq only (no OpenAI or Anthropic anywhere), model =
# llama-3.3-70b-versatile.
#
# IMPORTANT — why provider="openai" appears below even though this project
# uses ONLY Groq:
# CrewAI's LLM class normally talks to non-"native" providers (including
# Groq) through a package called LiteLLM. On Windows, LiteLLM has no
# prebuilt wheel and pip falls back to compiling it from source, which
# requires a working Rust toolchain — a common source of install failures
# that has nothing to do with this project's code.
# To avoid that entirely, this uses CrewAI's NATIVE OpenAI-compatible
# client (a thin wrapper around the standard `openai` Python package,
# which ships prebuilt wheels for every platform — no Rust required) and
# points it at Groq's own OpenAI-compatible endpoint via base_url.
# provider="openai" here selects the WIRE FORMAT/client shape only — every
# request still goes to Groq's servers (base_url below), authenticated
# with GROQ_API_KEY, running a Groq-hosted Llama model. No OpenAI API key,
# OpenAI server, or OpenAI model is ever used. Groq (like many providers)
# deliberately exposes an OpenAI-compatible endpoint for exactly this
# reason — see https://console.groq.com/docs/openai
#
# Heads-up: Groq announced (June 17, 2026) that llama-3.3-70b-versatile is
# deprecated for free/developer-tier accounts, with a full shutdown expected
# by August 2026. It still works as of this writing. If it stops working,
# swap the model in ONE place — set GROQ_MODEL in your .env — no code
# changes needed. Recommended fallback (still Groq): "qwen/qwen3.6-27b".
# Check https://console.groq.com/docs/models for the current lineup.
DEFAULT_GROQ_MODEL = "llama-3.3-70b-versatile"
GROQ_BASE_URL = "https://api.groq.com/openai/v1"


def get_llm(temperature: float = 0.2) -> LLM:
    """Builds the Groq-backed LLM CrewAI's agents will use."""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY is not set.\n"
            "Create a .env file next to this script (copy .env.example) with:\n"
            "  GROQ_API_KEY=your_key_here\n"
            "Get a free key at https://console.groq.com/keys"
        )
    model_name = os.getenv("GROQ_MODEL", DEFAULT_GROQ_MODEL)
    return LLM(
        model=model_name,
        provider="openai",  # wire format only — see note above; requests go to Groq, not OpenAI
        base_url=GROQ_BASE_URL,
        api_key=api_key,
        temperature=temperature,
    )


# ---------------------------------------------------------------------------
# Tool: deterministic hard-requirement check
# ---------------------------------------------------------------------------
def _parse_min_years(experience_required: str) -> int:
    """'1–3 years' / '2-4 years' -> 1 / 2 (first number = minimum)."""
    if not experience_required:
        return 0
    m = re.search(r"(\d+)", experience_required)
    return int(m.group(1)) if m else 0


@tool("check_hard_requirements")
def check_hard_requirements(candidate_json: str, job_json: str) -> str:
    """
    Compares a candidate's extracted years of experience and skills against
    a job's minimum experience and required skills.
    Input: candidate_json and job_json, each a JSON string.
    Returns a JSON string: {passed, meets_experience, required_years,
    candidate_years, missing_required_skills}.
    """
    candidate = json.loads(candidate_json)
    job = json.loads(job_json)

    required_skills = {
        s.strip().lower() for s in job.get("required_skills", "").split(",") if s.strip()
    }
    candidate_skills = {s.strip().lower() for s in candidate.get("skills_extracted", [])}
    missing_required_skills = sorted(required_skills - candidate_skills)

    required_years = _parse_min_years(job.get("experience_required", ""))
    candidate_years = int(candidate.get("years_exp_extracted", 0) or 0)
    meets_experience = candidate_years >= required_years

    passed = meets_experience and len(missing_required_skills) == 0

    return json.dumps(
        {
            "passed": passed,
            "meets_experience": meets_experience,
            "required_years": required_years,
            "candidate_years": candidate_years,
            "missing_required_skills": missing_required_skills,
        }
    )


# ---------------------------------------------------------------------------
# Agents
# ---------------------------------------------------------------------------
def build_agents(llm: LLM) -> dict:
    screening_agent = Agent(
        role="Recruitment Screening Specialist",
        goal=(
            "Determine whether a candidate meets a job's minimum hard requirements "
            "(experience and required skills) using the check_hard_requirements tool, "
            "and summarize the result in one or two plain-English sentences."
        ),
        backstory=(
            "You do an initial, objective pass on every application. You never guess "
            "at requirements — you always call the tool and report exactly what it finds."
        ),
        tools=[check_hard_requirements],
        llm=llm,
        verbose=False,
    )

    scoring_agent = Agent(
        role="Candidate Fit Scoring Analyst",
        goal=(
            "Assign a 0-100 fit score for how well a candidate matches a job, using only "
            "the structured facts provided (TF-IDF text-match score, screening result, "
            "skills overlap, years of experience). Never invent facts not given to you."
        ),
        backstory=(
            "You are a careful analyst who combines a quantitative text-match score with "
            "the screening outcome to produce one calibrated score. You explain your "
            "reasoning briefly and only reference facts you were actually given."
        ),
        llm=llm,
        verbose=False,
    )

    feedback_agent = Agent(
        role="Candidate Communication Specialist",
        goal=(
            "Write two short pieces of feedback for a scored candidate: a detailed internal "
            "note for the recruiter, and a short, professional, factual message suitable for "
            "sending directly to the candidate if they are not moving forward."
        ),
        backstory=(
            "You write clearly and kindly. The candidate-facing message never speculates "
            "about the person and cites only concrete skill/experience gaps already identified."
        ),
        llm=llm,
        verbose=False,
    )

    bias_check_agent = Agent(
        role="Fair Hiring Compliance Reviewer",
        goal=(
            "Review the candidate-facing message for any language that touches age, gender, "
            "name/ethnicity, nationality, disability, marital/family status, or any other "
            "protected characteristic — directly or indirectly. Rewrite it if needed so it is "
            "strictly limited to skills, experience, and qualifications."
        ),
        backstory=(
            "You are the last check before any message reaches a candidate. You are strict: "
            "when in doubt, you simplify the message down to the safest, most generic "
            "skills-based phrasing rather than risk problematic language."
        ),
        llm=llm,
        verbose=False,
    )

    return {
        "screening_agent": screening_agent,
        "scoring_agent": scoring_agent,
        "feedback_agent": feedback_agent,
        "bias_check_agent": bias_check_agent,
    }


# ---------------------------------------------------------------------------
# Tasks + Crew assembly (built fresh per candidate — CrewAI tasks are
# single-use once run)
# ---------------------------------------------------------------------------
def build_crew_for_candidate(agents: dict, candidate: dict, job: dict) -> Crew:
    candidate_json = json.dumps(candidate)
    job_json = json.dumps(job)

    screening_task = Task(
        description=(
            f"Candidate data (JSON): {candidate_json}\n"
            f"Job data (JSON): {job_json}\n\n"
            "Call the check_hard_requirements tool with these two JSON strings as "
            "candidate_json and job_json. Then summarize the result in plain English: "
            "does the candidate meet the minimum experience and have all required skills? "
            "List any missing required skills by name."
        ),
        expected_output=(
            "A short paragraph stating pass/fail, years of experience vs. required, and "
            "any missing required skills."
        ),
        agent=agents["screening_agent"],
    )

    scoring_task = Task(
        description=(
            f"Candidate data (JSON): {candidate_json}\n"
            f"Job data (JSON): {job_json}\n"
            f"The candidate's TF-IDF text-match score against this job is "
            f"{candidate.get('match_percent', 'N/A')}%.\n\n"
            "Using the screening result above plus this match score and the candidate's "
            "skills/experience, assign a single fit_score from 0-100 (integer) and give a "
            "2-3 sentence justification referencing only the facts provided. "
            "Respond ONLY with valid JSON in this exact shape, no other text: "
            '{"fit_score": <int 0-100>, "justification": "<2-3 sentences>"}'
        ),
        expected_output='Valid JSON: {"fit_score": <int>, "justification": "<text>"}',
        agent=agents["scoring_agent"],
        context=[screening_task],
    )

    feedback_task = Task(
        description=(
            f"Candidate name: {candidate.get('candidate_name', 'the candidate')}\n"
            f"Job role: {job.get('role', 'the role')}\n\n"
            "Using the screening result and fit score/justification above, write two things. "
            "Respond ONLY with valid JSON in this exact shape, no other text: "
            '{"internal_feedback": "<3-4 sentences for the recruiter, can be direct and detailed>", '
            '"candidate_feedback": "<2-3 sentences, professional and factual, suitable to send '
            'directly to the candidate if rejected, citing only concrete skill/experience gaps, '
            'never speculating about the person>"}'
        ),
        expected_output='Valid JSON: {"internal_feedback": "<text>", "candidate_feedback": "<text>"}',
        agent=agents["feedback_agent"],
        context=[screening_task, scoring_task],
    )

    bias_check_task = Task(
        description=(
            "Review the candidate_feedback message from the previous step. Check it does not "
            "reference age, gender, name/ethnicity, nationality, disability, or any other "
            "protected characteristic, directly or indirectly, and that it only discusses "
            "skills/experience/qualifications. Rewrite it if needed. "
            "Respond ONLY with valid JSON in this exact shape, no other text: "
            '{"final_candidate_feedback": "<the approved or rewritten message>", '
            '"was_revised": <true or false>, '
            '"revision_reason": "<short reason, or null if not revised>"}'
        ),
        expected_output=(
            'Valid JSON: {"final_candidate_feedback": "<text>", "was_revised": <bool>, '
            '"revision_reason": "<text or null>"}'
        ),
        agent=agents["bias_check_agent"],
        context=[feedback_task],
    )

    return Crew(
        agents=[
            agents["screening_agent"],
            agents["scoring_agent"],
            agents["feedback_agent"],
            agents["bias_check_agent"],
        ],
        tasks=[screening_task, scoring_task, feedback_task, bias_check_task],
        process=Process.sequential,
        verbose=False,
    )
