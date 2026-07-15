"""
pipeline/nlp_enrichment.py
--------------------------
Layer 2 (NLP): structured field extraction from resume_text.

Same logic developed and validated in notebooks/data.ipynb (regex-based
section parsing + spaCy PhraseMatcher skill extraction), extracted into
reusable functions.

Note on generalization: the section-based regex parsing (education,
experience, certifications, languages) assumes a templated resume format
(Name/Email/Phone/Location, then EDUCATION/SKILLS/EXPERIENCE/... section
headers) — true of this project's dataset, NOT guaranteed for an arbitrary
uploaded PDF resume. Functions here degrade gracefully (return None /
empty lists) rather than raising when a section isn't found, so a
free-form resume that doesn't match the template still gets partial
results instead of crashing. Skill extraction (PhraseMatcher) works on any
text, template or not.
"""

import re

import pandas as pd
import spacy
from spacy.matcher import PhraseMatcher


def load_nlp():
    """Tries the full pretrained pipeline first (for optional NER), falls
    back to a blank pipeline (rule-based extraction still works fully)."""
    try:
        nlp = spacy.load("en_core_web_sm")
        return nlp, True
    except OSError:
        return spacy.blank("en"), False


def extract_field(pattern: str, text) -> str | None:
    if not isinstance(text, str):
        return None
    m = re.search(pattern, text)
    return m.group(1).strip() if m else None


def extract_section(header: str, text) -> str | None:
    if not isinstance(text, str):
        return None
    m = re.search(rf"{header}\n(.*?)(?:\n\n|$)", text, re.DOTALL)
    return m.group(1).strip() if m else None


def parse_education(edu: str | None) -> dict:
    if not isinstance(edu, str):
        return {"degree": None, "university": None, "grad_year": None}
    m = re.match(r"^(.*?)\s*—\s*(.*?)\s*\((\d{4})\)", edu)
    if m:
        return {"degree": m.group(1).strip(), "university": m.group(2).strip(), "grad_year": int(m.group(3))}
    return {"degree": edu, "university": None, "grad_year": None}


def parse_experience(exp: str | None) -> dict:
    if not isinstance(exp, str):
        return {"job_title": None, "company": None, "years": 0, "has_experience": False}
    first_line = exp.split("\n")[0]
    m = re.match(r"^(.*?)\s*—\s*(.*?)\s*\((\d+)\s*years?\)", first_line)
    if m:
        return {
            "job_title": m.group(1).strip(),
            "company": m.group(2).strip(),
            "years": int(m.group(3)),
            "has_experience": True,
        }
    return {"job_title": first_line, "company": None, "years": 0, "has_experience": True}


def build_skill_vocab(resumes: pd.DataFrame, job_descriptions: pd.DataFrame) -> set:
    vocab = set()
    for s in resumes["skills"].dropna():
        vocab.update(x.strip() for x in s.split(","))
    for col in ["required_skills", "preferred_skills"]:
        for s in job_descriptions[col].dropna():
            vocab.update(x.strip() for x in s.split(","))
    return vocab


def build_skill_matcher(nlp, skills_vocab: set):
    skill_lookup = {s.lower(): s for s in skills_vocab}
    matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
    matcher.add("SKILL", [nlp.make_doc(skill) for skill in skills_vocab])
    return matcher, skill_lookup


def extract_skills(text, nlp, matcher, skill_lookup: dict) -> list:
    if not isinstance(text, str):
        return []
    doc = nlp.make_doc(text)
    matches = matcher(doc)
    found = {skill_lookup[doc[start:end].text.lower()] for _, start, end in matches}
    return sorted(found)


def enrich_resume_row(resume_text: str, nlp, matcher, skill_lookup: dict) -> dict:
    """Runs the full extraction pipeline on ONE resume_text string and
    returns a flat dict of extracted fields. Works for both the bundled
    dataset (fully populated) and an arbitrary uploaded resume (partial —
    missing sections simply come back as None/empty, never an error)."""
    email = extract_field(r"Email:\s*([^\s|]+)", resume_text)
    phone = extract_field(r"Phone:\s*([^\s|]+)", resume_text)
    location = extract_field(r"Location:\s*([^\n]+)", resume_text)

    education = parse_education(extract_section("EDUCATION", resume_text))
    experience = parse_experience(extract_section("EXPERIENCE", resume_text))

    cert_section = extract_section("CERTIFICATIONS", resume_text)
    certifications = (
        [x.lstrip("\u2022").strip() for x in cert_section.split("\n")] if cert_section else []
    )

    lang_section = extract_section("LANGUAGES", resume_text)
    languages = [x.strip() for x in lang_section.split(",")] if lang_section else []

    skills = extract_skills(resume_text, nlp, matcher, skill_lookup)

    return {
        "email": email,
        "phone": phone,
        "location": location,
        "degree_extracted": education["degree"],
        "university_extracted": education["university"],
        "grad_year_extracted": education["grad_year"],
        "job_title_extracted": experience["job_title"],
        "company_extracted": experience["company"],
        "years_exp_extracted": experience["years"],
        "has_prior_experience": experience["has_experience"],
        "certifications_extracted": certifications,
        "languages_extracted": languages,
        "skills_extracted": skills,
    }


def enrich_dataframe(resumes: pd.DataFrame, job_descriptions: pd.DataFrame, nlp=None, has_full_model=None) -> pd.DataFrame:
    """Runs enrich_resume_row over an entire resumes dataframe and merges
    the extracted fields in as new columns."""
    if nlp is None:
        nlp, has_full_model = load_nlp()

    skills_vocab = build_skill_vocab(resumes, job_descriptions)
    matcher, skill_lookup = build_skill_matcher(nlp, skills_vocab)

    enriched_rows = resumes["resume_text"].apply(
        lambda t: enrich_resume_row(t, nlp, matcher, skill_lookup)
    )
    enriched_df = pd.DataFrame(list(enriched_rows))
    result = pd.concat([resumes.reset_index(drop=True), enriched_df], axis=1)
    return result
