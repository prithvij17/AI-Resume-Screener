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


# General-purpose fallbacks, used when the templated "Email:"/"Phone:"/
# "EDUCATION" labels aren't found — real-world resumes rarely use this
# project's exact template, so these keep the app useful on them too.
GENERAL_EMAIL_RE = re.compile(r"([\w.\-+]+@[\w\-]+\.[\w.\-]+)")
GENERAL_PHONE_RE = re.compile(r"(\+?\d[\d\-. ]{7,}\d)")
DEGREE_KEYWORDS_RE = re.compile(
    r"\b(B\.?\s?Tech|M\.?\s?Tech|B\.?\s?E\.?|M\.?\s?E\.?|Bachelor of [A-Za-z ]+|"
    r"Master of [A-Za-z ]+|MBA|BBA|MCA|BCA|B\.?\s?Sc|M\.?\s?Sc|B\.?\s?Com|"
    r"M\.?\s?Com|Ph\.?\s?D|Diploma)\b",
    re.IGNORECASE,
)
# \b word boundaries here matter: without them this previously matched
# 4-digit substrings *inside* longer digit runs (e.g. found "1945" hiding
# inside a phone number "9319450067") — a real bug caught while testing
# against an actual uploaded resume, not a hypothetical.
YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")


def extract_email(text) -> str | None:
    strict = extract_field(r"Email:\s*([^\s|]+)", text)
    if strict:
        return strict
    if isinstance(text, str):
        m = GENERAL_EMAIL_RE.search(text)
        if m:
            return m.group(1).rstrip(".,;:")
    return None


def extract_phone(text) -> str | None:
    strict = extract_field(r"Phone:\s*([^\s|]+)", text)
    if strict:
        return strict
    if isinstance(text, str):
        m = GENERAL_PHONE_RE.search(text)
        if m:
            candidate = m.group(1).strip()
            if sum(c.isdigit() for c in candidate) >= 10:  # avoid matching years, scores, etc.
                return candidate
    return None


def _clean_degree_line(line: str) -> str:
    """PDF text extraction often flattens multi-column layouts onto one
    line (e.g. a degree name sharing a line with a contact email in a
    header). Strip obviously unrelated contact info back out — but only
    strings that actually look like a real phone number (10+ digits), not
    date ranges like '(2018 - 2022)' which can otherwise falsely match the
    same loose digit-and-separator shape."""
    line = GENERAL_EMAIL_RE.sub("", line)

    def _strip_if_real_phone(m):
        return "" if sum(c.isdigit() for c in m.group(0)) >= 10 else m.group(0)

    line = GENERAL_PHONE_RE.sub(_strip_if_real_phone, line)
    return re.sub(r"\s{2,}", " ", line).strip(" -—|")


def extract_degree_fallback(text) -> tuple[str | None, int | None]:
    """Best-effort: for resumes with no EDUCATION section header, finds
    every line mentioning a common degree keyword and returns the first
    one that also has a graduation year nearby (same line) — since a
    resume's degree is often mentioned twice (once in a header with no
    year, once in an education table/section that does have one), and the
    version WITH a year is the one worth keeping. Falls back to the first
    keyword mention with no year if none have one nearby, rather than
    guessing a year from somewhere unrelated in the document."""
    if not isinstance(text, str):
        return None, None

    candidates = []
    for m in DEGREE_KEYWORDS_RE.finditer(text):
        start = text.rfind("\n", 0, m.start()) + 1
        end = text.find("\n", m.end())
        if end == -1:
            end = len(text)
        line = _clean_degree_line(text[start:end])
        if not line:
            line = m.group(0)
        year_match = YEAR_RE.search(line)
        candidates.append((line, int(year_match.group(0)) if year_match else None))

    if not candidates:
        return None, None
    for line, year in candidates:
        if year is not None:
            return line, year
    return candidates[0]


def extract_location_fallback(text, nlp, has_full_model: bool) -> str | None:
    """Best-effort: uses spaCy's NER (GPE = geopolitical entity) to spot a
    place name when there's no 'Location:' label. Only works when the full
    en_core_web_sm model is loaded (has_full_model=True) — a blank pipeline
    has no NER component and will just find nothing, harmlessly."""
    if not has_full_model or not isinstance(text, str):
        return None
    doc = nlp(text[:2000])  # location is almost always near the top; keeps this fast
    for ent in doc.ents:
        if ent.label_ == "GPE":
            return ent.text
    return None


def parse_education(edu: str | None, full_text: str | None = None) -> dict:
    if isinstance(edu, str):
        m = re.match(r"^(.*?)\s*—\s*(.*?)\s*\((\d{4})\)", edu)
        if m:
            return {"degree": m.group(1).strip(), "university": m.group(2).strip(), "grad_year": int(m.group(3))}
        return {"degree": edu, "university": None, "grad_year": None}

    # no EDUCATION section header found — fall back to a whole-resume search
    degree_line, grad_year = extract_degree_fallback(full_text)
    return {"degree": degree_line, "university": None, "grad_year": grad_year}


COMMON_SECTION_HEADERS = {
    "experience", "work experience", "professional experience", "employment history",
    "education", "academic background", "projects", "skills", "technical skills",
    "certifications", "languages", "extracurriculars", "extra curriculars", "achievements",
}
EXPERIENCE_HEADER_ALIASES = {"experience", "work experience", "professional experience", "employment history"}


def extract_section_flexible(header_aliases: set, text) -> str | None:
    """Like extract_section, but case-insensitive and matches several
    possible header spellings — real resumes don't reliably use this
    project's exact 'EXPERIENCE' (all-caps) convention. Finds a line that
    IS one of header_aliases, then captures until the next line that looks
    like any other known section header, or end of text."""
    if not isinstance(text, str):
        return None
    lines = text.split("\n")
    start_idx = None
    for i, line in enumerate(lines):
        if line.strip().lower().rstrip(":") in header_aliases:
            start_idx = i + 1
            break
    if start_idx is None:
        return None
    end_idx = len(lines)
    for j in range(start_idx, len(lines)):
        clean = lines[j].strip().lower().rstrip(":")
        if clean in COMMON_SECTION_HEADERS and clean not in header_aliases:
            end_idx = j
            break
    section = "\n".join(lines[start_idx:end_idx]).strip()
    return section or None


def extract_years_experience_fallback(full_text: str | None) -> dict:
    """Best-effort, for resumes with no explicit 'Title — Company (N years)'
    template line: approximates years of experience as (current year -
    earliest year mentioned inside an experience-like section). Coarse —
    ignores months and gaps between jobs — but a reasonable stand-in, since
    real resumes almost always describe roles with date ranges rather than
    a stated year count. Deliberately scoped to an experience-like section
    only (not the whole document) so it doesn't pick up unrelated years
    from an education section."""
    section = extract_section_flexible(EXPERIENCE_HEADER_ALIASES, full_text)
    if not section:
        return {"years": 0, "has_experience": False}
    years_found = [int(y.group(0)) for y in YEAR_RE.finditer(section)]
    if not years_found:
        return {"years": 0, "has_experience": True}
    from datetime import datetime

    return {"years": max(0, datetime.now().year - min(years_found)), "has_experience": True}


def parse_experience(exp: str | None, full_text: str | None = None) -> dict:
    if isinstance(exp, str):
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

    # no strict "EXPERIENCE" section header found — fall back to a flexible,
    # case-insensitive section match + date-range-based approximation
    fallback = extract_years_experience_fallback(full_text)
    return {"job_title": None, "company": None, **fallback}


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


def guess_candidate_name(resume_text: str, fallback: str) -> str:
    """Best-effort: resumes conventionally put the candidate's name on the
    first non-empty line, sometimes sharing that line with contact info due
    to how PDF column layouts flatten to plain text (stripped out below).
    Falls back to a cleaned-up filename if nothing plausible is found —
    not perfect, but far better for a ranked-candidates table than showing
    a raw filename for every row."""
    if isinstance(resume_text, str):
        for raw_line in resume_text.splitlines():
            line = _clean_degree_line(raw_line.strip())  # strips embedded email/phone substrings
            if not line:
                continue
            words = line.split()
            looks_like_name = (
                1 <= len(words) <= 4
                and len(line) <= 40
                and "@" not in line
                and not any(c.isdigit() for c in line)
                and line.upper() not in {"RESUME", "CURRICULUM VITAE", "CV"}
            )
            return line if looks_like_name else fallback
    return fallback


def enrich_resume_row(resume_text: str, nlp, matcher, skill_lookup: dict, has_full_model: bool = False) -> dict:
    """Runs the full extraction pipeline on ONE resume_text string and
    returns a flat dict of extracted fields. Works for both the bundled
    dataset (fully populated via the strict template) and an arbitrary
    uploaded resume (partial, via general-purpose fallbacks — never an
    error, missing fields just come back as None/empty)."""
    email = extract_email(resume_text)
    phone = extract_phone(resume_text)
    location = extract_field(r"Location:\s*([^\n]+)", resume_text) or extract_location_fallback(
        resume_text, nlp, has_full_model
    )

    education = parse_education(extract_section("EDUCATION", resume_text), full_text=resume_text)
    experience = parse_experience(extract_section("EXPERIENCE", resume_text), full_text=resume_text)

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
        lambda t: enrich_resume_row(t, nlp, matcher, skill_lookup, has_full_model)
    )
    enriched_df = pd.DataFrame(list(enriched_rows))
    result = pd.concat([resumes.reset_index(drop=True), enriched_df], axis=1)
    return result
