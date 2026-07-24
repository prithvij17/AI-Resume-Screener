"""
app.py
------
Streamlit front-end for the AI-Powered Resume Screener & Job Matcher.

Ties together all four layers built earlier:
    Layer 1 (ML)     -> pipeline/ml_ranking.py       (TF-IDF + cosine similarity)
    Layer 2 (NLP)    -> pipeline/nlp_enrichment.py   (regex + spaCy extraction)
    Layer 3 (Agents) -> agents/crew_agents.py + agents/run_agents.py (CrewAI + Groq)
    Layer 4 (Email)  -> email/send_rejections.py     (SMTP rejection dispatch)

Run with:  streamlit run app.py
"""

import os
import sys
import ast
import smtplib
from datetime import datetime, timezone

import pandas as pd
import streamlit as st
import plotly.express as px
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Path setup — makes agents/ and email/ importable as plain modules, and
# pipeline/ importable as a package. Order matters: see the collision check
# in the project notes — this combination was verified safe (the real
# stdlib `email` package always wins over our email/ folder because it's a
# regular package with __init__.py, while ours has none).
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)
sys.path.insert(0, os.path.join(BASE_DIR, "agents"))
sys.path.insert(0, os.path.join(BASE_DIR, "email"))

from pipeline.ml_ranking import rank_all_jobs, score_single_resume
from pipeline.nlp_enrichment import (
    load_nlp,
    build_skill_vocab,
    build_skill_matcher,
    enrich_resume_row,
    enrich_dataframe,
)
import crew_agents
from pipeline.evaluation import (
    precision_at_k,
    judge_feedback_quality,
    evaluate_ranking_vs_fit_level,
    evaluate_extraction_completeness,
    evaluate_extraction_accuracy,
    compare_ml_vs_agent,
)
import run_agents
import send_rejections

load_dotenv(os.path.join(BASE_DIR, ".env"))

DATA_DIR = os.path.join(BASE_DIR, "data")
RESUMES_CSV = os.path.join(DATA_DIR, "resumes.csv")
JOBS_CSV = os.path.join(DATA_DIR, "job_descriptions.csv")
RANKED_CSV = os.path.join(DATA_DIR, "ranked_candidates.csv")
ENRICHED_CSV = os.path.join(DATA_DIR, "enriched_candidates.csv")
SCORED_CSV = os.path.join(DATA_DIR, "scored_candidates.csv")
# Tracks whichever job(s) were actually used in the most recent Tab 1 run
# (bundled, a custom-uploaded jobs CSV, or a PDF-batch job) — written every
# time Tab 1 runs, and what Tab 3 (Agent Scoring) reads from. Kept SEPARATE
# from JOBS_CSV on purpose: JOBS_CSV is the original bundled dataset file
# and is never overwritten, so "reset to bundled data" always stays available.
ACTIVE_JOBS_CSV = os.path.join(DATA_DIR, "active_jobs.csv")

LIST_COLUMNS = ["skills_extracted", "certifications_extracted", "languages_extracted"]


def parse_list_columns(df: pd.DataFrame) -> pd.DataFrame:
    """CSV round-trips turn Python lists into repr strings — convert back."""
    df = df.copy()
    for col in LIST_COLUMNS:
        if col in df.columns:
            df[col] = df[col].apply(
                lambda v: v if isinstance(v, list) else (ast.literal_eval(v) if isinstance(v, str) and v.startswith("[") else [])
            )
    return df


st.set_page_config(page_title="AI Resume Screener", layout="wide", page_icon="🧑‍💼")
st.title("🧑‍💼 AI-Powered Resume Screener & Job Matcher")
st.caption("ML ranking → NLP enrichment → CrewAI multi-agent scoring (Groq) → automated rejection emails")

# ---------------------------------------------------------------------------
# Sidebar: pipeline status + configuration
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Pipeline status")
    st.write("✅ Bundled data found" if os.path.exists(RESUMES_CSV) else "❌ No bundled data in data/")
    st.write("✅ Ranked" if os.path.exists(RANKED_CSV) else "⬜ Not ranked yet")
    st.write("✅ Enriched" if os.path.exists(ENRICHED_CSV) else "⬜ Not enriched yet")
    st.write("✅ Agent-scored" if os.path.exists(SCORED_CSV) else "⬜ Not scored yet")

    st.divider()
    st.header("Configuration")

    if os.getenv("GROQ_API_KEY"):
        st.success("Groq API key loaded (.env)")
    else:
        entered_key = st.text_input(
            "GROQ_API_KEY", type="password",
            help="Session-only — not written to disk. Get a free key at console.groq.com/keys",
        )
        if entered_key:
            os.environ["GROQ_API_KEY"] = entered_key
            st.success("Key set for this session")

    if os.getenv("SMTP_USERNAME") and os.getenv("SMTP_PASSWORD"):
        st.success(f"SMTP configured ({os.getenv('SMTP_USERNAME')})")
    else:
        smtp_user_in = st.text_input("SMTP_USERNAME (Gmail address)")
        smtp_pass_in = st.text_input("SMTP_PASSWORD (Gmail App Password)", type="password")
        if smtp_user_in and smtp_pass_in:
            os.environ["SMTP_USERNAME"] = smtp_user_in
            os.environ["SMTP_PASSWORD"] = smtp_pass_in
            os.environ.setdefault("FROM_EMAIL", smtp_user_in)
            st.success("SMTP set for this session")

    if os.getenv("TEST_OVERRIDE_EMAIL"):
        st.warning(f"TEST_OVERRIDE_EMAIL active — all real sends go to {os.getenv('TEST_OVERRIDE_EMAIL')}")

    st.divider()
    if st.button("🗑️ Clear generated data (start fresh)"):
        for f in [RANKED_CSV, ENRICHED_CSV, SCORED_CSV, ACTIVE_JOBS_CSV]:
            if os.path.exists(f):
                os.remove(f)
        for k in ["ranked_df", "enriched_df", "live_candidate"]:
            st.session_state.pop(k, None)
        st.success("Cleared. Reload the page.")

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["📊 Rank & Enrich", "🎯 Live Candidate Test", "🤖 Agent Scoring", "✉️ Rejection Emails", "📈 Evaluation Metrics"]
)

# ---------------------------------------------------------------------------
# Tab 1 — Layer 1 (ML) + Layer 2 (NLP)
# ---------------------------------------------------------------------------
with tab1:
    st.subheader("Layers 1 + 2 — TF-IDF Ranking + spaCy Enrichment")
    st.caption("Uses the bundled sample dataset by default, or upload your own resumes/job-descriptions CSVs.")

    col1, col2 = st.columns(2)
    with col1:
        resumes_upload = st.file_uploader("Resumes CSV (optional)", type="csv", key="resumes_upload")
    with col2:
        jobs_upload = st.file_uploader("Job descriptions CSV (optional)", type="csv", key="jobs_upload")

    if st.button("▶ Run ML + NLP Pipeline", type="primary"):
        resumes = pd.read_csv(resumes_upload) if resumes_upload else pd.read_csv(RESUMES_CSV)
        jobs = pd.read_csv(jobs_upload) if jobs_upload else pd.read_csv(JOBS_CSV)
        jobs.to_csv(ACTIVE_JOBS_CSV, index=False)  # keeps Tab 3 in sync with whichever jobs were just used

        with st.spinner("Ranking candidates (TF-IDF + cosine similarity)..."):
            ranked = rank_all_jobs(jobs, resumes)
            ranked.to_csv(RANKED_CSV, index=False)

        with st.spinner("Extracting structured fields (regex + spaCy PhraseMatcher)..."):
            enriched_resumes = enrich_dataframe(resumes, jobs)
            enrich_cols = [
                "candidate_name", "email", "phone", "location",
                "degree_extracted", "university_extracted", "grad_year_extracted",
                "job_title_extracted", "company_extracted", "years_exp_extracted", "has_prior_experience",
                "certifications_extracted", "languages_extracted", "skills_extracted",
            ]
            enriched = ranked.merge(enriched_resumes[enrich_cols], on="candidate_name", how="left")
            enriched.to_csv(ENRICHED_CSV, index=False)

        cleared_old_scores = os.path.exists(SCORED_CSV)
        if cleared_old_scores:
            os.remove(SCORED_CSV)  # new dataset -> any previous agent-scoring results are for different candidates now

        st.session_state["ranked_df"] = ranked
        st.session_state["enriched_df"] = enriched
        msg = f"Done — ranked and enriched {len(enriched)} candidates across {jobs['role'].nunique()} jobs."
        if cleared_old_scores:
            msg += " Previous Agent Scoring results were cleared since the dataset changed — re-run Tab 3."
        st.success(msg)

    st.divider()
    with st.expander("📎 Or: upload resume PDFs directly + define one job posting"):
        st.caption(
            "For a real pile of PDF resumes instead of a CSV. All uploaded resumes are scored against the "
            "ONE job you define below. Processing this **overwrites** the ranked/enriched results above — "
            "switch back to the bundled/CSV data by re-running the pipeline up top."
        )
        pdf_resumes = st.file_uploader(
            "Resume PDFs (multiple allowed)", type="pdf", accept_multiple_files=True, key="pdf_batch_resumes"
        )

        st.markdown("**Job details**")
        jc1, jc2 = st.columns(2)
        with jc1:
            custom_role = st.text_input("Role title*", placeholder="e.g. Backend Engineer")
            custom_category = st.text_input(
                "Category*", placeholder="e.g. Backend Development",
                help="Just a grouping label for this job — doesn't need to match the bundled dataset's categories.",
            )
            custom_experience = st.text_input("Experience required", placeholder="e.g. 2-4 years")
        with jc2:
            custom_required_skills = st.text_input("Required skills* (comma-separated)", placeholder="e.g. Python, Django, PostgreSQL")
            custom_preferred_skills = st.text_input("Preferred skills (comma-separated)", placeholder="e.g. AWS, Docker")
            custom_education = st.text_input("Education required", placeholder="e.g. B.Tech in CS or related field")

        jd_mode = st.radio("Job description text:", ["Paste text", "Upload PDF"], horizontal=True, key="jd_input_mode")
        custom_jd_text = None
        if jd_mode == "Paste text":
            custom_jd_text = st.text_area("Paste the full job description", height=150, key="jd_paste")
        else:
            jd_pdf = st.file_uploader("Job description PDF", type="pdf", key="jd_pdf_upload")
            if jd_pdf:
                import pdfplumber

                with pdfplumber.open(jd_pdf) as pdf:
                    custom_jd_text = "\n".join((p.extract_text() or "") for p in pdf.pages)
                with st.expander("Extracted job description text (preview)"):
                    st.text(custom_jd_text[:2000] if custom_jd_text else "(no extractable text found in this PDF)")

        if st.button("▶ Process PDF Batch", type="primary"):
            if not pdf_resumes:
                st.error("Upload at least one resume PDF.")
            elif not custom_role or not custom_category or not custom_required_skills:
                st.error("Role, Category, and Required skills are required.")
            else:
                import pdfplumber
                from pipeline.nlp_enrichment import guess_candidate_name

                with st.spinner(f"Extracting text from {len(pdf_resumes)} resume PDF(s)..."):
                    rows = []
                    for f in pdf_resumes:
                        with pdfplumber.open(f) as pdf:
                            text = "\n".join((p.extract_text() or "") for p in pdf.pages)
                        fallback_name = os.path.splitext(f.name)[0].replace("_", " ").replace("-", " ").title()
                        rows.append({
                            "candidate_name": guess_candidate_name(text, fallback_name),
                            "category": custom_category,
                            "resume_text": text,
                            "skills": None,  # unknown up front — enrichment fills skills_extracted from the text itself
                        })
                    custom_resumes_df = pd.DataFrame(rows)

                    # de-duplicate candidate names (e.g. two PDFs both guessed "John Smith")
                    dupe_mask = custom_resumes_df["candidate_name"].duplicated(keep=False)
                    if dupe_mask.any():
                        suffix = custom_resumes_df.groupby("candidate_name").cumcount().add(1).astype(str)
                        custom_resumes_df.loc[dupe_mask, "candidate_name"] = (
                            custom_resumes_df.loc[dupe_mask, "candidate_name"] + " (" + suffix[dupe_mask] + ")"
                        )

                custom_jobs_df = pd.DataFrame([{
                    "role": custom_role,
                    "category": custom_category,
                    "required_skills": custom_required_skills,
                    "preferred_skills": custom_preferred_skills or "",
                    "experience_required": custom_experience or "0 years",
                    "education_required": custom_education or "",
                    "job_description": custom_jd_text or "",
                }])
                custom_jobs_df.to_csv(ACTIVE_JOBS_CSV, index=False)  # so Tab 3 (Agent Scoring) can find this job

                with st.spinner("Ranking candidates (TF-IDF + cosine similarity)..."):
                    ranked = rank_all_jobs(custom_jobs_df, custom_resumes_df)
                    ranked.to_csv(RANKED_CSV, index=False)

                with st.spinner("Extracting structured fields (regex + spaCy PhraseMatcher)..."):
                    enriched_resumes = enrich_dataframe(custom_resumes_df, custom_jobs_df)
                    enrich_cols = [
                        "candidate_name", "email", "phone", "location",
                        "degree_extracted", "university_extracted", "grad_year_extracted",
                        "job_title_extracted", "company_extracted", "years_exp_extracted", "has_prior_experience",
                        "certifications_extracted", "languages_extracted", "skills_extracted",
                    ]
                    enriched = ranked.merge(enriched_resumes[enrich_cols], on="candidate_name", how="left")
                    enriched.to_csv(ENRICHED_CSV, index=False)

                cleared_old_scores = os.path.exists(SCORED_CSV)
                if cleared_old_scores:
                    os.remove(SCORED_CSV)  # new dataset -> any previous agent-scoring results are for different candidates now

                st.session_state["ranked_df"] = ranked
                st.session_state["enriched_df"] = enriched
                msg = f"Done — processed {len(enriched)} resume(s) against '{custom_role}'."
                if cleared_old_scores:
                    msg += " Previous Agent Scoring results were cleared since the dataset changed — re-run Tab 3."
                st.success(msg)

    enriched_df = st.session_state.get("enriched_df")
    if enriched_df is None and os.path.exists(ENRICHED_CSV):
        enriched_df = pd.read_csv(ENRICHED_CSV)

    if enriched_df is not None and not enriched_df.empty:
        st.divider()
        job_roles = sorted(enriched_df["job_role"].dropna().unique())
        selected_role = st.selectbox("View top candidates for:", job_roles)
        top_n = st.slider("How many to show", 3, 20, 5)
        subset = enriched_df[enriched_df["job_role"] == selected_role].sort_values("rank").head(top_n)
        display_cols = [c for c in [
            "rank", "candidate_name", "match_percent", "years_exp_extracted",
            "degree_extracted", "fit_level",
        ] if c in subset.columns]
        st.dataframe(subset[display_cols], width='stretch', hide_index=True)

        if "fit_level" in enriched_df.columns:
            st.divider()
            st.markdown("**Validation — does the TF-IDF score line up with the labeled `fit_level`?**")
            fig = px.box(
                enriched_df, x="fit_level", y="match_score",
                category_orders={"fit_level": ["high", "medium", "low"]},
                title="TF-IDF match score by ground-truth fit level",
            )
            st.plotly_chart(fig, width='stretch')
    else:
        st.info("Run the pipeline above to see ranked results.")

# ---------------------------------------------------------------------------
# Tab 2 — Live single-candidate test (paste text or upload a PDF)
# ---------------------------------------------------------------------------
with tab2:
    st.subheader("Test a single resume live")
    st.caption(
        "Section extraction (education/experience/certifications) works best on resumes following the "
        "Name/Email/Phone + EDUCATION/SKILLS/EXPERIENCE section template. Skill detection works on any text."
    )

    if not os.path.exists(JOBS_CSV) or not os.path.exists(RESUMES_CSV):
        st.warning("Bundled data/resumes.csv and data/job_descriptions.csv are required for this tab.")
    else:
        jobs_df = pd.read_csv(JOBS_CSV)
        resumes_df = pd.read_csv(RESUMES_CSV)

        selected_job_role = st.selectbox("Job to test against:", sorted(jobs_df["role"].unique()), key="live_job_select")
        job_row = jobs_df[jobs_df["role"] == selected_job_role].iloc[0].to_dict()

        input_mode = st.radio("Resume input:", ["Paste text", "Upload PDF"], horizontal=True)
        resume_text = None
        if input_mode == "Paste text":
            resume_text = st.text_area("Paste resume text here", height=220)
        else:
            pdf_file = st.file_uploader("Upload a resume PDF", type="pdf")
            if pdf_file:
                import pdfplumber

                with pdfplumber.open(pdf_file) as pdf:
                    resume_text = "\n".join((page.extract_text() or "") for page in pdf.pages)
                with st.expander("Extracted text (preview)"):
                    st.text(resume_text[:3000] if resume_text else "(no extractable text found in this PDF)")

        if resume_text and st.button("Score this candidate", type="primary"):
            with st.spinner("Computing TF-IDF match + extracting fields..."):
                match_pct = score_single_resume(resume_text, job_row, resumes_df)
                nlp, has_full_model = load_nlp()
                vocab = build_skill_vocab(resumes_df, jobs_df)
                matcher, lookup = build_skill_matcher(nlp, vocab)
                extracted = enrich_resume_row(resume_text, nlp, matcher, lookup, has_full_model)

            st.session_state["live_candidate"] = {
                "resume_text": resume_text, "job_row": job_row,
                "match_pct": match_pct, "extracted": extracted,
            }

        live = st.session_state.get("live_candidate")
        if live:
            st.divider()
            col1, col2 = st.columns([1, 2])
            with col1:
                st.metric("TF-IDF Match Score", f"{live['match_pct']}%")
            with col2:
                shown = {k: v for k, v in live["extracted"].items() if k != "skills_extracted"}
                st.json(shown)
            st.write("**Skills detected:**", ", ".join(live["extracted"]["skills_extracted"]) or "none found")

            st.divider()
            st.markdown("**Optional: score this candidate with the CrewAI agent pipeline (uses real Groq credits)**")
            if not os.getenv("GROQ_API_KEY"):
                st.warning("Add your GROQ_API_KEY in the sidebar first.")
            elif st.button("🤖 Run AI Agent Scoring on this candidate"):
                candidate_payload = {
                    "candidate_name": "Live Test Candidate",
                    "degree_extracted": live["extracted"]["degree_extracted"],
                    "years_exp_extracted": live["extracted"]["years_exp_extracted"],
                    "skills_extracted": live["extracted"]["skills_extracted"],
                    "certifications_extracted": live["extracted"]["certifications_extracted"],
                    "match_percent": live["match_pct"],
                }
                job_payload = {
                    "role": live["job_row"].get("role"),
                    "category": live["job_row"].get("category"),
                    "required_skills": live["job_row"].get("required_skills", ""),
                    "preferred_skills": live["job_row"].get("preferred_skills", ""),
                    "experience_required": live["job_row"].get("experience_required", ""),
                    "education_required": live["job_row"].get("education_required", ""),
                    "job_description": live["job_row"].get("job_description", ""),
                }
                with st.spinner("Running Screening → Scoring → Feedback → Bias-Check agents..."):
                    try:
                        llm = crew_agents.get_llm()
                        agents = crew_agents.build_agents(llm)
                        result = run_agents.run_pipeline_for_candidate(agents, candidate_payload, job_payload)
                    except Exception as e:  # noqa: BLE001
                        result = {"agent_error": str(e)}

                if result.get("agent_error"):
                    st.error(f"Agent pipeline failed: {result['agent_error']}")
                else:
                    st.metric("AI Fit Score", result["fit_score"])
                    st.write("**Screening:**", result["screening_summary"])
                    st.write("**Scoring justification:**", result["scoring_justification"])
                    st.write("**Internal feedback:**", result["internal_feedback"])
                    st.write("**Candidate-facing feedback:**", result["final_candidate_feedback"])

# ---------------------------------------------------------------------------
# Tab 3 — Layer 3 (CrewAI agents, batch)
# ---------------------------------------------------------------------------
with tab3:
    st.subheader("Layer 3 — CrewAI Multi-Agent Scoring (Groq)")

    if not os.path.exists(ENRICHED_CSV):
        st.warning("Run the ML + NLP pipeline in the first tab first.")
    else:
        enriched_df = parse_list_columns(pd.read_csv(ENRICHED_CSV))

        active_jobs_path = ACTIVE_JOBS_CSV if os.path.exists(ACTIVE_JOBS_CSV) else JOBS_CSV
        active_roles = pd.read_csv(active_jobs_path)["role"].tolist()
        st.caption(f"Scoring against: **{', '.join(active_roles)}** (from the most recent Tab 1 run)")

        max_per_job = st.slider("Candidates to score per job (cost control)", 1, 20, 3)
        process_all = st.checkbox("Process ALL candidates instead (slower, costs more)")

        if not os.getenv("GROQ_API_KEY"):
            st.warning("Add your GROQ_API_KEY in the sidebar to run this.")
        elif st.button("▶ Run Agent Scoring", type="primary"):
            done = run_agents.already_processed(SCORED_CSV)
            jobs_df = pd.read_csv(ACTIVE_JOBS_CSV) if os.path.exists(ACTIVE_JOBS_CSV) else pd.read_csv(JOBS_CSV)
            jobs_by_category = {row["category"]: row.to_dict() for _, row in jobs_df.iterrows()}

            to_process = []
            for category, group in enriched_df.groupby("category"):
                job = jobs_by_category.get(category)
                if job is None:
                    continue
                group_sorted = group.sort_values("match_score", ascending=False)
                if not process_all:
                    group_sorted = group_sorted.head(max_per_job)
                for _, row in group_sorted.iterrows():
                    if (row["candidate_name"], job["role"]) not in done:
                        to_process.append((row, job))

            if not to_process:
                st.info("Nothing new to process — already scored, or no candidates match a posted job.")
            else:
                llm = crew_agents.get_llm()
                agents = crew_agents.build_agents(llm)

                progress = st.progress(0.0, text="Starting...")
                results = []
                for i, (row, job) in enumerate(to_process, start=1):
                    candidate_payload = run_agents.build_candidate_payload(row)
                    job_payload = run_agents.build_job_payload(job)
                    progress.progress(
                        i / len(to_process),
                        text=f"[{i}/{len(to_process)}] {candidate_payload['candidate_name']} → {job_payload['role']}",
                    )
                    result = run_agents.run_pipeline_for_candidate(agents, candidate_payload, job_payload)
                    record = {
                        "candidate_name": row["candidate_name"],
                        "email": row.get("email"),
                        "job_role": job["role"],
                        "category": row["category"],
                        "match_percent": row.get("match_percent"),
                        "fit_level": row.get("fit_level"),
                        **result,
                    }
                    record["status"] = (
                        "rejected"
                        if (record["fit_score"] is not None and record["fit_score"] < run_agents.REJECTION_THRESHOLD)
                        else "shortlisted" if record["fit_score"] is not None else "error"
                    )
                    results.append(record)

                new_df = pd.DataFrame(results)
                if os.path.exists(SCORED_CSV):
                    existing = pd.read_csv(SCORED_CSV)
                    new_keys = set(zip(new_df["candidate_name"], new_df["job_role"]))
                    existing = existing[
                        ~existing.apply(lambda r: (r["candidate_name"], r["job_role"]) in new_keys, axis=1)
                    ]
                    combined = pd.concat([existing, new_df], ignore_index=True)
                else:
                    combined = new_df
                combined.to_csv(SCORED_CSV, index=False)
                progress.progress(1.0, text="Done.")
                st.success(f"Scored {len(results)} candidate/job pair(s).")

        if os.path.exists(SCORED_CSV):
            st.divider()
            scored_df = pd.read_csv(SCORED_CSV)
            display_cols = [c for c in [
                "candidate_name", "job_role", "fit_score", "status", "scoring_justification",
            ] if c in scored_df.columns]
            st.dataframe(scored_df[display_cols], width='stretch', hide_index=True)

# ---------------------------------------------------------------------------
# Tab 4 — Layer 4 (rejection email dispatch)
# ---------------------------------------------------------------------------
with tab4:
    st.subheader("Layer 4 — Rejection Email Dispatch")
    st.caption(
        "The dataset's emails are all @example.com/.net/.org (IANA-reserved, never deliver to real inboxes) — "
        "safe to test with --send. Set TEST_OVERRIDE_EMAIL in .env to redirect real sends to your own inbox."
    )

    if not os.path.exists(SCORED_CSV):
        st.warning("Run agent scoring in the previous tab first.")
    else:
        scored_df = pd.read_csv(SCORED_CSV)
        for col in ["email_sent", "email_sent_at", "email_error"]:
            if col not in scored_df.columns:
                scored_df[col] = None
            else:
                scored_df[col] = scored_df[col].astype("object")

        pending = send_rejections.get_pending_rejections(scored_df)
        st.write(f"**{len(pending)} pending rejection email(s)**")

        test_override = os.getenv("TEST_OVERRIDE_EMAIL")

        if not pending.empty:
            with st.expander(f"Preview messages ({min(len(pending), 10)} of {len(pending)} shown)"):
                preview_config = {
                    "test_override_email": test_override,
                    "from_name": os.getenv("FROM_NAME", "Recruiting Team"),
                    "from_email": os.getenv("FROM_EMAIL", "your_email@example.com"),
                }
                for _, row in pending.head(10).iterrows():
                    _, to_address, subject, body = send_rejections.build_message(row, preview_config)
                    if test_override:
                        st.markdown(
                            f"**Candidate:** {row['candidate_name']} ({row['email']}) — "
                            f"redirected to **{to_address}** because TEST_OVERRIDE_EMAIL is active  \n"
                            f"**Subject:** {subject}"
                        )
                    else:
                        st.markdown(f"**To:** {to_address} ({row['candidate_name']})  \n**Subject:** {subject}")
                    st.text(body)
                    st.divider()

            confirm = st.checkbox("I understand this will send real emails to these addresses.")
            if st.button("✉️ Send Rejection Emails", disabled=not confirm, type="primary"):
                try:
                    smtp_config = send_rejections.load_smtp_config()
                except RuntimeError as e:
                    st.error(str(e))
                else:
                    server = smtplib.SMTP(smtp_config["host"], smtp_config["port"])
                    server.starttls()
                    server.login(smtp_config["username"], smtp_config["password"])

                    sent, failed = 0, 0
                    progress = st.progress(0.0)
                    try:
                        for i, (idx, row) in enumerate(pending.iterrows(), start=1):
                            msg, to_address, _, _ = send_rejections.build_message(row, smtp_config)
                            try:
                                server.send_message(msg)
                                scored_df.loc[idx, "email_sent"] = True
                                scored_df.loc[idx, "email_sent_at"] = datetime.now(timezone.utc).isoformat()
                                scored_df.loc[idx, "email_error"] = None
                                sent += 1
                            except Exception as e:  # noqa: BLE001
                                scored_df.loc[idx, "email_sent"] = False
                                scored_df.loc[idx, "email_error"] = str(e)
                                failed += 1
                            progress.progress(i / len(pending))
                    finally:
                        server.quit()

                    scored_df.to_csv(SCORED_CSV, index=False)
                    st.success(f"Sent: {sent}, Failed: {failed}")
        else:
            st.info("No pending rejections to send.")

# ---------------------------------------------------------------------------
# Tab 5 — Evaluation Metrics
# ---------------------------------------------------------------------------
with tab5:
    st.subheader("Evaluation Metrics")
    st.caption(
        "Metrics from the project proposal, computed against the bundled dataset's ground-truth `fit_level` "
        "labels. Only meaningful on labeled data (the bundled dataset) — a custom PDF batch of real resumes "
        "has no ground truth to check against."
    )

    # --- Metric 1: Precision@5 (Scoring Layer) ---
    st.markdown("### Precision@5 — Scoring Layer")
    st.caption("Of the top 5 ranked candidates per job, the fraction that are actually labeled 'high' fit. Target: **> 0.60**")

    if not os.path.exists(RANKED_CSV):
        st.warning("Run the ML + NLP pipeline in Tab 1 first.")
    else:
        ranked_df = pd.read_csv(RANKED_CSV)
        if "fit_level" not in ranked_df.columns:
            st.info(
                "This dataset has no `fit_level` ground-truth column, so Precision@5 can't be computed — "
                "this metric only works on the bundled demo dataset. A custom PDF batch of real resumes has "
                "nothing to check against."
            )
        else:
            p5 = precision_at_k(ranked_df, k=5, score_col="match_score", group_col="job_role")
            col1, col2 = st.columns(2)
            with col1:
                st.metric(
                    "Overall Precision@5",
                    f"{p5['overall_precision']:.2f}",
                    delta=f"{p5['overall_precision'] - p5['target']:+.2f} vs target",
                )
            with col2:
                st.metric("Meets target (> 0.60)?", "✅ Yes" if p5["meets_target"] else "At target / below")

            per_job_df = pd.DataFrame([{"job_role": k, "precision_at_5": v} for k, v in p5["per_job_precision"].items()])
            fig = px.bar(per_job_df, x="job_role", y="precision_at_5", title="Precision@5 by job", range_y=[0, 1])
            fig.add_hline(y=0.60, line_dash="dash", annotation_text="target: 0.60")
            st.plotly_chart(fig, width="stretch")

    st.divider()

    # --- Metric 2: LLM-as-Judge Score (Agent Layer) ---
    st.markdown("### LLM-as-Judge Score — Agent Layer")
    st.caption(
        "A second, independent LLM call rates each candidate's internal feedback on Relevance, Accuracy, "
        "Clarity, and Actionability (1-5 each). Target: **> 3.5 / 5**. Uses real Groq API credits — one call "
        "per row evaluated, so cap the sample size below."
    )

    if not os.path.exists(SCORED_CSV):
        st.warning("Run Agent Scoring in Tab 3 first — this needs generated feedback to evaluate.")
    else:
        scored_df = pd.read_csv(SCORED_CSV)
        evaluable = scored_df.dropna(subset=["internal_feedback"]) if "internal_feedback" in scored_df.columns else pd.DataFrame()

        if evaluable.empty:
            st.info("No scored candidates with feedback text yet — run Agent Scoring in Tab 3 first.")
        elif not os.getenv("GROQ_API_KEY"):
            st.warning("Add your GROQ_API_KEY in the sidebar to run this.")
        else:
            sample_n = st.slider(
                "How many rows to evaluate (cost control)", 1, min(50, len(evaluable)), min(10, len(evaluable))
            )
            if st.button("▶ Run LLM-as-Judge Evaluation", type="primary"):
                with st.spinner(f"Judging {sample_n} feedback message(s)..."):
                    st.session_state["judge_result"] = judge_feedback_quality(
                        evaluable, feedback_col="internal_feedback", sample_size=sample_n
                    )

            judge_result = st.session_state.get("judge_result")
            if judge_result:
                if judge_result["average_score"] is None:
                    st.error("No valid judge responses were returned — check the errors in the results table below.")
                else:
                    c1, c2, c3 = st.columns(3)
                    c1.metric(
                        "Average Score", f"{judge_result['average_score']:.2f} / 5",
                        delta=f"{judge_result['average_score'] - judge_result['target']:+.2f} vs target",
                    )
                    c2.metric("Meets target (> 3.5)?", "✅ Yes" if judge_result["meets_target"] else "At target / below")
                    c3.metric("Evaluated / Failed", f"{judge_result['n_evaluated']} / {judge_result['n_failed']}")

                    dim_df = pd.DataFrame([{"dimension": k, "score": v} for k, v in judge_result["average_by_dimension"].items()])
                    fig = px.bar(dim_df, x="dimension", y="score", title="Average score by dimension", range_y=[0, 5])
                    fig.add_hline(y=3.5, line_dash="dash", annotation_text="target: 3.5")
                    st.plotly_chart(fig, width="stretch")

                st.dataframe(judge_result["results"], width="stretch", hide_index=True)

    st.divider()

    # --- Supplementary metrics ---
    with st.expander("📎 Supplementary metrics (extra depth beyond the proposal)"):
        if os.path.exists(RANKED_CSV):
            ranked_df = pd.read_csv(RANKED_CSV)
            if "fit_level" in ranked_df.columns:
                st.markdown("**ML Layer — TF-IDF match_score vs fit_level**")
                ml_eval = evaluate_ranking_vs_fit_level(ranked_df, score_col="match_score", group_col="job_role")
                c1, c2, c3 = st.columns(3)
                c1.metric("Avg. Spearman correlation", f"{ml_eval['average_correlation']:.2f}")
                c2.metric("Tier accuracy", f"{ml_eval['tier_accuracy'] * 100:.1f}%")
                c3.metric("Tier macro F1", f"{ml_eval['tier_macro_f1']:.2f}")
                st.text(ml_eval["classification_report"])

        if os.path.exists(ENRICHED_CSV):
            enriched_df = parse_list_columns(pd.read_csv(ENRICHED_CSV))
            st.markdown("**NLP Layer — extraction completeness**")
            st.dataframe(evaluate_extraction_completeness(enriched_df), width="stretch", hide_index=True)

            acc_df = evaluate_extraction_accuracy(enriched_df)
            if not acc_df.empty:
                st.markdown("**NLP Layer — extraction accuracy (bundled dataset only)**")
                st.dataframe(acc_df, width="stretch", hide_index=True)

        if os.path.exists(SCORED_CSV):
            scored_df = pd.read_csv(SCORED_CSV)
            if "fit_level" in scored_df.columns:
                agent_eval_df = scored_df.dropna(subset=["fit_score", "fit_level"])
                if not agent_eval_df.empty:
                    st.markdown("**Agent Layer — fit_score vs fit_level**")
                    agent_eval = evaluate_ranking_vs_fit_level(agent_eval_df, score_col="fit_score", group_col="job_role")
                    c1, c2 = st.columns(2)
                    c1.metric(
                        "Avg. Spearman correlation",
                        f"{agent_eval['average_correlation']:.2f}" if agent_eval["average_correlation"] is not None else "N/A",
                    )
                    c2.metric(
                        "Tier accuracy",
                        f"{agent_eval['tier_accuracy'] * 100:.1f}%" if agent_eval["tier_accuracy"] is not None else "N/A",
                    )

            st.markdown("**ML vs Agent score agreement**")
            ml_vs_agent = compare_ml_vs_agent(scored_df)
            if ml_vs_agent["correlation"] is not None:
                st.metric(
                    "Spearman correlation (match_percent vs fit_score)",
                    f"{ml_vs_agent['correlation']:.2f}",
                    help=f"n={ml_vs_agent['n']}",
                )
            else:
                st.info("Not enough data yet to compute this.")
